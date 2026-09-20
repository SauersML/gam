//! A rare-event binomial GAM on raw covariates fits with its smoothing-corrected
//! covariance (pyGAM audit F17).
//!
//! `y ~ factor(student) + s(balance) + s(income)` on the ISLR `default` data
//! (n = 2000, about 3% positives, raw covariates) used to certify an outer
//! minimum and then refuse the smoothing-corrected covariance, because the rho
//! Hessian had negative curvature. That Hessian omitted `∂²Δ_b` of the #784
//! block-local correction the criterion carries.
//!
//! Two defects stood behind it. First, the block took its eigenpairs from an
//! `eigh` of the assembled `H`. The balance smooth is penalised onto its rail
//! (`λ ≈ 1e13`), so `‖H‖ ≈ 9e12` while the soft modes the block lives on have
//! curvature `0.29` and `1.35`; that eigensolve resolves each eigenvalue only
//! to `O(ε·‖H‖) ≈ 2e-3`, and `Δ_b` changed from `6.062e-3` to `6.142e-3`
//! between two evaluations at the same rho. The block now reads the
//! criterion's own root-scale (#2644) eigensystem. Second, `Δ_b` had no
//! ρ-Hessian at all, so the corrected criterion's search ran on BFGS
//! curvature and its covariance had nothing exact to invert. `Δ_b` now carries
//! its exact analytic ρ-Hessian (`block_correction_hessian`), so the search
//! steps on the criterion's own curvature and the smoothing-corrected
//! covariance inverts it.
//!
//! The fixture is a synthetic analogue of `default`: the same covariate
//! scales, the same class balance and the same factor-by-income structure.
//! The cases are the (seed, formula) pairs of seeds 1 to 10 that stalled
//! before the block read the criterion's eigensystem (`|Pg|` from `4.4e-5` to
//! `5.6e-3` against the `3e-5` bound), and the pairs that then certified
//! without a smoothing-corrected covariance because `Δ_b` had no ρ-Hessian.
//! The two-smooth model is included because it is the minimal failing cell.

use csv::StringRecord;
use gam::{FitConfig, encode_recordswith_inferred_schema, init_parallelism};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// `n` rows: `student ~ Bernoulli(0.3)`, `balance ~ N(835, 480)` clipped at
/// zero, `income ~ N(17500, 4500)` for students and `N(40000, 10000)`
/// otherwise, clipped at 700, and
/// `logit P(y = 1) = −10.8 + 0.0057·balance − 0.65·student` — the scales and
/// the roughly 3% prevalence of `default`, on raw covariates.
fn default_like(seed: u64, n: usize) -> (gam::data::EncodedDataset, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");
    let balance_law = Normal::<f64>::new(835.0, 480.0).expect("balance law");
    let student_income = Normal::<f64>::new(17_500.0, 4_500.0).expect("student income law");
    let other_income = Normal::<f64>::new(40_000.0, 10_000.0).expect("income law");
    let mut rows: Vec<StringRecord> = Vec::with_capacity(n);
    let mut positives = 0usize;
    for _ in 0..n {
        let student = unit.sample(&mut rng) < 0.3;
        let balance = balance_law.sample(&mut rng).max(0.0);
        let income = if student {
            student_income.sample(&mut rng)
        } else {
            other_income.sample(&mut rng)
        }
        .max(700.0);
        let eta = -10.8 + 0.0057 * balance - if student { 0.65 } else { 0.0 };
        let y = unit.sample(&mut rng) < 1.0 / (1.0 + (-eta).exp());
        positives += usize::from(y);
        rows.push(StringRecord::from(vec![
            if student { "Yes" } else { "No" }.to_string(),
            balance.to_string(),
            income.to_string(),
            if y { "1" } else { "0" }.to_string(),
        ]));
    }
    let headers = ["student", "balance", "income", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode the fixture");
    (data, positives as f64 / n as f64)
}

#[test]
fn rare_event_binomial_on_raw_covariates_fits() {
    // Registers the Laplace marginal corrector; without it the #784 correction
    // declines before its diagnostic and the continuation is never exercised.
    init_parallelism();
    let config = FitConfig {
        family: Some("binomial-logit".to_string()),
        ..FitConfig::default()
    };
    const FACTOR_MODEL: &str = "y ~ factor(student) + s(balance) + s(income)";
    const ADDITIVE_MODEL: &str = "y ~ s(balance) + s(income)";
    for (seed, formula) in [
        (1_u64, FACTOR_MODEL),
        (2, FACTOR_MODEL),
        (7, FACTOR_MODEL),
        (9, FACTOR_MODEL),
        (1, ADDITIVE_MODEL),
        (2, ADDITIVE_MODEL),
        (5, ADDITIVE_MODEL),
        (6, ADDITIVE_MODEL),
        (7, ADDITIVE_MODEL),
        (9, ADDITIVE_MODEL),
        (10, ADDITIVE_MODEL),
    ] {
        let (data, prevalence) = default_like(seed, 2000);
        assert!(
            (0.01..0.06).contains(&prevalence),
            "seed {seed}: the fixture is a rare-event cell, prevalence {prevalence:.3}"
        );
        let fit = gam::fit_from_formula(formula, &data, &config).unwrap_or_else(|error| {
            panic!(
                "seed {seed} `{formula}` (prevalence {prevalence:.3}) must fit. A search \
                 whose criterion carries the #784 correction can only certify when the \
                 correction is a function of rho, and its smoothing-corrected covariance \
                 exists only when the correction carries its exact rho-Hessian: {error}"
            )
        });
        let gam::FitResult::Standard(standard) = &fit else {
            panic!("seed {seed} `{formula}` is a standard GLM fit");
        };
        let score = standard.fit.reml_score().unwrap_or(f64::NAN);
        assert!(
            score.is_finite(),
            "seed {seed} `{formula}` minted a fit with no finite REML/LAML criterion"
        );
        let corrected = standard.fit.beta_covariance_corrected().unwrap_or_else(|| {
            panic!("seed {seed} `{formula}` shipped no smoothing-corrected covariance")
        });
        assert!(
            corrected.iter().all(|v| v.is_finite()),
            "seed {seed} `{formula}`: the smoothing-corrected covariance is not finite"
        );
    }
}
