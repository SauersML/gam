//! gam#3452 acceptance: on the estimated-law route a calibrated Bernoulli
//! marginal-slope fit's generated-regressor-corrected covariance must match the
//! sampling variance of its own coefficients, from both sides.
//!
//! The fixture is gam#3030's receipt with its outcome drawn from the model an
//! estimated-law fit contains. Covariates are held fixed; each replicate redraws
//! the score and the outcome:
//!
//! * `z = 0.5·x₁ + 0.3·x₂ + e` with `Var e = 1`, so the calibrated score is `ζ = e`;
//! * `P(Y = 1 | x, ζ) = Φ(a(q(x)) + s·ζ)` with `q(x) = −0.5 + 0.4·x₁ − 0.3·x₂`,
//!   `s = 1.880`, and `a` the root of `Σ_c π_c Φ(a/√(1 + s²σ_c²)) = Φ(q)` under the
//!   TRUE law of `e`, so the estimated-law model is correctly specified.
//!
//! Two arms:
//!
//! 1. **Heavy-tailed.** `e` is a Gaussian scale mixture (`σ² = 0.5286` w.p. 0.9,
//!    `5.243` w.p. 0.1; `κ₄ = 9`). Every fit anchors on the equal-mass empirical
//!    grid of the calibrated score (`GlobalEmpirical`), whose nodes are
//!    standardized, so a first-stage shift or rescale moves the rows AND the grid.
//!    Before gam#3452 the correction moved only the rows against a fixed grid and
//!    left out the grid's own sampling error; the intercept's corrected variance
//!    stood 41% above its sampling variance (outside the 99.9% band at B = 400).
//! 2. **Gaussian control.** `e ~ N(0, 1)`; the fit anchors on the closed form, where
//!    the correction was already exact. The arm pins that the fix leaves it so.
//!
//! Each coefficient's mean reported variance must lie inside the two-sided
//! Wilson–Hilferty χ² band of the replicates' empirical variance, Bonferroni over
//! the coefficients at α = 10⁻³. The band fails a variance that is too large
//! exactly as it fails one too small.

use csv::StringRecord;
use gam::families::bms::LatentMeasureKind;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use gam::utils::splitmix64;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

fn next_unit(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

fn next_gauss(state: &mut u64) -> f64 {
    let u1 = next_unit(state);
    let u2 = next_unit(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn normal_cdf(x: f64) -> f64 {
    gam::probability::normal_cdf(x)
}

const N: usize = 2_000;
const SLOPE: f64 = 1.880;
const REPLICATES: usize = 400;

/// `(weight, variance)` of each normal component of `e`, a unit-variance law.
#[derive(Clone, Copy, Debug)]
enum Law {
    Gaussian,
    ScaleMixture,
}

impl Law {
    fn components(self) -> &'static [(f64, f64)] {
        match self {
            Law::Gaussian => &[(1.0, 1.0)],
            Law::ScaleMixture => &[(0.9, 0.5286), (0.1, 5.243)],
        }
    }

    fn draw(self, state: &mut u64) -> f64 {
        let pick = next_unit(state);
        let mut cumulative = 0.0;
        let components = self.components();
        let &(_, variance) = components
            .iter()
            .find(|&&(weight, _)| {
                cumulative += weight;
                pick < cumulative
            })
            .unwrap_or_else(|| components.last().expect("a component"));
        variance.sqrt() * next_gauss(state)
    }

    /// The anchor `a` with `Σ_c π_c Φ(a/√(1 + s²σ_c²)) = Φ(q)`, by bisection on a
    /// strictly increasing function.
    fn anchor(self, q: f64) -> f64 {
        let target = normal_cdf(q);
        let marginal = |a: f64| -> f64 {
            self.components()
                .iter()
                .map(|&(weight, variance)| {
                    weight * normal_cdf(a / (1.0 + SLOPE * SLOPE * variance).sqrt())
                })
                .sum()
        };
        let (mut low, mut high) = (-40.0_f64, 40.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if marginal(mid) < target {
                low = mid;
            } else {
                high = mid;
            }
        }
        0.5 * (low + high)
    }
}

/// The held covariates and their anchors under `law`.
fn design(law: Law) -> Vec<(f64, f64, f64)> {
    let mut state = 0x3452_C0DE_0000_0001;
    (0..N)
        .map(|_| {
            let x1 = next_gauss(&mut state);
            let x2 = next_gauss(&mut state);
            let q = -0.5 + 0.4 * x1 - 0.3 * x2;
            (x1, x2, law.anchor(q))
        })
        .collect()
}

fn replicate(
    law: Law,
    design: &[(f64, f64, f64)],
    state: &mut u64,
) -> gam::inference::data::EncodedDataset {
    let rows = design
        .iter()
        .map(|&(x1, x2, anchor)| {
            let e = law.draw(state);
            let z = 0.5 * x1 + 0.3 * x2 + e;
            let y = u8::from(next_unit(state) < normal_cdf(anchor + SLOPE * e));
            StringRecord::from(vec![
                y.to_string(),
                z.to_string(),
                x1.to_string(),
                x2.to_string(),
            ])
        })
        .collect();
    let headers = ["y", "z", "x1", "x2"].map(String::from).to_vec();
    encode_recordswith_inferred_schema(headers, rows).expect("encode the #3452 fixture")
}

fn config() -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        z_column: Some("z".to_string()),
        slope_formula: Some("1".to_string()),
        ..FitConfig::default()
    }
}

/// Wilson–Hilferty quantile of `χ²_k / k` at standard-normal quantile `z`.
fn chi2_over_df(k: f64, z: f64) -> f64 {
    let c = 2.0 / (9.0 * k);
    (1.0 - c + z * c.sqrt()).powi(3)
}

/// Standard-normal upper quantile by bisection on the library's CDF.
fn upper_normal_quantile(tail: f64) -> f64 {
    let (mut low, mut high) = (0.0_f64, 10.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if 1.0 - normal_cdf(mid) > tail {
            low = mid;
        } else {
            high = mid;
        }
    }
    0.5 * (low + high)
}

fn assert_covariance_matches_sampling_variance(law: Law, seed: u64) {
    init_parallelism();
    let design = design(law);
    // Each replicate draws from its own stream, so the replicates are
    // independent of the order the pool runs them in.
    let fits: Vec<(Vec<f64>, Vec<f64>, bool)> = (0..REPLICATES)
        .into_par_iter()
        .map(|r| {
            let mut state = seed ^ (r as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let data = replicate(law, &design, &mut state);
            let result = fit_from_formula("y ~ x1 + x2", &data, &config())
                .unwrap_or_else(|e| panic!("gam#3452 ({law:?}) replicate {r}: fit failed: {e}"));
            let FitResult::BernoulliMarginalSlope(fit) = result else {
                panic!("expected a BernoulliMarginalSlope fit");
            };
            assert!(
                fit.latent_z_conditional_calibration.is_some(),
                "gam#3452 ({law:?}) replicate {r}: the score's mean moves with x, so the fit must \
                 calibrate it and correct its covariance for that first stage; law consumed: {}",
                fit.latent_law_consumed.label()
            );
            let on_grid = matches!(
                fit.latent_measure,
                LatentMeasureKind::GlobalEmpirical { .. }
            );
            if matches!(law, Law::ScaleMixture) {
                assert!(
                    on_grid,
                    "gam#3452 heavy-tailed replicate {r}: the fit must anchor on the empirical \
                     grid (the route under test); law consumed: {}",
                    fit.latent_law_consumed.label()
                );
            }
            let covariance = fit
                .fit
                .beta_covariance()
                .unwrap_or_else(|| panic!("gam#3452 ({law:?}) replicate {r}: no covariance"));
            let p = fit.fit.beta.len();
            assert_eq!(covariance.dim(), (p, p));
            (
                fit.fit.beta.to_vec(),
                (0..p).map(|j| covariance[[j, j]]).collect(),
                on_grid,
            )
        })
        .collect();
    let empirical_fits = fits.iter().filter(|(_, _, on_grid)| *on_grid).count();
    let (betas, reported): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
        fits.into_iter().map(|(beta, variance, _)| (beta, variance)).unzip();
    let p = betas[0].len();
    assert!(betas.iter().all(|b| b.len() == p));
    // Two-sided, Bonferroni over the coefficients at α = 10⁻³.
    let z_tail = upper_normal_quantile(1.0e-3 / (2.0 * p as f64));
    let k = (REPLICATES - 1) as f64;
    let mut failures = Vec::new();
    for j in 0..p {
        let mean = betas.iter().map(|b| b[j]).sum::<f64>() / REPLICATES as f64;
        let sampling =
            betas.iter().map(|b| (b[j] - mean).powi(2)).sum::<f64>() / (REPLICATES - 1) as f64;
        let predicted = reported.iter().map(|v| v[j]).sum::<f64>() / REPLICATES as f64;
        // `sampling · k / σ²` is `χ²_k` at the true variance `σ²`, so the band on
        // the sampling variance around the reported one is `predicted · χ²_k/k`.
        let low = predicted * chi2_over_df(k, -z_tail);
        let high = predicted * chi2_over_df(k, z_tail);
        eprintln!(
            "[3452 {law:?}] coefficient {j}: sampling var {sampling:.4e}, mean reported \
             {predicted:.4e} (ratio reported/sampling {:.4}), band on sampling var \
             [{low:.4e}, {high:.4e}]",
            predicted / sampling
        );
        if !(sampling >= low && sampling <= high) {
            failures.push(format!(
                "coefficient {j}: sampling variance {sampling:.4e} outside [{low:.4e}, {high:.4e}] \
                 around the mean reported {predicted:.4e}"
            ));
        }
    }
    eprintln!("[3452 {law:?}] {empirical_fits}/{REPLICATES} fits on the empirical grid");
    assert!(
        failures.is_empty(),
        "gam#3452 ({law:?}): the corrected covariance does not match the coefficients' sampling \
         variance (z = {z_tail:.4}, {REPLICATES} replicates):\n  {}",
        failures.join("\n  ")
    );
}

#[test]
fn heavy_tailed_score_corrected_covariance_matches_sampling_variance_3452() {
    assert_covariance_matches_sampling_variance(Law::ScaleMixture, 0x3452_4EA7_0000_0001);
}

#[test]
fn gaussian_score_corrected_covariance_matches_sampling_variance_3452() {
    assert_covariance_matches_sampling_variance(Law::Gaussian, 0x3452_6A55_0000_0001);
}
