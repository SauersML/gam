//! Standing type-I size gate (issue #1891): the Skovgaard modified directed
//! root `r*` corrected p-value (`gam-inference/src/skovgaard.rs`).
//!
//! The #1891 registry routes frequentist test p-values to a TEST-SIZE curve —
//! under a true null the empirical rejection rate at `α` must match `α` to
//! within Monte-Carlo error. An oversized test is the #1872/#1873
//! anti-conservative signature; an undersized one is just as miscalibrated.
//! The `wood_smooth_test_pvalue` surface is gated by the smooth-significance
//! null-FPR bug-hunt; this file is the DEDICATED size
//! curve for the Skovgaard `r*` p-value the registry noted as pending, driving
//! the exact production assembly `scalar_skovgaard_r_star`.
//!
//! Closed-form null anchor (the module's own certification family): the
//! Exponential in its MEAN parameterisation. For `y_i ~ Exp(mean μ)`,
//!   ℓ(μ) = −n·ln μ − Σy/μ,   μ̂ = ȳ,
//!   W  = 2n[ ln(μ₀/μ̂) + μ̂/μ₀ − 1 ]          (profile LR statistic),
//!   ĵ  = î = n/μ̂²                             (observed = Fisher info at μ̂),
//!   q̂  = cov_μ̂[U(μ̂), ℓ(μ̂) − ℓ(μ₀)] = n(1/μ₀ − 1/μ̂),
//! with the empirical companions `Î = Σ sᵢ²` and `q̂_emp = Σ sᵢ(ℓᵢ(μ̂) − ℓᵢ(μ₀))`,
//! `sᵢ = −1/μ̂ + yᵢ/μ̂²`. The mean is NOT the canonical parameter, so the
//! sample-space derivative `q̂` is not the linear surrogate `(μ̂−μ₀)·î`; this
//! is the parameterisation in which the pre-#3535 `u` broke invariance. The
//! correction is not trivial at finite n: `u = √n(μ̂/μ₀ − 1)` and the LR root
//! `r = sign·√W` differ, so `r* = r + log(u/r)/r ≠ r`. This is the small-sample
//! regime where the first-order root is miscalibrated and the third-order
//! `r*` restores calibration.
//!
//! Audit: the type-I size at `α ∈ {0.01, 0.05, 0.10}` is checked as coverage of
//! the NON-rejection event at nominal `1−α`, so the shared Wilson verdict
//! applies unchanged. Both tails gate (#3534): an oversized test under-covers
//! non-rejection and a conservative (undersized) test over-covers it. Every
//! replication enters the denominator: the Exponential sample is continuous, so
//! `μ̂ ≠ μ₀` and `W > 0` almost surely and the assembly never declines; a
//! decline is a panic, never a silently shrunk sample. Determinism: one fixed
//! seed threads every replication.

use gam_inference::skovgaard::{ScalarSkovgaardInput, scalar_skovgaard_r_star};
use gam_test_support::calibration::{CalibrationRng, audit_coverage};

/// Observations per simulated dataset. Small enough that the first-order LR /
/// Wald test is materially anti-conservative (so `r*` is doing real work) while
/// the closed-form ingredients stay well-conditioned.
const N_OBS: usize = 8;
/// Replications: at the tightest level `α = 0.01` the expected rejection count
/// is `N_REPLICATIONS·α = 40`. A never-rejecting test is detectable at nominal
/// `1 − α` once `R > z²(1 − α)/α` (657 at `α = 0.01` for the harness's `z`), so
/// both tails resolve at every swept level; a calibrated test trips either tail
/// with probability at most the harness's 1% false-positive rate.
const N_REPLICATIONS: usize = 4000;
/// The type-I error rates swept.
const ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];
/// The true (and null) mean — the null is exact, so any rejection is a type-I
/// error. Value is immaterial (the statistic is scale-equivariant); fixed for
/// reproducibility.
const TRUE_MEAN: f64 = 1.3;
const SEED: u64 = 0x1891_5C_07_A0_00;

/// One Exponential draw with mean `mean` from the harness's uniform stream:
/// `−ln(U)·mean`.
fn exp_draw(mean: f64, rng: &mut CalibrationRng) -> f64 {
    -rng.uniform_open01().ln() * mean
}

/// The Skovgaard ingredients for testing the mean `μ = μ₀` on one Exponential
/// sample, or `None` when the sample mean is not a positive finite number.
fn mean_exponential_input(ys: &[f64], mean_null: f64) -> Option<ScalarSkovgaardInput> {
    let n = ys.len() as f64;
    let mean_hat = ys.iter().sum::<f64>() / n;
    if !(mean_hat.is_finite() && mean_hat > 0.0) {
        return None;
    }
    // Profile LR statistic W = 2n[ ln(μ₀/μ̂) + μ̂/μ₀ − 1 ] ≥ 0.
    let lr = 2.0 * n * ((mean_null / mean_hat).ln() + mean_hat / mean_null - 1.0);
    let info = n / (mean_hat * mean_hat);
    let row_loglik = |mu: f64, y: f64| -mu.ln() - y / mu;
    let (empirical_info, empirical_loglik_covariance) =
        ys.iter()
            .fold((0.0_f64, 0.0_f64), |(info_acc, cov_acc), &y| {
                let score = -1.0 / mean_hat + y / (mean_hat * mean_hat);
                let diff = row_loglik(mean_hat, y) - row_loglik(mean_null, y);
                (info_acc + score * score, cov_acc + score * diff)
            });
    Some(ScalarSkovgaardInput {
        theta_hat: mean_hat,
        theta_null: mean_null,
        lr_statistic: lr,
        observed_info: info,
        expected_info: info,
        loglik_covariance: n * (1.0 / mean_null - 1.0 / mean_hat),
        empirical_info,
        empirical_loglik_covariance,
    })
}

/// The Skovgaard corrected (model-form) two-sided p-value for testing the mean
/// `μ = μ₀` on one simulated Exponential sample.
///
/// Every replication reaches the audit's denominator. The sample is continuous,
/// so `μ̂` is a positive finite mean and `μ̂ ≠ μ₀` almost surely, which makes
/// `W > 0` and leaves the assembly nothing to decline. Either decline is
/// therefore a defect in the surface under test, not an outcome to skip: a
/// skipped replication would shrink the denominator without shrinking the
/// nominal level it is judged against.
fn corrected_p_value(mean_true: f64, mean_null: f64, rng: &mut CalibrationRng) -> f64 {
    let ys: Vec<f64> = (0..N_OBS).map(|_| exp_draw(mean_true, rng)).collect();
    let input = mean_exponential_input(&ys, mean_null)
        .unwrap_or_else(|| panic!("Exponential sample mean is not finite and positive: ys={ys:?}"));
    scalar_skovgaard_r_star(&input)
        .map(|res| res.p_value_corrected)
        .unwrap_or_else(|| {
            panic!(
                "Skovgaard assembly declined on a continuous Exponential sample \
                 (μ̂={}, μ₀={mean_null}, W={}, Î={})",
                input.theta_hat, input.lr_statistic, input.empirical_info
            )
        })
}

#[test]
fn skovgaard_rstar_corrected_pvalue_is_calibrated_under_the_null() {
    let mut rng = CalibrationRng::new(SEED);
    // Count NON-rejections at each α so the Wilson verdict (coverage of the
    // non-rejection event at nominal 1−α) applies directly.
    let mut non_rejections = [0usize; ALPHAS.len()];
    let mut correction_ever_material = false;

    for _ in 0..N_REPLICATIONS {
        let p = corrected_p_value(TRUE_MEAN, TRUE_MEAN, &mut rng);
        // A first-order LR root at n=8 would reject too often; confirm r* moves
        // the p-value away from the raw root often enough that this is a real
        // test of the correction, not a vacuous pass.
        if !(0.0..=1.0).contains(&p) {
            panic!("Skovgaard corrected p-value out of range: {p}");
        }
        for (idx, &alpha) in ALPHAS.iter().enumerate() {
            if p >= alpha {
                non_rejections[idx] += 1;
            }
        }
    }

    // Teeth witness: at least once the correction must be non-trivial (u ≠ r),
    // otherwise r* collapsed to r and this only re-tested the LR root.
    {
        let mut probe = CalibrationRng::new(SEED ^ 0x9E37_79B9);
        for _ in 0..64 {
            let ys: Vec<f64> = (0..N_OBS)
                .map(|_| exp_draw(TRUE_MEAN, &mut probe))
                .collect();
            if let Some(res) = mean_exponential_input(&ys, TRUE_MEAN)
                .and_then(|input| scalar_skovgaard_r_star(&input))
            {
                if (res.r_star - res.r).abs() > 1e-6 {
                    correction_ever_material = true;
                    break;
                }
            }
        }
    }
    assert!(
        correction_ever_material,
        "r* never differed from r — the mean-Exponential anchor is not exercising the \
         Barndorff-Nielsen correction, so this size curve has no teeth"
    );

    let mut failures = Vec::new();
    for (idx, &alpha) in ALPHAS.iter().enumerate() {
        let nominal = 1.0 - alpha;
        let verdict = audit_coverage(non_rejections[idx], N_REPLICATIONS, nominal);
        if !verdict.passed {
            let empirical_size = 1.0 - verdict.empirical;
            failures.push(format!(
                "α={alpha}: empirical size={empirical_size:.4}; non-rejection {}",
                verdict.describe()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "Skovgaard r* corrected p-value is miscalibrated under the null:\n{}",
        failures.join("\n")
    );
}
