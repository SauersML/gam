//! Standing type-I size gate (issue #1891): the Skovgaard modified directed
//! root `r*` corrected p-value (`gam-inference/src/skovgaard.rs`).
//!
//! The #1891 registry routes frequentist test p-values to a TEST-SIZE curve —
//! under a true null the empirical rejection rate at `α` must match `α` to
//! within Monte-Carlo error. An oversized test is the #1872/#1873
//! anti-conservative signature; an undersized one is just as miscalibrated. The `wood_smooth_test_pvalue` surface is gated by
//! the smooth-significance null-FPR bug-hunt; this file is the DEDICATED size
//! curve for the Skovgaard `r*` p-value the registry noted as pending, driving
//! the exact production assembly `scalar_skovgaard_r_star`.
//!
//! Closed-form null anchor (the module's own certification family): the
//! unit-rate Exponential. For `y_i ~ Exp(θ)` with rate `θ`,
//!   ℓ(θ) = n·ln θ − θ·Σy,   θ̂ = n/Σy,
//!   W  = 2n[ ln(θ̂/θ₀) − 1 + θ₀/θ̂ ]         (profile LR statistic),
//!   ĵ  = n/θ̂²   (observed info),  î = n/θ̂²  (Fisher info; canonical ⇒ î = ĵ),
//!   Î  = Σ_i (1/θ̂ − y_i)²                    (empirical score covariance).
//! Even though the family is canonical (`î = ĵ = Î`), the correction is NOT
//! trivial at finite n: the Wald root `u = (θ̂−θ₀)√ĵ` and the LR root
//! `r = sign·√W` differ, so `r* = r + log(u/r)/r ≠ r`. This is precisely the
//! small-sample regime where the first-order root is anti-conservative and the
//! third-order `r*` restores calibration — so the gate has real teeth: it would
//! fail if `r*` degraded to the (oversized) first-order test.
//!
//! Audit: the type-I size at `α ∈ {0.01, 0.05, 0.10}` is checked as coverage of
//! the NON-rejection event at nominal `1−α`, so the shared Wilson verdict
//! applies unchanged. Both tails gate (#3534): an oversized test under-covers
//! non-rejection and a conservative (undersized) test over-covers it. Every
//! replication enters the denominator: the Exponential sample is continuous, so
//! `θ̂ ≠ θ₀` and `W > 0` almost surely and the assembly never declines; a
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
/// The true (and null) rate — the null is exact, so any rejection is a type-I
/// error. Value is immaterial (the statistic is scale-equivariant); fixed for
/// reproducibility.
const TRUE_RATE: f64 = 1.3;
const SEED: u64 = 0x1891_5C_07_A0_00;

/// One Exponential(rate) draw from the harness's uniform stream: `−ln(U)/rate`.
fn exp_draw(rate: f64, rng: &mut CalibrationRng) -> f64 {
    -rng.uniform_open01().ln() / rate
}

/// The Skovgaard corrected (model-form) two-sided p-value for testing the rate
/// `θ = θ₀` on one simulated Exponential sample.
fn corrected_p_value(rate_true: f64, rate_null: f64, rng: &mut CalibrationRng) -> f64 {
    let ys: Vec<f64> = (0..N_OBS).map(|_| exp_draw(rate_true, rng)).collect();
    let sum_y: f64 = ys.iter().sum();
    let n = N_OBS as f64;
    let theta_hat = n / sum_y;
    // Profile LR statistic W = 2n[ ln(θ̂/θ₀) − 1 + θ₀/θ̂ ] ≥ 0.
    let lr = 2.0 * n * ((theta_hat / rate_null).ln() - 1.0 + rate_null / theta_hat);
    let observed_info = n / (theta_hat * theta_hat);
    // Canonical family: expected info equals observed info.
    let expected_info = observed_info;
    // Empirical score covariance Σ (1/θ̂ − y_i)² (score s_i = ∂ℓ_i/∂θ = 1/θ − y_i).
    let score_cov: f64 = ys.iter().map(|&y| (1.0 / theta_hat - y).powi(2)).sum();

    scalar_skovgaard_r_star(&ScalarSkovgaardInput {
        theta_hat,
        theta_null: rate_null,
        lr_statistic: lr,
        observed_info,
        expected_info,
        score_cov,
    })
    .map(|res| res.p_value_corrected)
    .unwrap_or_else(|| {
        panic!(
            "Skovgaard assembly declined on a continuous Exponential sample \
             (θ̂={theta_hat}, θ₀={rate_null}, W={lr}, Î={score_cov})"
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
        let p = corrected_p_value(TRUE_RATE, TRUE_RATE, &mut rng);
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
                .map(|_| exp_draw(TRUE_RATE, &mut probe))
                .collect();
            let sum_y: f64 = ys.iter().sum();
            let n = N_OBS as f64;
            let theta_hat = n / sum_y;
            let lr = 2.0 * n * ((theta_hat / TRUE_RATE).ln() - 1.0 + TRUE_RATE / theta_hat);
            let obs = n / (theta_hat * theta_hat);
            let sc: f64 = ys.iter().map(|&y| (1.0 / theta_hat - y).powi(2)).sum();
            if let Some(res) = scalar_skovgaard_r_star(&ScalarSkovgaardInput {
                theta_hat,
                theta_null: TRUE_RATE,
                lr_statistic: lr,
                observed_info: obs,
                expected_info: obs,
                score_cov: sc,
            }) {
                if (res.r_star - res.r).abs() > 1e-6 {
                    correction_ever_material = true;
                    break;
                }
            }
        }
    }
    assert!(
        correction_ever_material,
        "r* never differed from r — the Exponential anchor is not exercising the \
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
