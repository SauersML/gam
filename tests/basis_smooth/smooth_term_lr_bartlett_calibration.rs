//! #1063: the per-term smooth-significance test uses a genuine likelihood-ratio
//! statistic `W = 2(ℓ_full − ℓ_null)` (a constrained refit dropping the smooth),
//! Bartlett-corrected by the exact Lawley factor — and the correction IMPROVES
//! the χ² calibration of the test under the null.
//!
//! The summary table reports Wood's rank-truncated *Wald* statistic; the Lawley
//! factor corrects the *likelihood-ratio* statistic. Dividing the Wald T by the
//! LR factor would correct the wrong quantity (under penalization the Wald form
//! is already a weighted χ²). The principled fix (#1063 Option 1) is to compute
//! a real per-term LR statistic and correct *that*. This test proves two things
//! about `smooth_term_lr_inference_forspec`:
//!
//!   (a) PROVENANCE — for a Poisson/log smooth (closed-form Lawley jets) the
//!       reported significance is built from the Bartlett-corrected LR
//!       (`correction_provenance == "lawley_lr"`, `bartlett_factor > 1`,
//!       `statistic_corrected == statistic_lr / bartlett_factor`).
//!
//!   (b) CALIBRATION — under a NULL data-generating process (the smooth's
//!       covariate has no effect) the uncorrected LR is anti-conservative at
//!       modest n: `E[W] = d + Δε > d`, so its mean overshoots the χ²_d mean and
//!       its p-values skew small. The Bartlett-corrected statistic restores the
//!       mean toward d, so `mean(W*)` is closer to `d` than `mean(W)` and the
//!       corrected p-values are closer to Uniform — a strictly better χ²
//!       calibration than the uncorrected reference.
//!
//! This is the truth-recovery / calibration bar (not a reference-tool match):
//! the ground truth here is the exact null distribution of the LR statistic.

use gam::smooth::smooth_term_lr_inference_forspec;
use gam::{
    FitConfig, FitRequest, encode_recordswith_inferred_schema, init_parallelism, materialize,
};

use gam::inference::lawley::{
    RhoPenaltyComponent, RowExpectedJets, lawley_lr_mean_shift,
    lawley_lr_mean_shift_with_rho_variation,
};
use ndarray::Array2;

use csv::StringRecord;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

/// One null-DGP replicate: `y ~ Poisson(exp(0.3 + 0.8 x))`, with an independent
/// covariate `z ~ U(0,1)` that has NO effect on the mean. The smooth `s(z)` is
/// therefore null-true; its LR statistic should follow χ²_d.
fn null_replicate(n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let headers = vec!["y".to_string(), "x".to_string(), "z".to_string()];
    let mut rows = Vec::<StringRecord>::with_capacity(n);
    for i in 0..n {
        let x = i as f64 / (n as f64 - 1.0); // deterministic, spans [0,1]
        // z is an independent nuisance covariate with NO effect on the mean.
        let z: f64 = rng.random_range(0.0..1.0);
        let eta = 0.3 + 0.8 * x; // NOTE: no z term — the smooth is null-true.
        let lambda = eta.exp();
        let y = Poisson::new(lambda).expect("poisson rate").sample(&mut rng) as u64;
        rows.push(StringRecord::from(vec![
            y.to_string(),
            x.to_string(),
            z.to_string(),
        ]));
    }
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Run the per-term LR + Bartlett driver on one replicate and return the single
/// `s(z)` smooth-term report.
fn run_one(data: &gam::data::EncodedDataset) -> Option<gam::smooth::SmoothTermLrInference> {
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let mat = materialize("y ~ x + s(z)", data, &cfg).expect("materialize");
    let FitRequest::Standard(req) = mat.request else {
        panic!("expected a standard fit request");
    };
    let reports = smooth_term_lr_inference_forspec(
        req.data.view(),
        req.y.view(),
        req.weights.view(),
        req.offset.view(),
        &req.spec,
        req.family,
        &req.options,
    )
    .expect("smooth-term LR inference");
    reports.into_iter().find(|r| r.name.contains('z'))
}

#[test]
fn poisson_smooth_lr_is_bartlett_corrected_and_better_calibrated() {
    init_parallelism();

    const N: usize = 60;
    const REPS: usize = 120;

    // (a) Provenance + algebra check on a single representative replicate.
    let probe = run_one(&null_replicate(N, 1)).expect("s(z) report present");
    assert_eq!(
        probe.correction.label(),
        "lawley_lr",
        "Poisson/log smooth has closed-form Lawley jets — the correction must fire"
    );
    assert!(
        probe.bartlett_factor > 1.0,
        "small-n Lawley factor must inflate the χ² reference (c = 1 + Δε/d > 1); got {}",
        probe.bartlett_factor
    );
    assert!(
        (probe.statistic_corrected - probe.statistic_lr / probe.bartlett_factor).abs()
            < 1e-9 * (1.0 + probe.statistic_lr.abs()),
        "W* must equal W / c: {} vs {}",
        probe.statistic_corrected,
        probe.statistic_lr / probe.bartlett_factor
    );
    assert!(
        probe.p_value_corrected >= probe.p_value_uncorrected - 1e-12,
        "dividing W by c > 1 can only RAISE the p-value (less significant)"
    );
    // (a') Materiality diagnostic (#939 deliverable 4): a correction was applied,
    // so `material` must agree with the 10% rule it documents.
    {
        let factor_move = (probe.bartlett_factor - 1.0).abs();
        let p_hi = probe.p_value_uncorrected.max(probe.p_value_corrected);
        let p_lo = probe.p_value_uncorrected.min(probe.p_value_corrected);
        let p_move = (p_hi - p_lo) / p_hi.max(f64::MIN_POSITIVE);
        let expected_material = factor_move > 0.10 || p_move > 0.10;
        assert_eq!(
            probe.material, expected_material,
            "material flag must follow the 10% rule: c={}, factor_move={:.4}, p_move={:.4}",
            probe.bartlett_factor, factor_move, p_move
        );
    }

    // (b) Calibration sweep under the null DGP.
    let mut sum_w = 0.0;
    let mut sum_w_star = 0.0;
    let mut sum_p_unc = 0.0;
    let mut sum_p_cor = 0.0;
    let mut ref_df = 0.0;
    let mut count = 0usize;
    for rep in 0..REPS {
        let data = null_replicate(N, 1000 + rep as u64);
        let Some(r) = run_one(&data) else {
            continue;
        };
        if !(r.statistic_lr.is_finite()
            && r.statistic_corrected.is_finite()
            && r.p_value_uncorrected.is_finite()
            && r.p_value_corrected.is_finite()
            && r.ref_df.is_finite())
        {
            continue;
        }
        sum_w += r.statistic_lr;
        sum_w_star += r.statistic_corrected;
        sum_p_unc += r.p_value_uncorrected;
        sum_p_cor += r.p_value_corrected;
        ref_df += r.ref_df;
        count += 1;
    }
    assert!(
        count >= REPS * 8 / 10,
        "too many replicates failed to produce a finite LR report: {count}/{REPS}"
    );

    let mean_w = sum_w / count as f64;
    let mean_w_star = sum_w_star / count as f64;
    let mean_p_unc = sum_p_unc / count as f64;
    let mean_p_cor = sum_p_cor / count as f64;
    let mean_d = ref_df / count as f64;

    // The corrected statistic's mean must be closer to the χ²_d mean (= d) than
    // the uncorrected one — the defining property of the Bartlett correction.
    let err_unc = (mean_w - mean_d).abs();
    let err_cor = (mean_w_star - mean_d).abs();
    assert!(
        mean_w > mean_d,
        "uncorrected LR must be anti-conservative under the null at n={N}: \
         mean(W)={mean_w:.3} should exceed d={mean_d:.3}"
    );
    assert!(
        err_cor < err_unc,
        "Bartlett correction must move the LR mean toward d: \
         |mean(W*)−d|={err_cor:.4} must be < |mean(W)−d|={err_unc:.4} \
         (mean_w={mean_w:.3}, mean_w*={mean_w_star:.3}, d={mean_d:.3})"
    );

    // The corrected p-values must be closer to the Uniform(0,1) mean 0.5 than
    // the (small-skewed) uncorrected ones — a strictly better χ² calibration.
    let p_err_unc = (mean_p_unc - 0.5).abs();
    let p_err_cor = (mean_p_cor - 0.5).abs();
    assert!(
        p_err_cor <= p_err_unc + 1e-9,
        "corrected p-values must calibrate at least as well as uncorrected: \
         |mean(p*)−0.5|={p_err_cor:.4} must be ≤ |mean(p)−0.5|={p_err_unc:.4} \
         (mean_p={mean_p_unc:.3}, mean_p*={mean_p_cor:.3})"
    );
}
