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
    RhoPenaltyComponent, RowExpectedJets, RowKappas, lawley_lr_mean_shift,
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

/// #939 deliverable (2), the ρ̂-variation arm — VALIDATION BY SIMULATION over
/// the sampling distribution of ρ̂.
///
/// The conditional Lawley shift `Δε(ρ)` is `E[W | λ]` — the LR mean with the
/// smoothing parameter held FIXED at ρ. When λ is estimated, the relevant null
/// mean is the expectation over the sampling distribution of ρ̂:
///
/// ```text
/// E[W] = E_{ρ̂}[ Δε(ρ̂) ] = Δε(ρ₀) + ½ Δε''(ρ₀)·Var(ρ̂) + O(·)
/// ```
///
/// `lawley_lr_mean_shift_with_rho_variation` assembles exactly the right-hand
/// second-order term. This test takes the assembly's prediction as a HYPOTHESIS
/// and falsifies the alternative (fixed-λ only) against a Monte-Carlo ground
/// truth: it draws `ρ̂ ~ N(ρ₀, Var)` (the genuine sampling fluctuation of the
/// log-smoothing estimate), evaluates the *conditional* shift `Δε(ρ̂)` at each
/// draw by re-scaling the penalty, and averages. The claim is that the
/// ρ̂-variation assembly matches this simulated `E_{ρ̂}[Δε(ρ̂)]` to second order
/// — and matches it STRICTLY BETTER than the conditional (fixed-λ) shift, which
/// systematically misses the curvature term. That gap IS the size correction
/// attributable specifically to ρ̂-variation.
#[test]
fn rho_variation_assembly_matches_simulated_expectation_over_rho_hat() {
    // Poisson/log smooth substrate at a fixed null linear predictor: a 2-column
    // design (intercept + centered covariate) penalized on the second column.
    let n = 60usize;
    let mut x = Array2::<f64>::ones((n, 2));
    let mut kappas = Vec::<RowKappas>::with_capacity(n);
    for i in 0..n {
        let z = i as f64 / n as f64 - 0.5;
        x[[i, 1]] = z;
        let eta = 0.3 + 0.6 * z;
        kappas.push(
            RowExpectedJets::poisson_log(eta)
                .kappas()
                .expect("poisson kappas"),
        );
    }
    let tested = 1..2;

    // Population smoothing parameter ρ₀ = log λ₀ and its sampling variance. (In
    // the live engine Var(ρ̂) is the inverse REML outer Hessian; here it is a
    // fixed scenario value so the simulation is self-contained and exact.)
    let lambda0 = 3.0_f64;
    let rho0 = lambda0.ln();
    let var_rho = 0.6_f64;
    let mut s_comp = Array2::<f64>::zeros((2, 2));
    s_comp[[1, 1]] = lambda0;
    let penalty = s_comp.clone();
    let components = vec![RhoPenaltyComponent {
        s_component: s_comp,
    }];
    let rho_cov = Array2::from_shape_vec((1, 1), vec![var_rho]).unwrap();

    // Conditional shift Δε(ρ₀) (fixed-λ): the quantity the pre-#939-remainder
    // factor used.
    let conditional =
        lawley_lr_mean_shift(x.view(), &kappas, Some(penalty.view()), tested.clone())
            .expect("conditional Δε");

    // The ρ̂-variation assembly's predicted E[W]-shift.
    let assembled = lawley_lr_mean_shift_with_rho_variation(
        x.view(),
        &kappas,
        penalty.view(),
        tested.clone(),
        &components,
        rho_cov.view(),
    )
    .expect("assembled Δε(ρ̂)");

    // MONTE-CARLO ground truth: E_{ρ̂}[Δε(ρ̂)] over ρ̂ ~ N(ρ₀, var_rho). For each
    // draw, the conditional shift at ρ̂ scales the penalty by e^{ρ̂−ρ₀}.
    let mut rng = StdRng::seed_from_u64(20939);
    let normal = Normal::new(0.0, var_rho.sqrt()).expect("normal");
    let reps = 40_000usize;
    let mut sum = 0.0;
    for _ in 0..reps {
        let drho: f64 = normal.sample(&mut rng); // ρ̂ − ρ₀
        let lambda = (rho0 + drho).exp();
        let mut s = Array2::<f64>::zeros((2, 2));
        s[[1, 1]] = lambda;
        let de = lawley_lr_mean_shift(x.view(), &kappas, Some(s.view()), tested.clone())
            .expect("Δε(ρ̂) draw");
        sum += de;
    }
    let simulated = sum / reps as f64;

    // The ρ̂-variation correction must be genuinely non-zero (else the test is
    // vacuous): the simulated mean differs from the conditional shift.
    assert!(
        (simulated - conditional).abs() > 1e-6,
        "fixture must exhibit a non-trivial ρ̂-variation effect: \
         simulated E[Δε(ρ̂)]={simulated:.8}, conditional Δε(ρ₀)={conditional:.8}"
    );

    // (1) The assembly matches the simulated expectation to second order. The MC
    // standard error is ~ sd(Δε)/√reps; the residual is dominated by the
    // O(Var²·Δε'''') delta-method truncation plus MC noise — both small here.
    let err_assembled = (assembled - simulated).abs();
    let err_conditional = (conditional - simulated).abs();
    assert!(
        err_assembled < 0.2 * err_conditional.max(1e-12) + 1e-6,
        "ρ̂-variation assembly must match the simulated E[Δε(ρ̂)]: \
         |assembled−sim|={err_assembled:.3e} vs |conditional−sim|={err_conditional:.3e} \
         (assembled={assembled:.8}, conditional={conditional:.8}, sim={simulated:.8})"
    );

    // (2) The defining property: the ρ̂-variation assembly is STRICTLY closer to
    // the truth than the fixed-λ conditional shift. This is the size correction
    // attributable specifically to ρ̂-variation (vs the fixed-λ factor).
    assert!(
        err_assembled < err_conditional,
        "ρ̂-variation correction must improve on the fixed-λ shift: \
         |assembled−sim|={err_assembled:.3e} must be < |conditional−sim|={err_conditional:.3e}"
    );
}
