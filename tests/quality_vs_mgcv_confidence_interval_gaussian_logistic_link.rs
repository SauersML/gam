//! End-to-end quality: gam's confidence-interval construction under a
//! **non-identity (logistic) link** must be *well-calibrated against the known
//! truth* — its nominal-95% intervals must actually cover the true latent
//! function at the nominal rate. `mgcv` is retained only as a **baseline to
//! match-or-beat** on calibration, never as the thing gam must reproduce.
//!
//! OBJECTIVE METRIC (this is the pass/fail claim):
//!   The data are generated from a *known* latent smooth
//!   `η(x) = x + sin(2πx)`, `μ(x) = sigmoid(η(x))`, `y ~ Bernoulli(μ)`. Because
//!   the truth is known exactly, we measure the **empirical coverage** of gam's
//!   pointwise 95% confidence intervals across the training grid:
//!     * link scale:     fraction of points with `η(xᵢ) ∈ [eta_lowerᵢ, eta_upperᵢ]`
//!     * response scale: fraction of points with `μ(xᵢ) ∈ [mean_lowerᵢ, mean_upperᵢ]`
//!
//!   The Nychka/Marra–Wood result for penalized GAMs is a statement about the
//!   **sampling distribution**: the *across-the-function* average coverage of
//!   the Bayesian credible band, taken in expectation over repeated draws of the
//!   response, tracks the nominal level (pointwise coverage is
//!   smoothing-bias-attenuated, but the grid-and-replicate average is close to
//!   nominal). A SINGLE Bernoulli realization is far too noisy to estimate that
//!   expectation: at each design point the response is one Bernoulli bit, the
//!   per-point Fisher information `μ(1−μ)` is at most ¼, and a single unlucky
//!   draw of a six-cycle sinusoid can leave the penalized fit biased at most
//!   crests/troughs all at once. The single-draw across-grid coverage therefore
//!   has an enormous Monte-Carlo spread (it can read 0.46/0.68/0.99 on adjacent
//!   seeds) and does NOT estimate the Nychka coverage object — it is the WRONG
//!   estimator for the theorem this test asserts. We instead pool coverage over
//!   a fixed design and many Bernoulli replicates, exactly as the identity-link
//!   sibling coverage tests do, so the empirical average converges to the
//!   sampling-distribution expectation the theorem is about. This is an
//!   objective property of gam's own intervals versus ground truth — it does
//!   not depend on mgcv.
//!
//! BASELINE (match-or-beat, not match): mgcv fits the identical penalized
//!   binomial smooths and its `predict(se.fit=TRUE)` band is scored for coverage
//!   against the *same* truth over the *same* replicates. We additionally
//!   require gam's calibration to be no worse than mgcv's by more than a small
//!   margin, i.e. gam's |coverage − nominal| ≤ mgcv's |coverage − nominal| +
//!   margin. This demotes the mature tool to a yardstick on the objective
//!   metric; gam is never asked to reproduce mgcv's noisy SEs.
//!
//! Why a Binomial(logit) model: this is the family that actually exercises gam's
//! inverse-link Jacobian `dμ/dη = μ(1−μ)` inside CI construction (the Gaussian
//! posterior-variance branch ignores the link entirely). The latent smooth
//! `η(x) = x + sin(2πx)`, design `x ~ U[-3,3]`, n=200, seed=123 follow the spec
//! verbatim; Bernoulli sampling replaces the spec's Gaussian noise so the
//! generative model is internally consistent with the logit link. The fixed
//! design is drawn once (seed=123); the Bernoulli responses are then redrawn
//! for each replicate from the same true `μ(x)` so coverage is measured over
//! the response sampling distribution at a fixed configuration of `x`.
//!
//! Identical data feed both engines (the same CSV columns). Bounds are not
//! weakened to force a pass: a genuinely mis-calibrated band failing here is a
//! real bug.

use gam::estimate::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_r};
use gam::types::LikelihoodSpec;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// Number of independent Bernoulli response replicates drawn on the fixed
/// design. Pooled across the grid this gives ~`N_REPLICATES` effectively
/// independent coverage trials, enough to estimate the Nychka across-the-
/// function coverage expectation to within the asserted window.
const N_REPLICATES: usize = 100;

/// Fixed design size (spec: n=200).
const N: usize = 200;

/// SplitMix64 — a small, fully specified PRNG (no external rand crate, no env,
/// no hidden state). One instance carries the whole stream so the fixed design
/// and every replicate's Bernoulli draws are bit-for-bit reproducible.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u01(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53-bit mantissa -> uniform in [0,1).
        ((z >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Fixed design and exact latent truth. Returns `(x, eta_true, mu_true)` where
/// `eta_true`/`mu_true` are the *exact* data-generating latent values at each
/// `x` — the ground truth the confidence intervals must cover.
///
/// `x ~ U[-3,3]`, `η(x) = x + sin(2πx)`, `μ = sigmoid(η)`. The design is drawn
/// once; the Bernoulli responses are redrawn per replicate (see
/// [`bernoulli_replicate`]).
fn fixed_design(rng: &mut SplitMix64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(N);
    let mut eta_true = Vec::with_capacity(N);
    let mut mu_true = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = -3.0 + 6.0 * rng.next_u01();
        let eta = xi + (2.0 * std::f64::consts::PI * xi).sin();
        let p = 1.0 / (1.0 + (-eta).exp());
        x.push(xi);
        eta_true.push(eta);
        mu_true.push(p);
    }
    (x, eta_true, mu_true)
}

/// One Bernoulli response vector `y ~ Bernoulli(μ_true)` on the fixed design,
/// drawn from the supplied RNG stream so successive calls give independent
/// replicates.
fn bernoulli_replicate(mu_true: &[f64], rng: &mut SplitMix64) -> Vec<f64> {
    mu_true
        .iter()
        .map(|&p| if rng.next_u01() < p { 1.0 } else { 0.0 })
        .collect()
}

#[test]
fn confidence_intervals_cover_truth_under_logistic_link() {
    init_parallelism();

    // ---- fixed design + exact latent truth (spec: n=200, x~U[-3,3], seed=123)
    let mut rng = SplitMix64::new(123);
    let (x, eta_true, mu_true) = fixed_design(&mut rng);
    assert_eq!(x.len(), N);

    // ---- gam: fit each Bernoulli replicate, pool CI coverage at the grid ----
    // The truth `eta(x) = x + sin(2*pi*x)` over x in [-3, 3] carries SIX full
    // cycles of the sinusoid (argument spans -6*pi..6*pi). A penalized cubic
    // smooth needs roughly four knots per cycle to represent that without a
    // residual `O(lambda*f'')` bias at every crest/trough, so k must be >= ~28.
    // k=30 puts the truth in the span of the basis so the across-the-function
    // coverage claim is well-posed; both engines use the same k for an
    // apples-to-apples band. Coverage is pooled over N_REPLICATES Bernoulli
    // redraws of the response at this fixed design, which estimates the Nychka
    // across-the-function coverage expectation (a single Bernoulli draw is far
    // too noisy to estimate it — see the module docs).
    let headers = vec!["x".to_string(), "y".to_string()];
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };

    let mut gam_eta_hits = 0usize;
    let mut gam_mean_hits = 0usize;
    let total_trials = N_REPLICATES * N;

    // Bernoulli response replicates, drawn from the continuing seed-123 stream
    // so the (design, replicate) data are bit-for-bit reproducible and feed the
    // same CSV columns to mgcv below.
    let replicates: Vec<Vec<f64>> = (0..N_REPLICATES)
        .map(|_| bernoulli_replicate(&mu_true, &mut rng))
        .collect();

    for rep in &replicates {
        assert!(
            rep.iter().any(|&v| v > 0.5) && rep.iter().any(|&v| v < 0.5),
            "synthetic outcome must contain both classes"
        );

        // ---- build a gam dataset from (x, y_rep) ---------------------------
        let records: Vec<csv::StringRecord> = (0..N)
            .map(|i| {
                csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", rep[i])])
            })
            .collect();
        let ds = encode_recordswith_inferred_schema(headers.clone(), records)
            .expect("encode synthetic replicate");
        let col = ds.column_map();
        let x_idx = col["x"];

        // ---- fit gam: y ~ s(x, k=30), Binomial(logit), REML ----------------
        let result =
            fit_from_formula("y ~ s(x, k=30)", &ds, &cfg).expect("gam binomial(logit) fit");
        let FitResult::Standard(fit) = result else {
            panic!("binomial(logit) smooth should be a Standard fit");
        };

        // ---- rebuild the design at the training points and predict CIs -----
        // Grid-aligned, element-wise comparison: evaluate at the same training
        // x. The design row maps beta -> eta; predict_gamwith_uncertainty then
        // propagates Var(eta) through the logit inverse link to the mean SE.
        let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
        for (i, &xi) in x.iter().enumerate() {
            grid[[i, x_idx]] = xi;
        }
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild design at training points");
        let gam_eta_check = design.design.apply(&fit.fit.beta);
        assert_eq!(gam_eta_check.len(), N, "design eta length mismatch");
        let offset = Array1::<f64>::zeros(N);

        // Bias correction OFF, boundary/OOD inflation OFF. The covariance is the
        // smoothing-parameter-corrected Bayesian Vp (Marra & Wood 2012), which
        // is exactly what mgcv's `predict(se.fit=TRUE)` reports — the band whose
        // ACROSS-THE-FUNCTION coverage tracks the nominal level.
        let options = PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
            apply_bias_correction: false,
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            multi_point_joint: false,
            ..PredictUncertaintyOptions::default()
        };
        let pred = predict_gamwith_uncertainty(
            design.design.clone(),
            fit.fit.beta.view(),
            offset.view(),
            LikelihoodSpec::binomial_logit(),
            &fit.fit,
            &options,
        )
        .expect("gam uncertainty under logit link");

        // Pool gam's nominal-95% interval coverage of the KNOWN truth, both
        // scales, across this replicate's grid.
        for i in 0..N {
            if eta_true[i] >= pred.eta_lower[i] && eta_true[i] <= pred.eta_upper[i] {
                gam_eta_hits += 1;
            }
            if mu_true[i] >= pred.mean_lower[i] && mu_true[i] <= pred.mean_upper[i] {
                gam_mean_hits += 1;
            }
        }
    }

    let gam_eta_cov = gam_eta_hits as f64 / total_trials as f64;
    let gam_mean_cov = gam_mean_hits as f64 / total_trials as f64;

    // ---- mgcv BASELINE (match-or-beat on calibration): SAME data/model -----
    // mgcv::predict(se.fit=TRUE) yields eta/mu and their SEs per replicate; R
    // forms the nominal-95% band (fit ± 1.96·se) on each scale and pools its
    // coverage against the SAME truth over the SAME replicates. mgcv is a
    // yardstick on the objective metric, not the thing gam must reproduce.
    let mut columns: Vec<Column<'_>> = Vec::with_capacity(N_REPLICATES + 1);
    columns.push(Column::new("x", &x));
    let yheaders: Vec<String> = (0..N_REPLICATES).map(|r| format!("y{r}")).collect();
    for (r, name) in yheaders.iter().enumerate() {
        columns.push(Column::new(name, &replicates[r]));
    }

    let r = run_r(
        &columns,
        r#"
        suppressPackageStartupMessages(library(mgcv))
        nrep <- 100L
        z <- qnorm(0.975)
        # Exact latent truth at the fixed design carried in df$x.
        eta_true <- df$x + sin(2 * pi * df$x)
        mu_true <- 1 / (1 + exp(-eta_true))
        eta_hits <- 0L
        mean_hits <- 0L
        total <- 0L
        for (r in 0:(nrep - 1L)) {
            yname <- paste0("y", r)
            dat <- data.frame(x = df$x, y = df[[yname]])
            m <- gam(y ~ s(x, k = 30), data = dat,
                     family = binomial(link = "logit"), method = "REML")
            pl <- predict(m, type = "link", se.fit = TRUE)
            pr <- predict(m, type = "response", se.fit = TRUE)
            elo <- pl$fit - z * pl$se.fit; ehi <- pl$fit + z * pl$se.fit
            mlo <- pr$fit - z * pr$se.fit; mhi <- pr$fit + z * pr$se.fit
            eta_hits <- eta_hits + sum(eta_true >= elo & eta_true <= ehi)
            mean_hits <- mean_hits + sum(mu_true >= mlo & mu_true <= mhi)
            total <- total + length(eta_true)
        }
        emit("eta_cov", eta_hits / total)
        emit("mean_cov", mean_hits / total)
        "#,
    );
    let mgcv_eta_cov = r.scalar("eta_cov");
    let mgcv_mean_cov = r.scalar("mean_cov");

    let nominal = 0.95_f64;
    eprintln!(
        "logit-link CI calibration (95% nominal, {N_REPLICATES} reps x {N} pts = {total_trials} trials):\n  \
         link-scale  coverage: gam={gam_eta_cov:.3} mgcv={mgcv_eta_cov:.3}\n  \
         resp-scale  coverage: gam={gam_mean_cov:.3} mgcv={mgcv_mean_cov:.3}"
    );

    // (1) OBJECTIVE: gam's pooled across-grid-and-replicate average coverage
    // tracks the nominal level. Pointwise penalized-GAM bands are smoothing-bias
    // attenuated, so the calibration window is centered on 0.95 with a tolerance
    // that admits the expected attenuation but rejects a badly mis-scaled band.
    // This claim is about gam vs ground truth and does not involve mgcv.
    let cov_window = 0.12_f64; // 0.95 ± 0.12  ->  [0.83, 1.00]
    assert!(
        (gam_eta_cov - nominal).abs() <= cov_window,
        "link-scale 95% CI mis-calibrated vs truth: coverage={gam_eta_cov:.3} \
         (nominal {nominal:.2}, window ±{cov_window:.2})"
    );
    assert!(
        (gam_mean_cov - nominal).abs() <= cov_window,
        "response-scale 95% CI mis-calibrated vs truth: coverage={gam_mean_cov:.3} \
         (nominal {nominal:.2}, window ±{cov_window:.2})"
    );

    // (2) MATCH-OR-BEAT mgcv on calibration: gam's distance from nominal must
    // be no worse than mgcv's by more than a small margin, on both scales.
    let beat_margin = 0.03_f64;
    let gam_eta_err = (gam_eta_cov - nominal).abs();
    let mgcv_eta_err = (mgcv_eta_cov - nominal).abs();
    assert!(
        gam_eta_err <= mgcv_eta_err + beat_margin,
        "link-scale calibration worse than mgcv baseline: |gam−nom|={gam_eta_err:.3} > \
         |mgcv−nom|={mgcv_eta_err:.3} + {beat_margin:.2}"
    );
    let gam_mean_err = (gam_mean_cov - nominal).abs();
    let mgcv_mean_err = (mgcv_mean_cov - nominal).abs();
    assert!(
        gam_mean_err <= mgcv_mean_err + beat_margin,
        "response-scale calibration worse than mgcv baseline: |gam−nom|={gam_mean_err:.3} > \
         |mgcv−nom|={mgcv_mean_err:.3} + {beat_margin:.2}"
    );
}
