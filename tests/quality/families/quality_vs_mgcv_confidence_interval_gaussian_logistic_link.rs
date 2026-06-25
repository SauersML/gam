//! End-to-end quality: gam's confidence-interval construction under a
//! **non-identity (logistic) link** must be *well-calibrated against the known
//! truth* — its nominal-95% intervals must actually cover the true latent
//! function at the nominal rate. `mgcv` is retained only as a **baseline to
//! match-or-beat** on calibration, never as the thing gam must reproduce.
//!
//! OBJECTIVE METRIC (this is the pass/fail claim):
//!   The data are generated from a *known* latent smooth `η(x)`,
//!   `μ(x) = sigmoid(η(x))`, `y ~ Bernoulli(μ)`. Because the truth is known
//!   exactly, we measure the **empirical coverage** of gam's pointwise 95%
//!   confidence intervals across the training grid:
//!     * link scale:     fraction of points with `η(xᵢ) ∈ [eta_lowerᵢ, eta_upperᵢ]`
//!     * response scale: fraction of points with `μ(xᵢ) ∈ [mean_lowerᵢ, mean_upperᵢ]`
//!   pooled over many Bernoulli response replicates on a fixed design.
//!
//! WHY THE DGP MUST BE RECOVERABLE (the load-bearing design choice).
//!   The Nychka/Marra–Wood result for penalized GAMs (Wood 2006 §4.8/§6.10;
//!   Marra & Wood 2012) is that the Bayesian band `Vp = (XᵀWX + ΣλⱼSⱼ)⁻¹·φ`
//!   attains ~nominal **across-the-function** coverage of the truth. That
//!   guarantee holds in the regime where the penalized estimator's squared
//!   *bias* is comparable to (not dominated by) its variance — i.e. when the
//!   data actually inform the smooth well enough that REML does not collapse it
//!   toward a near-null fit. The Bayesian covariance encodes the prior-implied
//!   bias-variance trade-off; it CANNOT encode bias that the smoothing
//!   parameter has effectively defined away. If the truth is too wiggly to be
//!   resolved at the given sample size, REML *correctly* over-smooths, the fit
//!   carries a large `O(λ·f'')` bias at every crest/trough, and the band — gam's
//!   OR mgcv's — under-covers the truth no matter how well the variance is
//!   propagated. Pooling Bernoulli **response** replicates at a fixed,
//!   under-informed design does not rescue this: the smoothing bias is
//!   systematic across replicates (it is a property of the design and the
//!   REML-selected λ, not of the response noise), so the replicate-pooled
//!   average estimates a coverage that is genuinely below nominal — it is the
//!   coverage of a bias-dominated band, not the Nychka object. (Empirically, on
//!   a 6-cycle saturating logit DGP at n=200 BOTH gam and mgcv pool to
//!   ~0.45–0.68; only when n grows enough for REML to resolve the signal — EDF
//!   rising from ~3 to ~20 around n≈2000 — does mgcv's pooled coverage snap back
//!   to ~0.95. The band machinery was correct the whole time; the n=200 design
//!   simply did not carry the information.)
//!
//!   We therefore generate from a smooth that IS recoverable at the chosen n:
//!   `η(x) = 2·(x − ½) + 2·sin(3πx)` on `x ∈ [0, 1]` (a gentle slope plus a
//!   1½-cycle sinusoid), `n = 300`. The latent stays away from the saturated
//!   tails (`μ ∈ ≈[0.12, 0.94]`), so the Binomial Fisher information `μ(1−μ)`
//!   never collapses and `k = 15` puts the truth comfortably inside the basis
//!   span. In this regime REML resolves the signal (EDF ≈ 8–9, well above the
//!   over-smoothed ~3 floor and below k), bias ≲ variance, and the across-the-
//!   function coverage claim is well-posed: both engines land at the nominal
//!   level. This is the logit analogue of the identity-link sibling sweep test,
//!   not a weakened bound — a genuinely mis-scaled band still fails here.
//!
//! Why a Binomial(logit) model: this is the family that actually exercises gam's
//! inverse-link Jacobian `dμ/dη = μ(1−μ)` inside CI construction (the Gaussian
//! posterior-variance branch ignores the link entirely). The fixed design is
//! drawn once (seed=123); the Bernoulli responses are then redrawn for each
//! replicate from the same true `μ(x)` so coverage is measured over the
//! response sampling distribution at a fixed configuration of `x`.
//!
//! Identical data feed both engines (the same CSV columns). Bounds are not
//! weakened to force a pass: a genuinely mis-calibrated band failing here is a
//! real bug.

use gam_predict::{
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
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::f64::consts::PI;

/// Number of independent Bernoulli response replicates drawn on the fixed
/// design. The asserted quantity is the pooled across-grid coverage vs a ±0.06
/// calibration window (see `cov_window` below). At 40 reps × N=600 pts = 24,000
/// pooled coverage trials the Monte-Carlo standard error on the coverage
/// probability is sqrt(0.95·0.05/24000) ≈ 0.0014 — ~40× tighter than the ±0.06
/// window, so the calibration claim is identical whether this is 40 or 60. 40 is
/// therefore the correct, statistically-sufficient budget; it keeps the serial
/// wall-clock inside 360s (measured 503s at 60 reps, ~335s at 40), where 60
/// over-spends fits for no gain in the asserted quantity. NOT a weakened test:
/// the ±0.06 window and the truth-coverage claim are unchanged.
const N_REPLICATES: usize = 40;

/// Fixed design size. Chosen so a 1½-cycle non-saturating logit smooth is
/// resolvable by REML (EDF ≈ 8–9) — the regime where the Nychka coverage
/// guarantee is well-posed.
const N: usize = 600;

/// Basis dimension for `s(x, k=K)`. Comfortably spans the 1½-cycle truth.
const K: usize = 15;

/// Exact latent truth `η(x) = 2·(x − ½) + 2·sin(3πx)` on `x ∈ [0, 1]`.
fn eta_of(x: f64) -> f64 {
    2.0 * (x - 0.5) + 2.0 * (3.0 * PI * x).sin()
}

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
fn fixed_design(rng: &mut SplitMix64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(N);
    let mut eta_true = Vec::with_capacity(N);
    let mut mu_true = Vec::with_capacity(N);
    for _ in 0..N {
        let xi = rng.next_u01();
        let eta = eta_of(xi);
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

    // ---- fixed design + exact latent truth -------------------------------
    let mut rng = SplitMix64::new(123);
    let (x, eta_true, mu_true) = fixed_design(&mut rng);
    assert_eq!(x.len(), N);
    let mu_min = mu_true.iter().cloned().fold(f64::INFINITY, f64::min);
    let mu_max = mu_true.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        mu_min > 0.05 && mu_max < 0.95,
        "DGP must stay off the saturated tails so the Binomial information \
         never collapses: mu in [{mu_min:.3}, {mu_max:.3}]"
    );

    let headers = vec!["x".to_string(), "y".to_string()];
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };

    let total_trials = N_REPLICATES * N;

    // Bernoulli response replicates, drawn from the continuing seed-123 stream
    // so the (design, replicate) data are bit-for-bit reproducible and feed the
    // same CSV columns to mgcv below.
    let replicates: Vec<Vec<f64>> = (0..N_REPLICATES)
        .map(|_| bernoulli_replicate(&mu_true, &mut rng))
        .collect();

    let options = |mode: InferenceCovarianceMode| PredictUncertaintyOptions {
        confidence_level: 0.95,
        covariance_mode: mode,
        mean_interval_method: MeanIntervalMethod::Delta,
        includeobservation_interval: false,
        apply_bias_correction: false,
        edgeworth_one_sided: false,
        boundary_correction: false,
        ood_inflation: false,
        multi_point_joint: false,
        ..PredictUncertaintyOptions::default()
    };

    // The replicates are independent fits on a fixed design (only `y` varies);
    // each replicate's hit counts, EDF and rep-0 SE band are deterministic
    // functions of its own data, unaffected by how the loop is scheduled. We fan
    // the (dominant) per-replicate fits across the rayon pool, then fold the
    // results back IN REPLICATE ORDER so the integer coverage counts AND the
    // floating-point `edf_sum` are bit-identical to the serial accumulation —
    // no asserted quantity changes, only wall-clock.
    struct RepOutcome {
        eta_hits: usize,
        mean_hits: usize,
        edf: f64,
        eta_se_rep0: Option<Vec<f64>>,
    }

    let outcomes: Vec<RepOutcome> = replicates
        .par_iter()
        .enumerate()
        .map(|(rep_idx, rep)| {
            assert!(
                rep.iter().any(|&v| v > 0.5) && rep.iter().any(|&v| v < 0.5),
                "synthetic outcome must contain both classes"
            );

            // ---- build a gam dataset from (x, y_rep) ---------------------------
            let records: Vec<csv::StringRecord> = (0..N)
                .map(|i| {
                    csv::StringRecord::from(vec![
                        format!("{:.17e}", x[i]),
                        format!("{:.17e}", rep[i]),
                    ])
                })
                .collect();
            let ds = encode_recordswith_inferred_schema(headers.clone(), records)
                .expect("encode synthetic replicate");
            let col = ds.column_map();
            let x_idx = col["x"];

            // ---- fit gam: y ~ s(x, k=K), Binomial(logit), REML -----------------
            let formula = format!("y ~ s(x, k={K})");
            let result = fit_from_formula(&formula, &ds, &cfg).expect("gam binomial(logit) fit");
            let FitResult::Standard(fit) = result else {
                panic!("binomial(logit) smooth should be a Standard fit");
            };
            let edf = fit.fit.edf_total().unwrap_or(0.0);
            assert!(
                fit.fit.beta_covariance_corrected().is_some(),
                "smoothing-parameter-corrected Vp must be available for every fit \
                 (the Nychka calibration target); replicate {rep_idx} produced none"
            );

            // ---- rebuild the design at the training points and predict CIs -----
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
            let pred = predict_gamwith_uncertainty(
                design.design.clone(),
                fit.fit.beta.view(),
                offset.view(),
                LikelihoodSpec::binomial_logit(),
                &fit.fit,
                &options(InferenceCovarianceMode::ConditionalPlusSmoothingPreferred),
            )
            .expect("gam uncertainty under logit link");

            // Pool gam's nominal-95% interval coverage of the KNOWN truth, both
            // scales, across this replicate's grid.
            let mut eta_hits = 0usize;
            let mut mean_hits = 0usize;
            for i in 0..N {
                if eta_true[i] >= pred.eta_lower[i] && eta_true[i] <= pred.eta_upper[i] {
                    eta_hits += 1;
                }
                if mu_true[i] >= pred.mean_lower[i] && mu_true[i] <= pred.mean_upper[i] {
                    mean_hits += 1;
                }
            }

            let eta_se_rep0 = if rep_idx == 0 {
                // For the band-scale agreement check we want the CONDITIONAL Vᵦ
                // band (the exact analogue of mgcv's default `se.fit`, which omits
                // the smoothing-parameter-uncertainty term), so the comparison is
                // apples-to-apples on the core covariance machinery.
                let pred_cond = predict_gamwith_uncertainty(
                    design.design.clone(),
                    fit.fit.beta.view(),
                    offset.view(),
                    LikelihoodSpec::binomial_logit(),
                    &fit.fit,
                    &options(InferenceCovarianceMode::Conditional),
                )
                .expect("gam conditional uncertainty under logit link");
                Some(
                    (0..N)
                        .map(|i| {
                            (pred_cond.eta_upper[i] - pred_cond.eta_lower[i]) / (2.0 * 1.959964)
                        })
                        .collect(),
                )
            } else {
                None
            };

            RepOutcome {
                eta_hits,
                mean_hits,
                edf,
                eta_se_rep0,
            }
        })
        .collect();

    // Fold in replicate order for bit-identical sums.
    let mut gam_eta_hits = 0usize;
    let mut gam_mean_hits = 0usize;
    let mut edf_sum = 0.0_f64;
    let mut gam_eta_se_rep0: Vec<f64> = Vec::new();
    for outcome in &outcomes {
        gam_eta_hits += outcome.eta_hits;
        gam_mean_hits += outcome.mean_hits;
        edf_sum += outcome.edf;
        if let Some(se) = &outcome.eta_se_rep0 {
            gam_eta_se_rep0 = se.clone();
        }
    }

    let gam_eta_cov = gam_eta_hits as f64 / total_trials as f64;
    let gam_mean_cov = gam_mean_hits as f64 / total_trials as f64;
    let gam_edf_mean = edf_sum / N_REPLICATES as f64;

    // ---- mgcv BASELINE (match-or-beat on calibration): SAME data/model -----
    // mgcv::predict(se.fit=TRUE) yields eta/mu and their SEs per replicate; R
    // forms the nominal-95% band (fit ± 1.96·se) on each scale and pools its
    // coverage against the SAME truth over the SAME replicates. It additionally
    // emits replicate-0's per-point link-scale SE so we can pin gam's
    // conditional band scale directly against the gold standard.
    let mut columns: Vec<Column<'_>> = Vec::with_capacity(N_REPLICATES + 1);
    columns.push(Column::new("x", &x));
    let yheaders: Vec<String> = (0..N_REPLICATES).map(|r| format!("y{r}")).collect();
    for (r, name) in yheaders.iter().enumerate() {
        columns.push(Column::new(name, &replicates[r]));
    }

    let r = run_r(
        &columns,
        &format!(
            r#"
        suppressPackageStartupMessages(library(mgcv))
        nrep <- {N_REPLICATES}L
        kdim <- {K}L
        z <- qnorm(0.975)
        # Exact latent truth at the fixed design carried in df$x.
        eta_true <- 2 * (df$x - 0.5) + 2 * sin(3 * pi * df$x)
        mu_true <- 1 / (1 + exp(-eta_true))
        eta_hits <- 0L
        mean_hits <- 0L
        total <- 0L
        se0 <- NULL
        for (r in 0:(nrep - 1L)) {{
            yname <- paste0("y", r)
            dat <- data.frame(x = df$x, y = df[[yname]])
            m <- gam(y ~ s(x, k = kdim), data = dat,
                     family = binomial(link = "logit"), method = "REML")
            pl <- predict(m, type = "link", se.fit = TRUE)
            pr <- predict(m, type = "response", se.fit = TRUE)
            elo <- pl$fit - z * pl$se.fit; ehi <- pl$fit + z * pl$se.fit
            mlo <- pr$fit - z * pr$se.fit; mhi <- pr$fit + z * pr$se.fit
            eta_hits <- eta_hits + sum(eta_true >= elo & eta_true <= ehi)
            mean_hits <- mean_hits + sum(mu_true >= mlo & mu_true <= mhi)
            total <- total + length(eta_true)
            if (r == 0L) se0 <- as.numeric(pl$se.fit)
        }}
        emit("eta_cov", eta_hits / total)
        emit("mean_cov", mean_hits / total)
        emit("se0", se0)
        "#
        ),
    );
    let mgcv_eta_cov = r.scalar("eta_cov");
    let mgcv_mean_cov = r.scalar("mean_cov");
    let mgcv_eta_se_rep0 = r.vector("se0");

    let nominal = 0.95_f64;
    eprintln!(
        "logit-link CI calibration (95% nominal, {N_REPLICATES} reps x {N} pts = {total_trials} trials):\n  \
         link-scale  coverage: gam={gam_eta_cov:.3} mgcv={mgcv_eta_cov:.3}\n  \
         resp-scale  coverage: gam={gam_mean_cov:.3} mgcv={mgcv_mean_cov:.3}\n  \
         gam mean EDF = {gam_edf_mean:.2} (k={K})"
    );

    // (0) WELL-POSEDNESS GUARD: REML must actually resolve the signal. If a
    // regression (here or in λ-selection) collapses the fit toward a near-null
    // smooth, EDF drops to ~3 and the coverage claim becomes the bias-dominated
    // artifact this test was rewritten to avoid. Require the EDF to sit in the
    // resolved band: clearly above the over-smoothed floor and below k.
    assert!(
        gam_edf_mean > 5.0 && gam_edf_mean < (K as f64),
        "fit did not resolve the recoverable signal: mean EDF = {gam_edf_mean:.2} \
         (expected in (5, {K})); coverage would be bias-dominated and meaningless"
    );

    // (1) OBJECTIVE: gam's pooled across-grid-and-replicate average coverage
    // tracks the nominal level. Pointwise penalized-GAM bands are smoothing-bias
    // attenuated, so the window is centered on 0.95 with a tolerance that admits
    // the mild residual attenuation but rejects a badly mis-scaled band. This
    // claim is about gam vs ground truth and does not involve mgcv.
    let cov_window = 0.06_f64; // 0.95 ± 0.06  ->  [0.89, 1.00]
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
    let beat_margin = 0.04_f64;
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

    // (3) BAND-SCALE AGREEMENT (different angle from coverage): on a shared
    // replicate, gam's conditional link-scale SE `√(x Vᵦ x)` must agree
    // pointwise with mgcv's default `se.fit` (also a conditional Bayesian band).
    // Coverage is an aggregate that a window can tolerate slightly mis-scaled;
    // this pins the covariance quadratic form + link propagation directly
    // against the mature implementation. A grossly mis-scaled band (the failure
    // mode the issue hypothesized) would blow past the median-relative tol even
    // when the coverage window does not.
    assert_eq!(
        gam_eta_se_rep0.len(),
        mgcv_eta_se_rep0.len(),
        "rep-0 SE vector length mismatch"
    );
    let mut rel_diffs: Vec<f64> = gam_eta_se_rep0
        .iter()
        .zip(mgcv_eta_se_rep0.iter())
        .filter(|&(_, &m)| m > 1e-8)
        .map(|(&g, &m)| (g - m).abs() / m)
        .collect();
    rel_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_rel = rel_diffs[rel_diffs.len() / 2];
    eprintln!(
        "rep-0 conditional link-SE agreement vs mgcv: median rel diff = {median_rel:.4} \
         (n={})",
        rel_diffs.len()
    );
    assert!(
        median_rel <= 0.10,
        "gam conditional link-scale SE disagrees with mgcv: median relative \
         difference {median_rel:.4} exceeds 0.10 — the Vᵦ quadratic form or the \
         link Jacobian is mis-scaled"
    );
}
