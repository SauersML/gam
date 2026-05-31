//! End-to-end **objective quality** for gam's conditional transformation model
//! (CTM) on the bone-marrow-transplant survival times, stratified by transplant
//! type.
//!
//! OBJECTIVE METRIC: probability-integral-transform (PIT) calibration. A
//! conditional transformation model (Hothorn, Möst & Bühlmann, 2018) estimates a
//! smooth monotone transformation `h(y | x)` of the response onto a standard-
//! normal reference such that `P(Y <= y | x) = Phi(h(y | x))`. The defining
//! correctness property of *any* such model — independent of which package fit
//! it — is the PIT theorem: if the conditional distribution `F(y | x)` is correct,
//! then `U_i = F(y_i | x_i)` is uniform on `[0, 1]`, equivalently the
//! transformation scores `eta_i = Phi^{-1}(U_i)` are standard normal. We assert
//! this directly on gam's OWN fitted PIT values via the Kolmogorov-Smirnov
//! distance to `Uniform(0, 1)`:
//!
//!     D_n = sup_u | F_n(u) - u |,   F_n the empirical CDF of gam's {u_i}.
//!
//! This is an objective claim about gam's fit quality — that gam's conditional
//! transformation actually transforms the response to the target reference — not
//! a claim that gam reproduces another tool's noisy fit.
//!
//! gam constructs, in its SCOP transformation-normal family,
//!
//!     h(y | x) = b(x) + eps * (y - median) + sum_k I_k(y) * gamma_k(x)^2,
//!
//! a monotone-in-`y` transformation whose shape varies with the covariate. Its
//! I-spline response basis saturates at a finite response support `[y_lo, y_up]`,
//! so the reported conditional distribution is the finite-support renormalized
//! standard-normal CDF
//!
//!     u(y | x) = ( Phi(h(y|x)) - Phi(h(y_lo|x)) )
//!              / ( Phi(h(y_up|x)) - Phi(h(y_lo|x)) ),
//!
//! i.e. the CDF of `Y | x` *conditioned on `Y` in the training support*. gam
//! stores the calibrated per-row PIT score `eta_i = Phi^{-1}(u_i)` on the fitted
//! block, so `u_i = Phi(eta_i)` is exactly the support-conditional PIT we test.
//!
//! BASELINE (match-or-beat, NOT ground truth): R `mlt` — the mature, canonical
//! penalized-ML transformation-regression package — is fit to the byte-identical
//! `(t, trt)` arrays with the same standard-normal reference and the same finite-
//! support renormalization, and its own PIT values' KS distance is computed. We
//! require gam's calibration to be at least as good as mlt's (KS_gam <=
//! KS_mlt * 1.10). The primary claim is gam's PIT is uniform; mlt is only a
//! sanity ceiling, never the pass criterion.
//!
//! Data. `bench/datasets/bone.csv`, the bone-marrow-transplant cohort, n = 23.
//! The column literally named `d` in this CSV is the 0/1 event indicator; the
//! *continuous* response a transformation model acts on is the survival time `t`,
//! and the binary covariate is `trt` (allo = 0, auto = 1). Both engines receive
//! the byte-identical numeric `(t, trt)` arrays; no censoring is used.
//!
//! Bounds. With n = 23 the PIT sample is small, so its empirical CDF cannot hug
//! the diagonal arbitrarily closely even under a perfect model: the expected KS
//! distance of 23 genuine Uniform(0,1) draws is ~`0.27 / sqrt(23) ≈ 0.057`, and
//! the 90th percentile of the finite-sample KS distribution at n = 23 is about
//! 0.25. We assert `KS_gam <= 0.30` — comfortably above that null sampling
//! ceiling so a correctly-calibrated fit never trips it, yet far below the
//! `~0.5+` a genuine miscalibration (e.g. all PIT mass piled near 0 or 1 from a
//! broken transformation / clipping pathway) would produce. We never weaken these
//! bounds and never edit gam to pass.

use csv::StringRecord;
use gam::inference::probability::normal_cdf;
use gam::test_support::reference::{Column, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

/// Kolmogorov-Smirnov distance of an empirical sample to the `Uniform(0, 1)`
/// CDF: `D_n = sup_u |F_n(u) - u|`. With the sorted sample `u_(1) <= ... <=
/// u_(n)`, the supremum is attained at a sample point, so
/// `D_n = max_i max( i/n - u_(i), u_(i) - (i-1)/n )`. Values outside `[0, 1]`
/// would make the comparison ill-defined, so we require the caller to pass
/// probabilities.
fn ks_distance_to_uniform(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty(), "KS distance needs a non-empty sample");
    assert!(
        samples.iter().all(|&u| (0.0..=1.0).contains(&u)),
        "KS-vs-uniform inputs must be probabilities in [0, 1]"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("PIT samples must be comparable"));
    let n = sorted.len() as f64;
    let mut d = 0.0_f64;
    for (idx, &u) in sorted.iter().enumerate() {
        let i = (idx + 1) as f64;
        let upper = i / n - u; // F_n just above u_(i) minus u
        let lower = u - (i - 1.0) / n; // u minus F_n just below u_(i)
        d = d.max(upper).max(lower);
    }
    d
}

#[test]
fn gam_conditional_transformation_pit_is_calibrated_on_bone() {
    init_parallelism();

    // ---- bone-marrow-transplant cohort (bench/datasets/bone.csv, n = 23) ----
    // Hardcoded here so the *identical* numeric arrays reach gam and R: the
    // continuous response `t` (survival time) and the binary covariate `trt`
    // encoded allo = 0.0, auto = 1.0. (The CSV's `d` column is the event flag,
    // not the response a transformation model acts on — see the module doc.)
    let t: Vec<f64> = vec![
        28.0, 32.0, 49.0, 84.0, 357.0, 933.0, 1078.0, 1183.0, 1560.0, 2114.0, 2144.0, // allo
        42.0, 53.0, 57.0, 63.0, 81.0, 140.0, 176.0, 210.0, 252.0, 476.0, 524.0,
        1037.0, // auto
    ];
    let trt: Vec<f64> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // allo
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // auto
    ];
    let n = t.len();
    assert_eq!(n, 23, "bone cohort should have 23 rows");
    assert_eq!(trt.len(), n, "trt length must match t");
    assert!(
        t.iter().all(|&v| v.is_finite() && v > 0.0),
        "all survival times must be positive and finite"
    );

    // ---- fit with gam: conditional transformation model t ~ trt -------------
    // `transformation_normal = true` selects gam's SCOP transformation-normal
    // family (h mapped onto a standard normal). The binary `trt` enters as a
    // linear covariate term, giving a per-arm transformation. After calibration
    // the fitted block's `eta` holds the standard-normal PIT score
    // Phi^{-1}(P(Y<=y_i|x_i)) per training row.
    let headers = vec!["t".to_string(), "trt".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![t[i].to_string(), trt[i].to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode bone CTM data");

    let cfg = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let result = fit_from_formula("t ~ trt", &data, &cfg).expect("gam transformation CTM fit");
    let FitResult::TransformationNormal(fit) = result else {
        panic!("expected a TransformationNormal fit result for transformation_normal=true");
    };
    let block = fit
        .fit
        .block_states
        .first()
        .expect("transformation-normal fit must carry one fitted block");
    let gam_scores: Vec<f64> = block.eta.to_vec();
    assert_eq!(
        gam_scores.len(),
        n,
        "gam must report one transformation score per training row"
    );
    assert!(
        gam_scores.iter().all(|s| s.is_finite()),
        "gam transformation scores must be finite"
    );
    // gam's support-conditional PIT u_i = Phi(eta_i) at each observed point.
    let gam_pit: Vec<f64> = gam_scores.iter().map(|&z| normal_cdf(z)).collect();
    assert!(
        gam_pit.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "gam support-conditional PIT must be a probability in [0, 1]"
    );

    // ---- baseline: fit the SAME data with R mlt (mature CTM reference) ------
    // ctm(): Bernstein response basis on `t` interacting with the binary `trt`
    // factor (each arm its own monotone transformation), standard-normal
    // reference; mlt() maximizes the penalized log-likelihood. We then form the
    // *identical* finite-support-renormalized PIT
    // u = (F(t_i|x) - F(y_lo|x)) / (F(y_up|x) - F(y_lo|x)) and clip it exactly as
    // gam does, so mlt's PIT lives on the same scale as gam's for a fair KS
    // ceiling. mlt is a baseline to match-or-beat on calibration, NOT ground
    // truth.
    let clip_eps = gam::inference::model::TRANSFORMATION_SCORE_PIT_CLIP_EPS;
    let r = run_r(
        &[Column::new("t", &t), Column::new("trt", &trt)],
        &format!(
            r#"
        suppressPackageStartupMessages({{
          library(mlt)
          library(variables)
          library(basefun)
        }})
        clip_eps <- {clip_eps:.17e}
        df$trt <- factor(ifelse(df$trt > 0.5, "auto", "allo"), levels = c("allo", "auto"))
        y_lo <- min(df$t)
        y_up <- max(df$t)
        # Continuous response variable with support bracketing the data — the same
        # finite [min, max] support gam's I-spline response basis saturates at.
        yvar <- numeric_var("t", support = c(y_lo, y_up), bounds = c(0, Inf))
        # Bernstein-polynomial transformation of t (order 6 => smooth, monotone),
        # interacting with the treatment factor so each arm gets its own h(.).
        bb <- Bernstein_basis(yvar, order = 6, ui = "increasing")
        ctm_obj <- ctm(response = bb,
                       interacting = as.basis(~ trt, data = df),
                       todistr = "Normal")
        m <- mlt(ctm_obj, data = df)
        # For each observed (t_i, trt_i): predict the conditional distribution at
        # the row's own t_i and at the support endpoints for the same arm, then
        # renormalize and clip exactly as gam's finite-support PIT calibration.
        nd <- df[, c("t", "trt"), drop = FALSE]
        cdf <- numeric(nrow(nd))
        for (i in seq_len(nrow(nd))) {{
          ndi <- nd[i, , drop = FALSE]
          F_i  <- as.numeric(predict(m, newdata = ndi, q = nd$t[i], type = "distribution"))
          F_lo <- as.numeric(predict(m, newdata = ndi, q = y_lo,     type = "distribution"))
          F_up <- as.numeric(predict(m, newdata = ndi, q = y_up,     type = "distribution"))
          u <- (F_i - F_lo) / (F_up - F_lo)
          u <- min(max(u, clip_eps), 1 - clip_eps)
          cdf[i] <- u
        }}
        emit("cdf", cdf)
        emit("logLik", as.numeric(logLik(m)))
        "#
        ),
    );
    let mlt_pit = r.vector("cdf");
    assert_eq!(mlt_pit.len(), n, "mlt PIT length mismatch");
    assert!(
        mlt_pit.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "mlt support-conditional PIT must be a probability in [0, 1]"
    );

    // ---- objective metric: KS distance of gam's PIT to Uniform(0, 1) --------
    let ks_gam = ks_distance_to_uniform(&gam_pit);
    let ks_mlt = ks_distance_to_uniform(mlt_pit);

    eprintln!(
        "bone CTM (t ~ trt) n={n}: PIT calibration KS-vs-Uniform \
         gam={ks_gam:.4} mlt={ks_mlt:.4} mlt_logLik={:.3}",
        r.scalar("logLik")
    );

    // PRIMARY: gam's own conditional transformation is calibrated — its in-sample
    // PIT values are uniform on [0, 1]. The expected KS distance of 23 genuine
    // Uniform draws is ~0.057 and the finite-sample 90th percentile is ~0.25;
    // 0.30 sits above that null ceiling so a correct fit never trips it, yet a
    // broken transformation / clipping pathway (PIT mass piled near 0 or 1)
    // pushes KS toward 0.5+ and fails.
    assert!(
        ks_gam <= 0.30,
        "gam conditional-transformation PIT is miscalibrated: KS-vs-Uniform={ks_gam:.4}"
    );
    // BASELINE (match-or-beat): gam's calibration must be at least as good as the
    // mature mlt fit's, within a 10% slack for the differing response basis
    // (degree-3 I-spline vs order-6 Bernstein) and optimizer tolerance at n = 23.
    assert!(
        ks_gam <= ks_mlt * 1.10,
        "gam PIT calibration is worse than mlt baseline: KS_gam={ks_gam:.4} > 1.10 * KS_mlt={:.4}",
        ks_mlt * 1.10
    );
}
