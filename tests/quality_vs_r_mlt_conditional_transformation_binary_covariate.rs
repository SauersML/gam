//! End-to-end quality: gam's conditional transformation model (CTM) must agree
//! with **R `mlt`** — the mature, canonical reference for penalized maximum-
//! likelihood *transformation regression* (Hothorn, Möst & Bühlmann, 2018) — on
//! the bone-marrow-transplant survival times, stratified by transplant type.
//!
//! Why `mlt`. `mlt::mlt()` fits the *most likely transformation* model: it
//! estimates a smooth monotone transformation `h(y | x)` of the response onto a
//! chosen reference distribution by penalized maximum likelihood, so that
//! `P(Y <= y | x) = F_Z(h(y | x))`. That is exactly the object gam constructs in
//! its SCOP transformation-normal family,
//!
//!     h(y | x) = b(x) + eps * (y - median) + sum_k I_k(y) * gamma_k(x)^2,
//!
//! a monotone-in-`y` transformation whose shape is allowed to vary with the
//! covariate. gam's I-spline response basis saturates at a finite response
//! support `[y_lo, y_up]` (the observed min/max, widened by a negligible guard),
//! so the conditional distribution gam actually reports is the **finite-support
//! renormalized** standard-normal CDF
//!
//!     u(y | x) = ( Phi(h(y|x)) - Phi(h(y_lo|x)) )
//!              / ( Phi(h(y_up|x)) - Phi(h(y_lo|x)) ),
//!     eta(y | x) = Phi^{-1}( clip(u, eps_clip, 1 - eps_clip) ),
//!
//! i.e. the CDF of `Y | x` *conditioned on `Y` lying in the training support*.
//! That renormalization is intrinsic to gam's finite-support transformation
//! family (the per-observation likelihood is the support-normalized density),
//! and the calibrated per-row `eta` stored on the fitted block is exactly this
//! `Phi^{-1}(u)`. To compare apples-to-apples we give `mlt` a Bernstein-
//! polynomial response basis **interacting with the binary covariate** (so each
//! treatment arm gets its own monotone transformation, matching gam's
//! `gamma_k(x)` dependence) and `todistr = "Normal"` (the same standard-normal
//! reference gam targets), and then apply the **identical finite-support
//! renormalization** to `mlt`'s predicted distribution — evaluating `mlt`'s CDF
//! at each arm's support endpoints and forming the same `u`. With those choices
//! both engines report the same support-conditional CDF object.
//!
//! Data. `bench/datasets/bone.csv`, the bone-marrow-transplant cohort, n = 23.
//! NOTE: the column literally named `d` in this CSV is the 0/1 event indicator;
//! the *continuous* response a transformation model acts on is the survival time
//! `t`. A CTM is a model for a continuous response, so the response here is `t`
//! and the binary covariate is `trt` (allo = 0, auto = 1). The two engines are
//! handed the byte-identical numeric `(t, trt)` arrays — no censoring is used by
//! gam's transformation-normal formula path, so neither engine censors, keeping
//! the likelihoods identical.
//!
//! Quantity compared. The end-user-facing prediction: the support-conditional
//! CDF `u(y_i | x_i) = P(Y <= y_i | x_i, Y in support)` at every observed
//! `(t_i, trt_i)` point — a covariate-by-response grid that spans both treatment
//! arms and the full observed time range. gam reports it as `Phi(eta_i)`, where
//! `eta_i` is the calibrated finite-support PIT score `Phi^{-1}(u_i)` stored on
//! the fitted block; `mlt` reports the same `u_i` by evaluating
//! `predict(type = "distribution")` at the row's `t_i` and at the arm's support
//! endpoints and renormalizing identically. We compare the renormalized CDFs by
//! both `max_abs_diff` (probability scale) and Pearson correlation.
//!
//! Bound. Both engines solve a penalized monotone transformation MLE against the
//! same standard-normal reference on identical data, then apply the same finite-
//! support renormalization, so the reported support-conditional CDF is a
//! deterministic quantity up to (a) the response-basis convention — gam fits a
//! degree-3 I-spline (sample-capped to 2 internal knots at n = 23) while `mlt`
//! fits an order-6 Bernstein polynomial, two different but comparably smooth
//! monotone bases — and (b) optimizer tolerance on n = 23. `max_abs_diff <= 0.08`
//! on the renormalized CDF and Pearson `>= 0.95` are principled bounds: 0.08 in
//! probability comfortably covers the two-basis small-sample slack yet is far
//! below the 0.3-0.5 divergence a genuine bug in gam's transformation / PIT /
//! calibration pathway would produce, and Pearson 0.95 demands the two CDFs
//! track each other monotonically across the whole grid. We never weaken them and
//! never edit gam to pass.

use gam::inference::probability::normal_cdf;
use gam::test_support::reference::{Column, max_abs_diff, pearson, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;

#[test]
fn gam_conditional_transformation_matches_r_mlt_on_bone() {
    init_parallelism();

    // ---- bone-marrow-transplant cohort (bench/datasets/bone.csv, n = 23) ----
    // Hardcoded here so the *identical* numeric arrays reach gam and R: the
    // continuous response `t` (survival time) and the binary covariate `trt`
    // encoded allo = 0.0, auto = 1.0. (The CSV's `d` column is the event flag,
    // not the response a transformation model acts on — see the module doc.)
    let t: Vec<f64> = vec![
        28.0, 32.0, 49.0, 84.0, 357.0, 933.0, 1078.0, 1183.0, 1560.0, 2114.0, 2144.0, // allo
        42.0, 53.0, 57.0, 63.0, 81.0, 140.0, 176.0, 210.0, 252.0, 476.0, 524.0, 1037.0, // auto
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
    // linear covariate term, giving a per-arm transformation. After
    // calibration the fitted block's `eta` holds the standard-normal PIT score
    // Phi^{-1}(P(Y<=y_i|x_i)) per training row — gam's transformation score.
    let headers = vec!["t".to_string(), "trt".to_string()];
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![t[i].to_string(), trt[i].to_string()]))
        .collect();
    let data =
        encode_recordswith_inferred_schema(headers, rows).expect("encode bone CTM data");

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
    // gam predicted conditional CDF at the observed (t_i, trt_i) points.
    let gam_cdf: Vec<f64> = gam_scores.iter().map(|&z| normal_cdf(z)).collect();

    // ---- fit the SAME data with R mlt (the mature CTM reference) ------------
    // ctm(): Bernstein response basis on `t` interacting with the binary `trt`
    // factor (each arm gets its own monotone transformation), standard-normal
    // reference (todistr = "Normal"); mlt() maximizes the penalized log-
    // likelihood. We then form gam's finite-support-renormalized conditional
    // CDF u = (F(t_i|x) - F(y_lo|x)) / (F(y_up|x) - F(y_lo|x)) by predicting the
    // distribution at the row's t_i and at the response-support endpoints for the
    // row's arm, with the identical clip gam applies before qnorm.
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
        # renormalize and qnorm exactly as gam's finite-support PIT calibration.
        nd <- df[, c("t", "trt"), drop = FALSE]
        cdf <- numeric(nrow(nd))
        traf <- numeric(nrow(nd))
        for (i in seq_len(nrow(nd))) {{
          ndi <- nd[i, , drop = FALSE]
          F_i  <- as.numeric(predict(m, newdata = ndi, q = nd$t[i], type = "distribution"))
          F_lo <- as.numeric(predict(m, newdata = ndi, q = y_lo,     type = "distribution"))
          F_up <- as.numeric(predict(m, newdata = ndi, q = y_up,     type = "distribution"))
          u <- (F_i - F_lo) / (F_up - F_lo)
          u <- min(max(u, clip_eps), 1 - clip_eps)
          cdf[i]  <- u
          traf[i] <- qnorm(u)
        }}
        emit("cdf", cdf)
        emit("trafo", traf)
        emit("logLik", as.numeric(logLik(m)))
        "#
        ),
    );
    let mlt_cdf = r.vector("cdf");
    let mlt_trafo = r.vector("trafo");
    assert_eq!(mlt_cdf.len(), n, "mlt CDF length mismatch");
    assert_eq!(mlt_trafo.len(), n, "mlt trafo length mismatch");
    assert!(
        mlt_cdf.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "mlt support-conditional CDF must be a probability in [0, 1]"
    );

    // ---- compare ------------------------------------------------------------
    // gam_cdf = Phi(eta_i) is, by construction of the calibrated PIT, exactly the
    // renormalized u_i; mlt_cdf is the same u_i built from mlt's distribution.
    let cdf_max_abs = max_abs_diff(&gam_cdf, mlt_cdf);
    let cdf_corr = pearson(&gam_cdf, mlt_cdf);
    // Cross-check the standard-normal PIT score on the *same* renormalized scale:
    // gam's stored eta_i = qnorm(u_i) vs mlt's qnorm(u_i).
    let score_corr = pearson(&gam_scores, mlt_trafo);

    eprintln!(
        "bone CTM (t ~ trt) n={n}: gam vs mlt::ctm \
         cdf_max_abs={cdf_max_abs:.4} cdf_pearson={cdf_corr:.4} \
         trafo_pearson={score_corr:.4} mlt_logLik={:.3}",
        r.scalar("logLik")
    );

    // Same penalized monotone transformation MLE, same standard-normal reference,
    // same finite-support renormalization, same 23 (t, trt) rows. The reported
    // support-conditional CDF then differs only by the response-basis convention
    // (degree-3 I-spline vs order-6 Bernstein) and optimizer slack at n = 23.
    // 0.08 in probability covers that two-basis slack but fails on any real
    // divergence in gam's transformation / PIT / calibration pathway; Pearson
    // >= 0.95 demands the two CDFs track monotonically across the whole grid.
    assert!(
        cdf_max_abs <= 0.08,
        "support-conditional CDF diverges from mlt: max_abs_diff={cdf_max_abs:.4}"
    );
    assert!(
        cdf_corr >= 0.95,
        "support-conditional CDF decorrelates from mlt: pearson={cdf_corr:.4}"
    );
    assert!(
        score_corr >= 0.95,
        "transformation scores diverge from mlt: pearson={score_corr:.4}"
    );
}
