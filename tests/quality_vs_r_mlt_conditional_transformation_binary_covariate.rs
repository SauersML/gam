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
//!     P(Y <= y | x) = Phi( h(y | x) ),   (standard-normal reference)
//!
//! a monotone-in-`y` transformation whose shape is allowed to vary with the
//! covariate. We make the comparison apples-to-apples by giving `mlt` a
//! Bernstein-polynomial response basis **interacting with the binary covariate**
//! (so each treatment arm gets its own monotone transformation, matching gam's
//! `gamma_k(x)` dependence) and `todistr = "Normal"` (the same standard-normal
//! reference gam targets). With those choices both engines estimate the identical
//! penalized-MLE conditional-distribution object.
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
//! Quantity compared. The end-user-facing prediction: the conditional CDF
//! `P(Y <= y_i | x_i)` at every observed `(t_i, trt_i)` point — a covariate-by-
//! response grid that spans both treatment arms and the full observed time range.
//! gam reports it as `Phi(eta_i)`, where `eta_i` is the calibrated standard-
//! normal PIT score stored on the fitted block; `mlt` reports it as
//! `predict(type = "distribution")`. We additionally compare the *transformation
//! scores* themselves — gam's `eta_i = Phi^{-1}(P)` against `mlt`'s
//! `predict(type = "trafo")` (which, with `todistr = "Normal"`, is on the same
//! standard-normal scale) — via Pearson correlation.
//!
//! Bound. Both engines solve the same penalized transformation MLE against the
//! same standard-normal reference on identical data, so the predicted conditional
//! CDF is a deterministic parametric quantity up to (a) penalty/knot-placement
//! convention and (b) optimizer tolerance on n = 23. `max_abs_diff <= 0.03` on
//! the conditional CDF (a probability, so an absolute bound is the natural scale)
//! and Pearson `>= 0.92` on the transformation scores are tight, principled
//! bounds: 0.03 in probability is well inside the small-sample basis-convention
//! slack yet fails on any genuine divergence in gam's transformation / PIT /
//! prediction pathway. We never weaken them and never edit gam to pass.

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
    // likelihood. predict() returns the conditional distribution and the
    // transformation on the standard-normal scale at the same training points.
    let r = run_r(
        &[Column::new("t", &t), Column::new("trt", &trt)],
        r#"
        suppressPackageStartupMessages({
          library(mlt)
          library(variables)
          library(basefun)
        })
        df$trt <- factor(ifelse(df$trt > 0.5, "auto", "allo"), levels = c("allo", "auto"))
        # Continuous response variable with a generous support bracketing the data.
        yvar <- numeric_var("t", support = c(min(df$t), max(df$t)),
                            bounds = c(0, Inf))
        # Bernstein-polynomial transformation of t (order 6 => smooth, monotone),
        # interacting with the treatment factor so each arm gets its own h(.).
        bb <- Bernstein_basis(yvar, order = 6, ui = "increasing")
        ctm_obj <- ctm(response = bb,
                       interacting = as.basis(~ trt, data = df),
                       todistr = "Normal")
        m <- mlt(ctm_obj, data = df)
        # Conditional CDF P(Y<=t_i|trt_i) and the standard-normal-scale trafo at
        # the observed points (q = each row's own t, conditioned on its trt).
        nd <- df[, c("t", "trt"), drop = FALSE]
        cdf <- numeric(nrow(nd))
        traf <- numeric(nrow(nd))
        for (i in seq_len(nrow(nd))) {
          ndi <- nd[i, , drop = FALSE]
          cdf[i] <- as.numeric(predict(m, newdata = ndi, q = nd$t[i],
                                       type = "distribution"))
          traf[i] <- as.numeric(predict(m, newdata = ndi, q = nd$t[i],
                                        type = "trafo"))
        }
        emit("cdf", cdf)
        emit("trafo", traf)
        emit("logLik", as.numeric(logLik(m)))
        "#,
    );
    let mlt_cdf = r.vector("cdf");
    let mlt_trafo = r.vector("trafo");
    assert_eq!(mlt_cdf.len(), n, "mlt CDF length mismatch");
    assert_eq!(mlt_trafo.len(), n, "mlt trafo length mismatch");
    assert!(
        mlt_cdf.iter().all(|&p| (0.0..=1.0).contains(&p)),
        "mlt conditional CDF must be a probability in [0, 1]"
    );

    // ---- compare ------------------------------------------------------------
    let cdf_max_abs = max_abs_diff(&gam_cdf, mlt_cdf);
    let score_corr = pearson(&gam_scores, mlt_trafo);
    let cdf_corr = pearson(&gam_cdf, mlt_cdf);

    eprintln!(
        "bone CTM (t ~ trt) n={n}: gam vs mlt::ctm \
         cdf_max_abs={cdf_max_abs:.4} cdf_pearson={cdf_corr:.4} \
         trafo_pearson={score_corr:.4} mlt_logLik={:.3}",
        r.scalar("logLik")
    );

    // Same penalized transformation MLE, same standard-normal reference, same
    // 23 (t, trt) rows: the predicted conditional CDF is a deterministic
    // parametric quantity up to small-sample basis-convention / optimizer
    // slack. 0.03 in probability and Pearson >= 0.92 on the transformation
    // scores are tight bounds that still tolerate that slack but fail on any
    // real divergence in gam's transformation / PIT / prediction pathway.
    assert!(
        cdf_max_abs <= 0.03,
        "predicted conditional CDF diverges from mlt: max_abs_diff={cdf_max_abs:.4}"
    );
    assert!(
        score_corr >= 0.92,
        "transformation scores diverge from mlt: pearson={score_corr:.4}"
    );
}
