//! KNOWN-mgcv-DEFECT regression test — the second linear predictor's
//! response-scale standard error must be a genuine per-row quadratic form, not a
//! recycled scalar.
//!
//! The defect (mgcv 1.9-4, `gumbls()` predict block)
//! ------------------------------------------------
//! `mgcv:::predict.gam` for the Gumbel location-scale family computes the
//! response-scale SE of the SECOND (log-scale) linear predictor as
//!
//!     vp[,2] <- abs(family$linfo[[2]]$mu.eta(eta[,2])) * sqrt(ve[2])
//!
//! `ve` is the n-by-2 matrix of per-row link-scale variances of the two linear
//! predictors. `ve[2]` indexes that matrix in column-major order, so it is the
//! single scalar `ve[2,1]` — the variance of the FIRST predictor at row 2 — not
//! the intended column `ve[,2]`. Consequence: the entire second-predictor
//! response SE is built from ONE recycled variance value. The implied link-scale
//! variance is therefore constant across all prediction rows, even though the
//! true per-row variance of a smooth log-scale predictor varies substantially.
//! The correct line is `sqrt(ve[,2])`.
//!
//! What this test asserts
//! ----------------------
//! * REPLICATE THE mgcv BUG (gated on `Rscript`/`mgcv`): fit `gumbls()` on a
//!   genuinely heteroscedastic dataset, then show mgcv's implied second-predictor
//!   link variance has coefficient-of-variation ≈ 0 (a recycled scalar) while the
//!   TRUE per-row link variance varies by a large factor, so mgcv's comp-2
//!   response SE is materially wrong (>30% relative error somewhere on the grid).
//!   If a future mgcv release fixes this, the assertion fires — by design — and we
//!   update the defect record.
//! * PROVE gam IS IMMUNE (always runs): gam's Gaussian location-scale fit carries
//!   a full joint posterior covariance. The SE of the SECOND (log-σ) linear
//!   predictor, formed as the genuine per-row quadratic form
//!   `sqrt(x_rᵀ Σ_scale x_r)`, varies across the grid — it is NOT collapsed to a
//!   single recycled value, which is exactly the property mgcv's `gumbls()` loses.
//!
//! gam has no Gumbel location-scale family, so the mgcv arm exercises the family
//! that actually carries the bug while the gam arm exercises gam's nearest
//! location-scale construction (Gaussian, mean smooth + log-σ smooth). The point
//! of comparison is the DEFECT CLASS — "does the second predictor's response SE
//! degenerate to a recycled scalar?" — not a coefficient-for-coefficient match.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::solver::estimate::BlockRole;
use gam::test_support::reference::{Column, r_package_available, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use csv::StringRecord;
use ndarray::Array2;

/// Deterministic heteroscedastic Gaussian data: μ(x) = 2·sin(3x), and a log-scale
/// that genuinely varies with x so a fitted log-σ smooth has a non-constant
/// pointwise variance. Box-Muller standard normals from a fixed LCG keep the data
/// RNG-free and identical for the gam and mgcv arms.
fn heteroscedastic_xy(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    // Tiny deterministic LCG → uniforms → Box-Muller normals.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // top 53 bits → (0,1)
        (((state >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    };
    for i in 0..n {
        let xi = (i as f64 + 0.5) / n as f64;
        let u1 = next_unit();
        let u2 = next_unit();
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let mu = 2.0 * (3.0 * xi).sin();
        // σ(x) sweeps exp(-0.3)≈0.74 → exp(0.3)≈1.35 (variance varies ≈ 3.3×).
        let sigma = (0.6 * xi - 0.3).exp();
        x.push(xi);
        y.push(mu + sigma * z);
    }
    (x, y)
}

fn encode_xy(x: &[f64], y: &[f64]) -> gam::data::EncodedDataset {
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode dataset")
}

/// Coefficient of variation (sd/|mean|) of a strictly-positive vector.
fn coeff_of_variation(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    var.sqrt() / mean.abs()
}

#[test]
fn second_predictor_response_se_is_not_a_recycled_scalar() {
    init_parallelism();

    let n = 600;
    let (x, y) = heteroscedastic_xy(n);
    let data = encode_xy(&x, &y);

    // ---- gam: Gaussian location-scale (mean smooth + log-σ smooth) --------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("s(x)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &data, &cfg).expect("gam gaulss fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("expected a GaussianLocationScale fit for a Gaussian noise_formula model");
    };

    let uf = &fit.fit.fit;
    let p_mu = uf
        .block_by_role(BlockRole::Location)
        .expect("Location (mean) block")
        .beta
        .len();
    let scale_block = uf
        .block_by_role(BlockRole::Scale)
        .expect("Scale (log-σ) block");
    let beta_scale = scale_block.beta.clone();
    let p_sigma = beta_scale.len();
    assert!(
        p_sigma >= 3,
        "smooth log-σ block must carry a multi-column basis (got {p_sigma})"
    );
    let vb = uf
        .covariance_conditional
        .as_ref()
        .expect("location-scale fit must carry a joint posterior covariance");
    assert!(
        vb.nrows() >= p_mu + p_sigma,
        "joint covariance ({}×{}) too small for blocks p_mu={p_mu} p_sigma={p_sigma}",
        vb.nrows(),
        vb.ncols()
    );

    // ---- rebuild the log-σ (noise) design on an interior grid -------------
    let grid: Vec<f64> = (0..40).map(|i| 0.04 + 0.92 * i as f64 / 39.0).collect();
    let x_idx = data.column_map()["x"];
    let mut grid_rows = Array2::<f64>::zeros((grid.len(), data.headers.len()));
    for (i, &xg) in grid.iter().enumerate() {
        grid_rows[[i, x_idx]] = xg;
    }
    let noise_design = build_term_collection_design(grid_rows.view(), &fit.fit.noisespec_resolved)
        .expect("rebuild log-σ design at grid");
    assert_eq!(
        noise_design.design.ncols(),
        p_sigma,
        "noise design columns must match log-σ coefficient count"
    );
    let xs = noise_design.design.to_dense();
    let eta_scale = noise_design.design.apply(&beta_scale);

    // SE of the SECOND linear predictor as a genuine per-row quadratic form
    // sqrt(x_rᵀ Σ_scale x_r); response-scale σ SE via δ-method = exp(η)·SE(η).
    let mut link_var2 = Vec::with_capacity(grid.len());
    let mut resp_se2 = Vec::with_capacity(grid.len());
    for r in 0..xs.nrows() {
        let xr = xs.row(r);
        let mut var = 0.0;
        for i in 0..p_sigma {
            let xi = xr[i];
            if xi == 0.0 {
                continue;
            }
            for j in 0..p_sigma {
                var += xi * vb[[p_mu + i, p_mu + j]] * xr[j];
            }
        }
        assert!(
            var > 0.0 && var.is_finite(),
            "scale-block link variance must be finite and positive (row {r}: {var})"
        );
        link_var2.push(var);
        resp_se2.push(eta_scale[r].exp() * var.sqrt());
    }

    // gam IMMUNITY: the second predictor's link variance is a real function of x,
    // not a recycled scalar — its coefficient of variation is well above zero.
    let gam_var2_cv = coeff_of_variation(&link_var2);
    assert!(
        resp_se2.iter().all(|&s| s.is_finite() && s > 0.0),
        "gam second-predictor response SE must be finite and positive at every grid point"
    );
    assert!(
        gam_var2_cv > 0.10,
        "gam's second-predictor link variance is suspiciously flat (CV={gam_var2_cv:.4}); \
         a correct per-row quadratic form over a smooth basis must vary across x"
    );

    // ---- mgcv arm: replicate the gumbls recycled-scalar defect ------------
    if !r_package_available("mgcv") {
        eprintln!(
            "SKIP mgcv-replicate arm: Rscript/mgcv unavailable. gam immunity arm \
             PASSED (second-predictor link-variance CV={gam_var2_cv:.4})."
        );
        return;
    }

    let r = run_r(
        &[Column::new("x", &x), Column::new("y", &y)],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        b  <- gam(list(y ~ s(x), ~ s(x)), family = gumbls(), data = df)
        grid <- seq(0.04, 0.96, length.out = 40)
        nd <- data.frame(x = grid)
        pr <- predict(b, nd, type = "response", se.fit = TRUE)   # buggy comp-2 SE
        pl <- predict(b, nd, type = "link",     se.fit = TRUE)   # correct link SE
        mu.eta2 <- abs(b$family$linfo[[2]]$mu.eta(pl$fit[, 2]))
        mgcv_resp_se2    <- pr$se.fit[, 2]
        correct_resp_se2 <- mu.eta2 * pl$se.fit[, 2]             # delta-method truth
        # link variance mgcv *implied* for comp-2 (the bug => a single scalar):
        implied_var2 <- (mgcv_resp_se2 / mu.eta2)^2
        true_var2    <- pl$se.fit[, 2]^2
        emit("implied_var2_cv", sd(implied_var2) / mean(implied_var2))
        emit("true_var2_spread", max(true_var2) / min(true_var2))
        emit("max_rel_err", max(abs(mgcv_resp_se2 - correct_resp_se2) / correct_resp_se2))
        "#
        .to_string(),
    );
    let mgcv_implied_cv = r.scalar("implied_var2_cv");
    let mgcv_true_spread = r.scalar("true_var2_spread");
    let mgcv_max_rel_err = r.scalar("max_rel_err");

    eprintln!(
        "gumbls defect: mgcv implied-var2 CV={mgcv_implied_cv:.3e} (≈0 ⇒ recycled scalar), \
         true-var2 spread={mgcv_true_spread:.2}×, comp-2 response-SE max rel err={mgcv_max_rel_err:.3}; \
         gam second-predictor var CV={gam_var2_cv:.4}"
    );

    // REPLICATE: mgcv's implied comp-2 link variance is a recycled scalar …
    assert!(
        mgcv_implied_cv < 1e-6,
        "expected mgcv gumbls comp-2 link variance to be a recycled scalar (CV≈0), \
         got CV={mgcv_implied_cv:.3e} — has mgcv fixed the `sqrt(ve[2])` bug? Update this record."
    );
    // … even though the TRUE per-row variance varies by a large factor …
    assert!(
        mgcv_true_spread > 1.5,
        "the true comp-2 link variance must genuinely vary for this to be a real \
         defect (max/min spread={mgcv_true_spread:.2}, expected > 1.5)"
    );
    // … so mgcv's comp-2 response SE is materially wrong somewhere on the grid.
    assert!(
        mgcv_max_rel_err > 0.3,
        "mgcv gumbls comp-2 response SE should be materially wrong vs the delta-method \
         truth (max rel err={mgcv_max_rel_err:.3}, expected > 0.3)"
    );

    // CROSS-CHECK: gam's second-predictor variance varies by orders of magnitude
    // more than mgcv's recycled scalar — the defect class is absent in gam.
    assert!(
        gam_var2_cv > 1e3 * mgcv_implied_cv.max(1e-12),
        "gam's second-predictor variation (CV={gam_var2_cv:.4}) should dwarf mgcv's \
         recycled-scalar variation (CV={mgcv_implied_cv:.3e})"
    );
}
