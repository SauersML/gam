//! End-to-end quality: gam's *additive* combination of two heterogeneous smooth
//! types — an isotropic 2-D thin-plate `s(x1, x2, bs="tp")` plus a separable
//! anisotropic tensor `te(z, w)` — must RECOVER THE KNOWN ADDITIVE TRUTH on
//! noise-free synthetic data.
//!
//! OBJECTIVE METRIC ASSERTED (truth recovery, not tool agreement):
//!   The data are generated *without noise* from a known additive function, so
//!   the exact penalized-likelihood target on the training points is the truth
//!   vector itself. The primary, pass/fail claim is therefore
//!       RMSE(gam_fitted, truth) <= 2% of the truth's signal range,
//!   i.e. gam's total additive fit reconstructs the generating surface to within
//!   a small fraction of the signal range. Matching mgcv is NOT the criterion.
//!
//! Reference tool (BASELINE TO MATCH-OR-BEAT on accuracy, not ground truth):
//!   `mgcv::gam(y ~ s(x1, x2, bs="tp", k=10) + te(z, w, k=6), method="REML")`.
//!   We additionally require gam's reconstruction error to be no worse than
//!   1.10x mgcv's reconstruction error against the *same truth*. mgcv is the
//!   mature de-facto standard, so beating-or-matching its accuracy on the truth
//!   is meaningful — but mgcv's own fitted output is never the target; the
//!   known truth is.
//!
//! Why this combination matters: real GAM applications almost never use a single
//! smooth. An additive model stacks one penalty block per smooth term into a
//! single penalized objective, and the REML optimizer selects a smoothing
//! parameter *per block* simultaneously. This is precisely where multi-smooth
//! aggregation bugs hide — penalty-matrix assembly that lets blocks cross-talk,
//! rank deficiency from mis-stacked null spaces, or a lambda-selection coupling.
//! Here the two blocks are deliberately of *different* mathematical character:
//! an isotropic rotation-invariant thin-plate radial smooth over (x1, x2), and a
//! separable row-wise-Kronecker tensor smooth over (z, w). A cross-block penalty
//! assembly or lambda-selection bug would corrupt the reconstructed surface and
//! break the truth-recovery bound, even if each block in isolation is correct.
//!
//! Data: deterministic synthetic, n=500, fixed seed=20260530. Covariates
//! x1, x2, z, w are independent uniform draws on [0,1] from a reproducible
//! SplitMix64 stream (identical bytes fed to gam and to mgcv via the shared
//! CSV). The response is the additive truth
//!     f1(x1, x2) = sin(pi*x1) * exp(-x2)        (smooth, thin-plate-representable)
//!   + f2(z, w)   = z^2 * cos(pi*w)              (separable, tensor-representable)
//! with no added noise, so `truth_i = f1_i + f2_i` is exactly the target the
//! penalized fit should reproduce on the training points.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityPair, relative_l2, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use std::f64::consts::PI;

/// SplitMix64 -> uniform [0,1). Deterministic, seedable, no external RNG crate:
/// guarantees gam and mgcv receive byte-identical covariates.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_unit(state: &mut u64) -> f64 {
    // Top 53 bits -> [0,1).
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn gam_additive_tp_plus_te_matches_mgcv() {
    init_parallelism();

    // ---- deterministic synthetic data: n=500, seed=20260530 ----------------
    const N: usize = 500;
    let mut state: u64 = 20260530;
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    let mut w = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut truth = Vec::with_capacity(N);
    for _ in 0..N {
        let a = next_unit(&mut state);
        let b = next_unit(&mut state);
        let c = next_unit(&mut state);
        let d = next_unit(&mut state);
        // Additive truth: thin-plate-friendly f1 + separable tensor f2, no noise.
        let f1 = (PI * a).sin() * (-b).exp();
        let f2 = c * c * (PI * d).cos();
        x1.push(a);
        x2.push(b);
        z.push(c);
        w.push(d);
        // Noise-free response == the generating truth on this point.
        truth.push(f1 + f2);
        y.push(f1 + f2);
    }
    // Signal range of the known additive truth — the natural absolute scale for
    // a reconstruction-error bar.
    let truth_min = truth.iter().copied().fold(f64::INFINITY, f64::min);
    let truth_max = truth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let signal_range = truth_max - truth_min;
    assert!(
        signal_range.is_finite() && signal_range > 0.0,
        "degenerate truth signal range: {signal_range}"
    );

    // ---- fit with gam: y ~ s(x1,x2,bs="tp",k=10) + te(z,w,k=6), REML --------
    let headers = ["x1", "x2", "z", "w", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                format!("{:.17e}", x1[i]),
                format!("{:.17e}", x2[i]),
                format!("{:.17e}", z[i]),
                format!("{:.17e}", w[i]),
                format!("{:.17e}", y[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode additive dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];
    let z_idx = col["z"];
    let w_idx = col["w"];

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x1, x2, bs=\"tp\", k=10) + te(z, w, k=6)", &ds, &cfg)
        .expect("gam additive tp+te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for an additive gaussian tp+te model");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam total fitted mean at the training points: rebuild the design from the
    // frozen spec (all blocks + intercept) at the observed covariates. Identity
    // link => design*beta is exactly the additive fitted mean.
    let mut grid = Array2::<f64>::zeros((N, ds.headers.len()));
    for i in 0..N {
        grid[[i, x1_idx]] = x1[i];
        grid[[i, x2_idx]] = x2[i];
        grid[[i, z_idx]] = z[i];
        grid[[i, w_idx]] = w[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild additive design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME additive model with mgcv (the mature reference) ------
    let r = run_r(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("z", &z),
            Column::new("w", &w),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(x1, x2, bs = "tp", k = 10) + te(z, w, k = 6),
                 data = df, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_fitted = r.vector("fitted");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_fitted.len(), N, "mgcv fitted length mismatch");

    // ---- OBJECTIVE METRIC: reconstruction error against the KNOWN truth -----
    // Both fits target the same noise-free additive surface; the truth vector is
    // the exact thing each should reproduce on the training points. We measure
    // each engine's RMSE against that truth and normalize by the signal range.
    let gam_err = rmse(&gam_fitted, &truth);
    let mgcv_err = rmse(mgcv_fitted, &truth);
    let gam_err_frac = gam_err / signal_range;

    // Context-only (NOT a pass criterion): how close the two engines' fitted
    // surfaces happen to be to each other.
    let rel_to_mgcv = relative_l2(&gam_fitted, mgcv_fitted);

    eprintln!(
        "additive s(x1,x2,tp,k=10)+te(z,w,k=6): n={N} signal_range={signal_range:.4} \
         gam_rmse_vs_truth={gam_err:.6} ({:.3}% of range) mgcv_rmse_vs_truth={mgcv_err:.6} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} rel_l2_gam_vs_mgcv={rel_to_mgcv:.5}",
        gam_err_frac * 100.0
    );
    eprintln!(
        "{}",
        QualityPair::error(
            "smooths",
            "quality_vs_mgcv_tensor_additive_tp_te",
            "rmse_vs_truth",
            gam_err,
            "mgcv",
            mgcv_err,
        )
        .line()
    );

    // PRIMARY CLAIM (truth recovery): on noise-free data the penalized additive
    // fit must reconstruct the generating surface to within 2% of its signal
    // range. This is an absolute, tool-independent accuracy bar — a cross-block
    // penalty-assembly or lambda-selection bug corrupts the reconstructed
    // surface and inflates this error well past 2%.
    assert!(
        gam_err_frac <= 0.02,
        "additive tp+te does not recover the known truth: \
         rmse_vs_truth={gam_err:.6} = {:.3}% of signal range {signal_range:.4} (bar: 2%)",
        gam_err_frac * 100.0
    );

    // MATCH-OR-BEAT (mgcv as accuracy baseline, not as ground truth): gam's
    // reconstruction error must be no worse than 1.10x mgcv's error against the
    // *same* truth. We never assert gam reproduces mgcv's fitted output; we only
    // require gam to be at least as accurate as the mature reference.
    assert!(
        gam_err <= mgcv_err * 1.10,
        "additive tp+te less accurate than mgcv on the known truth: \
         gam_rmse={gam_err:.6} vs mgcv_rmse={mgcv_err:.6} (allowed 1.10x)"
    );

    // EDF sanity (range only, NOT matched to mgcv): the additive model has two
    // penalized blocks (tp k=10, te k=6) plus an intercept, so the total
    // effective degrees of freedom must be non-trivial (> 1, the intercept
    // alone) and below the unpenalized parameter count of ~10 + 36 = 46. This
    // guards against a degenerate over-shrunk (edf -> 1) or unpenalized
    // (edf -> k) fit without asserting agreement with the reference's edf.
    assert!(
        gam_edf > 1.0 && gam_edf < 46.0,
        "additive tp+te total edf outside a sane signal-appropriate range: \
         gam_edf={gam_edf:.3} (expected 1 < edf < 46)"
    );
}
