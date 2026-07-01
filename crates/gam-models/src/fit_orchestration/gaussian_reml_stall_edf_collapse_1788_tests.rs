//! Regression for #1788 at the real smooth-spline pipeline layer.
//!
//! For an ordinary Gaussian additive fit `y ~ s(x1)+s(x2)+s(x3)` the outer REML
//! optimizer can STALL at its iteration cap with every smooth block's `λ` railed
//! to its ceiling (~1e5–1e13). When that happens the trace-channel EDF assembly
//! collapses: `Σ_k λ_k·tr(H⁻¹S_k) → Σ block_cols`, so `edf_total → p −
//! Σ block_cols = 1` (the intercept-only value) and every per-term smooth EDF
//! reads 0 — even though the returned coefficients (taken from the inner P-IRLS
//! solve) stay wiggly and predict the response well (dozens of active columns,
//! correlation ≈ 0.93). That is internally inconsistent (a genuine penalized LS
//! solution with large `β` cannot have `tr(F) ≈ 0`) and it violated EDF
//! additivity while surfacing NO non-convergence signal on the reported EDF.
//!
//! The top-level `gam` crate cannot build in this environment (a `build.rs`
//! author tripwire), so the issue's `fit_from_formula` path is exercised here in
//! `gam-models`, which builds standalone.

use super::entry::fit_from_formula;
use super::request::{FitConfig, FitResult, StandardFitResult};
use csv::StringRecord;
use gam_data::encode_recordswith_inferred_schema;
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn truth(a: f64, b: f64, c: f64) -> f64 {
    (2.0 * std::f64::consts::PI * a).sin()
        + 0.6 * (2.0 * std::f64::consts::PI * b).cos()
        + (c - 0.5).powi(2) * 3.0
}

struct StallProbe {
    edf_total: f64,
    edf_by_block: Vec<f64>,
    outer_converged: bool,
    outer_iterations: usize,
    lambda_max: f64,
    /// Pearson correlation between fitted and observed response on the training
    /// rows — a coefficient-side "wiggliness" witness independent of the EDF.
    corr: f64,
    n_active_cols: usize,
}

fn fit_three_smooth(n: usize, seed: u64, noise_sd: f64) -> StallProbe {
    let mut rng = StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0_f64, 1.0).unwrap();
    let noise = Normal::new(0.0, noise_sd).unwrap();

    let headers: Vec<String> = ["x1", "x2", "x3", "y"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rows = Vec::with_capacity(n);
    let mut coords = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let a = unif.sample(&mut rng);
        let b = unif.sample(&mut rng);
        let c = unif.sample(&mut rng);
        let y = truth(a, b, c) + noise.sample(&mut rng);
        rows.push(StringRecord::from(vec![
            a.to_string(),
            b.to_string(),
            c.to_string(),
            y.to_string(),
        ]));
        coords.push((a, b, c));
        ys.push(y);
    }
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let col = ds.column_map();
    let (i1, i2, i3) = (col["x1"], col["x2"], col["x3"]);

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ smooth(x1) + smooth(x2) + smooth(x3)", &ds, &cfg)
        .expect("3-smooth gaussian fit");
    let StandardFitResult {
        fit, resolvedspec, ..
    } = match result {
        FitResult::Standard(s) => s,
        _ => panic!("expected Standard fit"),
    };

    // Rebuild the training design and form fitted values to measure the
    // coefficient-side signal directly (independent of the reported EDF).
    let mut mat = Array2::<f64>::zeros((n, ds.headers.len()));
    for (k, &(a, b, c)) in coords.iter().enumerate() {
        mat[[k, i1]] = a;
        mat[[k, i2]] = b;
        mat[[k, i3]] = c;
    }
    let design = build_term_collection_design(mat.view(), &resolvedspec).expect("design");
    let fitted: Array1<f64> = design.design.matrixvectormultiply(&fit.beta);
    let ybar = ys.iter().sum::<f64>() / n as f64;
    let fbar = fitted.iter().sum::<f64>() / n as f64;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (yy, ff) in ys.iter().zip(fitted.iter()) {
        let dy = yy - ybar;
        let df = ff - fbar;
        sxy += dy * df;
        sxx += df * df;
        syy += dy * dy;
    }
    let corr = if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx.sqrt() * syy.sqrt())
    } else {
        0.0
    };
    let n_active_cols = fit.beta.iter().filter(|&&b| b.abs() > 1e-6).count();
    let lambda_max = fit.lambdas.iter().cloned().fold(0.0_f64, f64::max);

    StallProbe {
        edf_total: fit.edf_total().expect("edf_total"),
        edf_by_block: fit.edf_by_block().to_vec(),
        outer_converged: fit.outer_converged,
        outer_iterations: fit.outer_iterations,
        lambda_max,
        corr,
        n_active_cols,
    }
}

/// #1788 core regression: `n=600, noise_sd=0.5, seed=2` deterministically drives
/// the outer REML into a flat-valley STALL — 200 iters, `λ` railed to ~1.07e13,
/// `outer_converged == false` — while the returned coefficients stay wiggly (34
/// active columns, fitted-vs-observed correlation ≈ 0.87). BEFORE the fix the
/// reported `edf_total` collapsed to ~1.0 (the intercept-only floor) with every
/// per-term smooth EDF = 0, contradicting those coefficients. AFTER the fix the
/// `untrusted_edf_collapse` guard substitutes the per-term dimension floor, so
/// the reported EDF is no longer self-contradictory.
#[test]
fn stalled_reml_edf_not_collapsed_to_intercept_1788() {
    let p = fit_three_smooth(600, 2, 0.5);

    // Precondition: this configuration reproduces the stall (railed λ + a
    // non-converged outer with demonstrably wiggly coefficients). If the
    // underlying optimizer ever starts converging here, the guard is moot — but
    // then the reproduction assumption is void, so make that loud.
    assert!(
        !p.outer_converged && p.lambda_max > 1e5,
        "#1788 setup expected an outer-REML stall with railed λ, got \
         converged={} lambda_max={:.3e} (iters={}); reproduction seed drifted",
        p.outer_converged,
        p.lambda_max,
        p.outer_iterations,
    );
    assert!(
        p.corr > 0.5 && p.n_active_cols >= 10,
        "#1788 setup expected wiggly coefficients (corr>0.5, many active cols), \
         got corr={:.3} active={}",
        p.corr,
        p.n_active_cols,
    );

    // The bug: edf_total collapsed to the intercept-only floor (~1.0) with every
    // per-term smooth EDF exactly 0, even though the fit predicts the response.
    // The guard must keep the reported EDF consistent with those coefficients.
    assert!(
        p.edf_total > 3.0,
        "#1788 BUG: edf_total={:.4} collapsed to the intercept-only floor on a \
         stalled fit whose coefficients are wiggly (corr={:.3}, active={}); \
         three active smooths cannot honestly report edf_total ≈ 1.0",
        p.edf_total,
        p.corr,
        p.n_active_cols,
    );
    // EDF additivity must hold: the reported total equals the per-term smooth
    // blocks plus the (small, non-negative) unpenalized-column offset — the
    // intercept — which `edf_by_block` does not itself carry. The correction must
    // preserve that offset, never dropping the intercept.
    let block_sum: f64 = p.edf_by_block.iter().sum();
    let offset = p.edf_total - block_sum;
    assert!(
        (0.0..=2.0).contains(&offset),
        "#1788 EDF additivity: edf_total={:.4} − Σ edf_by_block={block_sum:.4} = \
         {offset:.4} should be the small unpenalized-column (intercept) offset",
        p.edf_total,
    );
    // At least one smooth term must carry non-trivial EDF (not all-zero).
    assert!(
        p.edf_by_block.iter().any(|&e| e > 0.5),
        "#1788: every per-term smooth EDF is still ~0 on a wiggly stalled fit: {:?}",
        p.edf_by_block,
    );
}

/// The guard must be INERT on a healthy converged fit: `n=600, noise_sd=0.5,
/// seed=3` converges cleanly (~16 EDF, dozens of active columns). The reported
/// EDF must be untouched by the #1788 correction — no over-flooring, no
/// additivity drift.
#[test]
fn converged_fit_edf_untouched_by_guard_1788() {
    let p = fit_three_smooth(600, 3, 0.5);
    assert!(
        p.outer_converged,
        "#1788 sanity seed expected to converge, got converged=false (iters={})",
        p.outer_iterations,
    );
    assert!(
        p.edf_total > 8.0 && p.edf_total < 30.0,
        "#1788 converged fit reports an unexpected edf_total={:.3}",
        p.edf_total,
    );
    let block_sum: f64 = p.edf_by_block.iter().sum();
    let offset = p.edf_total - block_sum;
    assert!(
        (0.0..=2.0).contains(&offset),
        "#1788 additivity on converged fit: edf_total={:.4} − Σ={block_sum:.4} = \
         {offset:.4} should be the small unpenalized-column offset",
        p.edf_total,
    );
}
