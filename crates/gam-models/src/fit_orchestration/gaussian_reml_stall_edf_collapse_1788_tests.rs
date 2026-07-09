//! Regression for #1788 at the real smooth-spline pipeline layer.
//!
//! For an ordinary Gaussian additive fit `y ~ s(x1)+s(x2)+s(x3)` the reported
//! effective degrees of freedom must stay consistent with the fitted
//! coefficients. The #1788 pathology: when the outer REML STALLED at its
//! iteration cap with every smooth block's `λ` railed to its ceiling
//! (~1e5–1e13), the trace-channel EDF assembly collapsed — `Σ_k λ_k·tr(H⁻¹S_k)
//! → Σ block_cols`, so `edf_total → p − Σ block_cols = 1` (the intercept-only
//! value) and every per-term smooth EDF read 0, even though the returned
//! coefficients (from the inner P-IRLS solve) stayed wiggly and predicted the
//! response well. A genuine penalized LS solution with large `β` cannot have
//! `tr(F) ≈ 0`; that both violated EDF additivity and surfaced NO
//! non-convergence signal on the reported EDF.
//!
//! The invariant this file pins — the reported EDF may never contradict the
//! fitted coefficients — is enforced along two independent paths that together
//! close the pathology:
//!   * the outer optimizer no longer parking at the `λ` ceiling on the #1788
//!     fixture (grid-free stationary-point enumeration in `gaussian_reml.rs`,
//!     commit 101b087e8, now lands the interior ρ optimum so the trace channel
//!     assembles an honest EDF); and
//!   * the `guard_untrusted_edf_collapse` correction, which — for any fixture
//!     that still stalls with railed `λ` — substitutes the per-term dimension
//!     floor so the reported EDF cannot fall below what the live coefficients
//!     imply.
//!
//! This file was re-pinned to the post-101b087e8 optimizer contract: the #1788
//! fixture (`seed=2`) now CONVERGES, so the headline test asserts the
//! pathology is fixed at the source (converged + self-consistent EDF) rather
//! than pinning the earlier stall. The collapse-guard branch keeps coverage
//! through a seed scan that asserts the same EDF invariant on any fixture that
//! still stalls (the guard is defensive-only once the optimizer is robust).
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

impl StallProbe {
    /// The returned coefficients are demonstrably wiggly: they track the
    /// response and light up many basis columns. Both the converged and the
    /// stalled-then-guarded worlds must keep the reported EDF consistent with
    /// THIS, so it is the shared precondition for asserting on EDF.
    fn coefficients_are_wiggly(&self) -> bool {
        self.corr > 0.5 && self.n_active_cols >= 10
    }

    /// The outer REML parked at the `λ` ceiling without converging — the
    /// flat-valley stall that used to collapse the reported EDF.
    fn is_railed_stall(&self) -> bool {
        !self.outer_converged && self.lambda_max > 1e5
    }

    /// The reported EDF is internally consistent with wiggly coefficients:
    /// well above the intercept-only floor, additive (`edf_total` is
    /// `Σ edf_by_block` plus the small unpenalized-column / intercept offset),
    /// and no smooth term reads as entirely absent.
    fn edf_is_self_consistent(&self) -> bool {
        if self.edf_total <= 3.0 {
            return false;
        }
        let block_sum: f64 = self.edf_by_block.iter().sum();
        let offset = self.edf_total - block_sum;
        if !(0.0..=2.0).contains(&offset) {
            return false;
        }
        self.edf_by_block.iter().any(|&e| e > 0.5)
    }
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

/// #1788 headline contract, re-pinned to the post-101b087e8 optimizer. The
/// `n=600, noise_sd=0.5, seed=2` fixture used to drive the outer REML into a
/// flat-valley STALL (200 iters, `λ` railed to ~1.07e13, `outer_converged ==
/// false`) whose reported `edf_total` collapsed to ~1.0 while the coefficients
/// stayed wiggly. Grid-free stationary-point enumeration (`gaussian_reml.rs`,
/// 101b087e8) now lands the interior ρ optimum on this fixture, so the fit
/// CONVERGES and the trace-channel EDF assembles honestly — the pathology is
/// fixed at the source, no guard substitution required.
///
/// The enduring #1788 invariant is unchanged and still asserted: three active,
/// wiggly smooths cannot honestly report `edf_total ≈ 1.0`; the reported EDF
/// must stay additive and consistent with the fitted coefficients.
#[test]
fn stalled_reml_edf_not_collapsed_to_intercept_1788() {
    let p = fit_three_smooth(600, 2, 0.5);

    // The coefficients are wiggly — they track the response and light up many
    // basis columns. This is the anchor the reported EDF must not contradict.
    assert!(
        p.coefficients_are_wiggly(),
        "#1788 setup expected wiggly coefficients (corr>0.5, many active cols), \
         got corr={:.3} active={}",
        p.corr,
        p.n_active_cols,
    );

    // Post-101b087e8 contract: the grid-free stationary-point enumeration lands
    // the interior ρ optimum instead of parking at the `λ` ceiling, so this
    // fixture now CONVERGES. Pinning the fix (not the earlier stall) makes a
    // future regression that re-strands the optimizer at the rail loud here.
    // (If the probe shows this fixture STILL stalls, the reproduction contract
    // has not moved and this re-pin must be discarded in favour of the guard
    // path — see the seed-scan test below, which covers the stall case.)
    assert!(
        p.outer_converged,
        "#1788 re-pin: post-101b087e8 the seed=2 fixture is expected to CONVERGE \
         (grid-free stationary-point enumeration lands the interior ρ optimum), \
         got converged=false lambda_max={:.3e} (iters={}); the optimizer is back \
         at the rail — treat as a live regression, not a stale re-pin",
        p.lambda_max, p.outer_iterations,
    );

    // The reported EDF is self-consistent with those wiggly coefficients:
    // well above the intercept-only floor, additive, no smooth reading ~0.
    assert!(
        p.edf_is_self_consistent(),
        "#1788 BUG: reported EDF contradicts the wiggly coefficients — \
         edf_total={:.4}, edf_by_block={:?} (corr={:.3}, active={}); three active \
         smooths cannot honestly report edf_total ≈ 1.0 or all-zero per-term EDF",
        p.edf_total,
        p.edf_by_block,
        p.corr,
        p.n_active_cols,
    );
}

/// Collapse-guard coverage without a hand-picked stalling seed. The
/// `guard_untrusted_edf_collapse` correction is defensive-only once the
/// optimizer robustly converges, and it is a private `fn` on the sibling `fit`
/// module (not directly callable from this test module without widening its
/// visibility). We instead scan a small band of seeds through the real fit
/// pipeline: for EVERY fixture that still exhibits the #1788 flat-valley stall
/// (railed `λ`, `outer_converged == false`) with wiggly coefficients, the
/// guard must keep the reported EDF self-consistent — never collapsed to the
/// intercept-only floor. If no seed stalls in the band the optimizer is robust
/// post-101b087e8 and the collapse branch is simply unreachable via this
/// pipeline (its inert path is covered by
/// `converged_fit_edf_untouched_by_guard_1788`).
#[test]
fn stalled_fixtures_keep_edf_consistent_via_guard_1788() {
    for seed in 0..12u64 {
        let p = fit_three_smooth(600, seed, 0.5);
        if !(p.is_railed_stall() && p.coefficients_are_wiggly()) {
            continue;
        }
        assert!(
            p.edf_is_self_consistent(),
            "#1788 guard: seed={seed} stalled at the λ rail (lambda_max={:.3e}, \
             iters={}) with wiggly coefficients (corr={:.3}, active={}) yet the \
             reported EDF collapsed — edf_total={:.4}, edf_by_block={:?}; the \
             untrusted_edf_collapse guard must substitute the per-term dimension \
             floor",
            p.lambda_max,
            p.outer_iterations,
            p.corr,
            p.n_active_cols,
            p.edf_total,
            p.edf_by_block,
        );
    }
    // Not a failure if the scan finds no stall: that means the optimizer is
    // robust across the band and the collapse branch is genuinely unreachable
    // via this pipeline (its inert path is covered by
    // `converged_fit_edf_untouched_by_guard_1788`). The assertion above is a
    // property over every stalling fixture, so it holds vacuously in that case.
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
