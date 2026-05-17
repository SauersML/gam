//! Exact-equivalence regression for the Gaussian-Identity XᵀWX cache.
//!
//! The cache populates `XᵀWX` and `XᵀW(y − offset)` once before the REML
//! outer loop and reuses them inside every inner PIRLS solve, on the basis
//! that for Gaussian + Identity + constant prior weights both quantities
//! are λ-independent.  The penalty `λ·S` is still added per-λ, so the
//! linear system is identical bit-for-bit to the non-cached path up to
//! floating-point reordering of the GEMM contractions.
//!
//! This test calls `fit_model_for_fixed_rho` directly at a single ρ, once
//! with `gaussian_fixed_cache: None` (the baseline streaming GEMM path)
//! and once with a precomputed `GaussianFixedCache` (the new fast path),
//! and asserts the two coefficient vectors agree to better than 1e-10 in
//! relative norm — orders of magnitude tighter than PIRLS convergence so
//! any drift is caught immediately.

use gam::construction::CanonicalPenalty;
use gam::estimate::PenaltySpec;
use gam::pirls::{
    GaussianFixedCache, PenaltyConfig, PirlsConfig, PirlsProblem, fit_model_for_fixed_rho,
};
use gam::types::{
    GlmLikelihoodFamily, GlmLikelihoodSpec, InverseLink, LinkFunction, LogSmoothingParamsView,
};
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const N: usize = 10_000;
const P: usize = 32;
const SEED: u64 = 0xCACE_1DEE_BEEF_F00D;

fn make_problem() -> (Array2<f64>, Array1<f64>, Array1<f64>, Vec<CanonicalPenalty>) {
    // Deterministic design: intercept + (P-1) covariates uniform in [-1, 1].
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut x = Array2::<f64>::zeros((N, P));
    for i in 0..N {
        x[[i, 0]] = 1.0;
        for j in 1..P {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }
    let beta_true = Array1::from_shape_fn(P, |j| if j == 0 { 0.7 } else { 0.5 / (j as f64) });
    let eta = x.dot(&beta_true);
    // Light Gaussian noise so the fit is well-posed but not trivially zero
    // residual.  Identity link → y = eta + ε.
    let y = eta.mapv(|e| e + 0.1 * rng.random_range(-1.0..1.0));
    let w = Array1::<f64>::ones(N);

    // Ridge-style penalty on non-intercept coefficients, single block.
    let mut s = Array2::<f64>::zeros((P, P));
    for j in 1..P {
        s[[j, j]] = 1.0;
    }
    let canonical = canonicalize(&[s]);
    (x, y, w, canonical)
}

fn canonicalize(s_list: &[Array2<f64>]) -> Vec<CanonicalPenalty> {
    let p = s_list[0].nrows();
    s_list
        .iter()
        .enumerate()
        .filter_map(|(idx, s)| {
            gam::construction::canonicalize_penalty_spec(
                &PenaltySpec::Dense(s.clone()),
                p,
                idx,
                "gaussian_xtwx_cache_correctness",
            )
            .expect("canonicalize penalty spec")
        })
        .collect()
}

fn build_cache(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    offset: &Array1<f64>,
) -> GaussianFixedCache {
    // XᵀWX with full positive priorweights — symmetric by construction
    // since W is a non-negative diagonal.
    let mut wx = x.clone();
    for i in 0..wx.nrows() {
        let wi = w[i];
        for j in 0..wx.ncols() {
            wx[[i, j]] *= wi;
        }
    }
    let xtwx = x.t().dot(&wx);
    let mut wz = y.clone();
    wz -= offset;
    for i in 0..wz.len() {
        wz[i] *= w[i];
    }
    let xtwy = x.t().dot(&wz);
    let centered_weighted_y_sq = y
        .iter()
        .zip(offset.iter())
        .zip(w.iter())
        .map(|((&y, &offset), &wi)| {
            let centered = y - offset;
            wi * centered * centered
        })
        .sum();
    GaussianFixedCache {
        xtwx_orig: xtwx,
        xtwy_orig: xtwy,
        centered_weighted_y_sq,
        xtwx_sparse_orig: None,
    }
}

fn fit_at_rho(
    x: &Array2<f64>,
    y: &Array1<f64>,
    w: &Array1<f64>,
    penalties: &[CanonicalPenalty],
    rho: f64,
    cache: Option<&GaussianFixedCache>,
) -> Array1<f64> {
    let p = x.ncols();
    let offset = Array1::<f64>::zeros(y.len());
    let cfg = PirlsConfig {
        likelihood: GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GaussianIdentity),
        link_kind: InverseLink::Standard(LinkFunction::Identity),
        max_iterations: 200,
        convergence_tolerance: 1e-12,
        firth_bias_reduction: false,
        initial_lm_lambda: None,
    };
    let (result, _working) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(array![rho].view()),
        PirlsProblem {
            x: x.view(),
            offset: offset.view(),
            y: y.view(),
            priorweights: w.view(),
            covariate_se: None,
            gaussian_fixed_cache: cache,
        },
        PenaltyConfig {
            canonical_penalties: penalties,
            balanced_penalty_root: None,
            reparam_invariant: None,
            p,
            coefficient_lower_bounds: None,
            linear_constraints_original: None,
            penalty_shrinkage_floor: None,
            kronecker_factored: None,
        },
        &cfg,
        None,
    )
    .expect("fit_model_for_fixed_rho");
    // Convert back to original coordinates so the two paths can be compared
    // directly — Qs may differ between calls in principle, even though for
    // this problem the reparameterization is deterministic.
    match result.coordinate_frame {
        gam::pirls::PirlsCoordinateFrame::OriginalSparseNative => {
            result.beta_transformed.as_ref().clone()
        }
        gam::pirls::PirlsCoordinateFrame::TransformedQs => result
            .reparam_result
            .qs
            .dot(result.beta_transformed.as_ref()),
    }
}

fn rel_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "coefficient vectors must have same length"
    );
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        num += d * d;
        den += y * y;
    }
    (num / den.max(1e-30)).sqrt()
}

#[test]
fn cached_xtwx_matches_streaming_path_within_1em10() {
    let (x, y, w, penalties) = make_problem();
    let offset = Array1::<f64>::zeros(y.len());
    let cache = build_cache(&x, &y, &w, &offset);

    // Single ρ, ridge-like penalty: ensures the inner solver actually
    // exercises `XᵀWX + λ·S` rather than collapsing to `λ → ∞`.
    let rho = 0.3_f64;

    let beta_without = fit_at_rho(&x, &y, &w, &penalties, rho, None);
    let beta_with = fit_at_rho(&x, &y, &w, &penalties, rho, Some(&cache));

    let rel = rel_diff(&beta_with, &beta_without);
    eprintln!(
        "[gaussian-cache] N={N} p={P} rho={rho:.3} ‖Δβ‖/‖β‖ = {rel:.3e} \
         β_without[0..4]={:?} β_with[0..4]={:?}",
        &beta_without.as_slice().unwrap()[..4],
        &beta_with.as_slice().unwrap()[..4]
    );
    assert!(
        rel < 1e-10,
        "Gaussian XᵀWX cache must agree with streaming path within 1e-10 \
         relative; got {rel:.3e}"
    );
}

#[test]
fn cached_xtwx_matches_across_rho_grid() {
    // Sweep ρ to catch any interaction with the penalty addition path —
    // the cache stores the un-penalized XᵀWX, so a sign / scaling bug in
    // `penalty.add_to_hessian` would show up only at certain λ scales.
    let (x, y, w, penalties) = make_problem();
    let offset = Array1::<f64>::zeros(y.len());
    let cache = build_cache(&x, &y, &w, &offset);

    let mut worst = 0.0_f64;
    for &rho in &[-3.0, -1.0, 0.0, 1.0, 3.0, 6.0] {
        let beta_without = fit_at_rho(&x, &y, &w, &penalties, rho, None);
        let beta_with = fit_at_rho(&x, &y, &w, &penalties, rho, Some(&cache));
        let rel = rel_diff(&beta_with, &beta_without);
        eprintln!("[gaussian-cache] rho={rho:+.1}  rel={rel:.3e}");
        worst = worst.max(rel);
    }
    assert!(
        worst < 1e-10,
        "worst ‖Δβ‖/‖β‖ across rho grid = {worst:.3e}, expected < 1e-10"
    );
}
