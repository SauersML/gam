//! Bug hunt (#1027), companion invariants. The Wood–Pya–Säfken ρ-uncertainty
//! EDF correction came out negative because the inference block built the
//! influence matrix `F` and the weighted Gram `X'WX` from a penalty assembled in
//! the REPARAMETERIZED basis while pairing it with the ORIGINAL-basis Hessian /
//! inverse, and then symmetrized `F` (which is not symmetric). This test pins the
//! fix from three independent angles that the original repro did not:
//!
//!  1. **Trace identity.** The stored influence matrix is `F = H⁻¹X'WX`, whose
//!     trace is the conditional EDF. The basis mismatch made `tr(F) ≠ edf`; with
//!     the penalty rotated into the original basis, `tr(coefficient_influence)`
//!     must match `edf_total` to round-off. (Symmetrizing `F` preserves its
//!     trace, so this angle is blind to the symmetrization defect but exposes the
//!     basis defect directly.)
//!
//!  2. **Weighted Gram is symmetric PSD.** `fit.saved_frame_weighted_gram()` must be a
//!     genuine symmetric positive-semidefinite curvature — the property that
//!     makes `tr(X'WX·Σ_ρ) ≥ 0`. The old `H·F` reconstruction was asymmetric and
//!     indefinite.
//!
//!  3. **Non-negativity across smoothness regimes.** The ρ-uncertainty df must be
//!     ≥ 0 whether the truth is wiggly (light penalty, large EDF) or nearly
//!     linear (heavy penalty, small EDF — the large-λ regime where `X'WX = H − S`
//!     is most cancellation-prone).

use csv::StringRecord;
use faer::Side;
use gam::faer_ndarray::FaerEigh;
use gam::inference::model_comparison::model_comparison_from_unified;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Build `y ~ s(x)` data. `wiggle` scales the frequency of the underlying
/// signal: a high value forces a wiggly fit (light penalty), a value near zero a
/// nearly-linear fit (heavy penalty / large smoothing parameter).
fn build_dataset(n: usize, seed: u64, wiggle: f64) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 10.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let mut ys = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x: f64 = ux.sample(&mut rng);
            let y = (wiggle * x).sin() + noise.sample(&mut rng);
            ys.push(y);
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        ys,
    )
}

fn fit_standard(data: &gam::data::EncodedDataset) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(std_fit) =
        fit_from_formula("y ~ s(x)", data, &cfg).expect("Gaussian s(x) fit should succeed")
    else {
        panic!("expected a standard fit for y ~ s(x)");
    };
    std_fit
}

fn min_eigenvalue(m: &ndarray::Array2<f64>) -> f64 {
    let (evals, _) = m.eigh(Side::Lower).expect("eigendecomposition");
    evals.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_abs_asymmetry(m: &ndarray::Array2<f64>) -> f64 {
    let p = m.nrows();
    let mut worst = 0.0_f64;
    for i in 0..p {
        for j in 0..p {
            worst = worst.max((m[[i, j]] - m[[j, i]]).abs());
        }
    }
    worst
}

#[test]
fn influence_trace_matches_conditional_edf() {
    init_parallelism();
    // A wiggly and a near-linear regime, several seeds each.
    for &wiggle in &[1.0_f64, 0.08] {
        for seed in 0..3u64 {
            let (data, _ys) = build_dataset(300, 7 + seed, wiggle);
            let std_fit = fit_standard(&data);
            let fit = &std_fit.fit;

            let f = fit
                .coefficient_influence()
                .expect("influence matrix present on an inferential Gaussian fit");
            let edf = fit.edf_total().expect("edf_total present");
            let tr_f: f64 = (0..f.nrows()).map(|i| f[[i, i]]).sum();

            // tr(F) = tr(H⁻¹X'WX) is, by definition, the conditional EDF. A basis
            // mismatch in the penalty (the #1027 root cause) breaks this identity.
            let tol = 1e-6 * edf.abs().max(1.0);
            assert!(
                (tr_f - edf).abs() <= tol,
                "tr(coefficient_influence) = {tr_f:.8} must equal edf_total = {edf:.8} \
                 (wiggle={wiggle}, seed={seed}); a gap means the influence matrix \
                 F = H⁻¹X'WX was assembled in an inconsistent basis."
            );
        }
    }
}

#[test]
fn weighted_gram_is_symmetric_psd() {
    init_parallelism();
    for &wiggle in &[1.0_f64, 0.08] {
        for seed in 0..3u64 {
            let (data, _ys) = build_dataset(300, 21 + seed, wiggle);
            let std_fit = fit_standard(&data);
            let fit = &std_fit.fit;

            let gram = fit
                .saved_frame_weighted_gram()
                .expect("weighted Gram has a saved-frame form")
                .expect("weighted Gram present on an inferential Gaussian fit");

            // Symmetric to round-off.
            let asym = max_abs_asymmetry(&gram);
            let scale = gram
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            assert!(
                asym <= 1e-9 * scale,
                "weighted Gram must be symmetric; max asymmetry {asym:.3e} \
                 (scale {scale:.3e}, wiggle={wiggle}, seed={seed})"
            );

            // Positive semidefinite: the curvature X'WX (PSD by construction, and
            // PSD-floored on storage). The old H·F reconstruction had min-eig < 0.
            let min_eig = min_eigenvalue(&gram);
            assert!(
                min_eig >= -1e-8 * scale,
                "weighted Gram must be PSD; min eigenvalue {min_eig:.3e} \
                 (scale {scale:.3e}, wiggle={wiggle}, seed={seed})"
            );
        }
    }
}

#[test]
fn penalized_hessian_times_influence_equals_weighted_gram() {
    // H·F = H·(I − H⁻¹·S) = H − S = X'WX is the consistency identity that ties
    // the three stored matrices together. F = H⁻¹X'WX is a product of two
    // symmetric matrices and is generally NOT symmetric; symmetrizing it
    // leaves tr(F) and the basis untouched — so neither
    // `influence_trace_matches_conditional_edf` above nor any test that reads
    // the WPS correction via `weighted_gram` (which is now stored directly)
    // catches it — but it corrupts the frequentist Ve = F·H⁻¹·φ and distorts
    // the Wood-corrected reference d.f. `tr(F_jj)² / tr(F_jj²)` that
    // `inference::smooth_test::reference_df` consumes for every smooth's
    // p-value. This identity catches the symmetrization the moment F is
    // assembled.
    init_parallelism();
    for &wiggle in &[1.0_f64, 0.08] {
        for seed in 0..3u64 {
            let (data, _ys) = build_dataset(300, 41 + seed, wiggle);
            let std_fit = fit_standard(&data);
            let fit = &std_fit.fit;

            let h = fit
                .saved_frame_penalized_hessian()
                .expect("penalized Hessian has a saved-frame form")
                .expect("penalized Hessian present on an inferential Gaussian fit");
            let f = fit
                .coefficient_influence()
                .expect("influence matrix present on an inferential Gaussian fit");
            let xwx = fit
                .saved_frame_weighted_gram()
                .expect("weighted Gram has a saved-frame form")
                .expect("weighted Gram present on an inferential Gaussian fit");

            let hf = h.dot(f);
            let scale = xwx
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let mut worst = 0.0_f64;
            for i in 0..hf.nrows() {
                for j in 0..hf.ncols() {
                    worst = worst.max((hf[[i, j]] - xwx[[i, j]]).abs());
                }
            }
            // `H*F = H(I - H^-1 S) = H - S` is the RAW difference, while the
            // stored `weighted_gram` is `symmetrize(H - S)` (`optimizer.rs`
            // symmetrizes `xwx` in place and deliberately does NOT symmetrize
            // `F`). So this gap is `antisym(H - S)` plus whatever `H*H^-1 != I`
            // contributes. Both H and S are supposed to be symmetric, so measure
            // H's own asymmetry: if it is O(gap), the defect is upstream and
            // symmetrizing `xwx` is hiding it; if it is at round-off, the gap is
            // conditioning in `H^-1` instead. Two different repairs, and the
            // message could not tell them apart.
            let mut worst_h_asym = 0.0_f64;
            for i in 0..h.nrows() {
                for j in 0..h.ncols() {
                    worst_h_asym = worst_h_asym.max((h[[i, j]] - h[[j, i]]).abs());
                }
            }
            // Round-off only: H·F = X'WX is exact in real arithmetic, and
            // `optimizer.rs` now obtains `F` by SOLVING `H·F = X'WX` against
            // the strict Cholesky, so it holds to the factorization's backward
            // error by construction rather than by luck (#2668). Both causes
            // this message used to name are eliminated: `F` is never passed to
            // `symmetrize_in_place`, and the raw-coordinate rescalings compose
            // to `D⁻¹(H·F)D⁻¹`, preserving the identity. The historical 1.169e1
            // gap was neither — it was `|(H·H⁻¹ − I)·S|`, the forward error of
            // an explicit inverse at `cond(H) = 2.099e8`.
            assert!(
                worst <= 1e-8 * scale,
                "H·F must equal X'WX (the genuine PSD weighted Gram); max \
                 entrywise gap {worst:.3e} (scale {scale:.3e}, wiggle={wiggle}, \
                 seed={seed}); max|H-H^T|={worst_h_asym:.3e}. \
                 `F` comes from solving H·F = X'WX, so a gap now means either \
                 that solve stopped being certified, or that `F` and \
                 `weighted_gram` were assembled from different `H`/`S(λ)` \
                 (#1027 basis mismatch), or that something reintroduced an \
                 explicit `H⁻¹` product on this path."
            );
        }
    }
}

#[test]
fn rho_uncertainty_df_nonnegative_across_regimes() {
    init_parallelism();
    for &wiggle in &[1.5_f64, 1.0, 0.3, 0.08] {
        for seed in 0..3u64 {
            let (data, ys) = build_dataset(300, 101 + seed, wiggle);
            let std_fit = fit_standard(&data);
            let fit = &std_fit.fit;

            let n = ys.len();
            let y = Array1::from(ys);
            let eta_hat = Array1::from(
                fit.artifacts
                    .pirls
                    .as_ref()
                    .expect("Gaussian WPS fixture retains converged PIRLS geometry")
                    .final_eta
                    .to_vec(),
            );
            let cmp = model_comparison_from_unified(
                fit,
                y.view(),
                eta_hat.view(),
                Array1::ones(n).view(),
                None,
            )
            .expect("construct WPS comparison from a finite converged Gaussian fit");
            let corrected = cmp
                .edf
                .corrected
                .expect("WPS fixture must retain certified corrected EDF");
            let rho_df = cmp
                .edf
                .rho_uncertainty_df()
                .expect("WPS fixture must retain its smoothing-uncertainty EDF");
            let aic_corrected = cmp
                .aic_corrected
                .expect("WPS fixture must retain certified corrected AIC");
            let tol = 1e-6 * cmp.edf.conditional.abs().max(1.0);
            assert!(
                rho_df >= -tol,
                "ρ-uncertainty df must be ≥ 0 (corrected EDF ≥ conditional) in every \
                 smoothness regime; got {rho_df:.6e} at wiggle={wiggle}, seed={seed} \
                (conditional={:.4}, corrected={:.4}).",
                cmp.edf.conditional,
                corrected
            );
            // The corrected AIC must not under-penalize relative to conditional.
            assert!(
                aic_corrected >= cmp.aic_conditional - tol,
                "corrected AIC {:.4} must be ≥ conditional AIC {:.4} \
                 (wiggle={wiggle}, seed={seed})",
                aic_corrected,
                cmp.aic_conditional
            );
        }
    }
}

// ── #2672: the same three matrices, on a model that CONDITIONS its columns ──
//
// Every fixture above fits `y ~ s(x)`, which has no non-intercept parametric
// column, so `ParametricColumnConditioning::is_active()` is false and
// `backtransform_external_result` returns before it can touch the inference
// block. That is why this file — which exists to pin `H`, `F` and `X'WX`
// against each other — could not see that the back-transform DROPPED `F`
// outright on every model that does condition, with the note "we do not carry
// the similarity primitive here". Both halves of that primitive
// (`left_multiply_by_m` and `right_multiply_by_m_inv`) are defined on the same
// type; `F_orig = M·F_int·M⁻¹` is their composition.
//
// The consequence was not local: `F` is the sole input to Wood's
// smoothing-selection-corrected `edf1 = 2·tr(F_jj) − tr(F_jj²)`, so
// `wood_reference_df` returned `None` for every model with a parametric term and
// the smooth-term LR test fell back to the raw conditional EDF — the
// anti-conservative reference #1766 replaced.
//
// These arms are the same four claims as above with `y ~ x + s(z)` in place of
// `y ~ s(x)`, so the blind spot cannot come back.

/// `y = 0.7·x + sin(freq·2π·z) + ε` with a parametric `x` alongside the smooth,
/// so the parametric column conditioning is ACTIVE.
fn build_conditioned_dataset(
    n: usize,
    seed: u64,
    freq: f64,
) -> (gam::data::EncodedDataset, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let headers = ["y", "x", "z"].into_iter().map(String::from).collect();
    let mut ys = Vec::with_capacity(n);
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            let t = i as f64 / (n as f64 - 1.0);
            let z = ((i * 37) % n) as f64 / (n as f64 - 1.0);
            let y = 0.7 * t + (freq * std::f64::consts::TAU * z).sin() + noise.sample(&mut rng);
            ys.push(y);
            StringRecord::from(vec![y.to_string(), t.to_string(), z.to_string()])
        })
        .collect();
    (
        encode_recordswith_inferred_schema(headers, rows).expect("encode"),
        ys,
    )
}

fn fit_conditioned(data: &gam::data::EncodedDataset) -> gam::StandardFitResult {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let FitResult::Standard(std_fit) =
        fit_from_formula("y ~ x + s(z)", data, &cfg).expect("Gaussian x + s(z) fit should succeed")
    else {
        panic!("expected a standard fit for y ~ x + s(z)");
    };
    std_fit
}

#[test]
fn conditioned_model_retains_influence_and_its_identities_2672() {
    init_parallelism();
    for &freq in &[1.0_f64, 0.08] {
        for seed in 0..3u64 {
            let (data, _ys) = build_conditioned_dataset(300, 61 + seed, freq);
            let std_fit = fit_conditioned(&data);
            let fit = &std_fit.fit;

            let f = fit.coefficient_influence().unwrap_or_else(|| {
                panic!(
                    "influence matrix must survive the parametric-column back-transform \
                     (freq={freq}, seed={seed}); `None` here is the #2672 similarity-map drop"
                )
            });
            let h = fit
                .saved_frame_penalized_hessian()
                .expect("penalized Hessian has a saved-frame form")
                .expect("penalized Hessian present on an inferential Gaussian fit");
            let gram = fit
                .saved_frame_weighted_gram()
                .expect("weighted Gram has a saved-frame form")
                .expect("weighted Gram present on an inferential Gaussian fit");
            let edf = fit.edf_total().expect("edf_total present");

            // tr(F) = tr(H⁻¹X'WX) is the conditional EDF, and it is a SIMILARITY
            // invariant — so the back-transform cannot move it and this is the
            // sharpest single check that the right map was applied.
            let tr_f: f64 = (0..f.nrows()).map(|i| f[[i, i]]).sum();
            assert!(
                (tr_f - edf).abs() <= 1e-6 * edf.abs().max(1.0),
                "tr(coefficient_influence) = {tr_f:.8} must equal edf_total = {edf:.8} \
                 on a conditioned model (freq={freq}, seed={seed})"
            );

            // H·F = X'WX must survive the back-transform: `H` takes the congruence
            // `M⁻ᵀ(·)M⁻¹`, `F` the similarity `M(·)M⁻¹`, and their product is the
            // congruence the Gram itself takes.
            let hf = h.dot(f);
            let scale = gram
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let mut worst = 0.0_f64;
            for i in 0..hf.nrows() {
                for j in 0..hf.ncols() {
                    worst = worst.max((hf[[i, j]] - gram[[i, j]]).abs());
                }
            }
            assert!(
                worst <= 1e-8 * scale,
                "H·F must equal X'WX in the ORIGINAL basis after conditioning; max \
                 entrywise gap {worst:.3e} (scale {scale:.3e}, freq={freq}, seed={seed}). \
                 A gap means `F` and the pair (`H`, `X'WX`) were carried out of the \
                 internal basis by different maps."
            );

            // And the Gram itself is still a genuine symmetric PSD curvature.
            let asym = max_abs_asymmetry(&gram);
            assert!(
                asym <= 1e-9 * scale,
                "weighted Gram must stay symmetric on a conditioned model; max \
                 asymmetry {asym:.3e} (freq={freq}, seed={seed})"
            );
            let min_eig = min_eigenvalue(&gram);
            assert!(
                min_eig >= -1e-8 * scale,
                "weighted Gram must stay PSD on a conditioned model; min eigenvalue \
                 {min_eig:.3e} (freq={freq}, seed={seed})"
            );
        }
    }
}

#[test]
fn conditioned_model_rho_uncertainty_df_nonnegative_2672() {
    init_parallelism();
    for &freq in &[1.5_f64, 1.0, 0.3, 0.08] {
        for seed in 0..2u64 {
            let (data, ys) = build_conditioned_dataset(300, 141 + seed, freq);
            let std_fit = fit_conditioned(&data);
            let fit = &std_fit.fit;
            let n = ys.len();
            let y = Array1::from(ys);
            let eta_hat = Array1::from(
                fit.artifacts
                    .pirls
                    .as_ref()
                    .expect("converged PIRLS geometry retained")
                    .final_eta
                    .to_vec(),
            );
            let cmp = model_comparison_from_unified(
                fit,
                y.view(),
                eta_hat.view(),
                Array1::ones(n).view(),
                None,
            )
            .expect("construct WPS comparison from a converged Gaussian fit");
            let rho_df = cmp
                .edf
                .rho_uncertainty_df()
                .expect("conditioned fixture must retain its smoothing-uncertainty EDF");
            let tol = 1e-6 * cmp.edf.conditional.abs().max(1.0);
            assert!(
                rho_df >= -tol,
                "ρ-uncertainty df must be ≥ 0 on a conditioned model too; got \
                 {rho_df:.6e} at freq={freq}, seed={seed} (conditional={:.4})",
                cmp.edf.conditional
            );
        }
    }
}
