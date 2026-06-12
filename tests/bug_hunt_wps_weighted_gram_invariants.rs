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
//!  2. **Weighted Gram is symmetric PSD.** `fit.weighted_gram()` must be a
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
                .weighted_gram()
                .expect("weighted Gram present on an inferential Gaussian fit");

            // Symmetric to round-off.
            let asym = max_abs_asymmetry(gram);
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
            let min_eig = min_eigenvalue(gram);
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
                .penalized_hessian()
                .expect("penalized Hessian present on an inferential Gaussian fit");
            let f = fit
                .coefficient_influence()
                .expect("influence matrix present on an inferential Gaussian fit");
            let xwx = fit
                .weighted_gram()
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
            // Round-off only: H·F = X'WX is exact in real arithmetic.
            assert!(
                worst <= 1e-8 * scale,
                "H·F must equal X'WX (the genuine PSD weighted Gram); max \
                 entrywise gap {worst:.3e} (scale {scale:.3e}, wiggle={wiggle}, \
                 seed={seed}). A non-zero gap means F was either reassembled in \
                 the wrong basis (#1027 root cause) or symmetrized after the \
                 fact — `enforce_symmetry(F)` makes \
                 H·F_sym = (X'WX + H·X'WX·H⁻¹)/2 ≠ X'WX."
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
            let cmp = model_comparison_from_unified(
                fit,
                Array1::from(ys).view(),
                Array1::zeros(n).view(),
                Array1::ones(n).view(),
                None,
            );
            let rho_df = cmp.edf.rho_uncertainty_df();
            let tol = 1e-6 * cmp.edf.conditional.abs().max(1.0);
            assert!(
                rho_df >= -tol,
                "ρ-uncertainty df must be ≥ 0 (corrected EDF ≥ conditional) in every \
                 smoothness regime; got {rho_df:.6e} at wiggle={wiggle}, seed={seed} \
                 (conditional={:.4}, corrected={:.4}).",
                cmp.edf.conditional,
                cmp.edf.corrected
            );
            // The corrected AIC must not under-penalize relative to conditional.
            assert!(
                cmp.aic_corrected >= cmp.aic_conditional - tol,
                "corrected AIC {:.4} must be ≥ conditional AIC {:.4} \
                 (wiggle={wiggle}, seed={seed})",
                cmp.aic_corrected,
                cmp.aic_conditional
            );
        }
    }
}
