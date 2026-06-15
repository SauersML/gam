//! Bug hunt (companion to
//! `bug_hunt_gaussian_smooth_not_invariant_to_small_response_rescale.rs`,
//! issue #1127): the Gaussian-identity REML smoothing-parameter selection must
//! be *exactly* invariant to a multiplicative response rescale `y → a·y`, and
//! the profiled residual scale σ̂ must rescale by exactly `a`.
//!
//! Where the sibling test asserts the equivariance of the *fitted smooth* at a
//! single small `a`, this test attacks the same root cause from the angle of
//! the quantity that was actually corrupted — the **selected `λ̂`** — and does
//! so over a full two-sided sweep that brackets the absolute deviance floor.
//!
//! Math. For `y ~ s(x)`, Gaussian identity link, the penalized deviance
//! `D_p = Σ wᵢ(yᵢ−μ̂ᵢ)² + β̂ᵀSβ̂` is exactly quadratic in the response, so a
//! rescale sends `D_p → a²·D_p`. The profiled Gaussian REML criterion depends
//! on `D_p` only through `log D_p`, so the rescale shifts the objective by the
//! additive constant `ν·log a` and leaves its ρ-argmin — i.e. `λ̂`, and with it
//! the effective degrees of freedom — exactly unchanged. The profiled scale is
//! `σ̂² = D_p/ν → a²·σ̂²`, i.e. `σ̂ → a·σ̂`.
//!
//! The defect being guarded against. The `smooth_floor_dp` guard that keeps
//! `σ̂² > 0` used an *absolute* floor (`1e-12`, smoothing width `1e-8`). For a
//! small response `D_p = O(a²)` falls inside that fixed band — at `a = 1e-6`,
//! `D_p ≈ 3.6e-11` — and is spuriously inflated ~190×, so `log D_p` stops
//! tracking `2 log a + const`, the ρ-gradient is distorted, and the optimizer
//! settles at an over-smoothed `λ̂` (inflated ~40×, null-space penalty ~440×).
//! The fix makes the floor a fixed *fraction* of the weighted null deviance
//! `D₀ ∝ a²`, restoring the exact `λ̂` / EDF / σ̂ scaling tested here.

use csv::StringRecord;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Fixed synthetic Gaussian dataset `y0 = sin(2π x) + N(0, 0.3)`, response
/// multiplied by `a`. The covariate column and the noise realization are
/// identical for every `a` (same seed), so two datasets differ only by the
/// multiplicative response scale.
fn dataset_with_scale(a: f64, n: usize) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(20240615);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform");
    let noise = Normal::new(0.0, 0.3).expect("normal");
    let rows: Vec<StringRecord> = (0..n)
        .map(|_| {
            let x = u.sample(&mut rng);
            let y0 = (2.0 * std::f64::consts::PI * x).sin() + noise.sample(&mut rng);
            let y = y0 * a;
            StringRecord::from(vec![x.to_string(), y.to_string()])
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode")
}

/// Fit `y ~ s(x)` (Gaussian REML) and return `(λ̂, σ̂)`.
fn fit_lambdas_and_sigma(a: f64, n: usize) -> (Vec<f64>, f64) {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x)", &dataset_with_scale(a, n), &cfg).expect("fit ok");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard Gaussian fit");
    };
    (fit.fit.lambdas.to_vec(), fit.fit.standard_deviation)
}

#[test]
fn gaussian_reml_lambda_hat_is_invariant_to_a_response_rescale() {
    init_parallelism();

    let n = 400;
    let (lambda_ref, sigma_ref) = fit_lambdas_and_sigma(1.0, n);
    assert!(
        sigma_ref.is_finite() && sigma_ref > 0.0,
        "reference σ̂ must be a finite positive scale, got {sigma_ref}"
    );
    assert!(
        !lambda_ref.is_empty() && lambda_ref.iter().all(|l| l.is_finite() && *l > 0.0),
        "reference λ̂ must be finite and positive: {lambda_ref:?}"
    );

    // Two-sided sweep that brackets the old absolute floor. The up-scale side
    // was already correct; the down-scale side (a ≤ 1e-5) is where the absolute
    // `smooth_floor_dp` band corrupted the selection. a = 1e-8 drives D_p far
    // below the old floor — the most violent case in the issue.
    for &a in &[
        1.0e8_f64, 1.0e4, 1.0e2, 1.0e-2, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-8,
    ] {
        let (lambda_a, sigma_a) = fit_lambdas_and_sigma(a, n);

        assert_eq!(
            lambda_a.len(),
            lambda_ref.len(),
            "λ̂ dimension changed under rescale a={a:e}"
        );

        // λ̂ is scale-invariant: every component must match the unit-scale fit.
        // A pure linear rescale leaves the penalized normal equations and the
        // REML argmin untouched, so the only admissible drift is round-off in
        // the outer optimizer's log-λ golden/Newton steps (~1e-6 relative).
        for (k, (&la, &lref)) in lambda_a.iter().zip(&lambda_ref).enumerate() {
            let rel = (la.ln() - lref.ln()).abs();
            assert!(
                rel < 5.0e-3,
                "λ̂[{k}] not invariant under response rescale a={a:e}: \
                 λ̂={la:.6e} vs reference {lref:.6e} (|Δ log λ| = {rel:.3e}); \
                 the Gaussian REML λ-selection must be exactly scale-invariant. \
                 Full λ̂={lambda_a:?} vs reference {lambda_ref:?}."
            );
        }

        // σ̂ scales by exactly `a`: σ̂(a)/a must reproduce σ̂(1).
        let sigma_rescaled = sigma_a / a;
        let rel_sigma = (sigma_rescaled - sigma_ref).abs() / sigma_ref;
        assert!(
            rel_sigma < 1.0e-4,
            "σ̂ did not rescale linearly under a={a:e}: σ̂/a = {sigma_rescaled:.6e} vs \
             reference {sigma_ref:.6e} (rel = {rel_sigma:.3e}). The profiled Gaussian \
             scale is σ̂² = D_p/ν and D_p → a²·D_p, so σ̂ → a·σ̂ exactly."
        );
    }
}
