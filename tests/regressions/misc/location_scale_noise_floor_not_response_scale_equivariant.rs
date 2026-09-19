//! Bug hunt: the Gaussian location-scale (GAMLSS) noise model was NOT equivariant
//! under a global rescale of the response, because the soft σ-floor
//! (then the fixed `b = 0.01`) was added in RAW response units instead of being
//! scaled with the response.
//!
//! ## The invariance that must hold
//!
//! A Gaussian location-scale model `y_i ~ N(μ(x_i), σ(x_i)²)` is, by
//! construction, equivariant under `y → c·y` for any `c > 0`:
//!
//!   ℓ(cμ, cσ; cy) = ℓ(μ, σ; y) − n·log c,
//!
//! and the smoothing penalties act on the (centered) smooth coefficients, which
//! are unchanged by the pure intercept shift `log σ → log σ + log c`. So the
//! penalized REML/LAML optimum is exactly equivariant: the fitted standard
//! deviation surface must satisfy
//!
//!   σ̂_{c·y}(x) = c · σ̂_y(x)   for every x.
//!
//! The library *intends* to honor this. `families::sigma_link` standardizes the
//! response by `response_scale = sample_std(y)` before fitting so the fixed floor
//! `b = 0.01` is "operationally scale-relative" (≈ 1 % of the response spread,
//! mirroring mgcv's `gaulss(b = 0.01)`), then maps the fit back to raw units in
//! `rescale_gaussian_location_scale_to_raw` (`src/solver/workflow.rs`).
//!
//! ## The bug
//!
//! The back-mapping shifts the log-σ block intercept by `+ln(response_scale)`
//! (`src/solver/workflow.rs:1660`). That correctly scales the `exp(η)` part of
//! the link but leaves the ADDITIVE floor untouched:
//!
//!   σ_raw = b + exp(η_raw)
//!         = b + response_scale · exp(η_internal)
//!         = 0.01 + c · (σ̂_unit − 0.01),
//!
//! whereas equivariance requires `c · σ̂_unit = c·0.01 + c·(σ̂_unit − 0.01)`. The
//! residual error is therefore a constant `0.01·(1 − c)` on the σ scale,
//! INDEPENDENT of x. The former source comment on the fixed floor derived exactly
//! this `0.01·(response_scale − 1)` term and dismissed it as "negligible because
//! σ ≫ floor in every realistic fit" — but that is false whenever the response
//! spread is small (normalized features, data in small units, …): there the
//! 0.01 absolute floor dwarfs the true σ and the reported predictive
//! uncertainty is wrong by a large multiplicative factor.
//!
//! ## Why this test is confound-free
//!
//! Fitting on `y` standardizes by `S = sample_std(y)`; fitting on `c·y`
//! standardizes by `c·S`. The internal (standardized) problem is `y/S` in BOTH
//! cases, so the internal optimum η_internal — and hence the recovered smooth
//! SHAPE — is identical. The ONLY thing that differs between the two fits is how
//! the floor is carried back to raw units. The equivariance deviation is thus
//! exactly `0.01·(1 − c)`, free of optimizer / λ-selection drift.
//!
//! ## The fix
//!
//! σ is reconstructed from the persisted raw-unit log-σ coefficients with the
//! response-scale-relative floor `response_scale·sigma_floor` (so
//! `σ = response_scale·(sigma_floor + exp(η_internal))`), matching the
//! production predictor (`GaussianLocationScalePredictor::sigma_floor`). The
//! floor cannot ride the `+ln(response_scale)` intercept shift, so it is carried
//! as data (`GaussianLocationScaleFitResult::{response_scale, sigma_floor}`) and
//! applied at reconstruction. With the floor scaled, this assertion holds.
//!
//! The floor itself is no longer mgcv's fixed `0.01`: it is the recording-grid
//! bound `δ/√12` of the standardized response (`gaussian_resolution_sigma_floor`),
//! and a gap between standardized responses is unchanged by `y → c·y`, so the
//! floor is equivariant by construction as well. The extreme-rescale test below
//! pins that down at `c = 1e-4` and `c = 1e4`.

use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;

/// Reconstruct gam's fitted σ(x_i) at the training abscissae from a Gaussian
/// location-scale fit on the supplied responses, together with the fit's raw-unit
/// σ floor `response_scale·sigma_floor`:
/// `σ = response_scale·sigma_floor + exp(X_scale · β_scale)`.
fn fit_sigma_on_response(x: &[f64], y: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode loc-scale data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam location-scale fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult {
        fit,
        response_scale,
        sigma_floor,
        ..
    }) = result
    else {
        panic!("expected a Gaussian location-scale fit");
    };

    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-sigma) block present")
        .beta
        .clone();

    let mut grid = Array2::<f64>::zeros((n, ncols));
    for (i, &t) in x.iter().enumerate() {
        grid[[i, x_idx]] = t;
    }
    let scale_design = build_term_collection_design(grid.view(), &fit.noisespec_resolved)
        .expect("rebuild log-sigma design at grid");
    let eta_sigma = scale_design.design.apply(&beta_scale);
    // Equivariant σ reconstruction: the standardized→raw remap shifts the log-σ
    // intercept by `+ln(response_scale)` (scaling only the `exp(η)` term), so the
    // soft floor must be reconstructed at `response_scale·sigma_floor` for
    // σ = response_scale·(sigma_floor + exp(η_internal)) to scale with the
    // response (#884). The persisted production predictor uses exactly this floor.
    let raw_floor = response_scale * sigma_floor;
    let sigma = eta_sigma.iter().map(|&e| raw_floor + e.exp()).collect();
    (sigma, raw_floor)
}

/// Deterministic heteroscedastic recipe (seed-42 LCG): x ~ Uniform(0,1) sorted,
/// μ_true = sin(2πx), σ_true(x) = 0.1 + 0.2·sin(2πx), y = μ_true + σ_true·z.
fn heteroscedastic_recipe(n: usize) -> (Vec<f64>, Vec<f64>) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state: u64 = 42;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let z: Vec<f64> = (0..n)
        .map(|_| {
            let u1 = next_unit().max(1e-300);
            let u2 = next_unit();
            (-2.0 * u1.ln()).sqrt() * (two_pi * u2).cos()
        })
        .collect();
    let y = (0..n)
        .map(|i| (two_pi * x[i]).sin() + (0.1 + 0.2 * (two_pi * x[i]).sin()) * z[i])
        .collect();
    (x, y)
}

/// Largest relative deviation of `scaled` from `c · base`, with its index.
fn max_relative_equivariance_error(base: &[f64], scaled: &[f64], c: f64) -> (f64, usize) {
    let mut worst = (0.0_f64, 0usize);
    for (i, (&b, &s)) in base.iter().zip(scaled).enumerate() {
        let rel = (s - c * b).abs() / (c * b);
        if rel > worst.0 {
            worst = (rel, i);
        }
    }
    worst
}

/// The σ floor is derived from the standardized response's recording grid, so
/// neither it nor the fitted σ surface may depend on the unit the response is
/// recorded in. Rescaling by eight orders of magnitude each way must reproduce
/// σ̂ and the raw floor to within floating-point reassociation of the fit.
#[test]
fn location_scale_noise_and_floor_are_equivariant_at_extreme_rescales() {
    init_parallelism();
    let (x, y) = heteroscedastic_recipe(200);
    let (sigma_base, floor_base) = fit_sigma_on_response(&x, &y);
    assert!(
        floor_base > 0.0 && floor_base.is_finite(),
        "the fitted raw σ floor must be positive and finite, got {floor_base:e}"
    );
    for c in [1.0e-4_f64, 1.0e4] {
        let y_scaled: Vec<f64> = y.iter().map(|&v| c * v).collect();
        let (sigma_scaled, floor_scaled) = fit_sigma_on_response(&x, &y_scaled);
        let floor_rel = (floor_scaled - c * floor_base).abs() / (c * floor_base);
        assert!(
            floor_rel < 1.0e-6,
            "raw σ floor is not equivariant at c={c:e}: {floor_scaled:e} vs \
             c·floor={:e} (relative error {floor_rel:e})",
            c * floor_base
        );
        let (max_rel, worst) = max_relative_equivariance_error(&sigma_base, &sigma_scaled, c);
        assert!(
            max_rel < 1.0e-6,
            "σ̂ is not equivariant at c={c:e}: max relative error {max_rel:e} at i={worst} \
             (σ̂_scaled={:e}, c·σ̂_base={:e})",
            sigma_scaled[worst],
            c * sigma_base[worst]
        );
    }
}

#[test]
fn location_scale_noise_is_response_scale_equivariant() {
    init_parallelism();

    let n = 200usize;
    let (x, y) = heteroscedastic_recipe(n);

    // ---- fit at the native scale and at a SMALL response scale --------------
    // c = 0.05 makes the rescaled σ̂ (~0.005..0.03) comparable to mgcv's 0.01
    // floor, so a floor left in raw units shows up as a large RELATIVE error. The
    // model is genuinely usable at this scale (data measured in small units is
    // entirely ordinary); the predictive intervals it reports must still scale.
    let c = 0.05_f64;
    let y_scaled: Vec<f64> = y.iter().map(|&v| c * v).collect();

    let (sigma_base, floor_base) = fit_sigma_on_response(&x, &y);
    let (sigma_scaled, _) = fit_sigma_on_response(&x, &y_scaled);

    // ---- equivariance assertion: σ̂_{c·y}(x) == c · σ̂_y(x) ------------------
    let mut max_rel = 0.0_f64;
    let mut worst = (0usize, 0.0, 0.0);
    for i in 0..n {
        let expected = c * sigma_base[i];
        let rel = (sigma_scaled[i] - expected).abs() / expected;
        if rel > max_rel {
            max_rel = rel;
            worst = (i, sigma_scaled[i], expected);
        }
    }

    assert!(
        max_rel < 1.0e-2,
        "Gaussian location-scale noise model is not response-scale equivariant: \
         fitting on c·y (c={c}) gives σ̂ that differs from c·σ̂(y) by up to {:.1}% \
         (worst at i={}: σ̂_scaled={:.6e}, expected c·σ̂_base={:.6e}, \
         abs diff={:.6e}; a raw-unit floor would give floor·(1−c)={:.6e}). \
         Root cause (#884): the soft floor must be reconstructed at \
         response_scale·sigma_floor, not in raw units — the log-σ intercept shift \
         `+ln(response_scale)` only scales the exp(η) term.",
        max_rel * 100.0,
        worst.0,
        worst.1,
        worst.2,
        (worst.1 - worst.2).abs(),
        floor_base * (1.0 - c),
    );
}
