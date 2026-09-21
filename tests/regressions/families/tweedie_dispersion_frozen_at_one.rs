//! Regression lock for issue #771: Tweedie regression must estimate the
//! dispersion `φ` from the data rather than freezing it at 1.0.
//!
//! For a Tweedie GLM the response variance is `Var(y) = φ · μ^p` with `φ` a
//! genuine free dispersion parameter (this is the whole point of the family —
//! it models over-dispersed, zero-inflated positive data). Unlike Binomial and
//! Poisson, whose variance is fully pinned by the mean (`φ ≡ 1`), Tweedie's `φ`
//! must be estimated, exactly as Gamma's shape and the Gaussian residual
//! variance are. mgcv's `tw()` / statsmodels' Tweedie both estimate it.
//!
//! The original bug was rooted in `LikelihoodSpec::default_scale_metadata`
//! (src/types.rs), which lumped `Tweedie` in with `Binomial | Poisson` and
//! returned `FixedDispersion { phi: 1.0 }`. The fix makes Tweedie use
//! `EstimatedTweediePhi`, refreshed from converged-η Pearson residuals, so the
//! IRLS weight and fitted coefficient covariance reflect the data's dispersion.
//!
//! This test fits two Tweedie datasets that share the *same covariate values*
//! and the *same mean structure* but very different true dispersions
//! (`φ = 0.3` vs `φ = 8.0`) and checks, for each fit, the two things #771 is
//! about:
//!
//! 1. **φ̂ recovers the true φ.** The engine's estimator is the Pearson mean
//!    `φ̂ = (1/n) Σ (yᵢ − μ̂ᵢ)² / μ̂ᵢ^p`. Its sampling SD is derived below from
//!    the Tweedie cumulants, so the gate is `|φ̂ − φ| ≤ 4·SD(φ̂) + (p_β/n)·φ`
//!    (the last term is the `n` vs `n − p_β` divisor bias). A frozen φ = 1
//!    misses 0.3 and 8.0 by about 100 and 23 standard deviations.
//! 2. **SE(η̂) is `√φ̂` times the unit-dispersion SE.** For the unpenalized
//!    `y ~ x` fit the covariance is `φ̂ · I(β̂)⁻¹`. The test rebuilds both
//!    legitimate information matrices from the fit's own `μ̂`: expected
//!    (Fisher) weights `μ̂^{2−p}` and observed (Newton) weights
//!    `μ̂^{2−p} + (p − 1)(y − μ̂)μ̂^{1−p}`. The engine's SE must lie between the
//!    two SEs these give, widened by 1e-4 relative for the PIRLS stopping
//!    tolerance (β̂ is converged to ~1e-8, so SEs rebuilt from it agree far
//!    tighter than that).
//!
//! The old gate, `SE ratio ≥ 2.0` against an expected `√(8/0.3) ≈ 5.16`, was
//! one-sided and underived and never looked at φ̂. A φ̂ biased by a factor of
//! two, or SEs inflated by any amount, passed it.
//!
//! ## SD of φ̂ (derived, checked by simulation)
//!
//! Write `tᵢ = (yᵢ − μᵢ)²/μᵢ^p` and `sᵢ = (yᵢ − μᵢ)μᵢ^{1−p}`. The Tweedie
//! cumulants `κ₂ = φμ^p`, `κ₃ = pφ²μ^{2p−1}`, `κ₄ = p(2p−1)φ³μ^{3p−2}` give
//! `Var tᵢ = p(2p−1)φ³μᵢ^{p−2} + 2φ²`, `Var sᵢ = φμᵢ^{2−p}` and
//! `Cov(tᵢ, sᵢ) = pφ²`. With log link, `∂tᵢ/∂ηᵢ` has mean `−pφ`, so plugging
//! in the MLE `β̂ − β ≈ H⁻¹ (1/n) Σ xᵢ sᵢ` (with `H = (1/n) Σ μᵢ^{2−p} xᵢxᵢᵀ`)
//! gives the influence function `tᵢ − φ − pφ·x̄ᵀH⁻¹xᵢ·sᵢ` and
//!
//! `n·Var φ̂ = mean_i[p(2p−1)φ³μᵢ^{p−2} + 2φ²] − p²φ³·x̄ᵀH⁻¹x̄`.
//!
//! The second term matters: at φ = 8 it removes about two thirds of the
//! variance, because a large `y` both inflates its own `tᵢ` and pulls `μ̂` up.
//! In 2000 simulated replicates of this fixture the standardized error has SD
//! 1.03 (φ = 0.3) and 0.96 (φ = 8), with 0.00% and 0.05% of `|z|` above 4.

use csv::StringRecord;
use gam::smooth::build_term_collection_design;
use gam::types::{InverseLink, LikelihoodSpec, ResponseFamily, StandardLink};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use gam_predict::{
    InferenceCovarianceMode, MeanIntervalMethod, PredictUncertaintyOptions,
    predict_gamwith_uncertainty,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Poisson, Uniform};

const B0: f64 = 1.0;
const BX: f64 = 0.5;
const TWEEDIE_P: f64 = 1.5;
/// Number of mean coefficients in `y ~ x` (intercept and slope).
const N_COEF: usize = 2;
/// Standard deviations allowed between φ̂ and the true φ.
const PHI_Z: f64 = 4.0;
/// Relative slack on the SE hull for the PIRLS stopping tolerance.
const SE_REL_SLACK: f64 = 1e-4;

/// Draw a single compound Poisson–Gamma (Tweedie, `1 < p < 2`) response with
/// mean `mu` and dispersion `phi`.
fn tweedie_sample(rng: &mut StdRng, mu: f64, phi: f64) -> f64 {
    let p = TWEEDIE_P;
    let lambda = mu.powf(2.0 - p) / (phi * (2.0 - p));
    let alpha = (2.0 - p) / (p - 1.0);
    let scale = phi * (p - 1.0) * mu.powf(p - 1.0);
    let n: u64 = Poisson::new(lambda).expect("poisson rate").sample(rng) as u64;
    if n == 0 {
        return 0.0;
    }
    // Sum of `n` iid Gamma(alpha, scale) is Gamma(n*alpha, scale).
    Gamma::new(alpha * n as f64, scale)
        .expect("gamma(shape,scale)")
        .sample(rng)
}

/// `vᵀ A⁻¹ v` for a small symmetric positive-definite `A`, by Cholesky.
fn quad_inverse(a: &Array2<f64>, v: &[f64]) -> f64 {
    let k = v.len();
    assert_eq!(a.dim(), (k, k), "information matrix / vector size mismatch");
    let mut l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for m in 0..j {
                s -= l[[i, m]] * l[[j, m]];
            }
            if i == j {
                assert!(s > 0.0, "information matrix is not positive definite (pivot {s})");
                l[[i, i]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Solve L w = v; then vᵀA⁻¹v = |w|².
    let mut w = vec![0.0; k];
    for i in 0..k {
        let mut s = v[i];
        for m in 0..i {
            s -= l[[i, m]] * w[m];
        }
        w[i] = s / l[[i, i]];
    }
    w.iter().map(|x| x * x).sum()
}

/// `Σ wᵢ dᵢdᵢᵀ` over the rows `dᵢ` of `design`.
fn weighted_gram(design: &Array2<f64>, w: &[f64]) -> Array2<f64> {
    let k = design.ncols();
    let mut g = Array2::<f64>::zeros((k, k));
    for (row, &wi) in design.rows().into_iter().zip(w) {
        for a in 0..k {
            for b in 0..k {
                g[[a, b]] += wi * row[a] * row[b];
            }
        }
    }
    g
}

struct TweedieFit {
    /// Engine SE(η̂) on the evaluation grid.
    se: Vec<f64>,
    /// `√φ̂ ·` SE from expected (Fisher) information at `μ̂`.
    se_fisher: Vec<f64>,
    /// `√φ̂ ·` SE from observed (Newton) information at `μ̂`.
    se_observed: Vec<f64>,
    phi: f64,
}

/// Fit `y ~ x` as a Tweedie(log) model on `(x, y)` and return the engine's
/// linear-predictor SEs on `eval`, the two reference SEs rebuilt from the
/// fit's own `μ̂`, and the fitted dispersion.
fn fit_tweedie(x: &[f64], y: &[f64], eval: &[f64]) -> Result<TweedieFit, String> {
    let n = x.len();
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows)
        .map_err(|e| format!("encode: {e}"))?;
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some(format!("tweedie(p={TWEEDIE_P})")),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x", &ds, &cfg).map_err(|e| format!("fit: {e}"))?
    else {
        return Err("expected a standard fit".to_string());
    };

    let design_at = |points: &[f64]| -> Result<Array2<f64>, String> {
        let mut grid = Array2::<f64>::zeros((points.len(), ds.headers.len()));
        for (i, &xi) in points.iter().enumerate() {
            grid[[i, x_idx]] = xi;
        }
        build_term_collection_design(grid.view(), &fit.resolvedspec)
            .map(|d| d.design.to_dense())
            .map_err(|e| format!("design: {e}"))
    };
    let train = design_at(x)?;
    let dense = design_at(eval)?;
    let m = eval.len();

    let tweedie_log = LikelihoodSpec::new(
        ResponseFamily::Tweedie { p: TWEEDIE_P },
        InverseLink::Standard(StandardLink::Log),
    );
    let offset = Array1::<f64>::zeros(m);
    let pred = predict_gamwith_uncertainty(
        dense.clone(),
        fit.fit.beta.view(),
        offset.view(),
        tweedie_log,
        &fit.fit,
        &PredictUncertaintyOptions {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::Conditional,
            mean_interval_method: MeanIntervalMethod::Delta,
            includeobservation_interval: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .map_err(|e| format!("predict: {e}"))?;

    let phi = fit
        .fit
        .inference
        .as_ref()
        .map(|inf| inf.dispersion.phi())
        .ok_or("fit carries no inference block, so no dispersion")?;

    let p = TWEEDIE_P;
    let mu_hat: Vec<f64> = train
        .dot(&fit.fit.beta.view())
        .iter()
        .map(|e| e.exp())
        .collect();
    let w_fisher: Vec<f64> = mu_hat.iter().map(|m| m.powf(2.0 - p)).collect();
    let w_observed: Vec<f64> = mu_hat
        .iter()
        .zip(y)
        .map(|(m, yi)| m.powf(2.0 - p) + (p - 1.0) * (yi - m) * m.powf(1.0 - p))
        .collect();
    let info_fisher = weighted_gram(&train, &w_fisher);
    let info_observed = weighted_gram(&train, &w_observed);
    let se_from = |info: &Array2<f64>| -> Vec<f64> {
        dense
            .rows()
            .into_iter()
            .map(|d| (phi * quad_inverse(info, &d.to_vec())).sqrt())
            .collect()
    };

    Ok(TweedieFit {
        se: pred.eta_standard_error.to_vec(),
        se_fisher: se_from(&info_fisher),
        se_observed: se_from(&info_observed),
        phi,
    })
}

/// Derived sampling SD of the Pearson φ̂ at the true `(μ, φ)`; see the module
/// docs for the derivation.
fn pearson_phi_sd(x: &[f64], phi: f64) -> f64 {
    let p = TWEEDIE_P;
    let n = x.len() as f64;
    let mut mean_var_t = 0.0;
    let mut h = [[0.0_f64; 2]; 2];
    let mut xbar = [0.0_f64; 2];
    for &xi in x {
        let mu = (B0 + BX * xi).exp();
        mean_var_t += p * (2.0 * p - 1.0) * phi.powi(3) * mu.powf(p - 2.0) + 2.0 * phi * phi;
        let row = [1.0, xi];
        let w = mu.powf(2.0 - p);
        for a in 0..2 {
            xbar[a] += row[a];
            for b in 0..2 {
                h[a][b] += w * row[a] * row[b];
            }
        }
    }
    mean_var_t /= n;
    let h = Array2::from_shape_fn((2, 2), |(a, b)| h[a][b] / n);
    let xbar = [xbar[0] / n, xbar[1] / n];
    let var = (mean_var_t - p * p * phi.powi(3) * quad_inverse(&h, &xbar)) / n;
    assert!(var > 0.0, "derived Var(φ̂) must be positive, got {var}");
    var.sqrt()
}

#[test]
fn tweedie_prediction_se_scales_with_dispersion() {
    init_parallelism();

    let n = 4000usize;
    let phi_lo = 0.3_f64;
    let phi_hi = 8.0_f64;

    // Shared covariate values, so the only inferential difference between the
    // two fits is the true dispersion.
    let mut rng = StdRng::seed_from_u64(0x7_3D1E_u64);
    let ux = Uniform::new(-1.0_f64, 1.0_f64).expect("uniform -1..1");
    let x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();

    let mut rng_lo = StdRng::seed_from_u64(11);
    let mut rng_hi = StdRng::seed_from_u64(22);
    let mut y_lo = Vec::with_capacity(n);
    let mut y_hi = Vec::with_capacity(n);
    for &xi in &x {
        let mu = (B0 + BX * xi).exp();
        y_lo.push(tweedie_sample(&mut rng_lo, mu, phi_lo));
        y_hi.push(tweedie_sample(&mut rng_hi, mu, phi_hi));
    }

    let eval: Vec<f64> = vec![-0.8, -0.4, 0.0, 0.4, 0.8];

    let mut mean_se = Vec::new();
    for (y, phi_true, tag) in [(&y_lo, phi_lo, "lo"), (&y_hi, phi_hi, "hi")] {
        let fit = fit_tweedie(&x, y, &eval)
            .unwrap_or_else(|e| panic!("{tag}-dispersion Tweedie fit must succeed: {e}"));

        let sd = pearson_phi_sd(&x, phi_true);
        let tol = PHI_Z * sd + (N_COEF as f64 / n as f64) * phi_true;
        let z = (fit.phi - phi_true) / sd;
        eprintln!(
            "[tweedie-φ {tag}] true φ={phi_true} φ̂={:.5} SD(φ̂)={sd:.5} z={z:.2}; \
             SE engine={:?} fisher={:?} observed={:?}",
            fit.phi, fit.se, fit.se_fisher, fit.se_observed
        );
        assert!(
            (fit.phi - phi_true).abs() <= tol,
            "{tag}: Tweedie φ̂ = {:.5} does not recover the true φ = {phi_true}: \
             |error| = {:.5} > {tol:.5} ({PHI_Z} derived SDs of the Pearson estimator, \
             SD = {sd:.5}, plus the n vs n − p divisor bias). z = {z:.2}.",
            fit.phi,
            (fit.phi - phi_true).abs()
        );

        for i in 0..eval.len() {
            let (se, f, o) = (fit.se[i], fit.se_fisher[i], fit.se_observed[i]);
            let lo = f.min(o) * (1.0 - SE_REL_SLACK);
            let hi = f.max(o) * (1.0 + SE_REL_SLACK);
            assert!(
                se.is_finite() && se >= lo && se <= hi,
                "{tag}: SE(η̂) at x = {} is {se:.6e}, outside [{lo:.6e}, {hi:.6e}], the range \
                 spanned by √φ̂ × (Fisher SE {f:.6e}, observed-information SE {o:.6e}) with \
                 φ̂ = {:.5}. The covariance is not φ̂ · I(β̂)⁻¹.",
                eval[i],
                fit.phi
            );
        }
        mean_se.push(fit.se.iter().sum::<f64>() / fit.se.len() as f64);
    }

    eprintln!(
        "[tweedie-φ] mean η SE: lo={:.5} hi={:.5}; ratio={:.3} (≈ √(8/0.3) = {:.3})",
        mean_se[0],
        mean_se[1],
        mean_se[1] / mean_se[0],
        (phi_hi / phi_lo).sqrt()
    );
}
