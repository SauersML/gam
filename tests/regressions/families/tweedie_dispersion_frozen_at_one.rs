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
//! (`φ = 0.3` vs `φ = 8.0`). Because the working weights `W = μ^{2-p}` and the
//! design `X` are essentially identical across the two fits, the only thing that
//! should move the linear-predictor SE is `√φ`. A correct fit therefore yields
//! `mean SE(η̂)_hi / mean SE(η̂)_lo ≈ √(8.0/0.3) ≈ 5.2`. With `φ` frozen at 1.0
//! the ratio collapses to ~1.0 — the high-dispersion model reports the same
//! (wildly over-confident) uncertainty as the low-dispersion one.
//!
//! The SE ratio is only printed. Two derived gates carry the lock (#4131):
//! 1. `φ̂` itself, the quantity the frozen-φ bug got wrong, must recover each
//!    true `φ` within the Pearson estimator's own sampling band
//!    (`tweedie_phi_gate`).
//! 2. Every `SE(η̂)` must equal the one the fit's own `μ̂` and `φ̂` imply:
//!    `√(dᵀI⁻¹d)`, with `I` the Fisher information `Σ xxᵀμ̂^{2−p}/φ̂` or the
//!    observed information `Σ xxᵀ[μ̂^{2−p} + (p−1)(y−μ̂)μ̂^{1−p}]/φ̂`. The log link
//!    is not Tweedie's canonical link, so the two differ, by 0.06% (φ = 0.3)
//!    and 0.4% (φ = 8) on this fixture. The reported SE must lie in their hull.
//!    A covariance scaled by any dispersion other than `φ̂` misses the hull by
//!    `√(φ_used/φ̂)`.

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

use super::tweedie_phi_gate::TweedieLogFixture;

const B0: f64 = 1.0;
const BX: f64 = 0.5;
const TWEEDIE_P: f64 = 1.5;

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

/// Relative slack on the information hull of `SE(η̂)`. The fit's covariance is
/// read at its converged β̂, so the only gap left is the PIRLS stopping residual,
/// far below the 6e-4 width of the hull itself. The frozen-φ defect moves SE by
/// `√φ̂`, a factor of at least 1.8 on this fixture.
const SE_HULL_REL: f64 = 1e-4;

struct TweedieEtaFit {
    /// Reported `SE(η̂)` on the evaluation grid.
    se: Vec<f64>,
    /// `√(dᵀI⁻¹d)` from the Fisher information at the fit's own `μ̂`, `φ̂`.
    fisher_se: Vec<f64>,
    /// `√(dᵀI⁻¹d)` from the observed information at the fit's own `μ̂`, `φ̂`.
    observed_se: Vec<f64>,
    /// Fitted dispersion the covariance was scaled by.
    phi_hat: f64,
    /// Coefficient count of the fit.
    n_coefficients: usize,
}

/// `√(dᵀ I⁻¹ d)` for every row `d` of `rows`, with `I` symmetric positive
/// definite (dense Cholesky; the fits here have two coefficients).
fn information_se(info: &Array2<f64>, rows: &Array2<f64>) -> Vec<f64> {
    let p = info.nrows();
    let mut l = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        let mut diag = info[[j, j]];
        for k in 0..j {
            diag -= l[[j, k]] * l[[j, k]];
        }
        assert!(diag > 0.0, "information is not positive definite: {info:?}");
        l[[j, j]] = diag.sqrt();
        for i in (j + 1)..p {
            let mut v = info[[i, j]];
            for k in 0..j {
                v -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = v / l[[j, j]];
        }
    }
    rows.outer_iter()
        .map(|d| {
            // dᵀI⁻¹d = ‖L⁻¹d‖².
            let mut z = vec![0.0; p];
            for i in 0..p {
                let mut v = d[i];
                for k in 0..i {
                    v -= l[[i, k]] * z[k];
                }
                z[i] = v / l[[i, i]];
            }
            z.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect()
}

/// Fit `y ~ x` as a Tweedie(log) model on `(x, y)` and return the reported
/// linear-predictor standard errors on the supplied evaluation grid, the SEs
/// the fit's own `μ̂` and `φ̂` imply there, and the fitted dispersion `φ`.
fn fit_tweedie_eta_se(x: &[f64], y: &[f64], eval: &[f64]) -> TweedieEtaFit {
    let n = x.len();
    let headers: Vec<String> = ["y", "x"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| StringRecord::from(vec![y[i].to_string(), x[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode tweedie data");
    let col = ds.column_map();
    let x_idx = col["x"];

    let cfg = FitConfig {
        family: Some(format!("tweedie(p={TWEEDIE_P})")),
        ..FitConfig::default()
    };
    let FitResult::Standard(fit) =
        fit_from_formula("y ~ x", &ds, &cfg).expect("gam tweedie fit should succeed")
    else {
        panic!("expected a Standard Tweedie fit");
    };

    let design_at = |xs: &[f64]| -> Array2<f64> {
        let mut grid = Array2::<f64>::zeros((xs.len(), ds.headers.len()));
        for (i, &xi) in xs.iter().enumerate() {
            grid[[i, x_idx]] = xi;
        }
        build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("design on the supplied abscissae")
            .design
            .to_dense()
    };
    let dense = design_at(eval);
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
            edgeworth_one_sided: false,
            boundary_correction: false,
            ood_inflation: false,
            ..PredictUncertaintyOptions::default()
        },
    )
    .expect("tweedie predict with uncertainty");

    let phi_hat = fit
        .fit
        .inference
        .as_ref()
        .map(|inf| inf.dispersion.phi())
        .expect("Tweedie fit must carry an estimated dispersion");

    // Fisher and observed information at the fit's own β̂ and φ̂.
    let train = design_at(x);
    let beta = &fit.fit.beta;
    let n_coefficients = beta.len();
    let mut fisher = Array2::<f64>::zeros((n_coefficients, n_coefficients));
    let mut observed = Array2::<f64>::zeros((n_coefficients, n_coefficients));
    for (row, &yi) in train.outer_iter().zip(y.iter()) {
        let mu = row.dot(beta).exp();
        let w_fisher = mu.powf(2.0 - TWEEDIE_P) / phi_hat;
        let w_observed =
            w_fisher + (TWEEDIE_P - 1.0) * (yi - mu) * mu.powf(1.0 - TWEEDIE_P) / phi_hat;
        for a in 0..n_coefficients {
            for b in 0..n_coefficients {
                fisher[[a, b]] += w_fisher * row[a] * row[b];
                observed[[a, b]] += w_observed * row[a] * row[b];
            }
        }
    }

    TweedieEtaFit {
        se: pred.eta_standard_error.to_vec(),
        fisher_se: information_se(&fisher, &dense),
        observed_se: information_se(&observed, &dense),
        phi_hat,
        n_coefficients,
    }
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

    let fit_lo = fit_tweedie_eta_se(&x, &y_lo, &eval);
    let fit_hi = fit_tweedie_eta_se(&x, &y_hi, &eval);

    let mean_lo: f64 = fit_lo.se.iter().sum::<f64>() / fit_lo.se.len() as f64;
    let mean_hi: f64 = fit_hi.se.iter().sum::<f64>() / fit_hi.se.len() as f64;
    eprintln!(
        "[tweedie-φ] true φ: lo={phi_lo} hi={phi_hi}; fitted φ: lo={:.4} hi={:.4}; \
         mean η SE: lo={mean_lo:.5} hi={mean_hi:.5}; SE ratio={:.3} (√(φ_hi/φ_lo) = {:.3})",
        fit_lo.phi_hat,
        fit_hi.phi_hat,
        mean_hi / mean_lo,
        (phi_hi / phi_lo).sqrt()
    );

    let fixture = TweedieLogFixture {
        x: &x,
        b0: B0,
        bx: BX,
        p: TWEEDIE_P,
    };
    for (fit, phi, tag) in [(&fit_lo, phi_lo, "lo"), (&fit_hi, phi_hi, "hi")] {
        // 1. φ̂ recovers the data's dispersion.
        fixture.assert_phi_hat_recovers(tag, fit.phi_hat, phi, fit.n_coefficients);

        // 2. SE(η̂) is the one the fit's own μ̂ and φ̂ imply.
        for (k, &se) in fit.se.iter().enumerate() {
            let (f, o) = (fit.fisher_se[k], fit.observed_se[k]);
            let lo = f.min(o) * (1.0 - SE_HULL_REL);
            let hi = f.max(o) * (1.0 + SE_HULL_REL);
            eprintln!(
                "[tweedie-φ {tag}] x={}: SE(η̂)={se:.6e}; Fisher {f:.6e}; observed {o:.6e}",
                eval[k]
            );
            assert!(
                se.is_finite() && se >= lo && se <= hi,
                "Tweedie SE(η̂) ({tag}, x={}) = {se:.6e} is not the SE the fit's own \
                 μ̂ and φ̂ = {:.5} imply: Fisher {f:.6e}, observed {o:.6e} \
                 (hull ±{SE_HULL_REL:e}). The covariance is scaled by a dispersion \
                 other than the fitted φ̂.",
                eval[k],
                fit.phi_hat
            );
        }
    }
}
