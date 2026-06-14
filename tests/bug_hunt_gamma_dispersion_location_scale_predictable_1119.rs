//! #1119: a Gamma (and, by the same family path, NB / Tweedie) *dispersion*
//! location-scale model — `family="gamma"` with a `noise_formula` on the shape
//! channel — used to FIT but be completely UNPREDICTABLE: the joint
//! (mean + log-precision) posterior covariance and the EDF were never assembled
//! on the fit, so `predict` aborted with
//!
//!   "dispersion location-scale posterior mean requires covariance or
//!    penalized Hessian for posterior-mean prediction".
//!
//! The Beta dispersion sibling, routing through the SAME
//! `DispersionGlmLocationScalePredictor`, worked — Beta carries a genuinely
//! nonzero `(η_μ, η_φ)` Fisher cross block and so always assembled the joint
//! coefficient Hessian, whereas the Fisher-orthogonal members (NB / Gamma /
//! Tweedie) returned `None` for that Hessian, stranding the multi-block
//! outer-REML path and leaving `covariance_conditional` / EDF unset.
//!
//! The fix assembles the (block-diagonal, zero-cross) joint Hessian for all
//! four members and declares `likelihood_blocks_uncoupled()` for the orthogonal
//! ones, so the joint covariance + EDF are populated exactly as for Beta.
//!
//! This test is the missing predict-side gate. It is NOT a beta/mean-recovery
//! test (those exist and pass without exercising covariance, which is why they
//! never caught #1119); it asserts the three things #1119 reports as broken:
//!   1. the fit carries a JOINT posterior covariance of dim `p_mu + p_disp`;
//!   2. the fit carries a finite total EDF;
//!   3. the POSTERIOR-MEAN predict path — the exact call that hard-failed —
//!      returns finite means that recover the known `exp(0.5 + 0.6 x)` mean
//!      trend, with finite covariance-derived η / mean standard errors.
//!
//! Data-generating law (KNOWN): for covariate `x` on a grid,
//!   mu_true(x)    = exp(0.5 + 0.6 x)                  (log mean link)
//!   shape_true(x) = exp(0.7 + 0.4 cos(2 x))  = nu(x)  (genuine varying shape)
//!   y ~ Gamma(shape = nu(x), scale = mu(x)/nu(x))     (mean mu, Var = mu^2/nu)

use gam::estimate::BlockRole;
use gam::gamlss::DispersionFamilyKind;
use gam::predict::{
    DispersionLocationScalePredictor, InferenceCovarianceMode, PosteriorMeanOptions, PredictInput,
    PredictableModel,
};
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{pearson, rmse};
use gam::{
    DispersionLocationScaleFitResult, FitConfig, FitResult, encode_recordswith_inferred_schema,
    fit_from_formula, init_parallelism,
};
use ndarray::{Array1, Array2};

/// Deterministic seeded uniform in [0,1) (Numerical Recipes LCG, high bits).
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    /// Gamma(shape k>0, scale theta) via Marsaglia–Tsang (k>=1) with the
    /// Ahrens–Dieter boost for k<1, off a single reproducible LCG.
    fn gamma(&mut self, k: f64, theta: f64) -> f64 {
        if k < 1.0 {
            let g = self.gamma(k + 1.0, theta);
            let u = self.unit().max(1e-300);
            return g * u.powf(1.0 / k);
        }
        let d = k - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let u1 = self.unit().max(1e-300);
            let u2 = self.unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (1.0 + c * z).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.unit().max(1e-300);
            if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
                return d * v * theta;
            }
        }
    }
}

fn mu_true(x: f64) -> f64 {
    (0.5 + 0.6 * x).exp()
}
fn shape_true(x: f64) -> f64 {
    (0.7 + 0.4 * (2.0 * x).cos()).exp()
}

#[test]
fn gamma_dispersion_location_scale_assembles_covariance_and_is_predictable() {
    init_parallelism();

    // ---- synthetic heteroscedastic-dispersion Gamma (seed = 1119) ----------
    let n = 600usize;
    let mut rng = Lcg(1119);
    let x: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = mu_true(xi);
            let nu = shape_true(xi);
            rng.gamma(nu, mu / nu).max(1e-6)
        })
        .collect();

    // ---- gam Gamma dispersion location-scale fit (smooth mean + smooth noise),
    //      the exact #1119 repro shape: `s(x)` in BOTH channels so the outer
    //      REML smoothing path (which needed the joint Hessian) is exercised. --
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode gamma data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gamma".to_string()),
        noise_formula: Some("s(x, k=8)".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, k=8)", &ds, &cfg).expect("gam gamma dispersion fit");
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("expected a DispersionLocationScale fit result");
    };
    assert_eq!(
        kind,
        DispersionFamilyKind::Gamma,
        "gamma + noise_formula must route to the Gamma dispersion family"
    );

    let beta_mu = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location (mean) block present")
        .beta
        .clone();
    let beta_noise = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale (log-shape) block present")
        .beta
        .clone();
    let p_mu = beta_mu.len();
    let p_disp = beta_noise.len();
    let p_total = p_mu + p_disp;

    // ── 1. JOINT POSTERIOR COVARIANCE present and correctly sized ────────────
    //    This is the headline #1119 failure: `summary.covariance_n is None`.
    //    For an orthogonal dispersion family the joint Hessian is block-diagonal
    //    but MUST still be assembled and inverted into a `p_total × p_total`
    //    covariance — exactly as the Beta sibling does.
    let covariance = fit
        .fit
        .beta_covariance()
        .expect(
            "gamma dispersion location-scale fit must carry a joint posterior covariance \
             (was None before #1119: the orthogonal-family joint Hessian returned None and \
             stranded the covariance assembly)",
        )
        .clone();
    assert_eq!(
        covariance.dim(),
        (p_total, p_total),
        "joint covariance must span both blocks: expected {p_total}×{p_total} \
         (p_mu={p_mu} + p_disp={p_disp})"
    );
    assert!(
        covariance.iter().all(|v| v.is_finite()),
        "joint covariance must be finite"
    );
    // Posterior variances are positive on the diagonal.
    for i in 0..p_total {
        assert!(
            covariance[[i, i]] > 0.0,
            "joint covariance diagonal entry {i} must be a positive posterior variance, got {}",
            covariance[[i, i]]
        );
    }

    // ── 2. FINITE TOTAL EDF present ──────────────────────────────────────────
    //    `summary.edf_total is None` was the other #1119 symptom.
    let edf_total = fit
        .fit
        .edf_total()
        .expect("gamma dispersion location-scale fit must carry a finite total EDF (was None)");
    assert!(
        edf_total.is_finite() && edf_total > 0.0 && edf_total <= p_total as f64 + 1e-6,
        "total EDF must be finite in (0, p_total={p_total}], got {edf_total}"
    );

    // ---- build the dispersion predictor exactly as `Model::predictor` does --
    let predictor = DispersionLocationScalePredictor {
        beta_mu: beta_mu.clone(),
        beta_noise: beta_noise.clone(),
        likelihood: kind.likelihood_spec(),
        inverse_link: Some(kind.base_link()),
        covariance: Some(covariance),
    };

    // Held-out grid spanning the training range.
    let grid_n = 40usize;
    let grid_x: Vec<f64> = (0..grid_n)
        .map(|i| -1.8 + 3.6 * (i as f64) / (grid_n as f64 - 1.0))
        .collect();
    let mut test_grid = Array2::<f64>::zeros((grid_n, ncols));
    for i in 0..grid_n {
        test_grid[[i, x_idx]] = grid_x[i];
    }
    let mean_design = build_term_collection_design(test_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild mean design at grid");
    let disp_design = build_term_collection_design(test_grid.view(), &fit.noisespec_resolved)
        .expect("rebuild dispersion design at grid");
    assert_eq!(
        mean_design.design.ncols(),
        p_mu,
        "mean grid design must match the fitted mean block width"
    );
    assert_eq!(
        disp_design.design.ncols(),
        p_disp,
        "dispersion grid design must match the fitted log-precision block width"
    );

    let input = PredictInput {
        design: mean_design.design,
        offset: Array1::<f64>::zeros(grid_n),
        design_noise: Some(disp_design.design),
        offset_noise: Some(Array1::<f64>::zeros(grid_n)),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };

    // ── 3. POSTERIOR-MEAN PREDICT returns (with confidence bounds) ───────────
    //    `predict_posterior_mean` is the canonical response-scale predict path
    //    (the one the CLI / FFI drive). Its posterior-mean arm hard-failed in
    //    #1119 via `require_posterior_mean_backend` when the covariance was
    //    absent. With the joint covariance assembled it must now return finite
    //    means, covariance-derived η SEs, and finite confidence bounds.
    let pm_options = PosteriorMeanOptions {
        confidence_level: Some(0.95),
        covariance_mode: InferenceCovarianceMode::Conditional,
        include_observation_interval: true,
    };
    let pred = predictor
        .predict_posterior_mean(&input, &fit.fit, &pm_options)
        .expect(
            "posterior-mean predict must succeed for a gamma dispersion location-scale fit \
             (this is the call that aborted in #1119)",
        );

    assert_eq!(pred.mean.len(), grid_n, "one predicted mean per grid row");
    for (xi, m) in grid_x.iter().zip(pred.mean.iter()) {
        assert!(
            m.is_finite() && *m > 0.0,
            "gamma predicted mean at x={xi} must be a finite positive response, got {m}"
        );
    }

    // Covariance-derived η standard errors must be finite (they were
    // unavailable when the covariance was None).
    assert_eq!(
        pred.eta_standard_error.len(),
        grid_n,
        "one η standard error per grid row"
    );
    assert!(
        pred.eta_standard_error
            .iter()
            .all(|s| s.is_finite() && *s >= 0.0),
        "predict η standard errors must be finite and non-negative"
    );

    // Confidence bounds must be present (requested) and bracket the point mean.
    let mean_lower: &Array1<f64> = pred
        .mean_lower
        .as_ref()
        .expect("posterior-mean confidence lower bound must be available from the covariance");
    let mean_upper: &Array1<f64> = pred
        .mean_upper
        .as_ref()
        .expect("posterior-mean confidence upper bound must be available from the covariance");
    for i in 0..grid_n {
        assert!(
            mean_lower[i].is_finite()
                && mean_upper[i].is_finite()
                && mean_lower[i] <= pred.mean[i] + 1e-9
                && pred.mean[i] <= mean_upper[i] + 1e-9,
            "confidence interval at grid row {i} must bracket the mean: \
             [{}, {}] around {}",
            mean_lower[i],
            mean_upper[i],
            pred.mean[i]
        );
    }

    // ---- truth recovery on the held-out grid -------------------------------
    let truth_mean: Vec<f64> = grid_x.iter().map(|&xi| mu_true(xi)).collect();
    let pred_mean: Vec<f64> = pred.mean.to_vec();
    let log_pred: Vec<f64> = pred_mean.iter().map(|m| m.ln()).collect();
    let log_truth: Vec<f64> = truth_mean.iter().map(|m| m.ln()).collect();
    let corr = pearson(&pred_mean, &truth_mean);
    let rel_rmse = rmse(&log_pred, &log_truth);
    eprintln!(
        "[#1119 gamma dispersion predict] n={n} grid={grid_n} edf_total={edf_total:.3} \
         corr(pred,truth)={corr:.4} rmse(log mean)={rel_rmse:.4}"
    );
    assert!(
        corr > 0.9,
        "posterior-mean predictions must track the true mean trend: pearson={corr:.4} (<= 0.9)"
    );
    assert!(
        rel_rmse < 0.25,
        "posterior-mean predictions must recover the true log-mean: \
         rmse(log mean)={rel_rmse:.4} (>= 0.25)"
    );
}
