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

/// Regression for the posterior-mean *observation* (prediction) interval of a
/// heteroscedastic dispersion location-scale model.
///
/// THE BUG: `predict_posterior_mean`'s observation band was assembled by
/// `family_observation_band`, which reads a single fit-level scalar dispersion
/// (`observation_phi()` / `observation_theta()` / `observation_standard_deviation()`)
/// rather than the model's per-row precision submodel `precision = exp(eta_d)`.
/// For a genuine dispersion-LS fit (the entire point of which is a non-constant
/// variance) this collapsed the response-noise term to one constant, so the
/// observation band had a near-constant half-width across x — too narrow where
/// the true σ(x) is large and too wide where σ(x) is small. The companion
/// `predict_full_uncertainty` path got this right (it routes the per-row
/// `observation_noise(input)`), so the two prediction-interval APIs disagreed on
/// the SAME fit.
///
/// THE GATE: build a Gamma dispersion-LS fit whose shape (hence σ(x)) genuinely
/// varies across x, then assert
///   1. the posterior-mean observation half-width VARIES across rows in step
///      with the per-row `noise_sd` (the scalar-collapse made it ~constant), and
///   2. the σ implied by the posterior-mean band matches the per-row predictive
///      noise the full-uncertainty path uses, row by row.
#[test]
fn gamma_dispersion_posterior_mean_observation_band_is_per_row_not_scalar() {
    use gam::predict::PredictUncertaintyOptions;
    use gam::predict::interval_policy::PredictionTransform;

    init_parallelism();
    let n = 600usize;
    let mut rng = Lcg(7119);
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
    let covariance = fit
        .fit
        .beta_covariance()
        .expect("joint covariance present")
        .clone();

    let predictor = DispersionLocationScalePredictor {
        beta_mu,
        beta_noise,
        likelihood: kind.likelihood_spec(),
        inverse_link: Some(kind.base_link()),
        covariance: Some(covariance),
    };

    // Held-out grid spanning the training range so σ(x) genuinely varies.
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
    let input = PredictInput {
        design: mean_design.design,
        offset: Array1::<f64>::zeros(grid_n),
        design_noise: Some(disp_design.design),
        offset_noise: Some(Array1::<f64>::zeros(grid_n)),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };

    // The per-row response-noise σ(x) the model actually implies — the exact
    // quantity the full-uncertainty band consumes.
    let per_row_noise = predictor
        .observation_noise(&input)
        .expect("observation noise must be available")
        .expect("dispersion-LS exposes a per-row observation noise");
    assert_eq!(per_row_noise.len(), grid_n);
    // Sanity: σ(x) must genuinely vary (otherwise the test cannot distinguish a
    // per-row band from a scalar one).
    let noise_min = per_row_noise.iter().cloned().fold(f64::INFINITY, f64::min);
    let noise_max = per_row_noise
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        noise_max > 1.5 * noise_min,
        "test precondition: per-row σ(x) must vary materially across the grid \
         (min={noise_min:.4}, max={noise_max:.4})"
    );

    let pm_options = PosteriorMeanOptions {
        confidence_level: Some(0.95),
        covariance_mode: InferenceCovarianceMode::Conditional,
        include_observation_interval: true,
    };
    let pm = predictor
        .predict_posterior_mean(&input, &fit.fit, &pm_options)
        .expect("posterior-mean predict with observation interval");
    let obs_lower = pm
        .observation_lower
        .as_ref()
        .expect("observation lower band must be present");
    let obs_upper = pm
        .observation_upper
        .as_ref()
        .expect("observation upper band must be present");

    // Recover, per row, the σ implied by the posterior-mean observation band.
    // The band is μ ± z·√(SE(μ̂)² + σ²); with the per-row noise restored the
    // implied √(SE²+σ²) must track √(SE²+σ_row²) row-by-row, NOT a constant.
    let z = 1.959963984540054_f64; // Φ⁻¹(0.975)
    let mean_se = pm
        .eta_standard_error
        .iter()
        .zip(
            // mean SE = |dμ/dη| · SE(η); on the log link dμ/dη = μ.
            pm.mean.iter(),
        )
        .map(|(&se_eta, &mu)| mu.abs() * se_eta)
        .collect::<Vec<f64>>();

    let mut implied_sigma = Vec::with_capacity(grid_n);
    for i in 0..grid_n {
        // Use the wider (unclamped) tail as the band half-width: Gamma support
        // is [0, ∞) so the upper edge is never clamped, giving a clean readout.
        let half = (obs_upper[i] - pm.mean[i]) / z;
        let total_var = (half * half - mean_se[i] * mean_se[i]).max(0.0);
        implied_sigma.push(total_var.sqrt());
    }

    // (1) The band half-width must NOT be constant — the scalar-collapse bug made
    //     the implied σ identical across rows.
    let sig_min = implied_sigma.iter().cloned().fold(f64::INFINITY, f64::min);
    let sig_max = implied_sigma
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        sig_max > 1.3 * sig_min,
        "posterior-mean observation band must vary across rows with σ(x); a near-constant \
         implied σ (min={sig_min:.4}, max={sig_max:.4}) is the scalar-dispersion bug"
    );

    // (2) The implied σ must match the per-row model noise it should be built
    //     from (allowing a little slack for the small SE(μ̂)² term and the
    //     Gamma-mean curvature in the posterior-mean point).
    for i in 0..grid_n {
        let rel = (implied_sigma[i] - per_row_noise[i]).abs() / per_row_noise[i].max(1e-9);
        assert!(
            rel < 0.20,
            "row {i}: posterior-mean band implied σ={:.4} must match the per-row model \
             noise σ(x)={:.4} (rel err {:.3}); the scalar-dispersion band ignored σ(x)",
            implied_sigma[i],
            per_row_noise[i],
            rel
        );
    }

    // (3) Cross-API consistency: the full-uncertainty observation band on the
    //     SAME fit (which already used per-row noise) and the posterior-mean band
    //     must now broadly agree on the band WIDTH at each row.
    let unc_options = PredictUncertaintyOptions {
        confidence_level: 0.95,
        covariance_mode: InferenceCovarianceMode::Conditional,
        includeobservation_interval: true,
        ..PredictUncertaintyOptions::default()
    };
    let unc = predictor
        .predict_full_uncertainty(&input, &fit.fit, &unc_options)
        .expect("full-uncertainty predict with observation interval");
    let unc_lo = unc
        .observation_lower
        .as_ref()
        .expect("full-uncertainty observation lower present");
    let unc_hi = unc
        .observation_upper
        .as_ref()
        .expect("full-uncertainty observation upper present");
    for i in 0..grid_n {
        let pm_width = obs_upper[i] - obs_lower[i];
        let unc_width = unc_hi[i] - unc_lo[i];
        let rel = (pm_width - unc_width).abs() / unc_width.max(1e-9);
        assert!(
            rel < 0.25,
            "row {i}: posterior-mean band width {pm_width:.4} must broadly agree with the \
             full-uncertainty band width {unc_width:.4} (rel {rel:.3}); a large gap means the \
             two prediction-interval APIs disagree on the same fit (the #1119-sibling bug)"
        );
    }
}

/// Shared #1119 predict-side gate for an *orthogonal* dispersion mean family
/// (NegativeBinomial / Tweedie): the Fisher-orthogonal members returned `None`
/// for the joint coefficient Hessian before #1119, so — exactly like the Gamma
/// case above — `compute_joint_geometry` / `compute_joint_covariance` produced
/// no covariance, no penalized Hessian and no EDF, and the posterior-mean
/// predict path hard-failed. With the block-diagonal joint Hessian now assembled
/// (and `likelihood_blocks_uncoupled() = true`) all three must populate the
/// joint covariance + EDF and predict on a held-out grid.
fn assert_orthogonal_dispersion_family_predictable(
    family: &str,
    expected_kind: DispersionFamilyKind,
    x: &[f64],
    y: &[f64],
    mu_truth: impl Fn(f64) -> f64,
) {
    let n = x.len();
    assert_eq!(y.len(), n, "x/y length mismatch in #1119 orthogonal gate");
    let headers = vec!["x".to_string(), "y".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode dispersion data");
    let col = ds.column_map();
    let x_idx = col["x"];
    let ncols = ds.headers.len();

    let cfg = FitConfig {
        family: Some(family.to_string()),
        noise_formula: Some("s(x, k=8)".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("y ~ s(x, k=8)", &ds, &cfg).expect("gam dispersion location-scale fit");
    let FitResult::DispersionLocationScale(DispersionLocationScaleFitResult { fit, kind }) = result
    else {
        panic!("expected a DispersionLocationScale fit result for family={family}");
    };
    assert_eq!(
        kind, expected_kind,
        "{family} + noise_formula must route to {expected_kind:?}"
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
        .expect("scale (log-precision) block present")
        .beta
        .clone();
    let p_mu = beta_mu.len();
    let p_disp = beta_noise.len();
    let p_total = p_mu + p_disp;

    // ── joint covariance present + correctly sized (was None before #1119) ──
    let covariance = fit
        .fit
        .beta_covariance()
        .unwrap_or_else(|| {
            panic!(
                "{family} dispersion location-scale fit must carry a joint posterior covariance \
                 (orthogonal-family joint Hessian returned None before #1119)"
            )
        })
        .clone();
    assert_eq!(
        covariance.dim(),
        (p_total, p_total),
        "{family} joint covariance must span both blocks"
    );
    assert!(
        covariance.iter().all(|v| v.is_finite()),
        "{family} joint covariance must be finite"
    );
    for i in 0..p_total {
        assert!(
            covariance[[i, i]] > 0.0,
            "{family} covariance diagonal entry {i} must be a positive posterior variance"
        );
    }

    // ── finite total EDF (was None before #1119) ───────────────────────────
    let edf_total = fit
        .fit
        .edf_total()
        .unwrap_or_else(|| panic!("{family} dispersion fit must carry a finite total EDF"));
    assert!(
        edf_total.is_finite() && edf_total > 0.0 && edf_total <= p_total as f64 + 1e-6,
        "{family} total EDF must be finite in (0, p_total={p_total}], got {edf_total}"
    );

    // ── posterior-mean predict succeeds with finite bounds ──────────────────
    let predictor = DispersionLocationScalePredictor {
        beta_mu: beta_mu.clone(),
        beta_noise: beta_noise.clone(),
        likelihood: kind.likelihood_spec(),
        inverse_link: Some(kind.base_link()),
        covariance: Some(covariance),
    };
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

    let input = PredictInput {
        design: mean_design.design,
        offset: Array1::<f64>::zeros(grid_n),
        design_noise: Some(disp_design.design),
        offset_noise: Some(Array1::<f64>::zeros(grid_n)),
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };
    let pm_options = PosteriorMeanOptions {
        confidence_level: Some(0.95),
        covariance_mode: InferenceCovarianceMode::Conditional,
        include_observation_interval: true,
    };
    let pred = predictor
        .predict_posterior_mean(&input, &fit.fit, &pm_options)
        .unwrap_or_else(|e| {
            panic!("{family} posterior-mean predict must succeed (the #1119 abort): {e}")
        });

    assert_eq!(pred.mean.len(), grid_n, "one predicted mean per grid row");
    assert!(
        pred.mean.iter().all(|m| m.is_finite() && *m > 0.0),
        "{family} predicted means must be finite positive responses"
    );
    assert!(
        pred.eta_standard_error
            .iter()
            .all(|s| s.is_finite() && *s >= 0.0),
        "{family} predict η standard errors must be finite and non-negative"
    );
    pred.mean_lower
        .as_ref()
        .unwrap_or_else(|| panic!("{family} posterior-mean confidence lower bound must exist"));
    pred.mean_upper
        .as_ref()
        .unwrap_or_else(|| panic!("{family} posterior-mean confidence upper bound must exist"));

    // ── truth recovery on the held-out grid ─────────────────────────────────
    let truth_mean: Vec<f64> = grid_x.iter().map(|&xi| mu_truth(xi)).collect();
    let pred_mean: Vec<f64> = pred.mean.to_vec();
    let corr = pearson(&pred_mean, &truth_mean);
    eprintln!(
        "[#1119 {family} dispersion predict] n={n} grid={grid_n} edf_total={edf_total:.3} \
         corr(pred,truth)={corr:.4}"
    );
    assert!(
        corr > 0.85,
        "{family} posterior-mean predictions must track the true mean trend: pearson={corr:.4}"
    );
}

#[test]
fn negbin_dispersion_location_scale_assembles_covariance_and_is_predictable() {
    init_parallelism();
    // Heteroscedastic-dispersion NB2: mean exp(0.5 + 0.6 x), varying size θ(x).
    let n = 600usize;
    let mut rng = Lcg(1119);
    let x: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    // NB2 as a Gamma–Poisson mixture: λ ~ Gamma(shape=θ, scale=μ/θ), y ~ Poisson(λ).
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = mu_true(xi);
            let theta = shape_true(xi); // reuse a strictly positive varying precision
            let lambda = rng.gamma(theta, mu / theta).max(1e-9);
            // Poisson(lambda) via Knuth's product method off the same LCG.
            let l = (-lambda).exp();
            let mut k = 0u32;
            let mut p = 1.0_f64;
            loop {
                p *= rng.unit();
                if p <= l {
                    break;
                }
                k += 1;
                if k > 100_000 {
                    break;
                }
            }
            k as f64
        })
        .collect();
    assert_orthogonal_dispersion_family_predictable(
        "negbin",
        DispersionFamilyKind::NegativeBinomial,
        &x,
        &y,
        mu_true,
    );
}

#[test]
fn tweedie_dispersion_location_scale_assembles_covariance_and_is_predictable() {
    init_parallelism();
    // Heteroscedastic-dispersion Tweedie (compound Poisson–Gamma, 1<p<2):
    // mean exp(0.5 + 0.6 x); positive-mass + a genuine point mass at zero.
    let n = 600usize;
    let p = 1.5_f64;
    let mut rng = Lcg(1119);
    let x: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0))
        .collect();
    // Compound Poisson–Gamma with dispersion φ: N ~ Poisson(μ^{2−p}/(φ(2−p))),
    // jump sizes Gamma(α=(2−p)/(p−1), scale=φ(p−1)μ^{p−1}), y = Σ jumps (0 if N=0).
    let phi = 0.6_f64;
    let two_minus_p = 2.0 - p;
    let p_minus_1 = p - 1.0;
    let alpha = two_minus_p / p_minus_1;
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            let mu = mu_true(xi);
            let lambda = mu.powf(two_minus_p) / (phi * two_minus_p);
            // Poisson(lambda) via Knuth.
            let l = (-lambda).exp();
            let mut count = 0u32;
            let mut pp = 1.0_f64;
            loop {
                pp *= rng.unit();
                if pp <= l {
                    break;
                }
                count += 1;
                if count > 100_000 {
                    break;
                }
            }
            if count == 0 {
                0.0
            } else {
                let scale = phi * p_minus_1 * mu.powf(p_minus_1);
                let mut total = 0.0;
                for _ in 0..count {
                    total += rng.gamma(alpha, scale);
                }
                total.max(0.0)
            }
        })
        .collect();
    assert_orthogonal_dispersion_family_predictable(
        "tweedie",
        DispersionFamilyKind::Tweedie { p },
        &x,
        &y,
        mu_true,
    );
}
