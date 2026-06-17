//! Acceptance simulations for the Neyman-orthogonal, cross-fitted
//! marginal-slope calibration (#461 — design contract in
//! `src/families/marginal_slope_orthogonal_design.md`, §7).
//!
//! # The leakage these tests pin down
//!
//! The pipeline is a two-stage estimator. Stage 1 (CTN, conditional
//! transformation-normal) gaussianizes a continuous score conditional on
//! covariates `x` into a latent `z = Φ⁻¹(u)`. Stage 2 (Bernoulli marginal-
//! slope) fits a probit model `η = α(x) + s_f·β(x)·z` for a binary outcome,
//! where `β(x)` (the *logslope surface*) is the scientific target.
//!
//! Because `z` is a **generated regressor** that depends on the fitted Stage-1
//! parameters `θ̂₁`, an x-structured Stage-1 *miscalibration* (e.g. the
//! conditional scale `s(x)` wrong in one region of `x`) projects onto `β` and
//! manufactures spurious spatial heterogeneity in `β̂(x)` — even when the true
//! `β(x)` is flat. The orthogonalized estimator absorbs the realized Stage-1
//! influence directions `Z_infl = diag(s_f·β̂₀(x))·J`, `J = ∂z/∂θ₁`, computed
//! out-of-fold (cross-fitting), making the `β` estimating equation orthogonal
//! to `span(Z_infl)` — the discrete realization of `ψ − Π_η[ψ]`.
//!
//! # How the two arms are constructed (read this before the assertions)
//!
//! Both arms are driven through the **same shipped public entry points**, so
//! these tests verify exactly what the CLI / gamfit run — not a parallel
//! test-only path:
//!
//!   * **orthogonalized arm** — THE shipped path: a [`CtnStage1Recipe`] built
//!     from the Stage-1 CTN description (response column, covariate-formula RHS,
//!     CTN config), set on `FitConfig::ctn_stage1`, then plain
//!     [`fit_from_formula`]. There is no separate combined entry function:
//!     supplying the recipe *is* the request for orthogonalization. The
//!     materializer fits Stage-1 per fold from the recipe, **cross-fits to
//!     produce the out-of-fold `z`** (so NO `z_column` and no dose column in the
//!     data), and installs the A2 leakage-projection block (the marginal design
//!     is widened to `[M | Z̃_infl]`). This is exactly the call the CLI / lib /
//!     Python funnel through (design §5).
//!   * **naive arm (control)** — [`fit_from_formula`] with `ctn_stage1: None`
//!     on a **precomputed** `z` column (a single full-data CTN score). No
//!     cross-fit, no influence block ⇒ today's leaky free-warp behavior. This
//!     is the un-orthogonalized baseline #461 is about.
//!
//! Both arms share the identical `(x, y)` and the same Stage-1 *recipe* /
//! response basis, so they differ only by the orthogonalization — isolating the
//! effect under test. β̂(x) and its posterior covariance are read off the
//! returned [`BernoulliMarginalSlopeFitResult`] (block 1 = logslope; the joint
//! covariance offset to the logslope block is the marginal block's *actual*
//! widened coefficient count `blocks[0].beta.len()`, which the A2 absorber
//! grows by `p₁` on the orthogonalized arm).
//!
//! # Why these may fail before the implementation is correct
//!
//! Per the repo contract, quality tests assert OBJECTIVE quality metrics
//! (false-positive control, truth-recovery RMSE, bias/coverage), never
//! `gam ≈ reference output`. The DML library is a match-or-beat baseline, not
//! ground truth. No tolerance here may be weakened to make code pass.

use gam::families::bms::BernoulliMarginalSlopeFitResult;
use gam::families::marginal_slope_orthogonal::{influence_block_design, score_influence_jacobian};
use gam::terms::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, dml_partial_linear_reference, rmse};
use gam::transformation_normal::TransformationNormalConfig;
use gam::{
    CtnStage1Recipe, FitConfig, FitRequest, FitResult, encode_recordswith_inferred_schema,
    fit_from_formula, init_parallelism, materialize,
};
use ndarray::{Array1, Array2};

// Stage-1 CTN covariate RHS and Stage-2 marginal/logslope formulas. Kept in one
// place so the orthogonalized recipe and the naive control fit identical bases.
const COVARIATE_RHS: &str = "s(x, k=8)";
const STAGE2_FORMULA: &str = "y ~ s(x, k=8)";
const LOGSLOPE_FORMULA: &str = "s(x, k=8)";

// ---------------------------------------------------------------------------
// Deterministic RNG — SplitMix64, so every platform draws the identical data
// without pulling an RNG crate into the integration test.
// ---------------------------------------------------------------------------
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1).
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller (one of the pair, regenerated per call so
    /// downstream consumption order does not perturb the stream).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + statrs::function::erf::erf(x / std::f64::consts::SQRT_2))
}

// ---------------------------------------------------------------------------
// Dataset plumbing + the two shipped fit arms.
// ---------------------------------------------------------------------------

/// Build an in-memory `EncodedDataset` with columns `x`, `y`, `score` (and,
/// when `z` is `Some`, a `z` column for the naive control). The orthogonalized
/// arm needs `x`, `y`, `score` and must NOT carry a `z` column — the calibrated
/// entry synthesizes its own reserved score column and refuses if a clashing
/// one is present.
fn build_dataset(
    x: &[f64],
    y: &[f64],
    score: &[f64],
    z: Option<&[f64]>,
) -> gam::data::EncodedDataset {
    let n = x.len();
    assert_eq!(y.len(), n, "x/y length mismatch");
    assert_eq!(score.len(), n, "x/score length mismatch");
    let mut headers = vec!["x".to_string(), "y".to_string(), "score".to_string()];
    if z.is_some() {
        headers.push("z".to_string());
    }
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            let mut cols = vec![
                format!("{:.17e}", x[i]),
                format!("{:.17e}", y[i]),
                format!("{:.17e}", score[i]),
            ];
            if let Some(zc) = z {
                cols.push(format!("{:.17e}", zc[i]));
            }
            csv::StringRecord::from(cols)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode dataset")
}

/// Stage-1 CTN config for the orthogonalized arm: `TransformationNormalConfig::
/// default()` with a pinned response-basis size so `p_resp` (and hence `p₁`) is
/// fold-invariant across the cross-fit (the cross-fitter freezes the covariate
/// basis itself; this pins the response side).
fn stage1_config() -> TransformationNormalConfig {
    let mut config = TransformationNormalConfig::default();
    config.response_num_internal_knots = 6;
    config.response_degree = 3;
    config
}

/// Stage-2 marginal-slope `FitConfig` (Bernoulli probit base; logslope surface
/// `s(x)`). Both arms start from this base: the orthogonalized arm then sets
/// `ctn_stage1 = Some(recipe)` (and NO `z_column`), the naive arm sets
/// `z_column` and leaves `ctn_stage1` `None`.
fn stage2_config() -> FitConfig {
    FitConfig {
        family: Some("bernoulli-marginal-slope".to_string()),
        logslope_formula: Some(LOGSLOPE_FORMULA.to_string()),
        ..FitConfig::default()
    }
}

fn expect_bms(result: FitResult) -> BernoulliMarginalSlopeFitResult {
    match result {
        FitResult::BernoulliMarginalSlope(fit) => fit,
        _ => panic!("expected a FitResult::BernoulliMarginalSlope from the marginal-slope chain"),
    }
}

/// ORTHOGONALIZED arm — THE shipped path: a `CtnStage1Recipe` set on
/// `FitConfig::ctn_stage1`, then plain [`fit_from_formula`]. Supplying the
/// recipe *is* the request for orthogonalization (no flag, no combined entry
/// function): the materializer fits Stage-1 per fold from the recipe,
/// cross-fits to produce the out-of-fold `z` (so NO `z_column` and no dose
/// column in the data), and installs the A2 absorber. This is exactly the call
/// the CLI / lib / Python funnel through.
fn fit_ortho(x: &[f64], y: &[f64], score: &[f64]) -> BernoulliMarginalSlopeFitResult {
    let data = build_dataset(x, y, score, None); // {x, y, score}, NO z dose column
    let recipe = CtnStage1Recipe::new(
        "score",       // Stage-1 CTN response column
        COVARIATE_RHS, // Stage-1 covariate RHS (no ~, no response)
        stage1_config(),
        None, // Stage-1 weight column
        None, // Stage-1 offset column
    )
    .expect("build Stage-1 CTN recipe");
    let mut config = stage2_config();
    config.ctn_stage1 = Some(recipe); // do NOT set z_column — z is cross-fit OOF
    let result = fit_from_formula(STAGE2_FORMULA, &data, &config)
        .expect("orthogonalized marginal-slope fit (ctn_stage1 recipe + fit_from_formula)");
    expect_bms(result)
}

/// NAIVE arm (control) — plain Stage-2 on a precomputed `z` column with no
/// Stage-1 recipe ⇒ free-warp, no orthogonalization. `z` is the single
/// full-data CTN score computed by [`full_data_ctn_z`].
fn fit_naive(x: &[f64], y: &[f64], score: &[f64], z: &[f64]) -> BernoulliMarginalSlopeFitResult {
    let data = build_dataset(x, y, score, Some(z));
    let mut config = stage2_config(); // ctn_stage1 stays None ⇒ free-warp control
    config.z_column = Some("z".to_string());
    let result =
        fit_from_formula(STAGE2_FORMULA, &data, &config).expect("naive marginal-slope fit");
    expect_bms(result)
}

/// The naive control's precomputed `z`: a single full-data CTN fit's in-sample
/// latent score. We fit the CTN through the public formula path and read `z`
/// from the §6 `score_influence_jacobian` API (computing `J` already runs the
/// finite-support PIT, so `jac.z` is the single source of truth for `z` — no
/// separate PIT reconstruction). This mirrors exactly what a user does when
/// they hand a raw CTN score to Stage-2 without the calibrated chain.
fn full_data_ctn_z(x: &[f64], score: &[f64]) -> Array1<f64> {
    let n = x.len();
    let headers = vec!["x".to_string(), "score".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", score[i])])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode CTN dataset");

    let cfg = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let mut materialized = materialize(&format!("score ~ {COVARIATE_RHS}"), &ds, &cfg)
        .expect("materialize CTN Stage-1");
    let FitRequest::TransformationNormal(ref mut req) = materialized.request else {
        panic!("expected a TransformationNormal fit request");
    };
    req.config.response_degree = 3;
    req.config.response_num_internal_knots = 6;
    let result = gam::fit_model(materialized.request).expect("fit full-data CTN Stage-1");
    let FitResult::TransformationNormal(tn) = result else {
        panic!("expected a TransformationNormal fit result");
    };

    // Covariate design rows at the training x via the FROZEN resolved spec, so
    // the geometry matches the fit exactly.
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = x[i];
    }
    let design = build_term_collection_design(data.view(), &tn.covariate_spec_resolved)
        .expect("build CTN covariate design from frozen spec");
    let cov_rows = design.design.to_dense();
    let score_arr = Array1::from_vec(score.to_vec());
    let zero_offset = Array1::<f64>::zeros(score_arr.len());
    let jac = score_influence_jacobian(&tn, &score_arr, cov_rows.view(), &zero_offset)
        .expect("full-data Stage-1 score-influence Jacobian (for z)");
    jac.z
}

// ---------------------------------------------------------------------------
// β̂(x) and pointwise-variance readouts off the Stage-2 fit.
// ---------------------------------------------------------------------------

/// Evaluate the fitted logslope surface `β̂(x)` at a fresh grid. Reconstructed
/// exactly as the family does (block 1 = logslope, see `bms/block_specs.rs`):
/// `β̂(x_i) = baseline_logslope + design(x_i)·β_logslope` (offset zero here).
/// The A2 absorber columns ride the *marginal* block and are dropped at predict,
/// so they never touch this logslope readout — valid for both arms.
fn beta_of_x(fit: &BernoulliMarginalSlopeFitResult, x_grid: &[f64]) -> Vec<f64> {
    let n = x_grid.len();
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = x_grid[i];
    }
    let design = build_term_collection_design(data.view(), &fit.logslopespec_resolved)
        .expect("rebuild logslope design from resolved spec");
    let dense = design.design.to_dense();
    let beta_logslope = &fit.fit.blocks[1].beta;
    assert_eq!(
        dense.ncols(),
        beta_logslope.len(),
        "logslope design width {} != logslope beta length {}",
        dense.ncols(),
        beta_logslope.len()
    );
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = fit.baseline_logslope;
        for j in 0..dense.ncols() {
            acc += dense[[i, j]] * beta_logslope[j];
        }
        out[i] = acc;
    }
    out
}

/// Spatial-heterogeneity statistic of a fitted `β̂(x)`: the **spatial variance
/// ratio** `Var_x(β̂(x)) / σ̂_β²`, the across-grid sample variance of the fitted
/// logslope surface scaled by an estimate of its own sampling noise.
///
/// Interpretation: under a truly flat `β(x) ≡ const`, `β̂(x)` should vary across
/// `x` only by sampling noise, so this ratio sits near its null band; an
/// x-structured Stage-1 leakage inflates `Var_x(β̂(x))` well above that band —
/// the false positive #461 is about. `σ̂_β²` is the median across the grid of
/// the per-point posterior variance of `β̂(x)`, propagated from the joint
/// covariance through the logslope design, so the statistic is dimensionless
/// and self-calibrating.
fn spatial_heterogeneity_ratio(fit: &BernoulliMarginalSlopeFitResult, x_grid: &[f64]) -> f64 {
    let beta = beta_of_x(fit, x_grid);
    let m = beta.len() as f64;
    let mean = beta.iter().sum::<f64>() / m;
    let var_x = beta.iter().map(|b| (b - mean) * (b - mean)).sum::<f64>() / (m - 1.0).max(1.0);

    let sigma2 = median_pointwise_beta_variance(fit, x_grid);
    var_x / sigma2.max(1e-12)
}

/// Median over the grid of the pointwise posterior variance of `β̂(x)`:
/// `r(x)ᵀ Σ_ℓℓ r(x)`, where `r(x)` is the logslope design row at `x` and `Σ_ℓℓ`
/// is the logslope-block sub-covariance of the joint posterior.
fn median_pointwise_beta_variance(fit: &BernoulliMarginalSlopeFitResult, x_grid: &[f64]) -> f64 {
    let n = x_grid.len();
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = x_grid[i];
    }
    let design = build_term_collection_design(data.view(), &fit.logslopespec_resolved)
        .expect("rebuild logslope design for variance readout");
    let dense = design.design.to_dense();
    let p_logslope = dense.ncols();

    // Locate the logslope block's coefficient slice in the joint covariance:
    // block 0 = marginal, block 1 = logslope (bms/block_specs.rs). The joint
    // covariance is over the concatenated per-block β in block order, so the
    // logslope slice starts at the marginal block's ACTUAL coefficient count —
    // `blocks[0].beta.len()`, NOT `marginal_design.design.ncols()`. Under the
    // #461 orthogonalized arm the marginal block is widened to `[M | Z̃_infl]`
    // (its β grows by p₁), while `marginal_design` still reports the raw `M`
    // width; using the widened count keeps the offset correct for both arms.
    let p_marginal = fit.fit.blocks[0].beta.len();
    // Vb (conditional posterior covariance Var(β | λ) = H⁻¹·φ̂) is the natural
    // pointwise sampling-noise scale here; `compute_covariance: true` (set by
    // the materializer) populates it on the Stage-2 fit.
    let cov = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("Stage-2 fit must carry a conditional covariance");
    assert!(
        cov.nrows() >= p_marginal + p_logslope,
        "joint covariance {}x{} too small for marginal({p_marginal})+logslope({p_logslope})",
        cov.nrows(),
        cov.ncols()
    );

    let mut variances: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = 0.0;
        for a in 0..p_logslope {
            let ra = dense[[i, a]];
            if ra == 0.0 {
                continue;
            }
            for b in 0..p_logslope {
                v += ra * cov[[p_marginal + a, p_marginal + b]] * dense[[i, b]];
            }
        }
        variances.push(v.max(0.0));
    }
    variances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    variances[variances.len() / 2]
}

// ===========================================================================
// Sim A — false-heterogeneity CONTROL.
// ===========================================================================
//
// TRUE β(x) ≡ constant, but Stage-1 is given an x-DEPENDENT miscalibration: the
// continuous score's conditional SCALE is inflated for x in the right half of
// the domain. The naive arm projects that structured Stage-1 error onto β and
// reports spurious spatial heterogeneity; the orthogonalized arm absorbs it.
//
// ASSERTION: the spatial-heterogeneity ratio of the NAIVE β̂(x) exceeds its null
// band (false positive), while the ORTHOGONALIZED ratio sits inside it.
// ===========================================================================
#[test]
#[ignore = "MSI-only #979/#461 simulation: runs CTN crossfit plus paired BMS fits"]
fn sim_a_false_heterogeneity_is_controlled_by_orthogonalization() {
    init_parallelism();
    const N: usize = 3000;
    let mut rng = SplitMix64::new(0x5151_A1A1_2026_0461);

    // Constant true slope and a smooth true mean surface in x. The TRUE latent
    // driver `w ~ N(0,1)` is what the outcome depends on (constant slope); the
    // OBSERVED continuous score that Stage 1 ingests is `w` passed through an
    // x-dependent affine MIScalibration: conditional scale `s(x)` is 1 on the
    // left half of the domain and 1.9 on the right. Stage 1, fit on the mis-
    // scaled score, gaussianizes to a z whose error has x-structure — the
    // leakage #461 is about.
    let beta_true = 0.6_f64;
    let mut x = vec![0.0; N];
    let mut w = vec![0.0; N];
    let mut score = vec![0.0; N];
    for i in 0..N {
        let xi = rng.next_unit(); // x ~ U(0,1)
        let wi = rng.next_normal();
        let s_x = if xi > 0.5 { 1.9 } else { 1.0 };
        x[i] = xi;
        w[i] = wi;
        score[i] = 3.0 + 2.0 * xi + s_x * wi; // location smooth in x; scale mis-set
    }
    // Outcome: a single principled Bernoulli draw per row from the generative
    // law `η = α(x) + β_true·w`, α(x) a smooth surface in x.
    let mut rng_y = SplitMix64::new(0x9E13_55AA_7C0F_0461);
    let mut y = vec![0.0; N];
    for i in 0..N {
        let alpha_x = -0.3 + 0.8 * (std::f64::consts::PI * x[i]).sin();
        let eta = alpha_x + beta_true * w[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }

    let z_naive = full_data_ctn_z(&x, &score);
    let naive = fit_naive(&x, &y, &score, z_naive.as_slice().expect("contiguous z"));
    let ortho = fit_ortho(&x, &y, &score);

    // Evaluation grid spanning the domain.
    let grid: Vec<f64> = (0..41).map(|k| k as f64 / 40.0).collect();
    let ratio_naive = spatial_heterogeneity_ratio(&naive, &grid);
    let ratio_ortho = spatial_heterogeneity_ratio(&ortho, &grid);

    // Null band: under a flat truth with no leakage, Var_x(β̂)/σ̂_β² is O(1).
    // The naive arm must blow past it (the manufactured heterogeneity), the
    // orthogonalized arm must stay inside it. These are objective false-positive
    // bounds, not gam-vs-reference matches; do not loosen them to pass.
    const NULL_BAND: f64 = 6.0;
    const NAIVE_FALSE_POSITIVE_FLOOR: f64 = 12.0;

    assert!(
        ratio_naive > NAIVE_FALSE_POSITIVE_FLOOR,
        "Sim A precondition: naive arm should manufacture spatial heterogeneity \
         under x-dependent Stage-1 miscalibration, but its ratio {ratio_naive:.3} \
         did not exceed the false-positive floor {NAIVE_FALSE_POSITIVE_FLOOR}. \
         (If this fails, the injected miscalibration is too weak to exercise #461.)"
    );
    assert!(
        ratio_ortho < NULL_BAND,
        "Sim A: orthogonalized arm must keep the spatial-heterogeneity ratio inside \
         the null band {NULL_BAND}, but it was {ratio_ortho:.3}. The influence \
         projection failed to absorb the x-structured Stage-1 leakage (#461)."
    );
    assert!(
        ratio_ortho < 0.5 * ratio_naive,
        "Sim A: orthogonalization must at least halve the spurious heterogeneity \
         (naive {ratio_naive:.3} → ortho {ratio_ortho:.3})."
    );
}

// ===========================================================================
// Sim B — POWER preservation.
// ===========================================================================
//
// TRUE β(x) genuinely VARIES with x and Stage-1 is WELL-CALIBRATED (no x-
// dependent miscalibration). The orthogonalized projection must not eat the
// real signal: its RMSE to the true β(x) must be within a tight tolerance of
// the naive arm's RMSE.
// ===========================================================================
#[test]
#[ignore = "MSI-only #979/#461 simulation: runs CTN crossfit plus paired BMS fits"]
fn sim_b_orthogonalization_preserves_real_heterogeneity_signal() {
    init_parallelism();
    const N: usize = 3000;
    let mut rng = SplitMix64::new(0xB0B0_2026_0461_0001);

    // True slope varies smoothly with x; Stage-1 score is correctly scaled.
    let beta_fn = |xi: f64| 0.2 + 0.9 * xi; // monotone increasing slope in x
    let mut x = vec![0.0; N];
    let mut score = vec![0.0; N];
    let mut w = vec![0.0; N];
    for i in 0..N {
        let xi = rng.next_unit();
        let wi = rng.next_normal();
        // Well-calibrated Stage-1: constant unit conditional scale.
        x[i] = xi;
        w[i] = wi;
        score[i] = 3.0 + 2.0 * xi + wi;
    }
    let mut rng_y = SplitMix64::new(0xB0B0_2026_0461_0002);
    let mut y = vec![0.0; N];
    for i in 0..N {
        let alpha_x = -0.2 + 0.7 * (std::f64::consts::PI * x[i]).sin();
        let eta = alpha_x + beta_fn(x[i]) * w[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }

    let z_naive = full_data_ctn_z(&x, &score);
    let naive = fit_naive(&x, &y, &score, z_naive.as_slice().expect("contiguous z"));
    let ortho = fit_ortho(&x, &y, &score);

    let grid: Vec<f64> = (0..41).map(|k| k as f64 / 40.0).collect();
    let truth: Vec<f64> = grid.iter().map(|&g| beta_fn(g)).collect();

    // The latent z is on a standardized scale (CTN gaussianizes the score);
    // β̂(x) recovers the slope up to a constant rescaling between standardized z
    // and the unit-variance driver w. Align each arm to the truth by its own
    // best least-squares scale+shift so the RMSE measures SHAPE recovery on a
    // common footing — Sim B asserts ortho-vs-naive, not the absolute slope.
    let beta_naive = beta_of_x(&naive, &grid);
    let beta_ortho = beta_of_x(&ortho, &grid);
    let rmse_naive = rmse_after_affine_align(&beta_naive, &truth);
    let rmse_ortho = rmse_after_affine_align(&beta_ortho, &truth);

    // Power preservation: the projection removes only the Stage-1 error
    // geometry, which is ABSENT here (well-calibrated). The orthogonalized RMSE
    // must not exceed the naive RMSE by more than a tight margin.
    assert!(
        rmse_ortho <= 1.15 * rmse_naive + 0.02,
        "Sim B: orthogonalization ate real β(x) signal — ortho RMSE {rmse_ortho:.4} \
         exceeds naive RMSE {rmse_naive:.4} by more than the power-preservation \
         tolerance. The projection must not touch genuine heterogeneity."
    );
    // Both arms must actually recover the increasing trend (sanity that the
    // simulation has power at all): correlation of β̂ with truth is high.
    let corr = gam::test_support::reference::pearson(&beta_ortho, &truth);
    assert!(
        corr > 0.8,
        "Sim B: orthogonalized β̂(x) failed to track the true increasing slope \
         (Pearson {corr:.3} ≤ 0.8); the simulation has no real signal to preserve."
    );
}

// ===========================================================================
// Sim C — DML REFERENCE for the scalar target θ = E_x[β(x)].
// ===========================================================================
//
// True β(x) ≡ const (so θ = E_x[β(x)] = β_true is a clean scalar target). We
// build TWO Stage-1 score representations on the same outcome/covariate data:
//   * a CALIBRATED score — correctly scaled (oracle), and
//   * a MISCALIBRATED score — x-dependently mis-scaled (the Sim-A leakage).
// A Neyman-orthogonal estimator's scalar target must be STABLE across these two
// (first-order insensitive to first-stage error); a non-orthogonal one shifts.
//
// To stay SCALE-CORRECT (gam's θ̂ is on the probit-index scale; DoubleML's PLR
// θ̂ is on the linear-probability scale — different estimands, never compared as
// point values), each estimator is judged on ITS OWN scale by its
// *miscalibration-induced relative shift* Δ = |θ̂_miscal − θ̂_calib| / |θ̂_calib|.
// The DML library supplies the reference orthogonal Δ_dml. Assertions:
//   * gam ORTHOGONALIZED Δ_ortho ≲ Δ_dml (match-or-beat the mature orthogonal
//     baseline's robustness to first-stage error), and
//   * gam NAIVE Δ_naive ≫ Δ_ortho (the naive target really is biased).
//
// Skips with a clear message when no DML backend is importable.
// ===========================================================================
#[test]
fn sim_c_scalar_target_matches_dml_reference_under_miscalibration() {
    init_parallelism();
    const N: usize = 3000;
    let mut rng = SplitMix64::new(0xC0C0_2026_0461_0003);

    // Constant true slope; build BOTH a calibrated and a miscalibrated score
    // from the SAME latent driver w and outcome y, so the only thing that
    // changes between the two Stage-1 representations is the x-dependent scale.
    let beta_true = 0.6_f64;
    let mut x = vec![0.0; N];
    let mut w = vec![0.0; N];
    let mut score_calib = vec![0.0; N];
    let mut score_miscal = vec![0.0; N];
    for i in 0..N {
        let xi = rng.next_unit();
        let wi = rng.next_normal();
        let s_x = if xi > 0.5 { 1.9 } else { 1.0 };
        x[i] = xi;
        w[i] = wi;
        score_calib[i] = 3.0 + 2.0 * xi + wi; // correctly scaled (oracle)
        score_miscal[i] = 3.0 + 2.0 * xi + s_x * wi; // x-dependent mis-scale
    }
    let mut rng_y = SplitMix64::new(0xC0C0_2026_0461_0004);
    let mut y = vec![0.0; N];
    for i in 0..N {
        let alpha_x = -0.3 + 0.8 * (std::f64::consts::PI * x[i]).sin();
        let eta = alpha_x + beta_true * w[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }

    let grid: Vec<f64> = (0..101).map(|k| k as f64 / 100.0).collect();

    // θ̂ = E_x[β̂(x)] for each (score representation, arm).
    let theta_naive_for = |score: &[f64]| -> f64 {
        let z = full_data_ctn_z(&x, score);
        let fit = fit_naive(&x, &y, score, z.as_slice().expect("contiguous z"));
        mean(&beta_of_x(&fit, &grid))
    };
    let theta_ortho_for = |score: &[f64]| -> f64 {
        let fit = fit_ortho(&x, &y, score);
        mean(&beta_of_x(&fit, &grid))
    };

    let theta_naive_calib = theta_naive_for(&score_calib);
    let theta_naive_miscal = theta_naive_for(&score_miscal);
    let theta_ortho_calib = theta_ortho_for(&score_calib);
    let theta_ortho_miscal = theta_ortho_for(&score_miscal);

    let rel_shift =
        |miscal: f64, calib: f64| -> f64 { (miscal - calib).abs() / calib.abs().max(1e-9) };
    let delta_naive = rel_shift(theta_naive_miscal, theta_naive_calib);
    let delta_ortho = rel_shift(theta_ortho_miscal, theta_ortho_calib);

    // DML reference: the full-data CTN score is the treatment D, x the
    // confounder. Run it on BOTH score representations to read off the mature
    // orthogonal estimator's own relative shift Δ_dml.
    let z_calib = full_data_ctn_z(&x, &score_calib);
    let z_miscal = full_data_ctn_z(&x, &score_miscal);
    let dml_calib = dml_partial_linear_reference(
        &y,
        z_calib.as_slice().expect("contiguous calibrated z"),
        &[Column::new("x", &x)],
        5,
    );
    let dml_miscal = dml_partial_linear_reference(
        &y,
        z_miscal.as_slice().expect("contiguous miscalibrated z"),
        &[Column::new("x", &x)],
        5,
    );

    if !dml_calib.available || !dml_miscal.available {
        eprintln!(
            "Sim C SKIPPED: no Python DML backend (DoubleML/EconML) importable; \
             install `doubleml` or `econml` to run the orthogonal reference. The \
             gam-side naive-vs-orthogonalized contrast is still covered by Sim A."
        );
        return;
    }

    let delta_dml = rel_shift(dml_miscal.theta, dml_calib.theta);

    // (1) gam's orthogonalized target must match-or-beat the mature DML
    // estimator's robustness to first-stage error (small slack for Monte-Carlo
    // noise across the two independent reference fits).
    assert!(
        delta_ortho <= delta_dml + 0.05,
        "Sim C ({backend}): orthogonalized θ̂ shifted by Δ_ortho={delta_ortho:.4} under \
         x-dependent Stage-1 miscalibration, worse than the DML reference's orthogonal \
         shift Δ_dml={delta_dml:.4} (calib {tc:.4} → miscal {tm:.4}). The orthogonal \
         estimator must be first-order insensitive to first-stage error.",
        backend = dml_miscal.backend,
        tc = theta_ortho_calib,
        tm = theta_ortho_miscal,
    );
    // (2) The naive target really is biased by the leakage: its relative shift
    // is much larger than the orthogonalized arm's.
    assert!(
        delta_naive > 2.0 * delta_ortho + 0.02,
        "Sim C: naive θ̂ should be visibly biased by the x-structured Stage-1 \
         leakage (Δ_naive={delta_naive:.4}) relative to the orthogonalized arm \
         (Δ_ortho={delta_ortho:.4}); if not, the simulation has no leakage to correct."
    );
}

// ---------------------------------------------------------------------------
// Small numeric helpers.
// ---------------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// RMSE of `a` to `b` after fitting the best least-squares affine map
/// `a ↦ s·a + c` — measures shape recovery independent of an overall
/// scale/offset (the standardized-z rescaling that is common to both arms).
fn rmse_after_affine_align(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "affine-align length mismatch");
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
    }
    let s = if saa > 1e-300 { sab / saa } else { 0.0 };
    let c = mb - s * ma;
    let aligned: Vec<f64> = a.iter().map(|&v| s * v + c).collect();
    rmse(&aligned, b)
}

// ===========================================================================
// §6 helper contract — `influence_block_design`.
// ===========================================================================
//
// The orthogonalized arm above absorbs the Stage-1 influence directions
// internally; this focused test pins the public `influence_block_design` helper
// (design §6) against its documented closed form (design §3): row `i`, column
// `k` of the absorbed block must equal `s_f · β̂₀(x_i) · J[i,k]`. Exact math, not
// a tool comparison — it validates the building block the A2 absorber uses.
#[test]
fn influence_block_design_is_diag_scaled_jacobian() {
    let n = 6;
    let p1 = 3;
    let jac_cols =
        Array2::from_shape_fn((n, p1), |(i, k)| (i as f64 + 1.0) * 0.1 - (k as f64) * 0.07);
    let pilot_beta0 = Array1::from_shape_fn(n, |i| 0.3 + 0.2 * (i as f64) - 0.05 * (i * i) as f64);
    let s_f = 1.7_f64;

    // `influence_block_design` consumes only `columns` (× s_f·β̂₀); the co-located
    // `z` is carried for the cross-fit fold loop and is irrelevant here, so a
    // length-matched placeholder is fine for this closed-form check.
    let jac = gam::families::marginal_slope_orthogonal::ScoreInfluenceJacobian {
        columns: jac_cols.clone(),
        z: Array1::zeros(n),
    };
    let block = influence_block_design(&jac, &pilot_beta0, s_f);

    assert_eq!(
        block.nrows(),
        n,
        "influence block must have one row per observation"
    );
    assert_eq!(
        block.ncols(),
        p1,
        "influence block must have one column per Stage-1 parameter"
    );
    for i in 0..n {
        for k in 0..p1 {
            let expected = s_f * pilot_beta0[i] * jac_cols[[i, k]];
            assert!(
                (block[[i, k]] - expected).abs() <= 1e-12 * (1.0 + expected.abs()),
                "Z_infl[{i},{k}] = {got} != s_f·β̂₀·J = {expected} (design §3 closed form)",
                got = block[[i, k]],
            );
        }
    }
}
