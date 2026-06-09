//! Acceptance simulations for the Neyman-orthogonal, cross-fitted marginal-slope
//! calibration on the **survival** outcome (#461 — design contract in
//! `src/families/marginal_slope_orthogonal_design.md`, §3/§7).
//!
//! # Why a separate survival sim (not just the BMS sims)
//!
//! The shared §3 math (the Stage-1 influence directions
//! `Z_infl = diag(s_f·β̂₀(x))·J`, `J = ∂z/∂θ₁`, residualized and absorbed) is the
//! same on both families, but the *host kernel* differs: BMS folds the absorber
//! into its single additive η-index (A2 widen-marginal), whereas the survival
//! marginal-slope family enters η₁ through the **time-quantile location**
//! `q·c(g)` (scaled by `c(g) = √(1 + (s_f·g)²)`), so a plain-additive
//! `+Z̃_infl·γ` CANNOT ride the marginal design — it would be multiplied by
//! `c(g)`. The survival family therefore hosts the absorber as a **dedicated
//! additive η₁ channel** at the de-nested observed index (un-`c(g)`-scaled). This
//! test exercises that survival-specific kernel path end-to-end, on a real
//! survival outcome, through the shipped public entry points.
//!
//! # The leakage these tests pin down
//!
//! Stage 1 (CTN) gaussianizes a continuous score conditional on `x` into a
//! latent `z`. Stage 2 (survival marginal-slope) fits a probit-on-time index
//! whose log-slope surface `β(x)` is the scientific target. Because `z` is a
//! generated regressor depending on θ̂₁, an x-structured Stage-1 miscalibration
//! projects onto `β` and manufactures spurious spatial heterogeneity in `β̂(x)`
//! even when the true `β(x)` is flat. The orthogonalized arm absorbs the realized
//! Stage-1 influence directions out-of-fold (cross-fitting), keeping the `β`
//! estimating equation orthogonal to `span(Z_infl)`.
//!
//! # The two arms (shipped public paths)
//!
//!   * **orthogonalized arm** — a [`CtnStage1Recipe`] set on
//!     `FitConfig::ctn_stage1`, then plain [`fit_from_formula`] with
//!     `survival_likelihood = "marginal-slope"` and NO `z_column`. The
//!     materializer fits Stage-1 per fold, cross-fits the out-of-fold `z` and
//!     `J`, and installs the survival absorber (dedicated η₁ block, dropped at
//!     predict).
//!   * **naive arm (control)** — [`fit_from_formula`] with `ctn_stage1: None` on a
//!     precomputed full-data `z` column ⇒ leaky free-warp baseline.
//!
//! Per the repo contract these assert OBJECTIVE quality (false-positive control,
//! truth-recovery), never `gam ≈ reference output`; no tolerance is weakened to
//! make code pass.

use gam::families::survival_marginal_slope::SurvivalMarginalSlopeFitResult;
use gam::terms::smooth::build_term_collection_design;
use gam::test_support::reference::rmse;
use gam::transformation_normal::TransformationNormalConfig;
use gam::{
    CtnStage1Recipe, FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula,
    init_parallelism,
};
use ndarray::Array2;

// Stage-1 CTN covariate RHS and Stage-2 marginal/logslope formulas, kept in one
// place so the orthogonalized recipe and the naive control fit identical bases.
const COVARIATE_RHS: &str = "s(x, k=8)";
const SURVIVAL_FORMULA: &str = "Surv(entry, exit, event) ~ s(x, k=8)";
const LOGSLOPE_FORMULA: &str = "s(x, k=8)";

// ---------------------------------------------------------------------------
// Deterministic RNG — SplitMix64, identical draws on every platform.
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

    /// Standard normal via Box-Muller (one of the pair).
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit().max(1.0e-300);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Survival dataset plumbing + the two shipped fit arms.
// ---------------------------------------------------------------------------

/// Build an in-memory `EncodedDataset` with columns `x`, `entry`, `exit`,
/// `event`, `score` (and, when `z` is `Some`, a `z` column for the naive
/// control). The orthogonalized arm must NOT carry a `z` column — the calibrated
/// entry synthesizes its own reserved score column and refuses a clashing one.
fn build_dataset(
    x: &[f64],
    entry: &[f64],
    exit: &[f64],
    event: &[f64],
    score: &[f64],
    z: Option<&[f64]>,
) -> gam::data::EncodedDataset {
    let n = x.len();
    assert_eq!(entry.len(), n);
    assert_eq!(exit.len(), n);
    assert_eq!(event.len(), n);
    assert_eq!(score.len(), n);
    let mut headers = vec![
        "x".to_string(),
        "entry".to_string(),
        "exit".to_string(),
        "event".to_string(),
        "score".to_string(),
    ];
    if z.is_some() {
        headers.push("z".to_string());
    }
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            let mut cols = vec![
                format!("{:.17e}", x[i]),
                format!("{:.17e}", entry[i]),
                format!("{:.17e}", exit[i]),
                format!("{:.17e}", event[i]),
                format!("{:.17e}", score[i]),
            ];
            if let Some(zc) = z {
                cols.push(format!("{:.17e}", zc[i]));
            }
            csv::StringRecord::from(cols)
        })
        .collect();
    encode_recordswith_inferred_schema(headers, records).expect("encode survival dataset")
}

/// Stage-1 CTN config: pinned response-basis size so `p_resp` (and hence `p₁`) is
/// fold-invariant across the cross-fit.
fn stage1_config() -> TransformationNormalConfig {
    let mut config = TransformationNormalConfig::default();
    config.response_num_internal_knots = 6;
    config.response_degree = 3;
    config
}

/// Stage-2 survival marginal-slope `FitConfig` base. Both arms start here: the
/// orthogonalized arm then sets `ctn_stage1 = Some(recipe)` (NO `z_column`), the
/// naive arm sets `z_column` and leaves `ctn_stage1` `None`.
fn stage2_config() -> FitConfig {
    FitConfig {
        survival_likelihood: "marginal-slope".to_string(),
        logslope_formula: Some(LOGSLOPE_FORMULA.to_string()),
        baseline_target: "linear".to_string(),
        ..FitConfig::default()
    }
}

fn expect_survival(result: FitResult) -> SurvivalMarginalSlopeFitResult {
    match result {
        FitResult::SurvivalMarginalSlope(fit) => fit,
        _ => panic!("expected FitResult::SurvivalMarginalSlope from the marginal-slope chain"),
    }
}

/// ORTHOGONALIZED arm — the shipped path: a `CtnStage1Recipe` set on
/// `FitConfig::ctn_stage1`, then plain [`fit_from_formula`]. Supplying the recipe
/// IS the request for orthogonalization (cross-fit OOF `z`, survival absorber
/// installed). No `z_column`.
fn fit_ortho(
    x: &[f64],
    entry: &[f64],
    exit: &[f64],
    event: &[f64],
    score: &[f64],
) -> SurvivalMarginalSlopeFitResult {
    let data = build_dataset(x, entry, exit, event, score, None);
    let recipe = CtnStage1Recipe::new("score", COVARIATE_RHS, stage1_config(), None, None)
        .expect("build Stage-1 CTN recipe");
    let mut config = stage2_config();
    config.ctn_stage1 = Some(recipe); // do NOT set z_column — z is cross-fit OOF
    let result = fit_from_formula(SURVIVAL_FORMULA, &data, &config)
        .expect("orthogonalized survival marginal-slope fit (ctn_stage1 + fit_from_formula)");
    expect_survival(result)
}

/// NAIVE arm (control) — plain Stage-2 on a precomputed `z` column with no
/// Stage-1 recipe ⇒ free-warp, no orthogonalization.
fn fit_naive(
    x: &[f64],
    entry: &[f64],
    exit: &[f64],
    event: &[f64],
    score: &[f64],
    z: &[f64],
) -> SurvivalMarginalSlopeFitResult {
    let data = build_dataset(x, entry, exit, event, score, Some(z));
    let mut config = stage2_config(); // ctn_stage1 stays None ⇒ free-warp control
    config.z_column = Some("z".to_string());
    let result = fit_from_formula(SURVIVAL_FORMULA, &data, &config)
        .expect("naive survival marginal-slope fit");
    expect_survival(result)
}

/// The naive control's precomputed `z`: a single full-data CTN fit's in-sample
/// latent score, obtained through the orthogonalized arm's OWN cross-fit-free
/// fallback would be circular, so we read it from the shipped CTN score helper:
/// fit the CTN through the public formula path and emit `z` from the §6
/// `score_influence_jacobian` API (which already runs the finite-support PIT, so
/// `jac.z` is the single source of truth — no separate PIT reconstruction).
fn full_data_ctn_z(x: &[f64], score: &[f64]) -> ndarray::Array1<f64> {
    use gam::families::marginal_slope_orthogonal::score_influence_jacobian;
    use gam::{FitRequest, materialize};
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
    let mut materialized =
        materialize(&format!("score ~ {COVARIATE_RHS}"), &ds, &cfg).expect("materialize CTN");
    let FitRequest::TransformationNormal(ref mut req) = materialized.request else {
        panic!("expected a TransformationNormal fit request");
    };
    req.config.response_degree = 3;
    req.config.response_num_internal_knots = 6;
    let result = gam::fit_model(materialized.request).expect("fit full-data CTN Stage-1");
    let FitResult::TransformationNormal(tn) = result else {
        panic!("expected a TransformationNormal fit result");
    };
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = x[i];
    }
    let design = build_term_collection_design(data.view(), &tn.covariate_spec_resolved)
        .expect("build CTN covariate design from frozen spec");
    let cov_rows = design.design.to_dense();
    let score_arr = ndarray::Array1::from_vec(score.to_vec());
    let zero_offset = ndarray::Array1::<f64>::zeros(score_arr.len());
    let jac = score_influence_jacobian(&tn, &score_arr, cov_rows.view(), &zero_offset)
        .expect("full-data Stage-1 score-influence Jacobian (for z)");
    jac.z
}

// ---------------------------------------------------------------------------
// β̂(x) readout off the survival fit.
// ---------------------------------------------------------------------------

/// Evaluate the fitted log-slope surface `β̂(x)` at a grid. Survival block layout
/// is `[time, marginal, logslope, …]`, so block 2 is the logslope surface:
/// `β̂(x_i) = baseline_slope + design(x_i)·β_logslope`. The survival absorber is a
/// dedicated trailing block dropped at predict, so it never touches this readout.
fn beta_of_x(fit: &SurvivalMarginalSlopeFitResult, x_grid: &[f64]) -> Vec<f64> {
    let n = x_grid.len();
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = x_grid[i];
    }
    let design = build_term_collection_design(data.view(), &fit.logslopespec_resolved)
        .expect("rebuild logslope design from resolved spec");
    let dense = design.design.to_dense();
    let beta_logslope = &fit.fit.blocks[2].beta;
    assert_eq!(
        dense.ncols(),
        beta_logslope.len(),
        "logslope design width {} != logslope beta length {}",
        dense.ncols(),
        beta_logslope.len()
    );
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut acc = fit.baseline_slope;
        for j in 0..dense.ncols() {
            acc += dense[[i, j]] * beta_logslope[j];
        }
        out[i] = acc;
    }
    out
}

/// Spatial-heterogeneity statistic of a fitted `β̂(x)`: the sample variance of the
/// log-slope surface across the grid, normalized by its mean magnitude so it is
/// dimensionless. Under a truly flat `β(x) ≡ const` it sits near zero; an
/// x-structured Stage-1 leakage inflates the across-grid variance. (Survival
/// marginal-slope fits with `compute_covariance: false`, so unlike the BMS sim we
/// normalize by the surface scale rather than a posterior-variance estimate.)
fn spatial_heterogeneity_ratio(fit: &SurvivalMarginalSlopeFitResult, x_grid: &[f64]) -> f64 {
    let beta = beta_of_x(fit, x_grid);
    let m = beta.len() as f64;
    let mean = beta.iter().sum::<f64>() / m;
    let var_x = beta.iter().map(|b| (b - mean) * (b - mean)).sum::<f64>() / (m - 1.0).max(1.0);
    // Self-calibrating scale: the squared mean log-slope, floored so a near-zero
    // mean cannot blow the ratio up. This keeps the statistic comparable across
    // arms that recover the same overall slope magnitude.
    let scale = (mean * mean).max(1e-3);
    var_x / scale
}

// ---------------------------------------------------------------------------
// Survival outcome generation: a Gaussian-AFT log-time driven by the latent w.
// ---------------------------------------------------------------------------

/// Generate a right-censored survival outcome whose log-event-time depends on the
/// latent driver `w` with the (possibly x-varying) slope `beta_fn(x)`:
///   log T = μ(x) + β(x)·w + ε,   ε ~ N(0, σ²),
/// with an independent administrative + random censoring time. Returns
/// `(entry, exit, event)` with `entry ≡ 0`. The OBSERVED Stage-1 score is the
/// caller's responsibility (it is the (mis)calibrated `w` representation).
fn simulate_survival<F: Fn(f64) -> f64>(
    x: &[f64],
    w: &[f64],
    beta_fn: F,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = x.len();
    let mut rng = SplitMix64::new(seed);
    let mut entry = vec![0.0; n];
    let mut exit = vec![0.0; n];
    let mut event = vec![0.0; n];
    for i in 0..n {
        // Smooth baseline log-time surface in x + latent-driven slope.
        let mu_x = 1.0 + 0.5 * (std::f64::consts::PI * x[i]).sin();
        let eps = 0.35 * rng.next_normal();
        let log_t = mu_x + beta_fn(x[i]) * w[i] + eps;
        let t_event = log_t.exp();
        // Independent censoring: an exponential admin time with moderate rate so
        // ~20–30% of rows are censored (a realistic, well-powered survival set).
        let c = -(rng.next_unit().max(1e-12).ln()) / 0.15;
        entry[i] = 0.0;
        if t_event <= c {
            exit[i] = t_event;
            event[i] = 1.0;
        } else {
            exit[i] = c;
            event[i] = 0.0;
        }
    }
    (entry, exit, event)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// RMSE of `a` to `b` after the best least-squares affine map `a ↦ s·a + c`
/// (shape recovery, independent of the standardized-z rescaling common to both
/// arms).
fn rmse_after_affine_align(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    for (p, q) in a.iter().zip(b) {
        sab += (p - ma) * (q - mb);
        saa += (p - ma) * (p - ma);
    }
    let s = if saa > 1e-300 { sab / saa } else { 0.0 };
    let c = mb - s * ma;
    let aligned: Vec<f64> = a.iter().map(|&v| s * v + c).collect();
    rmse(&aligned, b)
}

// ===========================================================================
// Sim A — false-heterogeneity CONTROL (survival).
// ===========================================================================
//
// TRUE β(x) ≡ constant, but Stage-1 is given an x-DEPENDENT miscalibration: the
// continuous score's conditional SCALE is inflated for x in the right half of
// the domain. The naive arm projects that structured Stage-1 error onto the
// survival log-slope surface and reports spurious spatial heterogeneity; the
// orthogonalized arm absorbs it through the dedicated η₁ channel.
//
// ASSERTION: the spatial-heterogeneity ratio of the NAIVE β̂(x) exceeds its null
// band, while the ORTHOGONALIZED ratio sits inside it.
// ===========================================================================
#[test]
fn sim_a_false_heterogeneity_is_controlled_on_survival() {
    init_parallelism();
    const N: usize = 3000;
    let mut rng = SplitMix64::new(0x5A12_B0A1_2026_0461);

    let beta_true = 0.6_f64;
    let mut x = vec![0.0; N];
    let mut w = vec![0.0; N];
    let mut score = vec![0.0; N];
    for i in 0..N {
        let xi = rng.next_unit(); // x ~ U(0,1)
        let wi = rng.next_normal();
        // x-dependent miscalibration: conditional scale 1 on the left half, 1.9
        // on the right — Stage 1 fit on the mis-scaled score gaussianizes to a z
        // whose error has x-structure (the #461 leakage).
        let s_x = if xi > 0.5 { 1.9 } else { 1.0 };
        x[i] = xi;
        w[i] = wi;
        score[i] = 3.0 + 2.0 * xi + s_x * wi;
    }
    // Survival outcome from the latent driver w with a CONSTANT true slope.
    let (entry, exit, event) = simulate_survival(&x, &w, |_x| beta_true, 0x9E13_55AA_7C0F_0461);

    let z_naive = full_data_ctn_z(&x, &score);
    let naive = fit_naive(
        &x,
        &entry,
        &exit,
        &event,
        &score,
        z_naive.as_slice().expect("contiguous z"),
    );
    let ortho = fit_ortho(&x, &entry, &exit, &event, &score);

    let grid: Vec<f64> = (0..41).map(|k| k as f64 / 40.0).collect();
    let ratio_naive = spatial_heterogeneity_ratio(&naive, &grid);
    let ratio_ortho = spatial_heterogeneity_ratio(&ortho, &grid);

    // Null band: under a flat truth with no leakage, Var_x(β̂)/mean² is small.
    // The naive arm must blow past it (manufactured heterogeneity), the
    // orthogonalized arm must stay inside it. Objective false-positive bounds.
    const NULL_BAND: f64 = 0.04;
    const NAIVE_FALSE_POSITIVE_FLOOR: f64 = 0.10;

    assert!(
        ratio_naive > NAIVE_FALSE_POSITIVE_FLOOR,
        "Sim A precondition: naive survival arm should manufacture spatial \
         heterogeneity under x-dependent Stage-1 miscalibration, but its ratio \
         {ratio_naive:.4} did not exceed the false-positive floor \
         {NAIVE_FALSE_POSITIVE_FLOOR}. (If this fails the injected miscalibration \
         is too weak to exercise #461 on the survival kernel.)"
    );
    assert!(
        ratio_ortho < NULL_BAND,
        "Sim A: orthogonalized survival arm must keep the spatial-heterogeneity \
         ratio inside the null band {NULL_BAND}, but it was {ratio_ortho:.4}. The \
         dedicated-η₁ influence projection failed to absorb the x-structured \
         Stage-1 leakage (#461)."
    );
    assert!(
        ratio_ortho < 0.5 * ratio_naive,
        "Sim A: orthogonalization must at least halve the spurious heterogeneity \
         (naive {ratio_naive:.4} → ortho {ratio_ortho:.4})."
    );
}

// ===========================================================================
// Sim B — POWER preservation (survival).
// ===========================================================================
//
// TRUE β(x) genuinely VARIES with x and Stage-1 is WELL-CALIBRATED. The
// orthogonalized projection must not eat the real signal: its RMSE to the true
// β(x) (after a common affine alignment) must be within a tight tolerance of the
// naive arm's RMSE, and it must track the true increasing trend.
// ===========================================================================
#[test]
fn sim_b_orthogonalization_preserves_real_heterogeneity_on_survival() {
    init_parallelism();
    const N: usize = 3000;
    let mut rng = SplitMix64::new(0xB0B1_2026_0461_0001);

    let beta_fn = |xi: f64| 0.2 + 0.9 * xi; // monotone increasing slope in x
    let mut x = vec![0.0; N];
    let mut w = vec![0.0; N];
    let mut score = vec![0.0; N];
    for i in 0..N {
        let xi = rng.next_unit();
        let wi = rng.next_normal();
        // Well-calibrated Stage-1: constant unit conditional scale.
        x[i] = xi;
        w[i] = wi;
        score[i] = 3.0 + 2.0 * xi + wi;
    }
    let (entry, exit, event) = simulate_survival(&x, &w, beta_fn, 0xB0B1_2026_0461_0002);

    let z_naive = full_data_ctn_z(&x, &score);
    let naive = fit_naive(
        &x,
        &entry,
        &exit,
        &event,
        &score,
        z_naive.as_slice().expect("contiguous z"),
    );
    let ortho = fit_ortho(&x, &entry, &exit, &event, &score);

    let grid: Vec<f64> = (0..41).map(|k| k as f64 / 40.0).collect();
    let truth: Vec<f64> = grid.iter().map(|&g| beta_fn(g)).collect();

    let beta_naive = beta_of_x(&naive, &grid);
    let beta_ortho = beta_of_x(&ortho, &grid);
    let rmse_naive = rmse_after_affine_align(&beta_naive, &truth);
    let rmse_ortho = rmse_after_affine_align(&beta_ortho, &truth);

    assert!(
        rmse_ortho <= 1.15 * rmse_naive + 0.02,
        "Sim B: orthogonalization ate real β(x) signal on survival — ortho RMSE \
         {rmse_ortho:.4} exceeds naive RMSE {rmse_naive:.4} by more than the \
         power-preservation tolerance. The projection must not touch genuine \
         heterogeneity (Stage-1 is well-calibrated here)."
    );
    let corr = gam::test_support::reference::pearson(&beta_ortho, &truth);
    assert!(
        corr > 0.8,
        "Sim B: orthogonalized β̂(x) failed to track the true increasing slope on \
         survival (Pearson {corr:.3} ≤ 0.8); the simulation has no real signal to \
         preserve."
    );
}

// ===========================================================================
// Sim C — scalar-target robustness to first-stage error (survival).
// ===========================================================================
//
// True β(x) ≡ const, so θ = E_x[β(x)] = β_true is a clean scalar target. We build
// a CALIBRATED and a MISCALIBRATED Stage-1 score on the SAME outcome/covariate
// data; a Neyman-orthogonal estimator's scalar target must be first-order
// insensitive to first-stage error (stable across the two), while the naive
// target shifts. We assert the gam orthogonalized arm's miscalibration-induced
// relative shift is much smaller than the naive arm's — the survival analog of
// the BMS Sim C robustness contrast (no external DML backend needed: the
// naive-vs-orthogonalized contrast IS the orthogonality signature).
// ===========================================================================
#[test]
fn sim_c_scalar_target_is_robust_to_first_stage_error_on_survival() {
    init_parallelism();
    const N: usize = 3000;
    let mut rng = SplitMix64::new(0xC0C1_2026_0461_0003);

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
    // ONE survival outcome (constant true slope) shared by both score
    // representations — only the Stage-1 score scale changes between them.
    let (entry, exit, event) = simulate_survival(&x, &w, |_x| beta_true, 0xC0C1_2026_0461_0004);

    let grid: Vec<f64> = (0..101).map(|k| k as f64 / 100.0).collect();

    let theta_naive_for = |score: &[f64]| -> f64 {
        let z = full_data_ctn_z(&x, score);
        let fit = fit_naive(
            &x,
            &entry,
            &exit,
            &event,
            score,
            z.as_slice().expect("contiguous z"),
        );
        mean(&beta_of_x(&fit, &grid))
    };
    let theta_ortho_for = |score: &[f64]| -> f64 {
        let fit = fit_ortho(&x, &entry, &exit, &event, score);
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

    assert!(
        delta_naive > 2.0 * delta_ortho + 0.02,
        "Sim C (survival): the naive scalar target θ̂ should be visibly biased by \
         the x-structured Stage-1 leakage (Δ_naive={delta_naive:.4}) relative to \
         the orthogonalized arm (Δ_ortho={delta_ortho:.4}); the orthogonalized \
         estimator must be first-order insensitive to first-stage error \
         (calib {theta_ortho_calib:.4} → miscal {theta_ortho_miscal:.4})."
    );
    // Absolute robustness floor for the orthogonalized arm: its relative shift
    // must be small in its own right, not merely smaller than the naive arm's.
    assert!(
        delta_ortho < 0.15,
        "Sim C (survival): orthogonalized θ̂ shifted by Δ_ortho={delta_ortho:.4} \
         under first-stage miscalibration — too large for a Neyman-orthogonal \
         target (should be ≪ the naive Δ_naive={delta_naive:.4})."
    );
}
