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
//! to `span(Z_infl)`. That is the discrete realization of `ψ − Π_η[ψ]`.
//!
//! # How the two arms are constructed (read this before the assertions)
//!
//! Both arms fit Stage 2 through the *same* public terms entry point,
//! [`fit_bernoulli_marginal_slope_terms`] — the function `fit_model` ultimately
//! dispatches to for a `z_column` + `logslope_formula` config. They differ only
//! in the `score_influence_jacobian` field of the spec:
//!
//!   * **naive arm** — `score_influence_jacobian: None`. The Stage-1 `z` is
//!     fed in as a fixed, precomputed regressor with no influence projection.
//!     This is exactly the behavior of a raw `z_column` fit (no CTN chain in
//!     the workflow), and reproduces #461's failure mode.
//!   * **orthogonalized arm** — `score_influence_jacobian: Some(J_oof)`, the
//!     out-of-fold Stage-1 score-influence Jacobian from cross-fitting the CTN.
//!     This is the auto-enabled, magic-by-default path the workflow selects
//!     whenever a CTN Stage-1 produces the z-column (design §5).
//!
//! `J_oof` and the out-of-fold `z` are produced by [`crossfit_score_z_jacobian`]
//! below, which fits the CTN on each fold's complement, evaluates `z` and
//! `J = ∂z/∂θ₁` on the held-out fold via the public
//! `marginal_slope_orthogonal::score_influence_jacobian` API (design §6), and
//! concatenates. The `z` used by the naive arm is the *same* cross-fitted `z`,
//! so the two arms see identical regressors and differ *only* by the influence
//! projection — isolating the effect under test.
//!
//! # Why these may fail before the implementation lands
//!
//! Per the repo contract, quality tests assert OBJECTIVE quality metrics
//! (false-positive control, truth-recovery RMSE, bias/coverage), never
//! `gam ≈ reference output`. The DML library is a match-or-beat baseline, not
//! ground truth. The `marginal_slope_orthogonal` module and the
//! `score_influence_jacobian` spec wiring are part of #461; until they exist
//! and are correct, the orthogonalized assertions legitimately fail. That is
//! honest, and no tolerance here may be weakened to make code pass.

use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, LatentZPolicy,
};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::families::marginal_slope_orthogonal::{influence_block_design, score_influence_jacobian};
use gam::resource::ResourcePolicy;
use gam::terms::basis::{
    BSplineBasisSpec, BSplineBoundaryConditions, BSplineKnotSpec, OneDimensionalBoundary,
};
use gam::terms::smooth::{
    build_term_collection_design, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec,
};
use gam::test_support::reference::{dml_partial_linear_reference, rmse, Column};
use gam::transformation_normal::TransformationNormalFitResult;
use gam::types::{InverseLink, StandardLink};
use gam::{
    encode_recordswith_inferred_schema, fit_model, init_parallelism, materialize, FitConfig,
    FitRequest, FitResult,
};
use ndarray::{Array1, Array2};

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
// Stage-1 (CTN) fitting + cross-fitted z / influence-Jacobian.
// ---------------------------------------------------------------------------

/// Fit a conditional transformation-normal model of a continuous score `score`
/// on a single covariate `x` over the supplied row subset, via the public
/// formula path (`materialize` → `fit_model`). The covariate side is a single
/// penalized B-spline `s(x, k=8)`; the response basis complexity is pinned so
/// the fit is reproducible. Returns the fitted CTN result.
fn fit_ctn_stage1(x: &[f64], score: &[f64]) -> TransformationNormalFitResult {
    let n = x.len();
    assert_eq!(score.len(), n, "x/score length mismatch");
    let headers = vec!["x".to_string(), "score".to_string()];
    let records: Vec<csv::StringRecord> = (0..n)
        .map(|i| {
            csv::StringRecord::from(vec![
                format!("{:.17e}", x[i]),
                format!("{:.17e}", score[i]),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, records).expect("encode CTN dataset");

    let cfg = FitConfig {
        transformation_normal: true,
        ..FitConfig::default()
    };
    let mut materialized =
        materialize("score ~ s(x, k=8)", &ds, &cfg).expect("materialize CTN Stage-1");
    let FitRequest::TransformationNormal(ref mut req) = materialized.request else {
        panic!("expected a TransformationNormal fit request");
    };
    req.config.response_degree = 3;
    req.config.response_num_internal_knots = 6;

    let result = fit_model(materialized.request).expect("fit CTN Stage-1");
    let FitResult::TransformationNormal(tn) = result else {
        panic!("expected a TransformationNormal fit result");
    };
    tn
}

/// Build the covariate-design rows for a fitted CTN at the supplied `x`, using
/// the FROZEN training-resolved covariate spec (so knots/centers match the fit
/// exactly — this is the genuine out-of-sample evaluation basis).
fn ctn_covariate_rows(tn: &TransformationNormalFitResult, x: &[f64]) -> Array2<f64> {
    let n = x.len();
    // Layout mirrors the [x, score] training columns; only column 0 (x) is read
    // by `s(x)`, column 1 keeps the matrix width consistent for the design.
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = x[i];
    }
    let design = build_term_collection_design(data.view(), &tn.covariate_spec_resolved)
        .expect("build CTN covariate design from frozen spec");
    design.design.to_dense()
}

/// Cross-fitted out-of-fold latent `z` and score-influence Jacobian
/// `J = ∂z/∂θ₁` for a CTN Stage-1 → marginal-slope Stage-2 chain (design §4).
///
/// `k_folds` folds; for fold `f` the CTN is fit on the complement and `z`/`J`
/// are evaluated on `f`, then scattered back into the original row order. The
/// per-fold `J` is produced by the public `score_influence_jacobian` API
/// (design §6); `z` is read from the same CTN via the documented finite-support
/// PIT map. Returns `(z_oof, jac_oof)` with `z_oof.len() == n` and
/// `jac_oof.nrows() == n`. (Per-fold `J` may differ in column count if the
/// frozen Stage-1 dimensionality differs across folds; the chain here uses a
/// fixed spec so `p₁` is constant.)
struct CrossFitScoreZJac {
    z_oof: Array1<f64>,
    jac_oof: Array2<f64>,
}

fn crossfit_score_z_jacobian(x: &[f64], score: &[f64], k_folds: usize) -> CrossFitScoreZJac {
    let n = x.len();
    assert!(k_folds >= 2, "cross-fitting needs at least 2 folds");
    let mut z_oof = Array1::<f64>::from_elem(n, f64::NAN);
    // Discover p₁ from the first fold to size the J accumulator; deterministic
    // because every fold uses the same frozen Stage-1 spec.
    let mut jac_oof: Option<Array2<f64>> = None;

    for f in 0..k_folds {
        let mut train_x = Vec::new();
        let mut train_score = Vec::new();
        let mut test_idx = Vec::new();
        for i in 0..n {
            if i % k_folds == f {
                test_idx.push(i);
            } else {
                train_x.push(x[i]);
                train_score.push(score[i]);
            }
        }
        let tn = fit_ctn_stage1(&train_x, &train_score);

        let test_x: Vec<f64> = test_idx.iter().map(|&i| x[i]).collect();
        let test_score: Vec<f64> = test_idx.iter().map(|&i| score[i]).collect();
        let cov_rows = ctn_covariate_rows(&tn, &test_x);

        // z on the held-out fold via the documented finite-support PIT map.
        let z_fold = ctn_latent_z(&tn, &cov_rows, &test_score);
        // J = ∂z/∂θ₁ on the held-out fold (design §6 public API).
        let test_score_arr = Array1::from_vec(test_score.clone());
        let jac = score_influence_jacobian(&tn, &test_score_arr, cov_rows.view())
            .expect("Stage-1 score-influence Jacobian");
        let p1 = jac.columns.ncols();
        let acc = jac_oof.get_or_insert_with(|| Array2::<f64>::zeros((n, p1)));
        assert_eq!(
            acc.ncols(),
            p1,
            "cross-fit folds disagree on Stage-1 parameter dimension p₁"
        );
        for (row, &i) in test_idx.iter().enumerate() {
            z_oof[i] = z_fold[row];
            for c in 0..p1 {
                acc[[i, c]] = jac.columns[[row, c]];
            }
        }
    }

    let jac_oof = jac_oof.expect("at least one fold produced a Jacobian");
    assert!(
        z_oof.iter().all(|v| v.is_finite()),
        "cross-fit left an out-of-fold z unfilled"
    );
    CrossFitScoreZJac { z_oof, jac_oof }
}

/// Reconstruct the latent score `z_i = Φ⁻¹(u_i)` for a fitted CTN at covariate-
/// design rows `cov_rows` and response values `y`, with
/// `u_i = (Φ(h_i) − Φ(L_i)) / (Φ(U_i) − Φ(L_i))` (finite-support PIT). This is
/// the documented Stage-1 → Stage-2 hand-off (design §1) and reproduces the
/// exact transform the predict path applies; it is the math ground truth for
/// `z`, not a tool comparison.
fn ctn_latent_z(tn: &TransformationNormalFitResult, cov_rows: &Array2<f64>, y: &[f64]) -> Vec<f64> {
    use gam::terms::basis::{create_basis, BasisOptions, Dense, KnotSource};

    let family = &tn.family;
    let resp_knots = family.response_knots().clone();
    let resp_transform = family.response_transform();
    let degree = family.response_degree();
    let median = family.response_median();
    let eps = gam::transformation_normal::TRANSFORMATION_MONOTONICITY_EPS;

    let n = y.len();
    let p_cov = cov_rows.ncols();
    assert_eq!(cov_rows.nrows(), n, "cov_rows / y length mismatch");

    let beta = &tn.fit.blocks[0].beta;
    let p_shape = resp_transform.ncols();
    let p_resp = 1 + p_shape;
    assert_eq!(
        beta.len(),
        p_resp * p_cov,
        "beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
        beta.len()
    );
    let gamma = beta
        .view()
        .into_shape_with_order((p_resp, p_cov))
        .expect("reshape CTN beta into (p_resp, p_cov)");

    let y_arr = Array1::from_vec(y.to_vec());
    let (raw_val_arc, _) = create_basis::<Dense>(
        y_arr.view(),
        KnotSource::Provided(resp_knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .expect("I-spline value basis at response points");
    let shape_val = raw_val_arc.as_ref().dot(resp_transform);

    let mut upper_shape = vec![0.0; p_shape];
    for c in 0..p_shape {
        upper_shape[c] = resp_transform.column(c).sum();
    }
    let lower_floor = eps * (resp_knots[0] - median);
    let upper_floor = eps * (resp_knots[resp_knots.len() - 1] - median);

    let mut z = vec![0.0; n];
    for i in 0..n {
        let cov_row = cov_rows.row(i);
        let gamma0 = gamma.row(0).dot(&cov_row);
        let mut val = gamma0;
        let mut up = gamma0;
        for r in 1..p_resp {
            let g = gamma.row(r).dot(&cov_row);
            let g2 = g * g;
            val += shape_val[[i, r - 1]] * g2;
            up += upper_shape[r - 1] * g2;
        }
        let h = val + eps * (y[i] - median);
        let lower = gamma0 + lower_floor;
        let upper = up + upper_floor;
        let h_in = h.clamp(lower, upper);
        let u = if upper <= lower {
            0.5
        } else {
            ((normal_cdf(h_in) - normal_cdf(lower)) / (normal_cdf(upper) - normal_cdf(lower)))
                .clamp(1e-9, 1.0 - 1e-9)
        };
        // Φ⁻¹(u) via the erfinv route (statrs supplies erf_inv).
        z[i] = std::f64::consts::SQRT_2 * statrs::function::erf::erf_inv(2.0 * u - 1.0);
    }
    z
}

// ---------------------------------------------------------------------------
// Stage-2 (Bernoulli marginal-slope) — both arms, and β̂(x) readout.
// ---------------------------------------------------------------------------

/// Build a `TermCollectionSpec` with a single penalized B-spline `s(x)`.
fn xspline_spec(num_internal_knots: usize, data_range: (f64, f64)) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![SmoothTermSpec {
            name: "f_x".to_string(),
            basis: SmoothBasisSpec::BSpline1D {
                feature_col: 0,
                spec: BSplineBasisSpec {
                    degree: 3,
                    penalty_order: 2,
                    knotspec: BSplineKnotSpec::Generate {
                        data_range,
                        num_internal_knots,
                    },
                    double_penalty: false,
                    identifiability: Default::default(),
                    boundary: OneDimensionalBoundary::Open,
                    boundary_conditions: BSplineBoundaryConditions::default(),
                },
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

/// Fit Stage 2. When `jacobian` is `Some(J)`, the out-of-fold Stage-1 score-
/// influence Jacobian is wired into the spec's `score_influence_jacobian` field
/// (the field takes the RAW `J`; the family forms the absorbed block
/// `Z_infl = diag(s_f·β̂₀)·J` internally at its own rigid pilot, design §3) —
/// the orthogonalized arm. When `None`, this is the naive arm. Both arms share
/// the identical `z`, `x`, `y`, and term specs, so they differ ONLY by the
/// projection — that isolation is the whole point of the control.
fn fit_stage2(
    x: &[f64],
    z: &Array1<f64>,
    y: &[f64],
    x_range: (f64, f64),
    jacobian: Option<Array2<f64>>,
) -> BernoulliMarginalSlopeFitResult {
    let n = x.len();
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = x[i];
    }
    let y_arr = Array1::from_vec(y.to_vec());

    let spec = BernoulliMarginalSlopeTermSpec {
        y: y_arr,
        weights: Array1::ones(n),
        z: z.clone(),
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec: xspline_spec(8, x_range),
        logslopespec: xspline_spec(8, x_range),
        marginal_offset: Array1::zeros(n),
        logslope_offset: Array1::zeros(n),
        frailty: FrailtySpec::None,
        score_warp: None,
        link_dev: None,
        latent_z_policy: LatentZPolicy::exploratory_fit_weighted(),
        score_influence_jacobian: jacobian,
    };

    gam::families::bernoulli_marginal_slope::fit_bernoulli_marginal_slope_terms(
        data.view(),
        spec,
        &BlockwiseFitOptions {
            compute_covariance: true,
            ..Default::default()
        },
        &SpatialLengthScaleOptimizationOptions::default(),
        &ResourcePolicy::default_library(),
    )
    .expect("fit Bernoulli marginal-slope Stage 2")
}

/// Evaluate the fitted logslope surface `β̂(x)` at a fresh grid of covariate
/// values. Reconstructed exactly as the family does (block 1 = logslope, see
/// `bms/block_specs.rs`): `β̂(x_i) = baseline_logslope + logslope_offset_i +
/// design(x_i)·β_logslope`, with the offset zero in these sims.
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
/// the false positive #461 is about. `σ̂_β²` is taken as the median across the
/// grid of the per-point posterior variance of `β̂(x)`, propagated from the
/// joint covariance through the logslope design (the same linear map used to
/// evaluate `β̂(x)`), so the statistic is dimensionless and self-calibrating.
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
/// is the logslope-block sub-covariance of the joint posterior. This is the
/// natural sampling-noise scale for the spatial-variance numerator.
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
    // block 0 = marginal, block 1 = logslope (bms/block_specs.rs).
    let p_marginal = fit.marginal_design.design.ncols();
    // Vb (conditional posterior covariance Var(β | λ) = H⁻¹·φ̂) is the natural
    // pointwise sampling-noise scale here; `compute_covariance: true` populates
    // it on the Stage-2 fit.
    let cov = fit
        .fit
        .covariance_conditional
        .as_ref()
        .expect("Stage-2 fit must carry a conditional covariance (compute_covariance: true)");
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
// the domain. The CTN, fit on the mis-scaled score, produces a z whose error
// has x-structure. The naive Stage-2 projects that structured error onto β and
// reports spurious spatial heterogeneity; the orthogonalized arm absorbs it.
//
// ASSERTION: the spatial-heterogeneity ratio of the NAIVE β̂(x) exceeds its null
// band (false positive), while the ORTHOGONALIZED ratio sits inside it.
// ===========================================================================
#[test]
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

    let x_range = (0.0, 1.0);
    let cf = crossfit_score_z_jacobian(&x, &score, 5);

    let naive = fit_stage2(&x, &cf.z_oof, &y, x_range, None);
    let ortho = fit_stage2(&x, &cf.z_oof, &y, x_range, Some(cf.jac_oof.clone()));

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
//
// ASSERTION: |RMSE_ortho − RMSE_naive| small AND RMSE_ortho ≤ (1 + ε)·RMSE_naive.
// ===========================================================================
#[test]
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
        let score_i = 3.0 + 2.0 * xi + wi;
        x[i] = xi;
        w[i] = wi;
        score[i] = score_i;
    }
    let mut rng_y = SplitMix64::new(0xB0B0_2026_0461_0002);
    let mut y = vec![0.0; N];
    for i in 0..N {
        let alpha_x = -0.2 + 0.7 * (std::f64::consts::PI * x[i]).sin();
        let eta = alpha_x + beta_fn(x[i]) * w[i];
        let p = normal_cdf(eta).clamp(1e-9, 1.0 - 1e-9);
        y[i] = if rng_y.next_unit() < p { 1.0 } else { 0.0 };
    }

    let x_range = (0.0, 1.0);
    let cf = crossfit_score_z_jacobian(&x, &score, 5);

    let naive = fit_stage2(&x, &cf.z_oof, &y, x_range, None);
    let ortho = fit_stage2(&x, &cf.z_oof, &y, x_range, Some(cf.jac_oof.clone()));

    let grid: Vec<f64> = (0..41).map(|k| k as f64 / 40.0).collect();
    let truth: Vec<f64> = grid.iter().map(|&g| beta_fn(g)).collect();

    // The latent z is on a standardized scale (CTN gaussianizes the score to
    // N(0,1)); the fitted β̂(x) recovers the slope up to the constant rescaling
    // between the standardized z and the unit-variance driver w. Both arms
    // share the same z, so the rescaling is identical and cancels in the
    // RMSE COMPARISON — which is what Sim B asserts (ortho vs naive), not the
    // absolute slope value.
    let beta_naive = beta_of_x(&naive, &grid);
    let beta_ortho = beta_of_x(&ortho, &grid);

    // Align each arm to the truth by its own best least-squares scale+shift, so
    // the RMSE measures SHAPE recovery on a common footing for both arms.
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
//   * a CALIBRATED z — Stage 1 fit on a correctly-scaled score (oracle), and
//   * a MISCALIBRATED OOF z — Stage 1 fit on the x-dependently mis-scaled score
//     (the Sim-A leakage).
// A Neyman-orthogonal estimator's scalar target must be STABLE across these two
// (first-order insensitive to first-stage error); a non-orthogonal one shifts.
//
// To stay SCALE-CORRECT (gam's θ̂ is on the probit-index scale; DoubleML's PLR
// θ̂ is on the linear-probability scale — they are different estimands and must
// never be compared as point values), each estimator is judged on ITS OWN
// scale by its *miscalibration-induced relative shift*
//     Δ = |θ̂_miscal − θ̂_calib| / |θ̂_calib|.
// The DML library supplies the reference orthogonal Δ_dml. The assertions:
//   * gam ORTHOGONALIZED Δ_ortho ≲ Δ_dml (match-or-beat the mature orthogonal
//     baseline's robustness to first-stage error), and
//   * gam NAIVE Δ_naive ≫ Δ_ortho (the naive target really is biased by the
//     x-structured leakage).
//
// Skips with a clear message when no DML backend is importable (heavier
// optional dependency); the gam-side naive-vs-ortho contrast still runs
// unconditionally in Sim A.
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

    let x_range = (0.0, 1.0);
    let grid: Vec<f64> = (0..101).map(|k| k as f64 / 100.0).collect();

    // Cross-fit each Stage-1 score representation ONCE, then read off both the
    // naive and orthogonalized scalar targets θ̂ = E_x[β̂(x)] from the shared z/J.
    let cf_calib = crossfit_score_z_jacobian(&x, &score_calib, 5);
    let cf_miscal = crossfit_score_z_jacobian(&x, &score_miscal, 5);

    let theta_pair = |cf: &CrossFitScoreZJac| -> (f64, f64) {
        let naive = fit_stage2(&x, &cf.z_oof, &y, x_range, None);
        let theta_naive = mean(&beta_of_x(&naive, &grid));
        let ortho = fit_stage2(&x, &cf.z_oof, &y, x_range, Some(cf.jac_oof.clone()));
        let theta_ortho = mean(&beta_of_x(&ortho, &grid));
        (theta_naive, theta_ortho)
    };

    let (theta_naive_calib, theta_ortho_calib) = theta_pair(&cf_calib);
    let (theta_naive_miscal, theta_ortho_miscal) = theta_pair(&cf_miscal);

    let rel_shift =
        |miscal: f64, calib: f64| -> f64 { (miscal - calib).abs() / calib.abs().max(1e-9) };
    let delta_naive = rel_shift(theta_naive_miscal, theta_naive_calib);
    let delta_ortho = rel_shift(theta_ortho_miscal, theta_ortho_calib);

    // DML reference: the cross-fitted z is the treatment D, x the confounder.
    // Run it on BOTH the calibrated and miscalibrated z to read off the mature
    // orthogonal estimator's own relative shift Δ_dml.
    let dml_calib = dml_partial_linear_reference(
        &y,
        cf_calib.z_oof.as_slice().expect("contiguous calibrated z"),
        &[Column::new("x", &x)],
        5,
    );
    let dml_miscal = dml_partial_linear_reference(
        &y,
        cf_miscal.z_oof.as_slice().expect("contiguous miscalibrated z"),
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
    // estimator's robustness to first-stage error (a small slack accounts for
    // Monte-Carlo noise in the two independent reference fits).
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
    // is much larger than the orthogonalized arm's. This is the gam-side bias
    // witness that makes the orthogonality non-trivial.
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
// The orthogonalized Stage-2 arm above passes the RAW out-of-fold `J` through
// the spec field and lets the family form `Z_infl = diag(s_f·β̂₀)·J` internally.
// This focused test pins the public `influence_block_design` helper (design §6)
// against its documented closed form (design §3): row `i`, column `k` of the
// absorbed block must equal `s_f · β̂₀(x_i) · J[i,k]`. This is exact math, not a
// tool comparison — it validates the building block the workflow's internal
// path is expected to reproduce.
#[test]
fn influence_block_design_is_diag_scaled_jacobian() {
    let n = 6;
    let p1 = 3;
    let jac_cols = Array2::from_shape_fn((n, p1), |(i, k)| (i as f64 + 1.0) * 0.1 - (k as f64) * 0.07);
    let pilot_beta0 = Array1::from_shape_fn(n, |i| 0.3 + 0.2 * (i as f64) - 0.05 * (i * i) as f64);
    let s_f = 1.7_f64;

    let jac = gam::families::marginal_slope_orthogonal::ScoreInfluenceJacobian {
        columns: jac_cols.clone(),
    };
    let block = influence_block_design(&jac, &pilot_beta0, s_f);

    assert_eq!(block.nrows(), n, "influence block must have one row per observation");
    assert_eq!(block.ncols(), p1, "influence block must have one column per Stage-1 parameter");
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
