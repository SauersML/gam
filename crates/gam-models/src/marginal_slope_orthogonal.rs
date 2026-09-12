//! Neyman-orthogonal, cross-fitted marginal-slope calibration (closes #461):
//! the Stage-1 score-influence Jacobian and the absorbed influence block.
//!
//! Stage 1 (a conditional transformation-normal, "CTN", model) fits a monotone
//! `h(y | x; θ₁)` and emits a latent score `z_i = Φ⁻¹(u_i)` from the
//! finite-support PIT
//!
//! ```text
//! u_i = [Φ(h_i) − Φ(L_i)] / [Φ(U_i) − Φ(L_i)],
//! ```
//!
//! with (SCOP form, see `transformation_normal.rs`)
//!
//! ```text
//! h_i = b(x_i) + ε·(y_i − median) + Σ_{k≥1} I_k(y_i)·α_k(x_i)
//! L_i = h(y_min | x_i),  U_i = h(y_max | x_i)
//! b(x_i)   = Xᶜᵒᵛ_i · θ_b           (location column, response basis col 0)
//! α_k(x_i) = Xᶜᵒᵛ_i · θ_{αk}         (direct-α shape coordinates, cols k≥1,
//!                                    non-negative by the Khatri-Rao cone).
//! ```
//!
//! Stage 2 (marginal-slope) treats `z_i` as a **generated regressor**:
//! `z_i` depends on the Stage-1 estimate θ̂₁, so the β estimating equation is
//! not orthogonal to θ₁. This module exposes the score-influence Jacobian
//! `J = ∂z/∂θ₁` (design §2) and the absorbed influence block
//! `Z_infl = diag(s_f·β̂₀)·J` (design §3); Stage 2 appends `Z_infl` as a
//! null-penalized absorbed block, making the β estimating equation orthogonal
//! to `span(Z_infl)` — the discrete realization of `ψ − Π_η[ψ]`.
//!
//! ## Column ordering of `J`
//!
//! `θ₁` is the Stage-1 coefficient vector of length `p₁ = p_resp · p_cov`,
//! reshaped row-major to the `(p_resp, p_cov)` matrix `Γ` (`beta_mat`):
//! response component `k` (row) crossed with covariate column `j` (col). The
//! flat index of `Γ[k, j]` is `k·p_cov + j`. **`J`'s columns follow exactly
//! this order**: column `k·p_cov + j` holds `∂z_i/∂Γ[k, j]`. Response row
//! `k = 0` is the unconstrained location block `b(x)`; rows `k ≥ 1` are the
//! direct-α SCOP shape coordinates `α_k(x)`, non-negative on the fitted rows by
//! the Khatri-Rao monotonicity cone. The transform is AFFINE in those
//! coordinates, so both the value and the Jacobian below go through the one
//! chart evaluator in `transformation_normal::chart` (gam#2680).

use crate::inference::model::{
    TRANSFORMATION_SCORE_PIT_CLIP_EPS, TransformationNormalParameterization,
};
use crate::probability::standard_normal_quantile;
use crate::transformation_normal::{
    CtnRowBases, CtnRowFloors, TRANSFORMATION_MONOTONICITY_EPS, TransformationNormalFitResult,
    ctn_component_sensitivity, ctn_row_geometry, transformation_normal_pit_score,
};
use gam_linalg::faer_ndarray::{fast_ab, fast_abt, fast_xt_diag_x, fast_xt_diag_y};
use gam_linalg::roundoff::accumulation_growth;
use gam_linalg::utils::rank_certified_psd_pseudoinverse;
use gam_terms::smooth::build_term_collection_design;
use ndarray::{Array1, Array2, ArrayView2};

/// The coefficient chart a *fitted* (in-memory) CTN carries. A fit is the
/// definition of its own chart — the persisted marker exists so a saved model
/// can be *read back* under the chart it was written in, and it is checked on
/// the predict path. Naming it here keeps the generated-regressor Jacobian and
/// the likelihood on one evaluator (gam#2680).
const CTN_CHART: TransformationNormalParameterization =
    TransformationNormalParameterization::DirectAlpha;

/// REML seed for the absorber ridge with `n_rows` observations.
///
/// The §3 absorbed block `+Z_infl·γ` is an estimating-equation correction, not
/// a new outcome surface, so its identity penalty is *seeded* on the
/// likelihood-curvature scale rather than at the generic ρ₀ = 0 smooth seed.
/// Seeding γ's penalty at roughly one unit per observation starts the absorber
/// as a nuisance leakage correction rather than a competing flexible target
/// surface; the outer REML then moves λ wherever the evidence puts it (large λ
/// recovers the null correction, small λ engages a data-supported correction —
/// the residualized columns carry no marginal-span signal by construction).
/// `ln n ≥ 0` for every row count, so the seed needs no floor.
pub(crate) fn influence_absorber_log_lambda(n_rows: usize) -> f64 {
    (n_rows.max(1) as f64).ln()
}

/// Per-row, per-θ₁ score-influence Jacobian `∂z/∂θ₁` for a fitted CTN, plus the
/// latent score `z` itself on the same rows.
///
/// `columns` is `n × p₁` with `p₁ = p_resp · p_cov`; see the module-level
/// "Column ordering of `J`" note for the layout of the second axis. Computing
/// `J` already evaluates `h/L/U` and the finite-support PIT, so `z = Φ⁻¹(PIT)`
/// comes for free — exposing it here is the single source of truth for the
/// cross-fit fold loop, which needs the out-of-fold `z` alongside `J` and must
/// not re-run the PIT path to get it.
pub struct ScoreInfluenceJacobian {
    /// `n × p₁` matrix of `∂z_i/∂θ₁`.
    pub columns: Array2<f64>,
    /// `n` latent scores `z_i = Φ⁻¹(PIT_i)` at the same rows `J` was evaluated.
    pub z: Array1<f64>,
}

/// Compute `J = ∂z/∂θ₁` from a fitted CTN at the given `(x, y)` rows.
///
/// `response` (length `n`) and `covariate_data` (`n × d`) are the rows at which
/// to evaluate the Jacobian; they need not be the Stage-1 training rows (for
/// cross-fitting they are the held-out fold). The fitted CTN supplies the
/// coefficient vector θ̂₁, the response basis (re-evaluated at these `y` via the
/// fitted knots), and the resolved covariate spec (re-built at these `x`).
///
/// `offset` (length `n`) is the per-row additive transformation linear-predictor
/// offset on these rows. The CTN row build adds this offset identically to
/// `h`, `L`, and `U` (`row_quantities`: `h_acc = … + offset_i`, and likewise for
/// the lower/upper endpoints), so the finite-support PIT — and hence the latent
/// score `z` reported here — depends on it. The Jacobian itself (`∂z/∂θ₁`) does
/// NOT depend on the offset (it is θ₁-independent), but the *operating point*
/// at which the φ/Φ ratios are evaluated does; omitting a non-zero offset would
/// place `h/L/U` (and the emitted `z`) at the wrong point. For an offset-free
/// Stage-1 (`offset ≡ 0`) this is a no-op. Pass the held-out fold's offset rows.
///
/// Implements design §2:
///
/// ```text
/// ∂z_i/∂θ₁ = (1/φ(z_i)) · ∂u_i/∂θ₁
/// ∂u_i/∂θ₁ = [ φ(h_i)·∂h_i/∂θ₁
///              − u_i·(φ(U_i)·∂U_i/∂θ₁ − φ(L_i)·∂L_i/∂θ₁)
///              − φ(L_i)·∂L_i/∂θ₁ ] / (Φ(U_i) − Φ(L_i))
/// ∂h_i/∂A[0,j]   = Xᶜᵒᵛ_{i,j}
/// ∂h_i/∂A[k,j]   = I_k(y_i)·Xᶜᵒᵛ_{i,j}              (k ≥ 1)
/// ```
///
/// with `∂L_i`, `∂U_i` analogous (the response basis `I_k` evaluated at the
/// lower/upper support endpoints). The shape rows carry NO chart factor: the
/// direct-α transform is affine in `A`, so the sensitivity of every component is
/// the basis entry itself, uniformly in `k`. (Before gam#2680 this line read
/// `2·I_k(y_i)·γ_k(x_i)`, the derivative of the pre-`#2306` squared chart,
/// against a value path the fit had already moved off.) Non-finite rows,
/// support-order violations, and an under-resolvable endpoint mass return `Err`
/// with row context.
pub fn score_influence_jacobian(
    fit: &TransformationNormalFitResult,
    response: &Array1<f64>,
    covariate_data: ArrayView2<f64>,
    offset: &Array1<f64>,
) -> Result<ScoreInfluenceJacobian, String> {
    let family = &fit.family;
    let n = response.len();
    if covariate_data.nrows() != n {
        return Err(format!(
            "score_influence_jacobian: covariate rows ({}) != response rows ({n})",
            covariate_data.nrows()
        ));
    }
    if offset.len() != n {
        return Err(format!(
            "score_influence_jacobian: offset rows ({}) != response rows ({n})",
            offset.len()
        ));
    }
    if n == 0 {
        return Err("score_influence_jacobian: empty input rows".to_string());
    }

    let p_resp = family.p_resp();
    let p_cov = family.p_cov();
    let p1 = p_resp.checked_mul(p_cov).ok_or_else(|| {
        format!("score_influence_jacobian: p_resp({p_resp}) * p_cov({p_cov}) overflowed")
    })?;

    let beta = &fit
        .fit
        .block_states
        .first()
        .ok_or_else(|| "score_influence_jacobian: fitted CTN has no block states".to_string())?
        .beta;
    if beta.len() != p1 {
        return Err(format!(
            "score_influence_jacobian: beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
            beta.len()
        ));
    }
    // θ₁ reshaped row-major to Γ (p_resp × p_cov): row k = response component k,
    // col j = covariate column j. Matches the CTN fit reshape exactly.
    let beta_mat = beta
        .view()
        .into_shape_with_order((p_resp, p_cov))
        .map_err(|e| format!("score_influence_jacobian: beta reshape failed: {e}"))?;

    // Response value basis [1, I_1(y), …, I_K(y)] at the fitted knots, built by
    // the same helper the fit's own basis build uses.
    let (resp_val, resp_deriv) = family.evaluate_response_bases(response.view())?;
    if resp_val.nrows() != n || resp_val.ncols() != p_resp {
        return Err(format!(
            "score_influence_jacobian: response basis shape {}x{} != {n}x{p_resp}",
            resp_val.nrows(),
            resp_val.ncols()
        ));
    }

    // Covariate design at these rows, rebuilt from the fitted resolved spec so
    // the column geometry matches Stage-1. Materialize dense once.
    let cov_design = build_term_collection_design(covariate_data, &fit.covariate_spec_resolved)
        .map_err(|e| format!("score_influence_jacobian: covariate design build failed: {e}"))?;
    if cov_design.design.ncols() != p_cov {
        return Err(format!(
            "score_influence_jacobian: rebuilt covariate design has {} columns, fitted p_cov is {p_cov}",
            cov_design.design.ncols()
        ));
    }
    let x_cov = cov_design.design.try_row_chunk(0..n).map_err(|e| {
        format!("score_influence_jacobian: covariate design materialization failed: {e}")
    })?;
    let effective_offset = cov_design
        .compose_offset(offset.view(), "score influence Jacobian")
        .map_err(|e| e.to_string())?;

    // α_k(x_i) = Σ_j Xᶜᵒᵛ_{i,j}·A[k,j]  ⇒  alpha = Xᶜᵒᵛ · Aᵀ  (n × p_resp).
    let alpha = fast_abt(&x_cov, &beta_mat);
    if alpha.nrows() != n || alpha.ncols() != p_resp {
        return Err(format!(
            "score_influence_jacobian: alpha shape {}x{} != {n}x{p_resp}",
            alpha.nrows(),
            alpha.ncols()
        ));
    }

    // Row-independent endpoint response bases and floor offsets (fitted).
    let lower_basis = family.response_lower_basis();
    let upper_basis = family.response_upper_basis();
    let lower_floor = family.response_lower_floor_offset();
    let upper_floor = family.response_upper_floor_offset();
    let median = family.response_median();

    // Saturation boundaries of the PIT score. gam#2600: the fitted model's CDF
    // is `F = Φ(h)`, so `transformation_normal_pit_score` returns
    // `Φ⁻¹(Φ(h).clamp(clip_eps, 1−clip_eps)) = h` clamped to the same window.
    // Beyond a boundary the emitted `z` is EXACTLY the boundary quantile and is
    // locally constant in θ₁, so the Jacobian of the clamped function is
    // identically zero there — reporting an interior chain (as this routine
    // once did, with a floored `φ(z)`) would differentiate a different function
    // from the one whose value is consumed downstream. Computing the boundaries
    // with the same quantile kernel the score uses makes the test exact.
    let z_saturated_lo = standard_normal_quantile(TRANSFORMATION_SCORE_PIT_CLIP_EPS)
        .map_err(|e| format!("score_influence_jacobian: clip quantile failed: {e}"))?;
    let z_saturated_hi = standard_normal_quantile(1.0 - TRANSFORMATION_SCORE_PIT_CLIP_EPS)
        .map_err(|e| format!("score_influence_jacobian: clip quantile failed: {e}"))?;

    let mut columns = Array2::<f64>::zeros((n, p1));
    let mut z_scores = Array1::<f64>::zeros(n);

    for i in 0..n {
        let alpha_row = alpha.row(i);
        let val_row = resp_val.row(i);
        let deriv_row = resp_deriv.row(i);
        let x_row = x_cov.row(i);

        // h exactly as the CTN row-quantity build assembles it
        // (`row_quantities`): the additive linear-predictor offset enters h, and
        // the per-row monotonicity floor ε·(y − median) is recomputed from the
        // fitted median. The endpoints L and U are still evaluated — they are
        // the certified support the fitted model carries, and the order check
        // below is a real structural assertion about it — but since gam#2600
        // they are not part of the score, so they do not enter the chain.
        let geometry = ctn_row_geometry(
            CTN_CHART,
            alpha_row,
            CtnRowBases {
                value: val_row,
                // `h'` is not read by the PIT score or its Jacobian; the row's
                // derivative basis is supplied because the one-chart evaluator
                // computes all four components together.
                derivative: deriv_row,
                lower: lower_basis.view(),
                upper: upper_basis.view(),
            },
            CtnRowFloors {
                additive_offset: effective_offset[i],
                value_floor: TRANSFORMATION_MONOTONICITY_EPS * (response[i] - median),
                lower_floor,
                upper_floor,
            },
        );
        let (h, l, u) = (geometry.h, geometry.lower, geometry.upper);

        if !(h.is_finite() && l.is_finite() && u.is_finite()) {
            return Err(format!(
                "score_influence_jacobian: non-finite transform geometry at row {i}: h={h}, L={l}, U={u}"
            ));
        }
        if u <= l {
            return Err(format!(
                "score_influence_jacobian: support order violated at row {i}: L={l:.6e} >= U={u:.6e}"
            ));
        }

        // z is the PIT score, computed by the SAME canonical kernel the normal
        // Stage-2 path consumes (`calibrate_transformation_scores` →
        // `transformation_normal_pit_score`), so z is bit-identical to
        // Stage-2's z — single source of truth.
        let z = transformation_normal_pit_score(h, TRANSFORMATION_SCORE_PIT_CLIP_EPS)
            .map_err(|e| format!("score_influence_jacobian: PIT score failed at row {i}: {e}"))?;
        z_scores[i] = z;

        // At (or beyond) a clip boundary the score is the constant boundary
        // quantile: ∂z/∂θ₁ ≡ 0. The row was zero-initialized; skip the chain.
        if z <= z_saturated_lo || z >= z_saturated_hi {
            continue;
        }

        // Interior row: `z = h` identically, so the whole chain is the chart's
        // own sensitivity. There is no endpoint mass to differentiate, no
        // deep-tail `Φ(U)−Φ(L)` cancellation to defend against, and no `1/φ(z)`
        // to floor — all three were consequences of the endpoint-normalized PIT
        // that gam#2600 removed.
        let mut row = columns.row_mut(i);
        for k in 0..p_resp {
            // ∂h/∂A[k,j] shares the factor Xᶜᵒᵛ_{i,j}. The direct-α chart is
            // AFFINE in the coefficient matrix, so the response-side scalar is
            // the basis entry itself for every k — location block and shape
            // blocks alike, with no chart factor. The pre-gam#2680 code carried
            // `2·γ_k` on the shape rows here, the derivative of a squared chart
            // the value path had already left.
            let dh_scalar = ctn_component_sensitivity(CTN_CHART, val_row, k);
            let base = k * p_cov;
            for j in 0..p_cov {
                row[base + j] = dh_scalar * x_row[j];
            }
        }
    }

    if columns.iter().any(|v| !v.is_finite()) {
        return Err("score_influence_jacobian: produced non-finite Jacobian entries".to_string());
    }
    if z_scores.iter().any(|v| !v.is_finite()) {
        return Err("score_influence_jacobian: produced non-finite z scores".to_string());
    }

    Ok(ScoreInfluenceJacobian {
        columns,
        z: z_scores,
    })
}

/// Build the absorbed influence block `Z_infl = diag(s_f·β̂₀)·J` for Stage 2
/// (design §3): row-scale each Jacobian row `i` by `s_f · pilot_beta0[i]`,
/// where `pilot_beta0` is the rigid-pilot slope `β̂₀(x_i)` (length `n`).
///
/// The returned `n × p₁` matrix spans the realized η-space leakage directions
/// at the rigid pilot. Stage 2 appends it as a **plain additive** absorbed
/// parameter block `+Z_infl·γ` carrying a fixed small ridge `½·ρ·‖γ‖²` (γ is a
/// training-time leakage absorber, not a smooth/REML-learned block). This is
/// NOT routed through the multiplicative `score_warp` / `DeviationRuntime`
/// path — that path evaluates a scalar 1-D cubic in η and cannot carry the
/// arbitrary x-dependent `n × p₁` matrix. The absorber is orthogonalized
/// against the marginal block but deliberately overlaps slope, with gauge
/// priority above slope, and is dropped at predict time.
pub fn influence_block_design(
    jac: &ScoreInfluenceJacobian,
    pilot_beta0: &Array1<f64>,
    s_f: f64,
) -> Array2<f64> {
    let n = jac.columns.nrows();
    assert_eq!(
        pilot_beta0.len(),
        n,
        "influence_block_design: pilot_beta0 length must equal Jacobian rows"
    );
    let mut out = jac.columns.clone();
    for (i, mut row) in out.axis_iter_mut(ndarray::Axis(0)).enumerate() {
        let scale = s_f * pilot_beta0[i];
        row.mapv_inplace(|v| v * scale);
    }
    out
}

/// Residualize the influence columns `Z_infl` against the **marginal** design
/// span in the rigid-pilot row metric `W`, retaining the slope overlap
/// (#461, design §3 — single source of truth for the BMS and survival absorbed
/// blocks):
///
///   Z̃ = Z − M·(MᵀWM)⁺·MᵀW·Z.
///
/// Residualizing against **marginal only** deliberately keeps the
/// slope-aligned component, so the absorber soaks the leakage direction that
/// would otherwise manufacture spurious `β(x)` heterogeneity. `W` is the PIRLS
/// row inner product at the rigid pilot, so the resulting orthogonality
/// `MᵀW Z̃ ≈ 0` holds in the same metric the penalized joint solve sees, not
/// merely in the Euclidean sense.
///
/// `(MᵀWM)⁺` is the Moore–Penrose pseudo-inverse, so `M·(MᵀWM)⁺·MᵀW` is the
/// `W`-orthogonal projector onto `span(M)` whether or not the marginal design is
/// rank-deficient at the pilot: a dropped or aliased column adds a null
/// eigenvalue whose direction `M` maps to zero, so discarding it removes nothing
/// that is in the span. A ridge `εI` in its place leaks the fraction `ε/(λ+ε)` of
/// every direction with a small but resolvable eigenvalue `λ` back into the
/// absorber. The discarded eigenvalues are the ones the computed Gram cannot tell
/// from zero: each entry of `MᵀWM` accumulates `n` two-product terms
/// `wᵢ·mᵢₐ·mᵢᵦ` whose absolute sum is at most `√(GₐₐGᵦᵦ)`, so by Weyl the
/// formation moves every eigenvalue by at most `γ_{n+1}·tr(MᵀWM)`, and the
/// symmetric eigensolver adds its backward error `p·ε·λ_max`.
///
/// `z_infl` must already be the `influence_block_design` output (`n × p₁`).
/// When the marginal design has zero columns, or no column carries mass in the
/// `W` metric, the raw `z_infl` is returned (no span to project out).
pub(crate) fn residualize_influence_columns(
    z_infl: &Array2<f64>,
    marginal_design: ArrayView2<f64>,
    w_metric: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let n = marginal_design.nrows();
    assert_eq!(
        z_infl.nrows(),
        n,
        "residualize_influence_columns: Z_infl rows must equal marginal design rows"
    );
    assert_eq!(
        w_metric.len(),
        n,
        "residualize_influence_columns: row metric length must equal marginal design rows"
    );
    let p_m = marginal_design.ncols();
    if p_m == 0 {
        // No marginal span to residualize against; the raw directions are the
        // absorbed columns.
        return Ok(z_infl.clone());
    }
    // Weighted Gram MᵀWM in the pilot row metric.
    let gram = fast_xt_diag_x(&marginal_design, w_metric);
    if gram.iter().any(|value| !value.is_finite()) {
        return Err(
            "residualize_influence_columns: weighted marginal Gram has non-finite entries"
                .to_string(),
        );
    }
    let diagonal = gram.diag();
    if diagonal.iter().any(|&value| value < 0.0) {
        return Err(
            "residualize_influence_columns: weighted marginal Gram has a negative diagonal \
             entry, so the row metric is not positive semidefinite"
                .to_string(),
        );
    }
    let max_diagonal = diagonal.iter().copied().fold(0.0_f64, f64::max);
    if max_diagonal == 0.0 {
        // A PSD Gram with a zero diagonal is the zero matrix: no marginal column
        // has mass in the W metric, so there is no span to project out.
        return Ok(z_infl.clone());
    }
    // The resolution band above, relative to `λ_max`. `max_diagonal ≤ λ_max`, so
    // dividing the formation band by it never under-states the band.
    let relative_cutoff =
        accumulation_growth(n + 1) * diagonal.sum() / max_diagonal + p_m as f64 * f64::EPSILON;
    let pseudoinverse = rank_certified_psd_pseudoinverse(&gram, relative_cutoff)
        .map_err(|error| {
            format!("residualize_influence_columns: weighted marginal Gram pseudo-inverse: {error}")
        })?
        .into_pseudoinverse();
    // coeffs = (MᵀWM)⁺ MᵀW Z   (p_m × p₁)
    let cross = fast_xt_diag_y(&marginal_design, w_metric, z_infl);
    let coeffs = fast_ab(&pseudoinverse, &cross);
    // Z̃ = Z − M·coeffs.
    let projection = fast_ab(&marginal_design, &coeffs);
    Ok(z_infl - &projection)
}

/// The full §3 absorbed-block projection from the raw score-influence Jacobian —
/// the single shared entry point for both families.
///
/// Performs the entire sequence, so neither BMS (widened marginal design) nor
/// survival (dedicated η₁ channel) inlines any of it and both get byte-identical
/// numerics:
///
///  1. build the realized leakage directions `Z_infl = diag(s_f·β̂₀)·J`
///     ([`influence_block_design`]),
///  2. residualize against the marginal/primary span in the rigid-pilot
///     `W`-metric ([`residualize_influence_columns`]):
///     `Z̃ = Z_infl − M·(MᵀWM)⁺·MᵀW·Z_infl`, the pseudo-inverse truncated at the
///     computed Gram's own resolution rather than ridged.
///
/// Returns `Err` if the weighted marginal Gram cannot be pseudo-inverted or the
/// residualized columns are not all finite (e.g. a non-finite pilot slope or row
/// metric propagated through) — the guards are baked in so neither call site
/// repeats them. The two families differ
/// ONLY in how they install the returned `Z̃` (BMS widens `[M | Z̃]`; survival
/// adds a dedicated additive η₁ channel), never in this math.
///
/// `raw_jac` is the bare `n × p₁` score-influence Jacobian (`∂z/∂θ₁`) — i.e.
/// the value carried by the spec's `score_influence_jacobian` field — and
/// `oof_z` is the matching out-of-fold latent score; callers hold these two
/// arrays directly, so this entry point pairs them into a `ScoreInfluenceJacobian`
/// internally rather than asking every site to construct one.
pub(crate) fn residualized_influence_block(
    raw_jac: &Array2<f64>,
    oof_z: &Array1<f64>,
    pilot_beta0: &Array1<f64>,
    s_f: f64,
    marginal_design: ArrayView2<f64>,
    w_metric: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let jac = ScoreInfluenceJacobian {
        columns: raw_jac.clone(),
        z: oof_z.clone(),
    };
    let z_infl = influence_block_design(&jac, pilot_beta0, s_f);
    let residualized = residualize_influence_columns(&z_infl, marginal_design, w_metric)?;
    if residualized.iter().any(|v| !v.is_finite()) {
        return Err(
            "residualized_influence_block: residualized influence columns contain non-finite entries"
                .to_string(),
        );
    }
    Ok(residualized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn jac_from(columns: Array2<f64>) -> ScoreInfluenceJacobian {
        let n = columns.nrows();
        ScoreInfluenceJacobian {
            columns,
            z: Array1::zeros(n),
        }
    }

    // ---- influence_block_design ----

    #[test]
    fn influence_block_design_row_scales_by_sf_times_pilot() {
        // Z_infl[i, :] = (s_f * pilot_beta0[i]) * J[i, :].
        let cols = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let jac = jac_from(cols.clone());
        let pilot = array![2.0, -1.0, 0.5];
        let s_f = 3.0;
        let out = influence_block_design(&jac, &pilot, s_f);
        for i in 0..3 {
            let scale = s_f * pilot[i];
            for j in 0..2 {
                assert_eq!(out[[i, j]], cols[[i, j]] * scale);
            }
        }
    }

    #[test]
    fn influence_block_design_preserves_shape_and_does_not_mutate_input() {
        let cols = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let jac = jac_from(cols.clone());
        let out = influence_block_design(&jac, &array![1.0, 1.0], 1.0);
        assert_eq!(out.dim(), (2, 3));
        // s_f = 1, pilot = ones => Z_infl == J exactly.
        assert_eq!(out, cols);
        // Source columns untouched (function clones internally).
        assert_eq!(jac.columns, cols);
    }

    #[test]
    #[should_panic(expected = "pilot_beta0 length must equal Jacobian rows")]
    fn influence_block_design_panics_on_pilot_length_mismatch() {
        let jac = jac_from(array![[1.0], [2.0]]);
        influence_block_design(&jac, &array![1.0], 1.0);
    }

    // ---- residualize_influence_columns ----

    #[test]
    fn residualize_returns_input_when_no_marginal_columns() {
        // p_m == 0 => nothing to project out, raw columns returned verbatim.
        let z = array![[1.0, 2.0], [3.0, 4.0]];
        let m = Array2::<f64>::zeros((2, 0));
        let w = array![1.0, 1.0];
        let out = residualize_influence_columns(&z, m.view(), &w).expect("projection");
        assert_eq!(out, z);
    }

    #[test]
    fn residualize_returns_input_when_no_column_has_mass_in_the_metric() {
        // W ≡ 0 makes MᵀWM the zero matrix: there is no span in the metric, so
        // the raw columns come back verbatim rather than through a ridge.
        let z = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let m = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let w = Array1::<f64>::zeros(3);
        let out = residualize_influence_columns(&z, m.view(), &w).expect("projection");
        assert_eq!(out, z);
    }

    #[test]
    fn residualize_kills_columns_in_marginal_span() {
        // If Z lies entirely in the column span of M, the residual is ~0.
        // Build Z = M * C.
        let m = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let c = array![[2.0, -1.0], [0.5, 4.0]];
        let z = fast_ab(&m, &c);
        let w = Array1::<f64>::ones(4);
        let out = residualize_influence_columns(&z, m.view(), &w).expect("projection");
        let max_abs = out.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(max_abs < 1e-6, "residual of in-span Z too large: {max_abs}");
    }

    #[test]
    fn residualize_yields_w_orthogonal_residual() {
        // The defining property: MᵀW Z̃ ≈ 0 in the row metric W. Use a Z with a
        // component outside span(M) so the residual is nonzero but W-orthogonal.
        let m = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 4.0]];
        let z = array![[0.3, 1.0], [-2.0, 0.5], [4.0, -1.0], [0.7, 2.0]];
        let w = array![1.0, 2.0, 0.5, 1.5];
        let out = residualize_influence_columns(&z, m.view(), &w).expect("projection");
        // MᵀW Z̃ should be ~0.
        let mtwz = fast_xt_diag_y(&m, &w, &out);
        let max_abs = mtwz.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(max_abs < 1e-6, "MᵀW Z̃ not ~0: {max_abs}");
        // The residual is genuinely nonzero (Z had an out-of-span part).
        let resid_mag = out.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        assert!(resid_mag > 1e-3, "residual unexpectedly zero: {resid_mag}");
        // Shape preserved.
        assert_eq!(out.dim(), z.dim());
    }

    #[test]
    fn residualize_projects_out_a_resolvable_near_collinear_direction() {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;

        // Two marginal columns that differ by δ·t. The Gram's small eigenvalue
        // is ≈ 20δ²/λ_max ≈ 2.5e-6 against λ_max ≈ 8, far above the Gram's
        // resolution, so that direction IS in span(M), and Z = M·[−1, 1]ᵀ = δ·t
        // lies along it: the exact residual is zero.
        let delta = 1.0e-3;
        let m = array![
            [1.0, 1.0],
            [1.0, 1.0 + delta],
            [1.0, 1.0 + 2.0 * delta],
            [1.0, 1.0 + 3.0 * delta]
        ];
        let z = fast_ab(&m, &array![[-1.0], [1.0]]);
        let w = Array1::<f64>::ones(4);
        let (n, p) = m.dim();
        let gram = fast_xt_diag_x(&m, &w);
        let (evals, evecs) = gram.eigh(Side::Lower).expect("Gram eigh");
        let lambda_max = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lambda_min = evals.iter().copied().fold(f64::INFINITY, f64::min);
        // Non-vacuity: the near-collinear direction is resolvable, so it belongs
        // to the span the projection has to remove.
        let resolution =
            accumulation_growth(n + 1) * gram.diag().sum() / lambda_max + p as f64 * f64::EPSILON;
        assert!(
            lambda_min > resolution * lambda_max,
            "fixture direction is not resolvable: lambda_min={lambda_min:e}, band={:e}",
            resolution * lambda_max
        );
        // First-order forward error of a projection through the normal
        // equations: the Gram's relative resolution amplified by its condition.
        let bar = (lambda_max / lambda_min) * resolution;
        let z_scale = z.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let leak = |residual: &Array2<f64>| {
            residual.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())) / z_scale
        };

        let out = residualize_influence_columns(&z, m.view(), &w).expect("projection");
        assert!(
            leak(&out) <= bar,
            "pseudo-inverse projection kept {:e} of an in-span Z (bar {bar:e})",
            leak(&out)
        );

        // Control: the curvature-scaled ridge this projection replaced,
        // `1e-10·max diag(MᵀWM)`, keeps the fraction ε/(λ_min + ε) ≈ 1.6e-4 of
        // the same direction, which is the leak this fixture exists to detect.
        let retired_ridge = 1.0e-10 * gram.diag().iter().copied().fold(0.0_f64, f64::max);
        let mut ridged_inverse = Array2::<f64>::zeros((p, p));
        for k in 0..p {
            let scale = 1.0 / (evals[k] + retired_ridge);
            for i in 0..p {
                for j in 0..p {
                    ridged_inverse[[i, j]] += scale * evecs[[i, k]] * evecs[[j, k]];
                }
            }
        }
        let ridged_coeffs = fast_ab(&ridged_inverse, &fast_xt_diag_y(&m, &w, &z));
        let ridged = &z - &fast_ab(&m, &ridged_coeffs);
        assert!(
            leak(&ridged) > bar,
            "control: the retired ridge kept only {:e} of the in-span Z (bar {bar:e})",
            leak(&ridged)
        );
    }

    // ---- residualized_influence_block (end-to-end pure path) ----

    #[test]
    fn residualized_block_matches_manual_scale_then_residualize() {
        // The block builds Z_infl = diag(s_f·β̂₀)·J then residualizes against M in
        // the W-metric. Reconstruct that path manually.
        let raw_jac = array![[1.0, 0.5], [2.0, -1.0], [0.0, 3.0], [1.5, 1.0]];
        let oof_z = array![0.1, 0.2, 0.3, 0.4];
        let pilot = array![1.0, 2.0, -0.5, 0.5];
        let s_f = 1.5;
        let m = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let w = array![1.0, 1.0, 2.0, 0.5];

        let out =
            residualized_influence_block(&raw_jac, &oof_z, &pilot, s_f, m.view(), &w).unwrap();

        // Manual: scale rows, then residualize.
        let jac = ScoreInfluenceJacobian {
            columns: raw_jac.clone(),
            z: oof_z.clone(),
        };
        let z_infl = influence_block_design(&jac, &pilot, s_f);
        let expected =
            residualize_influence_columns(&z_infl, m.view(), &w).expect("projection");

        assert_eq!(out, expected);
        // And the result is W-orthogonal to the marginal span.
        let mtwz = fast_xt_diag_y(&m, &w, &out);
        let max_abs = mtwz.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_abs < 1e-6, "MᵀW Z̃ not ~0: {max_abs}");
    }
}
