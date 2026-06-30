//! Neyman-orthogonal, cross-fitted marginal-slope calibration (closes #461):
//! the Stage-1 score-influence Jacobian and the absorbed influence block.
//!
//! Stage 1 (a conditional transformation-normal, "CTN", model) fits a monotone
//! `h(y | x; θ₁)` and emits a latent score `z_i = Φ⁻¹(u_i)` from the
//! finite-support PIT
//!
//!     u_i = [Φ(h_i) − Φ(L_i)] / [Φ(U_i) − Φ(L_i)],
//!
//! with (SCOP form, see `transformation_normal.rs`)
//!
//!     h_i = b(x_i) + ε·(y_i − median) + Σ_{k≥1} I_k(y_i)·γ_k(x_i)²
//!     L_i = h(y_min | x_i),  U_i = h(y_max | x_i)
//!     b(x_i)   = Xᶜᵒᵛ_i · θ_b           (location column, response basis col 0)
//!     γ_k(x_i) = Xᶜᵒᵛ_i · θ_{γk}         (squared SCOP shape coeffs, cols k≥1).
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
//! squared SCOP shape blocks for `γ_k(x)`.

use gam_linalg::faer_ndarray::{
    FaerArrayView, factorize_symmetricwith_fallback, fast_ab, fast_abt, fast_xt_diag_x,
    fast_xt_diag_y,
};
use crate::transformation_normal::{
    TRANSFORMATION_MONOTONICITY_EPS, TransformationNormalFitResult, transformation_normal_pit_score,
};
use crate::inference::model::TRANSFORMATION_SCORE_PIT_CLIP_EPS;
use gam_linalg::matrix::FactorizedSystem;
use crate::probability::{
    log1mexp_positive, normal_cdf, normal_logcdf, normal_pdf, standard_normal_quantile,
};
use gam_terms::smooth::build_term_collection_design;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView2};

/// Fixed (NOT REML-learned) log-λ for the influence-absorber block's ridge.
///
/// The §3 absorbed block `+Z_infl·γ` carries a small fixed ridge `½·ρ·‖γ‖²`
/// (`ρ = exp(log_λ)`) so the joint solve soaks the `span(Z_infl)` component of
/// the η-residual into `γ` without that block being treated as a smooth/REML
/// surface. `log_λ = 0` ⇒ `ρ = 1`: enough to keep `γ` finite and bounded while
/// the much larger marginal/logslope likelihood curvature dominates the fit.
pub(crate) const INFLUENCE_ABSORBER_FIXED_LOG_LAMBDA: f64 = 0.0;

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
/// ∂h_i/∂Γ[0,j]   = Xᶜᵒᵛ_{i,j}
/// ∂h_i/∂Γ[k,j]   = 2·I_k(y_i)·γ_k(x_i)·Xᶜᵒᵛ_{i,j}   (k ≥ 1)
/// ```
///
/// with `∂L_i`, `∂U_i` analogous (the response basis `I_k` evaluated at the
/// lower/upper support endpoints). Non-finite rows, support-order violations,
/// and an under-resolvable endpoint mass return `Err` with row context.
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

    // Response value basis [1, I_1(y), …, I_K(y)] at the fitted knots.
    let resp_val = family.evaluate_response_value_basis(response.view())?;
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

    // γ_k(x_i) = Σ_j Xᶜᵒᵛ_{i,j}·Γ[k,j]  ⇒  gamma = Xᶜᵒᵛ · Γᵀ  (n × p_resp).
    let gamma = fast_abt(&x_cov, &beta_mat);
    if gamma.nrows() != n || gamma.ncols() != p_resp {
        return Err(format!(
            "score_influence_jacobian: gamma shape {}x{} != {n}x{p_resp}",
            gamma.nrows(),
            gamma.ncols()
        ));
    }

    // Row-independent endpoint response bases and floor offsets (fitted).
    let lower_basis = family.response_lower_basis();
    let upper_basis = family.response_upper_basis();
    let lower_floor = family.response_lower_floor_offset();
    let upper_floor = family.response_upper_floor_offset();
    let median = family.response_median();

    // Floor on φ(z): at the PIT clip the score saturates at a fixed extreme
    // quantile, so φ(z) is bounded below by φ(Φ⁻¹(clip_eps)). Clamping the
    // ∂z = ∂u/φ(z) denominator there keeps a saturated row's influence bounded
    // rather than infinite — the same bound the PIT clip already imposes on z.
    let pdf_z_floor = normal_pdf(
        standard_normal_quantile(TRANSFORMATION_SCORE_PIT_CLIP_EPS)
            .map_err(|e| format!("score_influence_jacobian: clip quantile failed: {e}"))?,
    );

    let mut columns = Array2::<f64>::zeros((n, p1));
    let mut z_scores = Array1::<f64>::zeros(n);

    for i in 0..n {
        let gamma_row = gamma.row(i);
        let val_row = resp_val.row(i);
        let x_row = x_cov.row(i);
        let g0 = gamma_row[0];

        // h, L, U exactly as the CTN row-quantity build assembles them
        // (`row_quantities`): the additive linear-predictor offset enters h, L,
        // and U identically, so it is added here at all three to place the PIT
        // operating point (and the emitted z) where the fitted model evaluates
        // it. The per-row monotonicity floor ε·(y − median) and the endpoint
        // floors are recomputed from the fitted median.
        let offset_i = offset[i];
        let value_floor = TRANSFORMATION_MONOTONICITY_EPS * (response[i] - median);
        let mut h = val_row[0] * g0 + offset_i + value_floor;
        let mut l = lower_basis[0] * g0 + offset_i + lower_floor;
        let mut u = upper_basis[0] * g0 + offset_i + upper_floor;
        for k in 1..p_resp {
            let gk = gamma_row[k];
            let gk_sq = gk * gk;
            h += val_row[k] * gk_sq;
            l += lower_basis[k] * gk_sq;
            u += upper_basis[k] * gk_sq;
        }

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

        // z is the finite-support PIT score, computed by the SAME canonical
        // kernel the normal Stage-2 path consumes (`calibrate_transformation_scores`
        // → `transformation_normal_pit_score`). That kernel forms the PIT ratio in
        // LOG space (`normal_logcdf` + `log1mexp_positive`), which stays accurate
        // when h/L/U all sit deep in a tail where the direct CDF subtraction
        // `(Φ(h)−Φ(L))/(Φ(U)−Φ(L))` would cancel catastrophically. Reusing it makes
        // z bit-identical to Stage-2's z — single source of truth.
        let z = transformation_normal_pit_score(h, l, u, TRANSFORMATION_SCORE_PIT_CLIP_EPS)
            .map_err(|e| format!("score_influence_jacobian: PIT score failed at row {i}: {e}"))?;
        z_scores[i] = z;

        // The ∂u/∂θ chain is evaluated at the SAME (clipped) operating point z
        // represents: u_pit = Φ(z) exactly inverts z = Φ⁻¹(u_clamped), so the
        // derivative coefficient and the reported score stay self-consistent
        // without recomputing the (less stable) direct ratio. The endpoint
        // φ/D ratios below are the analytic derivatives of that ratio.
        //
        // Compute log(D) = log(Φ(U)−Φ(L)) in log-space to avoid catastrophic
        // cancellation when L and U both sit deep in the same tail (e.g. L=5,
        // U=6 where normal_cdf returns 1.0 for both in direct form). This
        // mirrors the `log_normal_cdf_diff` approach in `transformation_normal.rs`:
        //
        //   If L > 0 : Φ(U)−Φ(L) = Φ(−L)−Φ(−U)  (reflection, both < 0.5)
        //              log_denom = normal_logcdf(−L) + log1mexp(normal_logcdf(−L) − normal_logcdf(−U))
        //   Otherwise: log_denom = normal_logcdf(U) + log1mexp(normal_logcdf(U) − normal_logcdf(L))
        let log_denom = if l > 0.0 {
            let log_neg_l = normal_logcdf(-l);
            let log_neg_u = normal_logcdf(-u);
            let gap = log_neg_l - log_neg_u;
            if !(gap.is_finite() && gap > 0.0) {
                return Err(format!(
                    "score_influence_jacobian: endpoint mass not resolvable at row {i}: l={l:.6e}, u={u:.6e}"
                ));
            }
            log_neg_l + log1mexp_positive(gap)
        } else {
            let log_cu = normal_logcdf(u);
            let log_cl = normal_logcdf(l);
            let gap = log_cu - log_cl;
            if !(gap.is_finite() && gap > 0.0) {
                return Err(format!(
                    "score_influence_jacobian: endpoint mass not resolvable at row {i}: l={l:.6e}, u={u:.6e}"
                ));
            }
            log_cu + log1mexp_positive(gap)
        };
        if !log_denom.is_finite() {
            return Err(format!(
                "score_influence_jacobian: log endpoint mass not finite at row {i}: l={l:.6e}, u={u:.6e}"
            ));
        }

        let u_pit = normal_cdf(z);

        // φ(x)/D = exp(log φ(x) − log D).  Using log-space for these ratios
        // keeps them accurate when D is tiny (deep-tail support intervals).
        // log φ(x) = −½x² − log(√(2π)).
        const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
        let log_phi = |x: f64| -0.5 * x * x - LOG_SQRT_2PI;

        // φ at h/L/U. The chain ∂u/∂θ uses φ at the *unclamped* h when
        // h is inside [L,U]; at the boundary (h clamped) φ(h)·∂h is the limiting
        // contribution and the clamp keeps it finite.
        let h_clamped = h.clamp(l, u);
        let c_h = (log_phi(h_clamped) - log_denom).exp();
        let c_l = (log_phi(l) - log_denom).exp();
        let c_u = (log_phi(u) - log_denom).exp();

        // ∂z = ∂u / φ(z). Where the score saturates (|z| large), φ(z) underflows;
        // the `pdf_z_floor` clamp keeps a saturated row's influence bounded.
        let pdf_z = normal_pdf(z).max(pdf_z_floor);

        let mut row = columns.row_mut(i);
        for k in 0..p_resp {
            // ∂h/∂Γ[k,j], ∂L/∂Γ[k,j], ∂U/∂Γ[k,j] share the factor Xᶜᵒᵛ_{i,j};
            // the response-side scalar differs between the location block
            // (k = 0, basis value, unsquared) and the shape blocks
            // (k ≥ 1, 2·basis·γ_k).
            let (dh_scalar, dl_scalar, du_scalar) = if k == 0 {
                (val_row[0], lower_basis[0], upper_basis[0])
            } else {
                let two_gk = 2.0 * gamma_row[k];
                (
                    val_row[k] * two_gk,
                    lower_basis[k] * two_gk,
                    upper_basis[k] * two_gk,
                )
            };
            let base = k * p_cov;
            for j in 0..p_cov {
                let xij = x_row[j];
                let dh = dh_scalar * xij;
                let dl = dl_scalar * xij;
                let du = du_scalar * xij;
                // ∂u = [φ(h)∂h − u·(φ(U)∂U − φ(L)∂L) − φ(L)∂L] / (Φ(U)−Φ(L))
                //     = c_h·∂h − u_pit·(c_u·∂U − c_l·∂L) − c_l·∂L
                let du_pit = c_h * dh - u_pit * (c_u * du - c_l * dl) - c_l * dl;
                row[base + j] = du_pit / pdf_z;
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
/// where `pilot_beta0` is the rigid-pilot logslope `β̂₀(x_i)` (length `n`).
///
/// The returned `n × p₁` matrix spans the realized η-space leakage directions
/// at the rigid pilot. Stage 2 appends it as a **plain additive** absorbed
/// parameter block `+Z_infl·γ` carrying a fixed small ridge `½·ρ·‖γ‖²` (γ is a
/// training-time leakage absorber, not a smooth/REML-learned block). This is
/// NOT routed through the multiplicative `score_warp` / `DeviationRuntime`
/// path — that path evaluates a scalar 1-D cubic in η and cannot carry the
/// arbitrary x-dependent `n × p₁` matrix. The absorber is orthogonalized
/// against the marginal block but deliberately overlaps logslope, with gauge
/// priority above logslope, and is dropped at predict time.
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
/// span in the rigid-pilot row metric `W`, retaining the logslope overlap
/// (#461, design §3 — single source of truth for the BMS and survival absorbed
/// blocks):
///
///   Z̃ = Z − M·(MᵀWM + εI)⁻¹·MᵀW·Z.
///
/// Residualizing against **marginal only** deliberately keeps the
/// logslope-aligned component, so the absorber soaks the leakage direction that
/// would otherwise manufacture spurious `β(x)` heterogeneity. `W` is the PIRLS
/// row inner product at the rigid pilot, so the resulting orthogonality
/// `MᵀW Z̃ ≈ 0` holds in the same metric the penalized joint solve sees, not
/// merely in the Euclidean sense. `eps` is the (caller-scaled) ridge added to
/// the weighted marginal Gram diagonal so the projection solve stays stable when
/// the marginal design is rank-deficient at the pilot.
///
/// `z_infl` must already be the `influence_block_design` output (`n × p₁`).
/// When the marginal design has zero columns the raw `z_infl` is returned (no
/// span to project out).
pub(crate) fn residualize_influence_columns(
    z_infl: &Array2<f64>,
    marginal_design: ArrayView2<f64>,
    w_metric: &Array1<f64>,
    eps: f64,
) -> Array2<f64> {
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
        return z_infl.clone();
    }
    // Weighted Gram MᵀWM and cross term MᵀW Z in the pilot row metric.
    let mut gram = fast_xt_diag_x(&marginal_design, w_metric);
    for i in 0..p_m {
        gram[[i, i]] += eps;
    }
    let cross = fast_xt_diag_y(&marginal_design, w_metric, z_infl);
    let gram_view = FaerArrayView::new(&gram);
    let factor = factorize_symmetricwith_fallback(gram_view.as_ref(), Side::Lower)
        .expect("residualize_influence_columns: weighted marginal Gram factorization failed");
    // coeffs = (MᵀWM + εI)⁻¹ MᵀW Z   (p_m × p₁)
    let coeffs = factor
        .solvemulti(&cross)
        .expect("residualize_influence_columns: marginal projection solve failed");
    // Z̃ = Z − M·coeffs.
    let projection = fast_ab(&marginal_design, &coeffs);
    z_infl - &projection
}

/// Relative magnitude (vs. the largest weighted marginal-Gram diagonal) of the
/// ridge added to `MᵀWM` in the §3 projection solve. Tiny — it only regularizes
/// a rank-deficient marginal design at the pilot (a dropped/aliased spatial
/// column leaves a zero pivot) and never perturbs a well-conditioned projection.
pub(crate) const INFLUENCE_PROJECTION_RELATIVE_RIDGE: f64 = 1.0e-10;
/// Absolute floor on the §3 projection ridge, so a degenerate (all-zero)
/// weighted marginal Gram still yields an invertible system.
pub(crate) const INFLUENCE_PROJECTION_RIDGE_FLOOR: f64 = 1.0e-12;

/// The full §3 absorbed-block projection from the raw score-influence Jacobian —
/// the single shared entry point for both families.
///
/// Performs the entire sequence, so neither BMS (widened marginal design) nor
/// survival (dedicated η₁ channel) inlines any of it and both get byte-identical
/// numerics:
///
///  1. build the realized leakage directions `Z_infl = diag(s_f·β̂₀)·J`
///     ([`influence_block_design`]),
///  2. derive the projection ridge from the weighted marginal Gram's own
///     magnitude — `eps = max(diag(MᵀWM))·INFLUENCE_PROJECTION_RELATIVE_RIDGE`
///     floored at `INFLUENCE_PROJECTION_RIDGE_FLOOR` — so it scales with the
///     design rather than being a fixed absolute the caller must guess,
///  3. residualize against the marginal/primary span in the rigid-pilot
///     `W`-metric ([`residualize_influence_columns`]):
///     `Z̃ = Z_infl − M·(MᵀWM + εI)⁻¹·MᵀW·Z_infl`.
///
/// Returns `Err` if the residualized columns are not all finite (e.g. a
/// non-finite pilot logslope or row metric propagated through) — the finite
/// guard is baked in so neither call site repeats it. The two families differ
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

    // Ridge scaled to the weighted marginal Gram's own magnitude. Only the
    // diagonal is needed to size it, so reuse the same MᵀWM the residualizer
    // forms internally (the cost is one extra weighted Gram; kept here so the
    // ε logic lives with the projection it regularizes).
    let p_m = marginal_design.ncols();
    let gram = fast_xt_diag_x(&marginal_design, w_metric);
    let gram_scale = (0..p_m).map(|i| gram[[i, i]]).fold(0.0_f64, f64::max);
    let eps =
        (gram_scale * INFLUENCE_PROJECTION_RELATIVE_RIDGE).max(INFLUENCE_PROJECTION_RIDGE_FLOOR);

    let residualized = residualize_influence_columns(&z_infl, marginal_design, w_metric, eps);
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
        let out = residualize_influence_columns(&z, m.view(), &w, 1e-12);
        assert_eq!(out, z);
    }

    #[test]
    fn residualize_kills_columns_in_marginal_span() {
        // If Z lies entirely in the column span of M, the residual is ~0
        // (with a tiny ridge). Build Z = M * C.
        let m = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let c = array![[2.0, -1.0], [0.5, 4.0]];
        let z = fast_ab(&m, &c);
        let w = Array1::<f64>::ones(4);
        let out = residualize_influence_columns(&z, m.view(), &w, 1e-12);
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
        let out = residualize_influence_columns(&z, m.view(), &w, 1e-12);
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

    // ---- residualized_influence_block (end-to-end pure path) ----

    #[test]
    fn residualized_block_matches_manual_scale_then_residualize() {
        // The block builds Z_infl = diag(s_f·β̂₀)·J then residualizes against M in
        // the W-metric with a Gram-scaled ridge. Reconstruct that path manually.
        let raw_jac = array![[1.0, 0.5], [2.0, -1.0], [0.0, 3.0], [1.5, 1.0]];
        let oof_z = array![0.1, 0.2, 0.3, 0.4];
        let pilot = array![1.0, 2.0, -0.5, 0.5];
        let s_f = 1.5;
        let m = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let w = array![1.0, 1.0, 2.0, 0.5];

        let out =
            residualized_influence_block(&raw_jac, &oof_z, &pilot, s_f, m.view(), &w).unwrap();

        // Manual: scale rows, derive eps from the weighted Gram diagonal, residualize.
        let jac = ScoreInfluenceJacobian {
            columns: raw_jac.clone(),
            z: oof_z.clone(),
        };
        let z_infl = influence_block_design(&jac, &pilot, s_f);
        let gram = fast_xt_diag_x(&m, &w);
        let gram_scale = (0..m.ncols()).map(|i| gram[[i, i]]).fold(0.0_f64, f64::max);
        let eps = (gram_scale * INFLUENCE_PROJECTION_RELATIVE_RIDGE)
            .max(INFLUENCE_PROJECTION_RIDGE_FLOOR);
        let expected = residualize_influence_columns(&z_infl, m.view(), &w, eps);

        assert_eq!(out.dim(), expected.dim());
        for (a, b) in out.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "mismatch: {a} vs {b}");
        }
        // And the result is W-orthogonal to the marginal span.
        let mtwz = fast_xt_diag_y(&m, &w, &out);
        let max_abs = mtwz.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(max_abs < 1e-6, "MᵀW Z̃ not ~0: {max_abs}");
    }
}
