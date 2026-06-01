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

use crate::faer_ndarray::{
    FaerArrayView, factorize_symmetricwith_fallback, fast_ab, fast_abt, fast_xt_diag_x,
    fast_xt_diag_y,
};
use crate::families::transformation_normal::{
    TRANSFORMATION_MONOTONICITY_EPS, TransformationNormalFitResult, transformation_normal_pit_score,
};
use crate::inference::model::TRANSFORMATION_SCORE_PIT_CLIP_EPS;
use crate::matrix::FactorizedSystem;
use crate::probability::{normal_cdf, normal_pdf, standard_normal_quantile};
use crate::smooth::build_term_collection_design;
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
) -> Result<ScoreInfluenceJacobian, String> {
    let family = &fit.family;
    let n = response.len();
    if covariate_data.nrows() != n {
        return Err(format!(
            "score_influence_jacobian: covariate rows ({}) != response rows ({n})",
            covariate_data.nrows()
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
        .ok_or_else(|| {
            "score_influence_jacobian: fitted CTN has no block states".to_string()
        })?
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
    let x_cov = cov_design
        .design
        .try_row_chunk(0..n)
        .map_err(|e| format!("score_influence_jacobian: covariate design materialization failed: {e}"))?;

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

        // h, L, U exactly as the CTN row-quantity build assembles them. At
        // out-of-sample rows the additive linear-predictor offset is absent
        // (predict semantics); the per-row monotonicity floor ε·(y − median)
        // and the endpoint floors are recomputed from the fitted median.
        let value_floor = TRANSFORMATION_MONOTONICITY_EPS * (response[i] - median);
        let mut h = val_row[0] * g0 + value_floor;
        let mut l = lower_basis[0] * g0 + lower_floor;
        let mut u = upper_basis[0] * g0 + upper_floor;
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
        // φ/Φ values below are the analytic derivatives of that ratio.
        let phi_l = normal_cdf(l);
        let phi_u = normal_cdf(u);
        let denom_mass = phi_u - phi_l;
        if !(denom_mass.is_finite() && denom_mass > 0.0) {
            return Err(format!(
                "score_influence_jacobian: endpoint mass not resolvable at row {i}: Φ(U)−Φ(L)={denom_mass:.6e}"
            ));
        }
        let u_pit = normal_cdf(z);

        // φ at h/L/U and at z. The chain ∂u/∂θ uses φ at the *unclamped* h when
        // h is inside [L,U]; at the boundary (h clamped) φ(h)·∂h is the limiting
        // contribution and the clamp keeps it finite.
        let pdf_h = normal_pdf(h.clamp(l, u));
        let pdf_l = normal_pdf(l);
        let pdf_u = normal_pdf(u);
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
                let du_pit = (pdf_h * dh - u_pit * (pdf_u * du - pdf_l * dl) - pdf_l * dl)
                    / denom_mass;
                row[base + j] = du_pit / pdf_z;
            }
        }
    }

    if columns.iter().any(|v| !v.is_finite()) {
        return Err(
            "score_influence_jacobian: produced non-finite Jacobian entries".to_string(),
        );
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
    debug_assert_eq!(
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
    debug_assert_eq!(
        z_infl.nrows(),
        n,
        "residualize_influence_columns: Z_infl rows must equal marginal design rows"
    );
    debug_assert_eq!(
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
