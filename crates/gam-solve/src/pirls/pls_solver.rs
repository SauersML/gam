//! Penalized least-squares solver and Gaussian fast paths.
//!
//! Owns:
//! - `GaussianFixedCache` — `XᵀWX`/`XᵀW(y−offset)` cache for the
//!   Gaussian-Identity short-circuit that the REML outer loop reuses across
//!   smoothing-parameter candidates.
//! - `SparseXtwxPrecomputed` — the sparse-pattern-aligned twin of the above
//!   for designs that take the sparse-native PIRLS path.
//! - `solve_penalized_least_squares_implicit` — identity/Gaussian implicit
//!   PLS, dense and sparse-native paths.

use super::loop_driver::max_symmetric_asymmetry;
use super::{
    FIXED_STABILIZATION_RIDGE, PirlsPenalty, PirlsWorkspace, SparseXtWxCache, StablePLSResult,
    WorkingReparamTransform, calculate_edf_from_sparse_factor,
    calculate_edfwithworkspace_from_factor, ensure_sparse_positive_definitewithridge,
    solve_sparse_spd,
};
use super::{
    calculate_deviance_from_eta, computeworkingweight_derivatives_from_eta,
    pirls_data_log_kernel_from_eta,
};
use crate::estimate::EstimationError;
use faer::sparse::SparseColMat;
use gam_linalg::faer_ndarray::{FaerLinalgError, array1_to_col_matmut};
use gam_linalg::matrix::{DesignMatrix, LinearOperator, SymmetricMatrix};
use gam_linalg::utils::{StableSolver, array_is_finite, inf_norm};
use gam_problem::{Coefficients, GlmLikelihoodSpec, InverseLink, LinkFunction};
use ndarray::{ArcArray1, Array1, Array2, ArrayView1, ShapeBuilder};
use std::sync::Arc;

/// #1868 / #1033: the once-built, ψ-invariant length-`n` row bundle for the
/// Gaussian-identity n-free κ-trial *skip* path.
///
/// On that path the inner "solve" is a zero-iteration synthesis whose every
/// length-`n` array is a trial-INVARIANT placeholder — the row predictions are
/// not recomputed, so `η ≡ μ ≡ offset`, the working response `z ≡ y`, the
/// score/Hessian weights `w ≡ priorweights`, and the working-weight
/// derivatives are `computeworkingweight_derivatives_from_eta(offset)` — all
/// functions of the frozen `(offset, y, weights)` and the fixed link, never of
/// the trial ψ. Re-materialising them on every κ callback is the O(n)-per-call
/// regression #1868 tracks (~16·n element touches per trial).
///
/// Building them **once** and sharing them by `ArcArray1` (a reference-counted
/// ndarray whose `.clone()` is O(1)) lets each trial's `PirlsResult` reuse the
/// same rows with zero per-callback row work, so the κ outer loop touches only
/// k×k objects per trial — the #1033 architectural invariant. The two cached
/// scalars (`log_likelihood` at `μ=offset`, `max_abs_eta = ‖offset‖∞`) are the
/// only other length-`n` reductions the synthesis performed per trial.
#[derive(Debug, Clone)]
pub struct GaussianFrozenRows {
    /// `η ≡ μ ≡ offset` (identity link, stale rows) — shared by the
    /// `final_offset`, `final_eta`, `finalmu`, and `solvemu` result fields.
    pub eta: ArcArray1<f64>,
    /// Working response `z ≡ y` — shared by `solveworking_response`.
    pub z: ArcArray1<f64>,
    /// Score/Hessian weights `w ≡ priorweights` — shared by `finalweights`
    /// and `solveweights`.
    pub weights: ArcArray1<f64>,
    /// `dμ/dη` at `η=offset`.
    pub solve_dmu_deta: ArcArray1<f64>,
    /// `d²μ/dη²` at `η=offset`.
    pub solve_d2mu_deta2: ArcArray1<f64>,
    /// `d³μ/dη³` at `η=offset`.
    pub solve_d3mu_deta3: ArcArray1<f64>,
    /// `dW_H/dη` at `η=offset`.
    pub solve_c_array: ArcArray1<f64>,
    /// `d²W_H/dη²` at `η=offset`.
    pub solve_d_array: ArcArray1<f64>,
    /// `log L(y; μ=offset)` — the trial-invariant zero-iteration log-likelihood.
    pub log_likelihood: f64,
    /// `‖offset‖∞` — the trial-invariant `max_abs_eta`.
    pub max_abs_eta: f64,
}

impl GaussianFrozenRows {
    /// Build the ψ-invariant frozen row bundle ONCE from the fit's frozen
    /// `(offset, y, weights)` and fixed link. This is the single O(n) reduction
    /// the n-free κ loop is allowed to pay (it is amortised across every trial),
    /// so every subsequent skip-path callback shares these rows O(1) and touches
    /// zero length-`n` objects (#1868).
    ///
    /// The values are bit-identical to what the loop_driver stale-row synthesis
    /// used to re-materialise per trial: `η ≡ μ ≡ offset` (the tensor path is
    /// Gaussian-identity, so the row predictions are stale placeholders), the
    /// working-weight derivatives are `computeworkingweight_derivatives_from_eta`
    /// at `η=offset` (constant `(1,0,0,0,0)` for Gaussian-identity), and the two
    /// scalars are the zero-iteration `log L(y; μ=offset)` and `‖offset‖∞`.
    pub(crate) fn build(
        offset: ArrayView1<'_, f64>,
        y: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        likelihood: &GlmLikelihoodSpec,
        inverse_link: &InverseLink,
    ) -> Result<Self, EstimationError> {
        let eta_owned = offset.to_owned();
        let (solve_c_array, solve_d_array, solve_dmu_deta, solve_d2mu_deta2, solve_d3mu_deta3) =
            computeworkingweight_derivatives_from_eta(
                likelihood,
                inverse_link,
                &eta_owned,
                weights,
            )?;
        let deviance = calculate_deviance_from_eta(
            y.view(),
            &eta_owned,
            likelihood,
            inverse_link,
            weights.view(),
        )?;
        let log_likelihood = pirls_data_log_kernel_from_eta(
            y,
            &eta_owned,
            likelihood,
            inverse_link,
            weights,
            deviance,
        )?;
        let max_abs_eta = inf_norm(eta_owned.iter().copied());
        Ok(Self {
            eta: eta_owned.into_shared(),
            z: y.to_owned().into_shared(),
            weights: weights.to_owned().into_shared(),
            solve_dmu_deta: solve_dmu_deta.into_shared(),
            solve_d2mu_deta2: solve_d2mu_deta2.into_shared(),
            solve_d3mu_deta3: solve_d3mu_deta3.into_shared(),
            solve_c_array: solve_c_array.into_shared(),
            solve_d_array: solve_d_array.into_shared(),
            log_likelihood,
            max_abs_eta,
        })
    }
}

/// Reusable `XᵀWX` and `XᵀW(y − offset)` for Gaussian + Identity REML fits.
///
/// The Gaussian-identity P-IRLS short-circuit solves a single linear system
/// `(XᵀWX + Σ λ_k S_k + ρ·I) β = XᵀW(y − offset)`. The right-hand-side matrix
/// and vector are independent of the smoothing parameters `λ`, so when the
/// outer REML loop evaluates the same problem at many `(λ_1, …, λ_k)`
/// candidates we only need to assemble them **once** before the loop and
/// reuse them inside every inner PIRLS call.
///
/// Stored in *original* coordinates (no Qs rotation applied). When the
/// inner solver uses a `WorkingReparamTransform`, it conjugates / projects
/// these matrices on the fly — that step is O(p³) / O(p²), independent of N.
#[derive(Debug)]
pub struct GaussianFixedCache {
    /// `XᵀWX` in the original coefficient basis. Symmetric, p × p.
    pub xtwx_orig: Array2<f64>,
    /// `XᵀW(y − offset)` in the original basis. Length p.
    pub xtwy_orig: Array1<f64>,
    /// `(y − offset)ᵀW(y − offset)`.
    ///
    /// Together with `xtwx_orig` and `xtwy_orig`, this is the last scalar
    /// sufficient statistic needed to evaluate the Gaussian penalized RSS
    /// exactly at any λ without re-streaming the rows.
    pub centered_weighted_y_sq: f64,
    /// When true, the caller is deliberately serving a design-moving trial from
    /// sufficient statistics and the `DesignMatrix` rows on the current REML
    /// surface may be a stale reference surface. Consumers must not apply those
    /// rows for fitted values, RSS, or likelihood summaries.
    pub row_prediction_is_stale: bool,
    /// `XᵀWX` precomputed for the sparse path, aligned with the symbolic
    /// pattern of `SparseXtWxCache::new(x)` on the original sparse design.
    /// `None` when the design has no sparse form (e.g. dense-only fits).
    ///
    /// The sparse REML path rebuilds `H = XᵀWX + Sλ + δI` per outer
    /// evaluation. For Gaussian-Identity the weights never change, so the
    /// `XᵀWX` contribution is invariant across the outer loop and can be
    /// scattered from this cached values vector instead of re-doing the
    /// O(nnz²/n) SpGEMM each call.
    pub xtwx_sparse_orig: Option<Arc<SparseXtwxPrecomputed>>,
    /// #1868 / #1033: the once-built ψ-invariant frozen row bundle for the
    /// n-free κ-trial skip path. Present exactly when `row_prediction_is_stale`
    /// is `true` and the producer (`gaussian_fixed_cache_at` via
    /// `install_psi_gram_statistics`) attached it. When present the Gaussian
    /// zero-iteration inner synthesis shares these length-`n` placeholders O(1)
    /// instead of re-materialising `offset`/`y`/`weights` and the working-weight
    /// derivatives per trial. `None` on the exact (non-stale) path, where the
    /// rows are freshly realised from the design.
    pub frozen_rows: Option<Arc<GaussianFrozenRows>>,
}

/// Precomputed numerical values of `XᵀWX` aligned with the symbolic pattern
/// that `SparseXtWxCache::new(x)` produces on its first call. Two such caches
/// built from the same sparse `x` produce byte-identical symbolic patterns
/// (faer's `sparse_sparse_matmul_symbolic` is deterministic), so the cached
/// values can be installed back into a fresh `SparseXtWxCache` for the same
/// `x` without rerunning the SpGEMM.
///
/// We snapshot the symbolic pattern (`col_ptr` / `row_idx`) alongside the
/// values so the consumer can verify pattern equivalence and fall through to
/// the per-call recomputation if anything diverges (e.g. an `x` with a
/// different symbolic shape sneaks in).
#[derive(Debug, Clone)]
pub struct SparseXtwxPrecomputed {
    pub xtwx_symbolic_col_ptr: Vec<usize>,
    pub xtwx_symbolic_row_idx: Vec<usize>,
    pub xtwxvalues: Vec<f64>,
}

impl SparseXtwxPrecomputed {
    /// Build the precomputed `XᵀWX` value layout for `x` at the given
    /// `weights`. The output reuses the same construction path the inner
    /// PIRLS workspace uses, so it lands in exactly the symbolic pattern
    /// the consumer expects.
    pub fn build(
        x: &SparseColMat<usize, f64>,
        weights: &Array1<f64>,
    ) -> Result<Self, EstimationError> {
        let mut cache = SparseXtWxCache::new(x)?;
        cache.compute_numeric(x, weights)?;
        Ok(Self {
            xtwx_symbolic_col_ptr: cache.xtwx_symbolic.col_ptr().to_vec(),
            xtwx_symbolic_row_idx: cache.xtwx_symbolic.row_idx().to_vec(),
            xtwxvalues: cache.xtwxvalues,
        })
    }
}

/// Identity-link solver that operates in original or QS-transformed coordinates
/// without materializing X·Qs.  When the design is sparse and `qs` is `None`
/// (sparse-native path), uses sparse Cholesky for O(nnz^{1.5}) cost instead
/// of the O(p³) dense Cholesky.
pub(super) fn solve_penalized_least_squares_implicit(
    x_original: &DesignMatrix,
    transform: Option<&WorkingReparamTransform>,
    z: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    penalty: &PirlsPenalty,
    workspace: &mut PirlsWorkspace,
    y: ArrayView1<f64>,
    link_function: LinkFunction,
    gaussian_fixed_cache: Option<&GaussianFixedCache>,
) -> Result<(StablePLSResult, usize), EstimationError> {
    let p_dim = penalty.dim();

    // ── Sparse-native fast path ──────────────────────────────────────────
    // When design is sparse and we are in original coordinates (qs = None),
    // assemble the penalized Hessian in sparse format and solve with sparse
    // Cholesky.  This avoids O(p²) dense X'WX and O(p³) dense factorization.
    if transform.is_none()
        && let Some(x_sparse) = x_original.as_sparse()
    {
        let PirlsPenalty::Dense { s_transformed, .. } = penalty else {
            crate::bail_invalid_estim!(
                "sparse-native PIRLS requires a dense transformed penalty matrix"
            );
        };
        let weights_owned = weights.to_owned();

        // Gaussian-Identity fast path: the inner sparse `XᵀWX` is invariant
        // across the outer REML loop because the IRLS weights are constant
        // (W = priorweights). The cached values land in the inner workspace
        // and bypass the per-eval SpGEMM.
        let precomputed_xtwx =
            gaussian_fixed_cache.and_then(|c| c.xtwx_sparse_orig.as_ref().map(|arc| arc.as_ref()));

        // 1. Sparse penalized Hessian: H = X'diag(w)X + S_λ + ridge·I.
        //    The Cholesky factor is reused from the SPD check so we avoid
        //    factorizing the same matrix twice.
        let (h_sparse, factor, ridge_used) = ensure_sparse_positive_definitewithridge(|ridge| {
            let ridge = if ridge == 0.0 {
                FIXED_STABILIZATION_RIDGE
            } else {
                ridge
            };
            workspace.assemble_sparse_penalized_hessian(
                x_sparse,
                &weights_owned,
                s_transformed,
                ridge,
                precomputed_xtwx,
            )
        })?;

        // 2. RHS = X'W(z - offset) + S_λ μ + ridge_used · μ.
        // The `ridge_used · μ` term matches the diagonal ridge added to
        // the Hessian in step 1, keeping the augmented system a
        // Tikhonov regularization centered at the prior mean target
        // rather than at zero (see `prior_mean_target` field docs).
        let mut wz = z.to_owned();
        wz -= &offset;
        wz *= &weights_owned;
        let mut rhs = x_original.transpose_vector_multiply(&wz);
        rhs += penalty.linear_shift();
        if ridge_used > 0.0 {
            let prior_mean_target = penalty.prior_mean_target();
            if prior_mean_target.len() == rhs.len() {
                rhs.scaled_add(ridge_used, prior_mean_target);
            }
        }

        // 3. Sparse Cholesky solve (factor reused from step 1)
        let betavec = solve_sparse_spd(&factor, &rhs)?;

        // 4. EDF — reuse the sparse Cholesky factor from step 1 to avoid a
        // second O(nnz·…) factorization of the identical penalized Hessian.
        let h_sym = SymmetricMatrix::Sparse(h_sparse);
        let edf = calculate_edf_from_sparse_factor(&factor, penalty)?;

        // 5. Scale. When Gaussian sufficient statistics are installed, compute
        // RSS from k-space only; the design rows may be a stale reference
        // surface on the #1033 ψ-tensor fast path.
        let standard_deviation = match link_function {
            LinkFunction::Identity => {
                let weighted_rss = if let Some(cache) = gaussian_fixed_cache {
                    let quadratic = betavec.dot(&cache.xtwx_orig.dot(&betavec));
                    (cache.centered_weighted_y_sq - 2.0 * betavec.dot(&cache.xtwy_orig) + quadratic)
                        .max(0.0)
                } else {
                    let fitted_vals = {
                        let xb = x_original.apply(&betavec);
                        let mut f = xb;
                        f += &offset;
                        f
                    };
                    let residuals = &y - &fitted_vals;
                    weights
                        .iter()
                        .zip(residuals.iter())
                        .map(|(&w, &r)| w * r * r)
                        .sum()
                };
                let effective_n = y.len() as f64;
                (weighted_rss / (effective_n - edf).max(1.0)).sqrt()
            }
            _ => 1.0,
        };

        return Ok((
            StablePLSResult {
                beta: Coefficients::new(betavec),
                penalized_hessian: h_sym,
                edf,
                standard_deviation,
                ridge_used,
            },
            p_dim,
        ));
    }

    // ── Dense / QS-rotated path ──────────────────────────────────────────

    // 1. Prepare weighted buffers
    if workspace.wz.len() != z.len() {
        workspace.wz = Array1::zeros(z.len());
    }
    workspace.wz.assign(&z);
    workspace.wz -= &offset;
    workspace.wz *= &weights;

    // 2. Form X'WX: compute in original coordinates, then rotate by Qs.
    //
    // Gaussian + Identity REML reuses a precomputed `XᵀWX` (the weights and
    // design never change across the outer loop in that family), so when the
    // caller supplied a `GaussianFixedCache` we skip the O(N·p²) dense
    // assembly here and adopt the cached matrix as-is.
    let weights_owned = weights.to_owned();
    let xtwx_orig = if let Some(cache) = gaussian_fixed_cache {
        // Cache hit: weights and design are invariant for Gaussian-Identity
        // across the outer REML loop, so adopt the precomputed XᵀWX directly
        // and avoid the O(N·p²) dense assembly entirely.
        let p = x_original.ncols();
        if cache.xtwx_orig.nrows() != p || cache.xtwx_orig.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "GaussianFixedCache XᵀWX shape {}×{} does not match design p={}",
                cache.xtwx_orig.nrows(),
                cache.xtwx_orig.ncols(),
                p,
            )));
        }
        cache.xtwx_orig.clone()
    } else {
        match x_original {
            // Only materialized dense designs can use the shared dense assembly path.
            // Lazy operator-backed dense designs route to diag_xtw_x like sparse.
            DesignMatrix::Dense(x_dense) if x_dense.is_materialized_dense() => {
                let p = x_dense.ncols();
                let x_dense = x_dense.to_dense_arc();
                if workspace.hessian_buf.nrows() != p || workspace.hessian_buf.ncols() != p {
                    workspace.hessian_buf = Array2::zeros((p, p).f());
                } else {
                    workspace.hessian_buf.fill(0.0);
                }
                PirlsWorkspace::add_dense_xtwx_signed(
                    &weights_owned,
                    &mut workspace.weighted_x_chunk,
                    x_dense.as_ref(),
                    &mut workspace.hessian_buf,
                );
                std::mem::take(&mut workspace.hessian_buf)
            }
            _ => {
                // Operator-form fallback: sparse designs and lazy operator-backed
                // dense designs cannot be densified, so route through the signed
                // XᵀWX operator.
                gam_linalg::matrix::xt_diag_x_signed(
                    x_original,
                    gam_linalg::matrix::FiniteSignedWeightsView::try_from_array(&weights_owned)
                        .map_err(EstimationError::InvalidInput)?,
                )
                .map(|h| h.to_dense())
                .map_err(EstimationError::InvalidInput)?
            }
        }
    };
    let xtwx_orig_asym = max_symmetric_asymmetry(&xtwx_orig);
    let xtwx_transformed = if let Some(transform) = transform {
        transform.conjugate_matrix(&xtwx_orig)
    } else {
        xtwx_orig
    };
    let mut penalized_hessian = xtwx_transformed.clone();
    penalty.add_to_hessian(&mut penalized_hessian);

    // 3. Form X'Wz: compute in original coordinates, then rotate.
    //    With the Gaussian-Identity cache `z = y` and `wz = W·(y − offset)`
    //    is identical across outer iterations, so reuse the precomputed
    //    `XᵀW(y − offset)` directly.
    let xtwy_orig = if let Some(cache) = gaussian_fixed_cache {
        assert_eq!(
            cache.xtwy_orig.len(),
            x_original.ncols(),
            "GaussianFixedCache XᵀW(y−offset) length must match design p"
        );
        cache.xtwy_orig.clone()
    } else {
        x_original.transpose_vector_multiply(&workspace.wz)
    };
    if workspace.vec_buf_p.len() != p_dim {
        workspace.vec_buf_p = Array1::zeros(p_dim);
    }
    if let Some(transform) = transform {
        workspace
            .vec_buf_p
            .assign(&transform.apply_transpose(&xtwy_orig));
    } else {
        workspace.vec_buf_p.assign(&xtwy_orig);
    }
    workspace.vec_buf_p += penalty.linear_shift();

    {
        // The penalized Hessian is assembled from symmetric pieces (XᵀWX and
        // the penalty), so any asymmetry is pure floating-point accumulation
        // error; anything above this floor signals a genuine assembly bug.
        const PENALIZED_HESSIAN_ASYMMETRY_TOL: f64 = 1e-8;
        let xtwx_asym = max_symmetric_asymmetry(&xtwx_transformed);
        let penalty_asym = match penalty {
            PirlsPenalty::Dense { s_transformed, .. } => max_symmetric_asymmetry(s_transformed),
            PirlsPenalty::Diagonal { .. } => 0.0,
        };
        let total_asym = max_symmetric_asymmetry(&penalized_hessian);
        assert!(
            total_asym <= PENALIZED_HESSIAN_ASYMMETRY_TOL,
            "implicit PLS penalized Hessian asymmetry too large: total={total_asym:.3e}, xtwx_orig={xtwx_orig_asym:.3e}, xtwx={xtwx_asym:.3e}, penalty={penalty_asym:.3e}, tol={PENALIZED_HESSIAN_ASYMMETRY_TOL:.3e}",
        );
    }

    // 4. Ridge stabilization — CONDITIONAL, matching the sparse path
    // (`ensure_sparse_positive_definitewithridge`) and the dense Newton path
    // (`ensure_positive_definitewithridge`). A penalized Hessian assembled from
    // `XᵀWX + S_λ` is mathematically PSD; a fixed tiny nugget is only needed to
    // cure round-off when the bare matrix narrowly fails Cholesky. Applying the
    // nugget UNCONDITIONALLY (the previous behaviour) made β̂ the stationary
    // point of the RIDGED objective `½βᵀ(H+δI)β`, so the inner residual was
    // `Xᵀu − S_λβ̂ = δβ̂` rather than 0. The outer REML ψ-gradient differentiates
    // the BARE objective via the envelope theorem (it assumes exact
    // stationarity), so the gratuitous δ broke the envelope identity: the
    // analytic datafit derivative `a` was short by `½·δ·βᵀ(dβ̂/dψ)` and the
    // β-independent `log|H|` term was differentiated on the un-ridged surface
    // while the criterion VALUE used `log|H+δI|`. For the Matérn iso-κ joint
    // REML at θ₀ (`TransformedQs` frame, δ_eff ≈ 1.75e-6 in the original basis)
    // this is exactly the residual outer-gradient↔FD DESYNC of #1122 (gap
    // 2.565e-2, with `cos(Xᵀu−S_λβ̂, β̂) = 1.0000` pinning the residual to the
    // ridge gradient). Try the bare matrix first so the well-conditioned common
    // case carries NO ridge (`ridge_used = 0`) and the envelope identity holds
    // exactly; fall back to the Tikhonov nugget only when the bare factorization
    // actually fails. The augmented RHS `r + δμ` keeps the fallback a Tikhonov
    // regularization centered at the prior-mean target.
    let bare_factor = StableSolver::new()
        .factorize(&penalized_hessian)
        .ok();
    let (factor, ridge_used) = if let Some(factor) = bare_factor {
        (factor, 0.0)
    } else {
        let nugget = FIXED_STABILIZATION_RIDGE;
        let mut regularizedhessian = penalized_hessian.clone();
        if nugget > 0.0 {
            for i in 0..p_dim {
                regularizedhessian[[i, i]] += nugget;
            }
        }
        let factor = StableSolver::new()
            .factorize(&regularizedhessian)
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        (factor, nugget)
    };

    // 5. Solve
    if workspace.rhs_full.len() != p_dim {
        workspace.rhs_full = Array1::zeros(p_dim);
    }
    workspace.rhs_full.assign(&workspace.vec_buf_p);
    if ridge_used > 0.0 {
        let prior_mean_target = penalty.prior_mean_target();
        if prior_mean_target.len() == p_dim {
            workspace.rhs_full.scaled_add(ridge_used, prior_mean_target);
        }
    }
    let mut rhsview = array1_to_col_matmut(&mut workspace.rhs_full);
    factor.solve_in_place(rhsview.as_mut());
    if !array_is_finite(&workspace.rhs_full) {
        return Err(EstimationError::LinearSystemSolveFailed(
            FaerLinalgError::FactorizationFailed {
                context: "PIRLS implicit PLS non-finite solve",
            },
        ));
    }
    let betavec = workspace.rhs_full.clone();

    // 6. EDF — reuse the factor already produced in step 5 to avoid a second
    // O(p³) factorization of the identical regularized Hessian.
    let edf = calculate_edfwithworkspace_from_factor(&factor, penalty, workspace)?;

    // 7. Scale (composed: eta = offset + X Qs beta). When Gaussian sufficient
    // statistics are installed, compute RSS from k-space only; the design rows
    // may be a stale reference surface on the #1033 ψ-tensor fast path.
    let qbeta = if let Some(transform) = transform {
        transform.apply(&betavec)
    } else {
        betavec.clone()
    };
    let standard_deviation = match link_function {
        LinkFunction::Identity => {
            let weighted_rss = if let Some(cache) = gaussian_fixed_cache {
                let quadratic = qbeta.dot(&cache.xtwx_orig.dot(&qbeta));
                (cache.centered_weighted_y_sq - 2.0 * qbeta.dot(&cache.xtwy_orig) + quadratic)
                    .max(0.0)
            } else {
                let xqbeta = x_original.apply(&qbeta);
                let mut fitted = xqbeta;
                fitted += &offset;
                let residuals = &y - &fitted;
                weights
                    .iter()
                    .zip(residuals.iter())
                    .map(|(&w, &r)| w * r * r)
                    .sum()
            };
            let effective_n = y.len() as f64;
            (weighted_rss / (effective_n - edf).max(1.0)).sqrt()
        }
        _ => 1.0,
    };

    Ok((
        StablePLSResult {
            beta: Coefficients::new(betavec),
            penalized_hessian: SymmetricMatrix::Dense(penalized_hessian),
            edf,
            standard_deviation,
            ridge_used,
        },
        p_dim,
    ))
}
