use super::inner_strategy::GeometryBackendKind;
use super::penalty_logdet::PenaltyPseudologdet;
use super::*;
use crate::linalg::utils::enforce_symmetry;

// Relative scale of the diagonal ridge added to the ρ-Hessian before
// inverting it for sigma-point construction. Matches the analogous IFT
// regularisation: tiny enough to leave well-conditioned Hessians intact,
// large enough that a near-singular Hessian still yields a usable V_ρ.
const AUTO_CUBATURE_HESSIAN_RIDGE_REL: f64 = 1e-8;
// Absolute floor for the diagonal ridge (prevents zero ridge when the
// Hessian diagonal is degenerate / all-zero).
const AUTO_CUBATURE_HESSIAN_RIDGE_ABS: f64 = 1e-8;
// Inset from RHO_BOUND when clamping sigma points so the inner PIRLS
// fit at a sigma point is strictly interior to the box constraint
// (the box edge is unreachable by IRLS without barrier intervention).
const AUTO_CUBATURE_RHO_CLAMP_INSET: f64 = 1e-8;
// Skip cubature when the first-order rho-Hessian inverse already shows
// negligible posterior variance on rho (max diag < this threshold) and
// neither boundary contact nor large outer-gradient flags fired.
const AUTO_CUBATURE_RHOVAR_TRIGGER: f64 = 0.1;

/// Severity classifier for first-order fallbacks taken by
/// [`RemlState::compute_smoothing_correction_auto`].
///
/// `Routine` covers by-design eligibility gates (dimension limits, the
/// near-boundary/highgrad linearization gate, rank-deficient `V_ρ` where
/// cubature would inject spurious variance, `n_rho == 0`, etc.). These
/// log at `info` and do not count as failures.
///
/// `NumericalFailure` covers situations where cubature was requested by
/// the eligibility logic but a downstream numerical step refused to
/// produce a usable second-order correction: Hessian compute / inversion
/// failed, the inverse Hessian's spectrum is non-positive, a sigma-point
/// inner PIRLS diverged, or the assembled total covariance is
/// non-finite. These log at `warn` and increment
/// [`SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT`] so they are visible
/// in long-running fits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingCorrectionFallbackSeverity {
    Routine,
    NumericalFailure,
}

/// Structured outcome of [`RemlState::compute_smoothing_correction_auto`].
///
/// The variant tells the caller exactly which branch produced the
/// returned matrix: a successful cubature upgrade, a principled
/// first-order linearization (the gradient-only correction at ρ̂), or an
/// unavailable correction (the only branch that yields no matrix —
/// reserved for `n_rho == 0` where the unified corrected covariance
/// equals `H⁻¹` and no separate additive correction is meaningful, and
/// for the case where the caller did not supply a base covariance to
/// upgrade).
#[derive(Clone, Debug)]
pub enum SmoothingCorrectionOutcome {
    /// Cubature upgrade succeeded.
    Cubature {
        correction: Array2<f64>,
        rank: usize,
        n_points: usize,
        near_boundary: bool,
        grad_norm: f64,
        max_rho_var: f64,
    },
    /// Principled first-order linearization was returned.
    FirstOrder {
        correction: Option<Array2<f64>>,
        reason: &'static str,
        severity: SmoothingCorrectionFallbackSeverity,
    },
}

impl SmoothingCorrectionOutcome {
    /// Extract the additive correction matrix, if any.
    ///
    /// Returns `None` only when the first-order path itself produced
    /// nothing (e.g. `n_rho == 0` where no separate correction is
    /// meaningful, or when no base covariance was supplied to upgrade).
    pub fn into_correction(self) -> Option<Array2<f64>> {
        match self {
            SmoothingCorrectionOutcome::Cubature { correction, .. } => Some(correction),
            SmoothingCorrectionOutcome::FirstOrder { correction, .. } => correction,
        }
    }

    /// Human-readable label naming the branch taken.
    pub fn branch_label(&self) -> &'static str {
        match self {
            SmoothingCorrectionOutcome::Cubature { .. } => "cubature",
            SmoothingCorrectionOutcome::FirstOrder { severity, .. } => match severity {
                SmoothingCorrectionFallbackSeverity::Routine => "first-order (routine)",
                SmoothingCorrectionFallbackSeverity::NumericalFailure => {
                    "first-order (numerical failure)"
                }
            },
        }
    }
}

/// Process-wide count of numerical failures inside
/// [`RemlState::compute_smoothing_correction_auto`]. Incremented whenever
/// cubature was requested by the eligibility gate but a downstream numerical
/// step refused to produce a usable second-order correction.
pub static SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Accumulate the sigma-point cubature total covariance `V̂_p` from per-point
/// `(A_m, b_m)` pairs.
///
/// Math: with equal weights `w_m = 1/M`,
///   `mean_hinv = Σ w_m A_m`
///   `mean_beta = Σ w_m b_m`
///   `second_beta = Σ w_m b_m b_mᵀ`
///   `var_beta = second_beta − mean_beta · mean_betaᵀ`
///   `V̂_p = mean_hinv + var_beta`
///
/// This is the law of total covariance applied to the per-sigma Laplace
/// approximation: `V_p = φ[E_ρ H(ρ)⁻¹ + Cov_ρ β̂(ρ)]`. Returned matrix is
/// not yet symmetry-enforced; the caller does that.
///
/// Pulled out as a free function so the sigma-cubature math has a single,
/// directly-testable implementation, independent of the (CPU Rayon / future
/// GPU stream-pool) execution model that produced `points`.
pub(crate) fn accumulate_sigma_cubature_total_covariance(
    points: &[(Array2<f64>, Array1<f64>)],
    p: usize,
) -> Array2<f64> {
    let w = 1.0 / (points.len() as f64);
    let mut mean_hinv = Array2::<f64>::zeros((p, p));
    let mut mean_beta = Array1::<f64>::zeros(p);
    let mut second_beta = Array2::<f64>::zeros((p, p));
    for (cov_point, beta_point) in points {
        // scaled_add avoids allocating intermediate scaled arrays per sigma
        // point; numerically equivalent to `mean += &arr.mapv(|v| w * v)`.
        mean_hinv.scaled_add(w, cov_point);
        mean_beta.scaled_add(w, beta_point);
        let beta_col = beta_point.view().insert_axis(ndarray::Axis(1));
        let beta_row = beta_point.view().insert_axis(ndarray::Axis(0));
        let outer = beta_col.dot(&beta_row);
        second_beta.scaled_add(w, &outer);
    }
    let mean_outer = mean_beta
        .view()
        .insert_axis(ndarray::Axis(1))
        .dot(&mean_beta.view().insert_axis(ndarray::Axis(0)));
    let var_beta = second_beta - mean_outer;
    mean_hinv + var_beta
}

/// Process-wide count of cubature upgrades that succeeded inside
/// [`RemlState::compute_smoothing_correction_auto`]. Paired with
/// [`SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT`] for visibility.
pub static SMOOTHING_CORRECTION_CUBATURE_COUNT: AtomicU64 = AtomicU64::new(0);

impl<'a> RemlState<'a> {
    fn cached_penalty_block_structural_nullities(
        &self,
    ) -> Result<super::penalty_logdet::PenaltyBlockStructuralNullities, EstimationError> {
        if let Some(cached) = self
            .penalty_block_structural_nullities
            .read()
            .unwrap()
            .clone()
        {
            return Ok(cached);
        }
        let computed = PenaltyPseudologdet::structural_block_nullities(&self.canonical_penalties)
            .map_err(EstimationError::LayoutError)?;
        *self.penalty_block_structural_nullities.write().unwrap() = Some(computed.clone());
        Ok(computed)
    }

    /// Compute first and second derivatives of the exact pseudo-logdet
    /// log|S|₊ with respect to ρ.
    ///
    /// Uses eigendecomposition to identify the positive eigenspace, then
    /// computes exact derivatives on that subspace:
    ///
    ///   ∂_k L = tr(S⁺ Aₖ)
    ///   ∂²_kl L = δ_{kl} ∂_k L − λₖ λₗ tr(S⁺ Sₖ S⁺ Sₗ)
    ///
    /// where Aₖ = λₖ Sₖ and S⁺ is the pseudoinverse on the positive eigenspace.
    pub(super) fn structural_penalty_logdet_derivatives(
        &self,
        rs_transformed: &[Array2<f64>],
        lambdas: &Array1<f64>,
        ridge: f64,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        let k_count = lambdas.len();
        if rs_transformed.len() != k_count {
            return Err(EstimationError::LayoutError(format!(
                "Penalty root/lambda count mismatch in structural logdet derivatives: roots={}, lambdas={}",
                rs_transformed.len(),
                k_count
            )));
        }
        if k_count == 0 {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        // Build S_k = R_k^T R_k for each penalty component.
        let s_k_matrices: Vec<Array2<f64>> = rs_transformed
            .iter()
            .map(|r_k| crate::faer_ndarray::fast_atb(r_k, r_k))
            .collect();

        let lambdas_slice = lambdas.as_slice().unwrap();

        let pld = PenaltyPseudologdet::from_components(&s_k_matrices, lambdas_slice, ridge)
            .map_err(EstimationError::LayoutError)?;

        let (det1, det2) = pld.rho_derivatives(&s_k_matrices, lambdas_slice);
        Ok((det1, det2))
    }

    /// Block-local penalty logdet derivatives using `CanonicalPenalty`.
    ///
    /// When all penalties are block-disjoint, the eigendecomposition factorizes
    /// per-block at O(block_p³) instead of O(p³). Falls back to the dense path
    /// when blocks overlap.
    pub(super) fn structural_penalty_logdet_derivatives_block_local(
        &self,
        lambdas: &Array1<f64>,
        ridge: f64,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        // Kronecker fast path: compute logdet derivatives directly from the
        // marginal eigenvalue grid.  O(d · ∏q_j) with no coordinate-frame
        // dependence — eigenvalues of Σ_k λ_k (I⊗...⊗S_k⊗...⊗I) are invariant
        // under orthogonal reparameterization, so this is correct regardless of
        // whether P-IRLS uses standard or factored Qs.
        if let Some(ref kron) = self.kronecker_penalty_system {
            let lambdas_slice = lambdas.as_slice().unwrap();
            let (_, det1, det2) = kron.logdet_and_derivatives(lambdas_slice, ridge);
            return Ok((det1, det2));
        }

        let k_count = self.canonical_penalties.len();
        if k_count == 0 || lambdas.len() != k_count {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        let lambdas_slice = lambdas.as_slice().unwrap();

        let cached_block_nullities = if ridge > 0.0 {
            Some(self.cached_penalty_block_structural_nullities()?)
        } else {
            None
        };
        let pld = PenaltyPseudologdet::from_penalties_with_cached_block_nullities(
            &self.canonical_penalties,
            lambdas_slice,
            ridge,
            self.p,
            cached_block_nullities.as_ref(),
        )
        .map_err(EstimationError::LayoutError)?;

        let (det1, det2) =
            pld.rho_derivatives_from_penalties(&self.canonical_penalties, lambdas_slice);
        Ok((det1, det2))
    }

    pub(super) fn compute_lamlhessian_exact_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array2<f64>, EstimationError> {
        let mode = super::unified::EvalMode::ValueGradientHessian;
        let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse(rho, bundle, mode)?
        } else {
            self.evaluate_unified(rho, bundle, mode)?
        };
        result
            .hessian
            .materialize_dense()
            .map_err(EstimationError::RemlOptimizationFailed)?
            .ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "Unified Hessian returned no analytic representation for VGH mode".into(),
                )
            })
    }

    pub(crate) fn compute_lamlhessian_consistent(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        let decision = self.selecthessian_strategy_policy(&bundle);
        match decision.strategy {
            super::inner_strategy::HessianEvalStrategyKind::SpectralExact => {
                self.compute_lamlhessian_exact_from_bundle(rho, &bundle)
            }
        }
    }

    pub(crate) fn compute_smoothing_correction_auto(
        &self,
        final_rho: &Array1<f64>,
        final_fit: &PirlsResult,
        base_covariance: Option<&Array2<f64>>,
        finalgrad_norm: f64,
    ) -> SmoothingCorrectionOutcome {
        use SmoothingCorrectionFallbackSeverity::{NumericalFailure, Routine};

        let first_order_routine = |correction: Option<Array2<f64>>, reason: &'static str| {
            SmoothingCorrectionOutcome::FirstOrder {
                correction,
                reason,
                severity: Routine,
            }
        };
        let first_order_numerical = |correction: Option<Array2<f64>>, reason: &'static str| {
            SmoothingCorrectionOutcome::FirstOrder {
                correction,
                reason,
                severity: NumericalFailure,
            }
        };

        // Always compute the fast first-order correction first.
        let first_order = super::compute_smoothing_correction(self, final_rho, final_fit);
        let first_order_correction = first_order.correction.clone();
        let n_rho = final_rho.len();
        if n_rho == 0 {
            // No hyperparameters: the unified corrected covariance equals H^{-1}.
            // Validate the unified path using the spectral operator.
            if let Some(base_cov) = base_covariance
                && let Ok(hop) = super::unified::DenseSpectralOperator::from_symmetric(base_cov)
            {
                let outer = Array2::<f64>::zeros((0, 0));
                let unified_diag =
                    super::unified::compute_corrected_covariance_diagonal(&[], &[], &outer, &hop);
                if let Ok(diag) = unified_diag {
                    let p = base_cov.nrows();
                    let max_dev = (0..p)
                        .map(|i| (base_cov[[i, i]] - diag[i]).abs())
                        .fold(0.0_f64, f64::max);
                    log::trace!(
                        "[corrected-cov] unified diagonal validation: max_dev={:.4e}",
                        max_dev,
                    );
                }
                let unified_full =
                    super::unified::compute_corrected_covariance(&[], &[], &outer, &hop);
                if let Ok(full) = unified_full {
                    log::trace!(
                        "[corrected-cov] unified full norm: {:.4e}",
                        full.iter().map(|v| v * v).sum::<f64>().sqrt(),
                    );
                }
            }
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "n_rho == 0: unified corrected covariance equals H^{-1}",
            ));
        }
        if n_rho > AUTO_CUBATURE_MAX_RHO_DIM {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "n_rho exceeds AUTO_CUBATURE_MAX_RHO_DIM: cubature cost prohibitive",
            ));
        }
        if final_fit.beta_transformed.len() > AUTO_CUBATURE_MAX_BETA_DIM {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "beta dimension exceeds AUTO_CUBATURE_MAX_BETA_DIM: cubature cost prohibitive",
            ));
        }
        let near_boundary = final_rho
            .iter()
            .any(|&v| (RHO_BOUND - v.abs()) <= AUTO_CUBATURE_BOUNDARY_MARGIN);
        let grad_norm = if finalgrad_norm.is_finite() {
            finalgrad_norm
        } else {
            0.0
        };
        // Scale-invariant "high gradient" certificate. The first-order
        // smoothing correction is the local linearization at ρ̂; cubature
        // upgrades it when the linearization is suspect (boundary contact, or
        // the outer gradient is genuinely large). An absolute ‖g‖>1e-3 gate
        // is wrong at every scale: biobank deviance ≈ 10⁵–10⁶ makes ‖g‖≈1
        // perfectly fine but trips the gate unconditionally, while tiny CI
        // problems with deviance ≈ 10–100 stay under 1e-3 even when actually
        // unconverged. Use the same `τ·(1+|F|)` rescaling the OUTER paths use
        // (BFGS / ARC / trust-region via `outer_scaled_tolerance`); deviance
        // is the dominant term in the REML cost at every scale and is the
        // natural cost proxy reachable from `PirlsResult`.
        const HIGHGRAD_REL_TOL: f64 = 1e-3;
        let cost_scale = 1.0 + final_fit.deviance.abs();
        let highgrad = grad_norm > HIGHGRAD_REL_TOL * cost_scale;
        if !near_boundary && !highgrad {
            // Keep the hot path cheap when the local linearization is likely sufficient.
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "linearization sufficient: not near boundary and outer gradient is small",
            ));
        }

        // If the first-order path used a rank-deficient pseudo-inverse, the
        // ρ-Hessian was indefinite or near-singular and the matrix-free ridged
        // inverse used below would silently impute spurious variance along the
        // dropped (unidentified) directions. Cubature sigma points propagated
        // through that spurious V_ρ would manufacture higher-order corrections
        // that are not supported by the data. The principled response is to
        // honor the rank deficiency: return the first-order correction (which
        // is already the correct rank-deficient inflation on the identified
        // subspace) and skip cubature entirely.
        if let Some(rank) = first_order.active_rank
            && rank < n_rho
        {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "first-order V_rho rank-deficient: cubature would impute spurious variance",
            ));
        }

        // Build V_rho from the outer Hessian around rho_hat.
        let mut hessian_rho = if let Some(h) = first_order.hessian_rho {
            h
        } else {
            match self.compute_lamlhessian_consistent(final_rho) {
                Ok(h) => h,
                Err(_) => {
                    return self.finalize_smoothing_outcome(first_order_numerical(
                        first_order_correction,
                        "rho Hessian compute_lamlhessian_consistent failed",
                    ));
                }
            }
        };
        enforce_symmetry(&mut hessian_rho);
        let ridge = AUTO_CUBATURE_HESSIAN_RIDGE_REL
            * hessian_rho
                .diag()
                .iter()
                .map(|&v| v.abs())
                .fold(0.0, f64::max)
                .max(AUTO_CUBATURE_HESSIAN_RIDGE_ABS);
        for i in 0..n_rho {
            hessian_rho[[i, i]] += ridge;
        }
        let Some(hessian_rho_inv) =
            matrix_inversewith_regularization(&hessian_rho, "auto cubature rho Hessian")
        else {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "rho Hessian inversion failed after ridge regularization",
            ));
        };

        let max_rhovar = hessian_rho_inv
            .diag()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !near_boundary && !highgrad && max_rhovar < AUTO_CUBATURE_RHOVAR_TRIGGER {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "post-inversion rho posterior variance below trigger threshold",
            ));
        }

        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let (evals, evecs) = match hessian_rho_inv.eigh(Side::Lower) {
            Ok(x) => x,
            Err(_) => {
                return self.finalize_smoothing_outcome(first_order_numerical(
                    first_order_correction,
                    "eigendecomposition of inverse rho-Hessian failed",
                ));
            }
        };
        let max_eval = evals
            .iter()
            .copied()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let eigenvalue_floor = max_eval * (n_rho.max(1) as f64) * f64::EPSILON;
        let mut eig_pairs: Vec<(usize, f64)> = evals
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite() && *v > eigenvalue_floor)
            .collect();
        if eig_pairs.is_empty() {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "inverse rho-Hessian has no positive eigenvalues above numerical floor",
            ));
        }
        eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let totalvar: f64 = eig_pairs.iter().map(|(_, v)| *v).sum();
        if !totalvar.is_finite() || totalvar <= 0.0 {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "positive-eigenvalue total mass non-finite or non-positive",
            ));
        }

        let mut rank = 0usize;
        let mut captured = 0.0_f64;
        for (_, eig) in eig_pairs
            .iter()
            .take(AUTO_CUBATURE_MAX_EIGENVECTORS.min(eig_pairs.len()))
        {
            captured += *eig;
            rank += 1;
            if captured / totalvar >= AUTO_CUBATURE_TARGET_VAR_FRAC {
                break;
            }
        }
        // `rank == 0` would require the truncation loop to not execute
        // despite a non-empty `eig_pairs`. The loop always runs at least
        // once when there is at least one positive eigenvalue, so this
        // branch is unreachable in practice. Treat as a
        // NumericalFailure guard rather than a routine fallback so any
        // future regression surfaces visibly.
        if rank == 0 {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "variance-truncation produced rank 0 (unreachable guard)",
            ));
        }

        let Some(base_cov) = base_covariance else {
            // Caller did not supply a base covariance to upgrade. This
            // is a configuration choice (the caller has nothing to add
            // the cubature correction onto), not a numerical failure;
            // the first-order delta is the documented outcome.
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "no base covariance supplied: nothing for cubature to upgrade",
            ));
        };
        let p = base_cov.nrows();
        let radius = (rank as f64).sqrt();
        let mut sigma_points: Vec<Array1<f64>> = Vec::with_capacity(2 * rank);
        for (eig_idx, eigval) in eig_pairs.iter().take(rank) {
            let axis = evecs.column(*eig_idx).to_owned();
            let scale = radius * eigval.sqrt();
            let delta = axis.mapv(|v| v * scale);

            let lo = -RHO_BOUND + AUTO_CUBATURE_RHO_CLAMP_INSET;
            let hi = RHO_BOUND - AUTO_CUBATURE_RHO_CLAMP_INSET;
            for sign in [1.0_f64, -1.0_f64] {
                let mut rho_point = final_rho.clone();
                rho_point
                    .iter_mut()
                    .zip(delta.iter())
                    .for_each(|(r, &d)| *r = (*r + sign * d).clamp(lo, hi));
                sigma_points.push(rho_point);
            }
        }
        // Unreachable: `rank >= 1` ensures at least two sigma points
        // (one positive, one negative) per eigenvector. Treat as a
        // NumericalFailure guard so any future regression surfaces.
        if sigma_points.is_empty() {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "empty sigma-point set (unreachable guard)",
            ));
        }

        // Evaluate sigma points in parallel via the stateless cubature PIRLS
        // entry. Unlike `execute_pirls_if_needed`, that callee performs no
        // PIRLS-cache lookup/insert, no warm-start read/write, no LM-lambda
        // hint read/write, no adaptive-cap or IFT-quality feedback writes —
        // so multiple sigma fits run concurrently without serializing on the
        // shared PIRLS-cache lock and without contaminating the production
        // outer trajectory's warm-start / LM / IFT state. This replaces the
        // previous `AtomicFlagGuard`-based opt-out: process-wide atomic
        // flips were a leaky proxy that still let writes through (e.g. the
        // adaptive-cap feedback and last_pirls_lm_lambda paths) and serialized
        // unrelated REML evaluations racing the cubature window.
        let point_results: Vec<Option<(Array2<f64>, Array1<f64>)>> = (0..sigma_points.len())
            .into_par_iter()
            .map(|idx| {
                let fit_point = self
                    .execute_pirls_stateless_for_cubature(&sigma_points[idx])
                    .ok()?;
                let h_point = map_hessian_to_original_basis(fit_point.as_ref()).ok()?;
                let cov_point = matrix_inversewith_regularization(&h_point, "auto cubature point")?;
                let beta_point = fit_point
                    .reparam_result
                    .qs
                    .dot(fit_point.beta_transformed.as_ref());
                Some((cov_point, beta_point))
            })
            .collect();

        if point_results.iter().any(|r| r.is_none()) {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "one or more sigma-point inner PIRLS fits failed",
            ));
        }

        let point_pairs: Vec<(Array2<f64>, Array1<f64>)> =
            point_results.into_iter().flatten().collect();
        let mut total_cov = accumulate_sigma_cubature_total_covariance(&point_pairs, p);
        if !total_cov.iter().all(|v| v.is_finite()) {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "assembled total covariance contains non-finite entries",
            ));
        }
        enforce_symmetry(&mut total_cov);

        let mut corr = total_cov - base_cov;
        enforce_symmetry(&mut corr);

        self.finalize_smoothing_outcome(SmoothingCorrectionOutcome::Cubature {
            correction: corr,
            rank,
            n_points: sigma_points.len(),
            near_boundary,
            grad_norm,
            max_rho_var: max_rhovar,
        })
    }

    /// Emit the canonical `[smoothing-correction]` log line, update the
    /// process-wide counters, and return the outcome unchanged.
    fn finalize_smoothing_outcome(
        &self,
        outcome: SmoothingCorrectionOutcome,
    ) -> SmoothingCorrectionOutcome {
        let branch_label = outcome.branch_label();
        match &outcome {
            SmoothingCorrectionOutcome::Cubature {
                rank,
                n_points,
                near_boundary,
                grad_norm,
                max_rho_var,
                ..
            } => {
                SMOOTHING_CORRECTION_CUBATURE_COUNT.fetch_add(1, Ordering::Relaxed);
                log::info!(
                    "[smoothing-correction] branch={} rank={} points={} near_boundary={} \
                     grad_norm={:.3e} max_rho_var={:.3e}",
                    branch_label,
                    rank,
                    n_points,
                    near_boundary,
                    grad_norm,
                    max_rho_var,
                );
            }
            SmoothingCorrectionOutcome::FirstOrder {
                reason,
                severity,
                correction,
            } => {
                let has_matrix = correction.is_some();
                match severity {
                    SmoothingCorrectionFallbackSeverity::Routine => {
                        log::info!(
                            "[smoothing-correction] branch=first-order severity=routine \
                             has_matrix={} reason=\"{}\"",
                            has_matrix,
                            reason
                        );
                    }
                    SmoothingCorrectionFallbackSeverity::NumericalFailure => {
                        SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT
                            .fetch_add(1, Ordering::Relaxed);
                        log::warn!(
                            "[smoothing-correction] branch=first-order severity=numerical-failure \
                             has_matrix={} reason=\"{}\" failure_count={}",
                            has_matrix,
                            reason,
                            SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.load(Ordering::Relaxed),
                        );
                    }
                }
            }
        }
        outcome
    }
}

#[cfg(test)]
mod sigma_cubature_accumulation_tests {
    //! Math-spec validation tests for the sigma-cubature accumulation
    //! formula (Block 6 validation test #1, "cubature linear exactness").
    //!
    //! These tests pin the math of
    //! [`accumulate_sigma_cubature_total_covariance`] independently of the
    //! execution model that produced the per-sigma `(A_m, b_m)` pairs.
    //! Pulled out so the same parity oracle covers both the CPU Rayon
    //! sigma loop and any future GPU stream-pool sigma executor.
    use super::accumulate_sigma_cubature_total_covariance;
    use ndarray::{Array1, Array2};

    /// Cubature linear exactness: if `b_m = b_0 + J·(ρ_m − ρ̂)` is linear
    /// in `ρ` and `A_m = A_0` is constant, the cubature output must equal
    /// `A_0 + J · V_ρ,r · J^T` exactly, where `V_ρ,r` is the empirical
    /// covariance of the sigma points themselves (equal-weighted, by the
    /// usual 2r-point symmetric rule with `M = 2r` and weights `1/M`).
    ///
    /// This is the conservation law the cubature formula was designed to
    /// satisfy; any drift away from it is a math bug, not a numerics
    /// issue, so the tolerance is at f64 round-off (1e-12 relative).
    #[test]
    fn cubature_linear_exactness_recovers_jvjt() {
        // Pick a non-trivial (p, d_ρ, r) shape: p=4 outputs, d_ρ=3 inputs,
        // r=3 retained eigendirections → 2r = 6 sigma points. Use a
        // hand-built `V_ρ,r` with three distinct eigenvalues so the test
        // genuinely exercises off-diagonal covariance entries.
        let p = 4;
        let d_rho = 3;
        let r = 3;
        let m_points = 2 * r;

        // Hand-picked eigendecomposition of V_ρ,r: orthonormal U from
        // QR of a simple block, diagonal eigenvalues d. Sigma points:
        // ρ_m − ρ̂ = ±√(r · d_j) · u_j for j = 0..r and sign ∈ {+,−}, so
        // the empirical covariance under equal weights 1/M equals V_ρ,r.
        let eigenvalues = [0.25_f64, 0.49, 0.81];
        // Use a simple orthonormal matrix (a 3×3 Householder-like
        // construction) for U so the test does not depend on any RNG.
        // U columns are the eigenvectors of V_ρ,r.
        let u: Array2<f64> = ndarray::array![
            [1.0 / 3f64.sqrt(), 1.0 / 2f64.sqrt(), 1.0 / 6f64.sqrt()],
            [1.0 / 3f64.sqrt(), -1.0 / 2f64.sqrt(), 1.0 / 6f64.sqrt()],
            [1.0 / 3f64.sqrt(), 0.0, -2.0 / 6f64.sqrt()],
        ];
        // sanity: U is orthonormal
        let ut_u = u.t().dot(&u);
        for i in 0..d_rho {
            for j in 0..d_rho {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ut_u[[i, j]] - want).abs() < 1e-12,
                    "U is not orthonormal at ({i},{j}): got {} expected {}",
                    ut_u[[i, j]],
                    want,
                );
            }
        }

        // V_ρ,r = U · diag(d) · Uᵀ
        let mut v_rho_r = Array2::<f64>::zeros((d_rho, d_rho));
        for k in 0..d_rho {
            let col = u.column(k);
            let scaled = col.mapv(|v| v * eigenvalues[k]);
            for i in 0..d_rho {
                for j in 0..d_rho {
                    v_rho_r[[i, j]] += scaled[i] * col[j];
                }
            }
        }

        // Build the 2r sigma displacements: ρ_m − ρ̂ = ±√(r · d_j) · u_j.
        // Equal weights 1/M and the symmetric ± pairing make the
        // empirical mean zero and the empirical second-moment matrix
        // sum to V_ρ,r exactly.
        let mut sigma_displacements: Vec<Array1<f64>> = Vec::with_capacity(m_points);
        for k in 0..r {
            let scale = (r as f64 * eigenvalues[k]).sqrt();
            let axis = u.column(k).to_owned();
            for sign in [1.0_f64, -1.0_f64] {
                sigma_displacements.push(axis.mapv(|v| v * sign * scale));
            }
        }

        // Pick a non-degenerate (p × d_ρ) Jacobian J and constants b_0,
        // A_0. Use plain integers so the synthetic test data is
        // exactly representable in f64.
        let b0: Array1<f64> = ndarray::array![1.0, -2.0, 3.5, 0.5];
        let jacobian: Array2<f64> = ndarray::array![
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        // A_0 is SPD by construction so the cubature output is a real
        // covariance matrix; pick a simple diagonal + small off-diagonal
        // structure so the assertion is not vacuous on the A_0 term.
        let mut a0 = Array2::<f64>::eye(p);
        a0[[0, 1]] = 0.25;
        a0[[1, 0]] = 0.25;
        a0[[2, 3]] = -0.10;
        a0[[3, 2]] = -0.10;

        // Synthesize per-sigma (A_m, b_m) with A_m = A_0 (constant),
        // b_m = b_0 + J · (ρ_m − ρ̂) (linear).
        let points: Vec<(Array2<f64>, Array1<f64>)> = sigma_displacements
            .iter()
            .map(|drho| {
                let bm = &b0 + &jacobian.dot(drho);
                (a0.clone(), bm)
            })
            .collect();

        // Expected: V̂_p = A_0 + J · V_ρ,r · Jᵀ. Symmetric by
        // construction, so no enforce_symmetry needed for the oracle.
        let jvjt = jacobian.dot(&v_rho_r).dot(&jacobian.t());
        let expected = &a0 + &jvjt;

        let actual = accumulate_sigma_cubature_total_covariance(&points, p);

        // f64 round-off bound: every entry is a sum of <= 32 products
        // of single-digit magnitudes, so 1e-12 relative is very safe.
        let mut max_rel_dev = 0.0_f64;
        let mut max_abs_dev = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                let diff = (actual[[i, j]] - expected[[i, j]]).abs();
                let denom = expected[[i, j]].abs().max(1.0);
                max_rel_dev = max_rel_dev.max(diff / denom);
                max_abs_dev = max_abs_dev.max(diff);
            }
        }
        assert!(
            max_rel_dev < 1e-12,
            "cubature linear-exactness violation: max_rel_dev={:.3e}, max_abs_dev={:.3e}",
            max_rel_dev,
            max_abs_dev,
        );
    }

    /// Degenerate sanity: a single sigma point with `M = 1` collapses
    /// `var_beta` to zero (the symmetric ± pairing degenerates), so
    /// the cubature output equals exactly `A_0`. Guards the formula
    /// against a stray off-by-one in the variance subtraction.
    #[test]
    fn cubature_single_point_collapses_to_a0() {
        let p = 3;
        let a0: Array2<f64> = ndarray::array![[2.0, 0.5, 0.0], [0.5, 1.5, 0.25], [0.0, 0.25, 1.0]];
        let b0: Array1<f64> = ndarray::array![0.1, -0.2, 0.3];
        let points = vec![(a0.clone(), b0.clone())];
        let actual = accumulate_sigma_cubature_total_covariance(&points, p);
        // With M=1: mean_beta = b0, second_beta = b0 b0ᵀ,
        //          var_beta = b0 b0ᵀ - b0 b0ᵀ = 0,
        //          mean_hinv = a0  ⇒ total = a0.
        for i in 0..p {
            for j in 0..p {
                let diff = (actual[[i, j]] - a0[[i, j]]).abs();
                assert!(
                    diff < 1e-14,
                    "single-point cubature did not collapse to A_0 at ({i},{j}): \
                     actual={}, expected={}, diff={:.3e}",
                    actual[[i, j]],
                    a0[[i, j]],
                    diff,
                );
            }
        }
    }
}

#[cfg(test)]
mod smoothing_correction_outcome_tests {
    //! Unit tests for the structured [`SmoothingCorrectionOutcome`] type
    //! introduced by issue #201. These tests cover variant
    //! classification helpers, the routine-vs-numerical-failure
    //! severity distinction, that `None` correction is only possible
    //! in `FirstOrder` outcomes, and that the failure-reason strings
    //! used in the function body are non-empty and distinct (a
    //! tripwire so future refactors cannot silently lose a
    //! classification). End-to-end tests of the fallback paths inside
    //! `compute_smoothing_correction_auto` live with the broader REML
    //! integration suite; the tests here are the targeted local
    //! coverage of the new structured-return contract.
    use super::*;
    use ndarray::array;
    use std::sync::atomic::Ordering;

    fn make_first_order(
        reason: &'static str,
        severity: SmoothingCorrectionFallbackSeverity,
        with_matrix: bool,
    ) -> SmoothingCorrectionOutcome {
        let correction = if with_matrix {
            Some(array![[1.0, 0.0], [0.0, 1.0]])
        } else {
            None
        };
        SmoothingCorrectionOutcome::FirstOrder {
            correction,
            reason,
            severity,
        }
    }

    #[test]
    fn cubature_branch_label_and_extraction() {
        let outcome = SmoothingCorrectionOutcome::Cubature {
            correction: array![[2.0, 0.0], [0.0, 2.0]],
            rank: 2,
            n_points: 4,
            near_boundary: true,
            grad_norm: 1.5,
            max_rho_var: 0.7,
        };
        assert_eq!(outcome.branch_label(), "cubature");
        let mat = outcome
            .into_correction()
            .expect("cubature always has a matrix");
        assert_eq!(mat.dim(), (2, 2));
        assert_eq!(mat[[0, 0]], 2.0);
    }

    #[test]
    fn first_order_routine_branch_label_and_extraction() {
        let outcome = make_first_order(
            "n_rho == 0",
            SmoothingCorrectionFallbackSeverity::Routine,
            true,
        );
        assert_eq!(outcome.branch_label(), "first-order (routine)");
        assert!(outcome.into_correction().is_some());
    }

    #[test]
    fn first_order_numerical_branch_label_and_extraction() {
        let outcome = make_first_order(
            "rho Hessian inversion failed after ridge regularization",
            SmoothingCorrectionFallbackSeverity::NumericalFailure,
            true,
        );
        assert_eq!(outcome.branch_label(), "first-order (numerical failure)");
        assert!(outcome.into_correction().is_some());
    }

    #[test]
    fn first_order_without_matrix_returns_none() {
        let outcome = make_first_order(
            "no base covariance supplied",
            SmoothingCorrectionFallbackSeverity::Routine,
            false,
        );
        assert!(outcome.into_correction().is_none());
    }

    #[test]
    fn severity_counter_is_monotonic() {
        let before = SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.load(Ordering::Relaxed);
        SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.fetch_add(1, Ordering::Relaxed);
        let after = SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.load(Ordering::Relaxed);
        assert!(
            after > before,
            "numerical-failure counter must be monotonic ({} -> {})",
            before,
            after
        );
    }

    #[test]
    fn cubature_counter_is_observable() {
        let before = SMOOTHING_CORRECTION_CUBATURE_COUNT.load(Ordering::Relaxed);
        SMOOTHING_CORRECTION_CUBATURE_COUNT.fetch_add(1, Ordering::Relaxed);
        let after = SMOOTHING_CORRECTION_CUBATURE_COUNT.load(Ordering::Relaxed);
        assert!(after > before);
    }

    #[test]
    fn classification_reason_strings_are_nonempty_and_distinct() {
        let reasons = [
            // Routine gates.
            "n_rho == 0: unified corrected covariance equals H^{-1}",
            "n_rho exceeds AUTO_CUBATURE_MAX_RHO_DIM: cubature cost prohibitive",
            "beta dimension exceeds AUTO_CUBATURE_MAX_BETA_DIM: cubature cost prohibitive",
            "linearization sufficient: not near boundary and outer gradient is small",
            "first-order V_rho rank-deficient: cubature would impute spurious variance",
            "post-inversion rho posterior variance below trigger threshold",
            "no base covariance supplied: nothing for cubature to upgrade",
            // Numerical failures.
            "rho Hessian compute_lamlhessian_consistent failed",
            "rho Hessian inversion failed after ridge regularization",
            "eigendecomposition of inverse rho-Hessian failed",
            "inverse rho-Hessian has no positive eigenvalues above numerical floor",
            "positive-eigenvalue total mass non-finite or non-positive",
            "variance-truncation produced rank 0 (unreachable guard)",
            "empty sigma-point set (unreachable guard)",
            "one or more sigma-point inner PIRLS fits failed",
            "assembled total covariance contains non-finite entries",
        ];
        for r in reasons.iter() {
            assert!(!r.is_empty(), "classification reason must not be empty");
            let routine = make_first_order(r, SmoothingCorrectionFallbackSeverity::Routine, true);
            let numerical = make_first_order(
                r,
                SmoothingCorrectionFallbackSeverity::NumericalFailure,
                true,
            );
            assert_eq!(routine.branch_label(), "first-order (routine)");
            assert_eq!(numerical.branch_label(), "first-order (numerical failure)");
        }

        let mut sorted: Vec<&'static str> = reasons.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            reasons.len(),
            "classification reasons must be distinct so callers can disambiguate"
        );
    }
}
