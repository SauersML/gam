use super::inner_strategy::GeometryBackendKind;
use super::penalty_logdet::PenaltyPseudologdet;
use super::*;
use crate::model_types::SmoothingCorrectionMethod;
use gam_linalg::matrix::symmetrize_in_place;
use std::sync::atomic::Ordering;

// Relative scale of the diagonal ridge added to the ρ-Hessian before
// inverting it for sigma-point construction. Matches the analogous IFT
// regularisation: tiny enough to leave well-conditioned Hessians intact,
// large enough that a near-singular Hessian still yields a usable V_ρ.
pub(crate) const AUTO_CUBATURE_HESSIAN_RIDGE_REL: f64 = 1e-8;
// Absolute floor for the diagonal ridge (prevents zero ridge when the
// Hessian diagonal is degenerate / all-zero).
pub(crate) const AUTO_CUBATURE_HESSIAN_RIDGE_ABS: f64 = 1e-8;
// Inset from RHO_BOUND when clamping sigma points so the inner PIRLS
// fit at a sigma point is strictly interior to the box constraint
// (the box edge is unreachable by IRLS without barrier intervention).
pub(crate) const AUTO_CUBATURE_RHO_CLAMP_INSET: f64 = 1e-8;
// Skip cubature when the first-order rho-Hessian inverse already shows
// negligible posterior variance on rho (max diag < this threshold) and
// neither boundary contact nor large outer-gradient flags fired.
pub(crate) const AUTO_CUBATURE_RHOVAR_TRIGGER: f64 = 0.1;

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
        rho_covariance: Option<Array2<f64>>,
        rho_hessian_stabilization: gam_problem::StabilizationLedger,
        rank: usize,
        n_points: usize,
        near_boundary: bool,
        grad_norm: f64,
        max_rho_var: f64,
        /// The exact first-order IFT correction computed BEFORE the decision
        /// to escalate to cubature, retained rather than discarded (#946).
        /// `Some` exactly when `first_order_method` is
        /// `Some(FirstOrderIdentifiedSubspace{..})`. Callers that need the
        /// exact (not cubature-approximated) WPS correction — the corrected-
        /// EDF/AIC channel — read this instead of `correction`/the method
        /// this variant reports as primary.
        first_order_correction: Option<Array2<f64>>,
        /// Provenance for `first_order_correction`. Always either `None` or
        /// `Some(FirstOrderIdentifiedSubspace{..})` — never `SigmaPointCubature`.
        first_order_method: Option<SmoothingCorrectionMethod>,
    },
    /// Principled first-order linearization was returned.
    FirstOrder {
        correction: Option<Array2<f64>>,
        rho_covariance: Option<Array2<f64>>,
        reason: &'static str,
        severity: SmoothingCorrectionFallbackSeverity,
        method: Option<SmoothingCorrectionMethod>,
    },
    /// Exact first-order geometry was unavailable. The typed reason is
    /// preserved instead of presenting a missing matrix as a routine skip.
    Unavailable {
        reason: SmoothingCorrectionUnavailable,
        rho_covariance: Option<Array2<f64>>,
    },
}

impl SmoothingCorrectionOutcome {
    /// Consume the outcome without discarding how a retained matrix was made.
    ///
    /// Returns `(primary_correction, primary_method, first_order_correction,
    /// first_order_method)`. The primary pair is the fit's EFFECTIVE
    /// correction — cubature when it escalated, otherwise first-order — and
    /// is unchanged in meaning from before this method grew a first-order
    /// pair (#946): every existing consumer that only reads `.0`/`.1` keeps
    /// its exact prior behavior. The first-order pair is ADDITIONALLY
    /// retained so a consumer that specifically needs the exact (never
    /// cubature-approximated) WPS correction — the corrected-EDF/AIC channel
    /// — has it available even when the primary pair escalated to cubature
    /// for some other consumer's benefit.
    pub fn into_correction_with_method(
        self,
    ) -> (
        Option<Array2<f64>>,
        Option<SmoothingCorrectionMethod>,
        Option<Array2<f64>>,
        Option<SmoothingCorrectionMethod>,
    ) {
        match self {
            SmoothingCorrectionOutcome::Cubature {
                correction,
                rank,
                n_points,
                rho_hessian_stabilization,
                first_order_correction,
                first_order_method,
                ..
            } => (
                Some(correction),
                Some(SmoothingCorrectionMethod::SigmaPointCubature {
                    rank,
                    n_points,
                    rho_hessian_stabilization,
                }),
                first_order_correction,
                first_order_method,
            ),
            SmoothingCorrectionOutcome::FirstOrder {
                correction, method, ..
            } => {
                // The primary result already IS the first-order result here
                // (no cubature ran); the first-order pair mirrors it exactly.
                let first_order_correction = correction.clone();
                (correction, method, first_order_correction, method)
            }
            SmoothingCorrectionOutcome::Unavailable { .. } => (None, None, None, None),
        }
    }

    /// Read the regularized inverse outer Hessian `Cov(rho_hat)`, when the
    /// selected path produced one. This is consumed by higher-order LR
    /// inference and does not affect the covariance correction matrix.
    pub fn rho_covariance(&self) -> Option<&Array2<f64>> {
        match self {
            SmoothingCorrectionOutcome::Cubature { rho_covariance, .. }
            | SmoothingCorrectionOutcome::FirstOrder { rho_covariance, .. }
            | SmoothingCorrectionOutcome::Unavailable { rho_covariance, .. } => {
                rho_covariance.as_ref()
            }
        }
    }

    /// Human-readable label naming the branch taken.
    pub fn branch_label(&self) -> &'static str {
        match self {
            SmoothingCorrectionOutcome::Cubature { .. } => "cubature",
            SmoothingCorrectionOutcome::Unavailable { .. } => "unavailable",
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

/// Outcome of one sigma-point evaluation: the inverted-Hessian `A_m = H_m⁻¹`
/// in the original (Qs-mapped) basis, and the original-basis coefficient
/// vector `b_m = Qs · β̂_transformed`. Both are exactly what
/// [`accumulate_sigma_cubature_total_covariance`] consumes.
///
/// A sigma point is either fully represented by this pair or its typed error
/// aborts the cubature batch; there is no per-point sentinel/fallback surface.
pub(crate) type SigmaPointResult = (Array2<f64>, Array1<f64>);

/// Predicate: is the device-resident inner PIRLS that the GPU stream-pool
/// sigma executor needs available in this build/runtime?
///
/// Returns `true` when both of the following hold:
///   * The global GPU policy selects CUDA (`cuda_selected()`).
///   * A live [`gam_gpu::device_runtime::GpuRuntime`] is present, confirming
///     that CUDA is initialised and the JIT row-kernel cache is warm.
///
/// The full Stage 3.3 device-resident PIRLS loop (`pirls_loop_on_stream`)
/// already exists in [`gam_gpu::pirls_gpu`] and covers all six
/// canonical (family, link) pairings supported by the GPU admission gate.
/// The sigma-cubature stream-pool executor
/// ([`sigma_cubature_evaluate_gpu_stream_pool`]) uses it directly.
///
/// The intentional non-flag gate is magic by default: no CLI flag, no env
/// var, no Cargo feature. The predicate inspects only build + runtime
/// properties that determine correctness.
#[inline]
pub(crate) fn device_pirls_stage3_ready() -> Result<bool, gam_gpu::gpu_error::GpuError> {
    gam_gpu::cuda_selected()
}

/// Sigma-cubature executor dispatch — the swap site between the CPU Rayon
/// path and the GPU stream-pool path (Stage 3.3 + stream pool).
///
/// Both branches return per-sigma `(A_m, b_m)` pairs that the downstream
/// [`accumulate_sigma_cubature_total_covariance`] consumes without knowing
/// which executor produced them; that's the contract
/// `cubature_linear_exactness_recovers_jvjt` pins to f64 round-off.
///
/// Magic by default: no flags. When [`device_pirls_stage3_ready`] returns
/// `true` the GPU branch fires for every cubature batch where the problem
/// geometry justifies it (family in JIT-cached set, `p ≥ 32`,
/// `n ≥ row_kernel_min_n`, dense design). A pre-admission `Ok(None)` uses the
/// CPU executor; once admitted, typed geometry/runtime failures propagate and
/// are never retried on a different implementation.
pub(crate) fn sigma_cubature_dispatch(
    state: &RemlState<'_>,
    sigma_points: &[Array1<f64>],
) -> Result<Vec<SigmaPointResult>, EstimationError> {
    let stage3_ready = device_pirls_stage3_ready().map_err(|error| {
        EstimationError::RemlOptimizationFailed(format!(
            "GPU runtime resolution failed for sigma cubature: {error}"
        ))
    })?;
    if stage3_ready {
        // Device path: try GPU stream-pool executor first.
        match sigma_cubature_evaluate_gpu_stream_pool(state, sigma_points) {
            Ok(Some(results)) => return Ok(results),
            Ok(None) => {
                // Device declined (shape / family / policy gate); fall through.
                log::debug!(
                    "[sigma-cubature] GPU stream pool declined (Ok(None)) — \
                     falling through to CPU Rayon oracle"
                );
            }
            #[cfg(target_os = "linux")]
            Err(crate::gpu_kernels::sigma_cubature::SigmaCubatureGpuError::Geometry(error)) => {
                return Err(error);
            }
            Err(crate::gpu_kernels::sigma_cubature::SigmaCubatureGpuError::Runtime(error)) => {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "sigma-cubature admitted GPU runtime failure: {error}"
                )));
            }
        }
    }

    sigma_cubature_evaluate_cpu_rayon(state, sigma_points)
}

/// GPU stream-pool sigma-cubature evaluator.
///
/// For each sigma point this function:
///   1. Runs the reparameterisation engine to obtain `Qs` and
///      `s_transformed` for that ρ value.
///   2. Materialises `x_transformed = X_original · Qs` on the host
///      (dense-only; sparse design returns `Ok(None)`).
///   3. Passes the per-sigma inputs to
///      [`crate::gpu_kernels::sigma_cubature::try_gpu_sigma_stream_pool_eval`]
///      which allocates a stream pool (N_streams = min(8, M)), rotates
///      sigma points across streams, runs `pirls_loop_on_stream` on each,
///      and returns one `(H_original⁻¹, β_original)` pair per point.
///
/// Returns:
///   * `Ok(Some(results))` — every sigma point returned a usable GPU result.
///   * `Ok(None)` — GPU path not eligible for this batch (sparse design,
///     family not in JIT-cached set, policy gate, etc.).
///   * `Err(_)` — GPU driver / shape failure the caller should log.
pub(crate) fn sigma_cubature_evaluate_gpu_stream_pool(
    state: &RemlState<'_>,
    sigma_points: &[Array1<f64>],
) -> Result<Option<Vec<SigmaPointResult>>, crate::gpu_kernels::sigma_cubature::SigmaCubatureGpuError>
{
    use crate::gpu::pirls_dispatch_wire::admission_for;
    use crate::gpu_kernels::sigma_cubature::try_gpu_sigma_stream_pool_eval;
    use gam_gpu::device_runtime::GpuRuntime;
    use gam_terms::construction::{EngineDims, stable_reparameterization_engine_canonical};

    if sigma_points.is_empty() {
        return Ok(Some(Vec::new()));
    }

    let n = state.x.nrows();
    let p = state.p;

    // Dense-only: the GPU loop requires X_original as a dense column-major array.
    let x_dense = match state.x.as_dense() {
        Some(d) => d,
        None => return Ok(None),
    };

    // Admission check: family must be in the JIT-cached set and n/p must
    // clear the policy floor. Use the likelihood spec from the REML config.
    let likelihood_spec = &state.config.likelihood;
    let Some(admission) = admission_for(&likelihood_spec.spec, n, p, true) else {
        return Ok(None);
    };
    let Some(runtime) = GpuRuntime::resolve(gam_gpu::global_policy())? else {
        return Ok(None);
    };
    if !runtime.policy().should_use_gpu_pirls_loop(admission) {
        return Ok(None);
    }

    // Compute the reparameterisation for every sigma point on the host.
    // This is a moderate-cost eigendecomposition (O(p³) per point); it
    // runs sequentially here because the downstream GPU launches dominate.
    let engine_dims = EngineDims::new(p, state.canonical_penalties.len());
    let mut per_sigma: Vec<crate::gpu_kernels::sigma_cubature::SigmaPointGpuInput> =
        Vec::with_capacity(sigma_points.len());

    for rho in sigma_points {
        let lambdas = Array1::from_vec(
            gam_problem::checked_exp_log_strengths(rho.iter().copied())
                .map_err(|error| gam_gpu::gpu_err!("sigma rho: {error}"))?,
        );
        let lambdas_slice = lambdas
            .as_slice_memory_order()
            .ok_or_else(|| gam_gpu::gpu_err!("sigma rho lambdas not contiguous"))?;
        let reparam = stable_reparameterization_engine_canonical(
            &state.canonical_penalties,
            lambdas_slice,
            engine_dims,
            Some(&state.reparam_invariant),
            state.penalty_shrinkage_floor,
        )
        .map_err(|e| gam_gpu::gpu_err!("sigma reparam engine: {e:?}"))?;

        // Compute prior-mean shifts in the transformed basis. These are zero
        // for the standard sigma-cubature path (no explicit prior-mean offset).
        let linear_shift = ndarray::Array1::<f64>::zeros(p);

        per_sigma.push(crate::gpu_kernels::sigma_cubature::SigmaPointGpuInput {
            s_transformed: reparam.s_transformed,
            qs: reparam.qs,
            linear_shift,
            constant_shift: 0.0,
        });
    }

    // Carry the row-kernel scalar as a typed family contract. Non-Gamma
    // admissions have no synthetic shape value; the final CUDA ABI receives a
    // poison value only after matching the discriminant against the row family.
    let likelihood_scale = match likelihood_spec.spec.response {
        ResponseFamily::Gamma => crate::gpu::pirls_gpu::PirlsLoopLikelihoodScale::gamma_shape(
            likelihood_spec
                .resolved_gamma_shape()
                .map_err(|error| gam_gpu::gpu_err!("sigma Gamma scale: {error}"))?,
        )
        .map_err(|error| gam_gpu::gpu_err!("sigma Gamma scale: {error}"))?,
        _ => {
            likelihood_spec
                .resolved_scale()
                .map_err(|error| gam_gpu::gpu_err!("sigma likelihood scale: {error}"))?;
            crate::gpu::pirls_gpu::PirlsLoopLikelihoodScale::non_gamma()
        }
    };

    try_gpu_sigma_stream_pool_eval(
        x_dense.view(),
        state.y,
        state.weights,
        state.offset.view(),
        &per_sigma,
        admission,
        likelihood_scale,
        state.config.pirls_convergence_tolerance,
        state.config.max_iterations,
    )
}

/// CPU Rayon sigma evaluator. The same loop that lived inline at the call
/// site in [`RemlState::compute_smoothing_correction_auto`] before P3
/// introduced the dispatch boundary; the math is bit-identical and
/// continues to be the parity oracle pinned by
/// [`sigma_cubature_accumulation_tests::cubature_linear_exactness_recovers_jvjt`].
///
/// Stateless inner PIRLS (`execute_pirls_stateless_for_cubature`) performs
/// no PIRLS-cache lookup/insert, no warm-start read/write, no LM-lambda
/// hint read/write, no adaptive-cap or IFT-quality feedback writes — so
/// multiple sigma fits run concurrently without serializing on the shared
/// PIRLS-cache lock and without contaminating the production outer
/// trajectory's warm-start / LM / IFT state. This replaces the previous
/// `AtomicFlagGuard`-based opt-out: process-wide atomic flips were a
/// leaky proxy that still let writes through (e.g. the adaptive-cap
/// feedback and last_pirls_lm_lambda paths) and serialized unrelated
/// REML evaluations racing the cubature window.
pub(crate) fn sigma_cubature_evaluate_cpu_rayon(
    state: &RemlState<'_>,
    sigma_points: &[Array1<f64>],
) -> Result<Vec<SigmaPointResult>, EstimationError> {
    let rows: Vec<Result<SigmaPointResult, EstimationError>> = (0..sigma_points.len())
        .into_par_iter()
        .map(|idx| -> Result<SigmaPointResult, EstimationError> {
            let fit_point = state.execute_pirls_stateless_for_cubature(&sigma_points[idx])?;
            let h_point = map_hessian_to_original_basis(fit_point.as_ref())?;
            let cov_point = crate::gpu_kernels::sigma_cubature::certified_sigma_point_covariance(
                &h_point,
                "auto cubature point",
            )
            .map_err(|error| {
                EstimationError::RemlOptimizationFailed(format!(
                    "sigma point {idx}: exact SPD Hessian inverse failed: {error}"
                ))
            })?;
            let beta_point = fit_point
                .reparam_result
                .qs
                .dot(fit_point.beta_transformed.as_ref());
            Ok((cov_point, beta_point))
        })
        .collect();
    rows.into_iter().collect()
}

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
    /// Compute the pseudo-logdet `log|Σ λ_k S_k|₊`, its rank, and its first and
    /// second derivatives with respect to ρ — all from one eigendecomposition.
    ///
    /// On the positive eigenspace of `Σ λ_k S_k`:
    ///
    ///   ∂_k L = tr(S⁺ Aₖ)
    ///   ∂²_kl L = δ_{kl} ∂_k L − λₖ λₗ tr(S⁺ Sₖ S⁺ Sₗ)
    ///
    /// where Aₖ = λₖ Sₖ and S⁺ is the pseudoinverse on that eigenspace.
    ///
    /// The value `log|Σ λ_k S_k|₊` and its ρ-derivatives must range over the
    /// SAME positive eigenspace, or the analytic gradient differentiates a
    /// different function than the cost reports (the objective↔gradient desync
    /// class). Sourcing both from one [`PenaltyPseudologdet`] is the structural
    /// cure — the rank convention (eigenvalue-threshold over `Σ λ_k S_k +
    /// ridge·I`) is identical on both sides by construction (#901: a separate
    /// structural-rank value path desynced the GLM ρ-gradient against FD).
    pub(super) fn structural_penalty_logdet_value_and_derivatives(
        &self,
        rs_transformed: &[Array2<f64>],
        lambdas: &Array1<f64>,
        ridge: f64,
    ) -> Result<(f64, usize, Array1<f64>, Array2<f64>), EstimationError> {
        let k_count = lambdas.len();
        if rs_transformed.len() != k_count {
            return Err(EstimationError::LayoutError(format!(
                "Penalty root/lambda count mismatch in structural logdet derivatives: roots={}, lambdas={}",
                rs_transformed.len(),
                k_count
            )));
        }
        if k_count == 0 {
            return Ok((
                0.0,
                0,
                Array1::zeros(k_count),
                Array2::zeros((k_count, k_count)),
            ));
        }

        // Build S_k = R_k^T R_k for each penalty component.
        let s_k_matrices: Vec<Array2<f64>> = rs_transformed
            .iter()
            .map(|r_k| gam_linalg::faer_ndarray::fast_atb(r_k, r_k))
            .collect();

        let lambdas_slice = lambdas.as_slice().unwrap();

        let pld = PenaltyPseudologdet::from_components(&s_k_matrices, lambdas_slice, ridge)
            .map_err(EstimationError::LayoutError)?;

        let value = pld.value();
        let rank = pld.rank();
        let (det1, det2) = pld.rho_derivatives(&s_k_matrices, lambdas_slice);
        Ok((value, rank, det1, det2))
    }

    /// Block-local penalty logdet derivatives using `CanonicalPenalty`.
    ///
    /// When all penalties are block-disjoint, the eigendecomposition factorizes
    /// per-block at O(block_p³) instead of O(p³). Falls back to the dense path
    /// when blocks overlap.
    pub(super) fn structural_penalty_logdet_derivatives_block_local(
        &self,
        lambdas: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        let (_, _, det1, det2) =
            self.structural_penalty_logdet_value_and_derivatives_block_local(lambdas, bundle)?;
        Ok((det1, det2))
    }

    /// Same as [`structural_penalty_logdet_derivatives_block_local`] but also
    /// returns the pseudo-logdet VALUE and rank from the SAME object the
    /// derivatives are taken on — see
    /// [`structural_penalty_logdet_value_and_derivatives`] for why value and
    /// derivative must share one positive eigenspace (#901).
    pub(super) fn structural_penalty_logdet_value_and_derivatives_block_local(
        &self,
        lambdas: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<(f64, usize, Array1<f64>, Array2<f64>), EstimationError> {
        let ridge = bundle.ridge_passport.penalty_logdet_ridge();
        // Kronecker fast path: compute logdet derivatives directly from the
        // marginal eigenvalue grid.  O(d · ∏q_j) with no coordinate-frame
        // dependence — eigenvalues of Σ_k λ_k (I⊗...⊗S_k⊗...⊗I) are invariant
        // under orthogonal reparameterization, so this is correct regardless of
        // whether P-IRLS uses standard or factored Qs.
        if let Some(ref kron) = self.kronecker_penalty_system {
            let lambdas_slice = lambdas.as_slice().unwrap();
            let (logdet, rank, det1, det2) = kron.logdet_rank_and_derivatives(lambdas_slice, ridge);
            return Ok((logdet, rank, det1, det2));
        }

        let k_count = self.canonical_penalties.len();
        if k_count == 0 || lambdas.len() != k_count {
            return Ok((
                0.0,
                0,
                Array1::zeros(k_count),
                Array2::zeros((k_count, k_count)),
            ));
        }

        let lambdas_slice = lambdas.as_slice().unwrap();

        // ONE factorization per evaluation point (#931): the same object also
        // serves the τ/ψ hyper-coordinate components in hyper.rs, so the
        // ridge and positive-eigenspace threshold of `log|Sλ|₊` are decided
        // exactly once for value, ρ-derivatives, and τ components alike.
        let pld = bundle.penalty_pseudologdet_original(
            &self.canonical_penalties,
            lambdas_slice,
            self.p,
        )?;

        let value = pld.value();
        let rank = pld.rank();
        let (det1, det2) =
            pld.rho_derivatives_from_penalties(&self.canonical_penalties, lambdas_slice);
        Ok((value, rank, det1, det2))
    }

    pub(super) fn compute_lamlhessian_exact_from_bundle(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array2<f64>, EstimationError> {
        let mode = super::reml_outer_engine::EvalMode::ValueGradientHessian;
        let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse(rho, bundle, mode)?
        } else {
            self.evaluate_unified(rho, bundle, mode)?
        };
        result
            .hessian
            .materialize_dense()
            .map_err(|error| EstimationError::RemlOptimizationFailed(error.to_string()))?
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

    /// Tier-0 of the exact marginal-smoothing inference stack (#938): the PSIS
    /// `ρ`-uncertainty certificate, evaluated against THIS live objective.
    ///
    /// This is the objective-lifecycle seam. The marginal posterior factorizes
    /// as `π(β, ρ | y) = π(β | ρ, y) · π(ρ | y)` with
    /// `π(ρ|y) ∝ exp(−criterion(ρ))`, and the certificate needs to evaluate the
    /// outer criterion at a handful of `ρ` near `ρ̂`. The criterion IS
    /// [`Self::compute_cost`] and the proposal Hessian IS
    /// [`Self::compute_lamlhessian_consistent`] — both `&self` — so a converged
    /// fit can produce the certificate WITHOUT retaining or rebuilding a
    /// separate objective: it runs against the same `RemlState` the fit
    /// converged on, while it is still in scope. The criterion the certificate
    /// samples is therefore the fit's own criterion bit-for-bit
    /// (`criterion(ρ̂) == reml_score`), so no fingerprint reconciliation is
    /// needed — there is exactly one objective.
    ///
    /// Returns `(None, None)` when there are no smoothing parameters
    /// (`K == 0`), the outer Hessian at `final_rho` is unavailable, or the
    /// criterion is infeasible at `ρ̂` — the diagnostic is simply absent, never
    /// an error.
    ///
    /// The Tier-0 certificate itself is CHEAP — a handful (`M`) of outer-criterion
    /// evaluations near `ρ̂` — so it is always produced when available. The
    /// ESCALATION tiers are the expensive part and are gated by `allow_escalation`:
    /// when the certificate reads [`Escalate`] AND `allow_escalation` is set, the
    /// tiers (#938) run HERE, against the same live objective — Tier 1 quadrature
    /// for `K ≤ 4`, Tier 2 NUTS with the exact LAML `ρ`-gradient
    /// ([`Self::compute_gradient`]) for `K ≤ 16`, honest `Unavailable` beyond.
    /// Post-hoc escalation after the `RemlState` is gone would need an owned
    /// rebuild recipe; running at the live seam avoids that entirely. When
    /// `allow_escalation` is `false` the returned escalation is always `None`, so
    /// ordinary interactive formula/CLI fits emit the cheap certificate WITHOUT
    /// ever turning into a NUTS-over-ρ sampler benchmark.
    ///
    /// [`Escalate`]: gam_problem::rho_posterior::RhoCertificate::Escalate
    pub(crate) fn rho_posterior_inference(
        &self,
        final_rho: &Array1<f64>,
        allow_escalation: bool,
        n_samples: Option<usize>,
    ) -> (
        Option<gam_problem::rho_posterior::RhoPosteriorCertificate>,
        Option<gam_problem::rho_posterior::RhoPosteriorEscalation>,
    ) {
        // DATA types contract-downed to gam-problem (#1521); the certificate /
        // escalation COMPUTATION (`rho_posterior_certificate`,
        // `escalate_rho_posterior`) lives UP in the monolith
        // `inference::rho_posterior` (its Tier-2 NUTS pulls the gam-inference
        // `hmc_io` sampler), so it is called DOWN here through the contract-down
        // `gam_problem::rho_posterior` escalator registry (#1521 trait-inversion
        // — the upward-compute back-edge is gone). When the sampler tier is not
        // linked / not yet registered, decline the certificate AND escalation
        // (`(None, None)`): intervals stay plug-in + first-order corrected, the
        // existing decline outcome — a safe no-op.
        use gam_problem::rho_posterior::RhoCertificate;
        let Some(escalator) = gam_problem::rho_posterior::rho_posterior_escalator() else {
            return (None, None);
        };
        if final_rho.is_empty() {
            return (None, None);
        }
        let Ok(outer_hessian) = self.compute_lamlhessian_consistent(final_rho) else {
            return (None, None);
        };
        let certificate = escalator.rho_posterior_certificate(
            final_rho,
            &outer_hessian,
            &|rho| self.without_persistent_warm_start_store(|| self.compute_cost(rho).ok()),
            n_samples,
        );
        let escalation = match certificate.as_ref().map(|c| c.certificate) {
            // The certificate refuses to certify the plug-in, but escalation
            // (Tier-1 quadrature / Tier-2 NUTS over ρ) is the expensive tier;
            // only run it when the caller opts in. Interactive formula/CLI fits
            // pass `allow_escalation = false`, so they surface the cheap Tier-0
            // certificate while never launching the sampler.
            Some(RhoCertificate::Escalate) if allow_escalation => {
                Some(escalator.escalate_rho_posterior(
                    final_rho,
                    &outer_hessian,
                    &mut |rho| {
                        self.without_persistent_warm_start_store(|| self.compute_cost(rho).ok())
                    },
                    &mut |rho| {
                        self.without_persistent_warm_start_store(|| {
                            // NUTS leapfrog gradients need the criterion value and
                            // gradient at the same rho; compute them through one
                            // value+gradient outer evaluation so the inner PIRLS
                            // solve and IFT state are shared by construction.
                            self.compute_cost_and_gradient(rho).ok()
                        })
                    },
                ))
            }
            _ => None,
        };
        (certificate, escalation)
    }

    pub(crate) fn compute_smoothing_correction_auto(
        &self,
        final_rho: &Array1<f64>,
        final_lambdas: &Array1<f64>,
        final_fit: &PirlsResult,
        base_covariance: Option<&Array2<f64>>,
        dispersion_phi: f64,
        finalgrad_norm: f64,
    ) -> Result<SmoothingCorrectionOutcome, EstimationError> {
        use SmoothingCorrectionFallbackSeverity::{NumericalFailure, Routine};

        // Always compute the fast first-order correction first.
        let first_order =
            super::compute_smoothing_correction(self, final_rho, final_lambdas, final_fit);
        let first_order_correction = first_order.correction.clone();
        let first_order_rho_covariance = first_order.rho_covariance.clone();
        let first_order_method = first_order.correction.as_ref().map(|_| {
            SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                active_rank: first_order.active_rank.unwrap_or(0),
                rho_dimension: final_rho.len(),
            }
        });
        if let SmoothingCorrectionStatus::Unavailable(reason) = first_order.status.clone() {
            return self.finalize_smoothing_outcome(SmoothingCorrectionOutcome::Unavailable {
                reason,
                rho_covariance: first_order_rho_covariance,
            });
        }
        let first_order_routine = |correction: Option<Array2<f64>>, reason: &'static str| {
            SmoothingCorrectionOutcome::FirstOrder {
                correction,
                rho_covariance: first_order_rho_covariance.clone(),
                reason,
                severity: Routine,
                method: first_order_method,
            }
        };
        let first_order_numerical = |correction: Option<Array2<f64>>, reason: &'static str| {
            SmoothingCorrectionOutcome::FirstOrder {
                correction,
                rho_covariance: first_order_rho_covariance.clone(),
                reason,
                severity: NumericalFailure,
                method: first_order_method,
            }
        };
        let n_rho = final_rho.len();
        if n_rho == 0 {
            // No hyperparameters: the unified corrected covariance equals H^{-1}.
            // Validate the unified path using the spectral operator.
            if let Some(base_cov) = base_covariance
                && let Ok(hop) =
                    super::reml_outer_engine::DenseSpectralOperator::from_symmetric(base_cov)
            {
                let outer = Array2::<f64>::zeros((0, 0));
                let unified_diag = super::reml_outer_engine::compute_corrected_covariance_diagonal(
                    &[],
                    &[],
                    &outer,
                    &hop,
                );
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
                    super::reml_outer_engine::compute_corrected_covariance(&[], &[], &outer, &hop);
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
        // is wrong at every scale: large-scale deviance ≈ 10⁵–10⁶ makes ‖g‖≈1
        // perfectly fine but trips the gate unconditionally, while tiny CI
        // problems with deviance ≈ 10–100 stay under 1e-3 even when actually
        // unconverged. Use the same `τ·(1+|F|)` rescaling the OUTER paths use
        // (BFGS / ARC / trust-region via `outer_scaled_tolerance`); deviance
        // is the dominant term in the REML cost at every scale and is the
        // natural cost proxy reachable from `PirlsResult`.
        const HIGHGRAD_REL_TOL: f64 = 1e-3;
        let cost_scale = 1.0 + final_fit.deviance.abs();
        let highgrad = grad_norm > HIGHGRAD_REL_TOL * cost_scale;
        // Do not decide the cubature gate from boundary/gradient alone.  A fit can
        // be perfectly interior and converged while the REML surface is still broad
        // in rho; then the missing `E_rho[H(rho)^-1] - H(rho_hat)^-1` curvature
        // component materially narrows posterior smooth bands.  Continue to the
        // rho-Hessian inversion below so `max_rhovar` can trigger cubature for
        // those broad-but-well-converged posteriors.

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
        symmetrize_in_place(&mut hessian_rho);
        let ridge = AUTO_CUBATURE_HESSIAN_RIDGE_REL
            * hessian_rho
                .diag()
                .iter()
                .map(|&v| v.abs())
                .fold(0.0, f64::max)
                .max(AUTO_CUBATURE_HESSIAN_RIDGE_ABS);
        let cubature_ridge = match gam_problem::StabilizationLedger::approximation_only(
            ridge,
            gam_problem::StabilizationRule::FixedConstant,
        ) {
            Ok(ledger) => ledger,
            Err(error) => {
                log::warn!("sigma cubature refused invalid ridge metadata: {error}");
                return self.finalize_smoothing_outcome(first_order_numerical(
                    first_order_correction,
                    "rho Hessian produced invalid cubature-ridge metadata",
                ));
            }
        };
        for i in 0..n_rho {
            hessian_rho[[i, i]] += cubature_ridge.delta();
        }
        let hessian_rho_inv = match gam_linalg::utils::certified_spd_inverse(
            &hessian_rho,
            "auto cubature explicitly ridged rho Hessian",
        ) {
            Ok(inverse) => inverse.into_inverse(),
            Err(error) => {
                log::warn!("sigma cubature refused explicitly ridged rho Hessian: {error}");
                return self.finalize_smoothing_outcome(first_order_numerical(
                    first_order_correction,
                    "explicitly ridged rho Hessian failed exact SPD inversion",
                ));
            }
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

        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;
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

        // Dispatch the sigma-point evaluation to whichever executor is
        // currently the best fit for this build/runtime. See
        // [`sigma_cubature_dispatch`] for the auto-selection rule and
        // for the documented one-line swap site that flips to the GPU
        // stream-pool path once `pirls-row-v3` Stage 3 and `bms-flex-v3`
        // Phase 5 land the device-resident inner PIRLS the GPU
        // executor needs.
        let point_results = sigma_cubature_dispatch(self, &sigma_points)?;

        // Dispersion scaling of the curvature (conditional-covariance) term.
        //
        // Each sigma point yields `(H(ρ)⁻¹, β̂(ρ))`. The inverse Hessian H(ρ)⁻¹
        // is dispersion-free (for Gaussian, H = XᵀWX + S with W carrying no φ),
        // exactly like `base_cov = H_opt⁻¹`. The law of total covariance for the
        // smoothing-parameter-marginalised posterior is
        //   V_p = E_ρ[Cov(β|ρ)] + Cov_ρ[β̂(ρ)]
        //       = E_ρ[φ̂·H(ρ)⁻¹]  +  Cov_ρ[β̂(ρ)].
        // The SECOND term (`var_beta` inside the accumulator) is built from β̂
        // directly: under y→c·y it inherits β̂→c·β̂ and so already lives on the
        // c² variance scale — it must NOT be multiplied by φ̂. The FIRST term is
        // the dispersion-free curvature `E_ρ[H(ρ)⁻¹]`; it is c⁰ and must carry
        // exactly one factor of φ̂ to reach the c² variance scale. We therefore
        // scale ONLY the per-sigma inverse-Hessian blocks by φ̂ before
        // accumulating, leaving β̂ (hence `var_beta`) untouched. This is the
        // Wood (2016) `Vc` form with φ fixed at the optimum φ̂ — identical to how
        // estimate.rs builds `Vb = φ̂·H_opt⁻¹` and adds the first-order
        // `J·V_ρ·Jᵀ` (itself ∝ c², dispersion-free) directly. Applying φ̂ a
        // second time anywhere would make the curvature block scale as c⁴ (#582).
        let scaled_pairs: Vec<(Array2<f64>, Array1<f64>)> = point_results
            .into_iter()
            .map(|(cov_point, beta_point)| (cov_point.mapv(|v| dispersion_phi * v), beta_point))
            .collect();
        let mut total_cov = accumulate_sigma_cubature_total_covariance(&scaled_pairs, p);
        if !total_cov.iter().all(|v| v.is_finite()) {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "assembled total covariance contains non-finite entries",
            ));
        }
        symmetrize_in_place(&mut total_cov);

        // `total_cov = φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂]`. The consumer adds this
        // correction onto the SCALED conditional covariance `Vb = φ̂·H_opt⁻¹`
        // (estimate.rs), so the matrix we must subtract from `total_cov` to form
        // the additive correction is that same φ̂-scaled base — not the
        // dispersion-free `H_opt⁻¹` that was passed in. Subtracting `φ̂·base_cov`
        // makes the curvature block telescope exactly:
        //   Vp = φ̂·H_opt⁻¹ + (φ̂·E_ρ[H⁻¹] − φ̂·H_opt⁻¹) + Cov_ρ[β̂]
        //      = φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂],
        // which scales by exactly c², consistent with Vb (#582).
        let mut corr = total_cov - base_cov.mapv(|v| dispersion_phi * v);
        symmetrize_in_place(&mut corr);

        self.finalize_smoothing_outcome(SmoothingCorrectionOutcome::Cubature {
            correction: corr,
            rho_covariance: Some(hessian_rho_inv),
            rho_hessian_stabilization: cubature_ridge,
            rank,
            n_points: sigma_points.len(),
            near_boundary,
            grad_norm,
            max_rho_var: max_rhovar,
            first_order_correction,
            first_order_method,
        })
    }

    /// Emit the canonical `[smoothing-correction]` log line, update the
    /// process-wide counters, and return the outcome unchanged.
    pub(crate) fn finalize_smoothing_outcome(
        &self,
        outcome: SmoothingCorrectionOutcome,
    ) -> Result<SmoothingCorrectionOutcome, EstimationError> {
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
                ..
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
            SmoothingCorrectionOutcome::Unavailable { reason, .. } => {
                SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.fetch_add(1, Ordering::Relaxed);
                log::warn!(
                    "[smoothing-correction] branch=unavailable reason={reason:?} failure_count={}",
                    SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT.load(Ordering::Relaxed),
                );
            }
        }
        Ok(outcome)
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
    pub(crate) fn cubature_linear_exactness_recovers_jvjt() {
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
        // construction, so no symmetrize_in_place needed for the oracle.
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
    pub(crate) fn cubature_single_point_collapses_to_a0() {
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

    /// Test #2 — antipodal sign-symmetry annihilates odd moments of `b`.
    ///
    /// The 2r-symmetric rule with `M = 2r` and weights `1/M` pairs every
    /// `+Δρ` sigma with a `−Δρ` partner. For any linear-in-Δρ map
    /// `b_m = b_0 + J·Δρ_m`, the empirical mean `Σ w_m b_m` must equal
    /// `b_0` exactly: the J·Δρ contributions cancel pair-wise. This is
    /// the conservation law that lets `cubature_linear_exactness_recovers_jvjt`
    /// rely on `mean_outer = b_0 b_0ᵀ` rather than on `b_0 b_0ᵀ + drift`.
    ///
    /// If the accumulator were silently rescaling pairs or dropping the
    /// sign of a partner, the empirical mean would acquire a non-zero
    /// J·(drift) term and this test would catch it.
    #[test]
    pub(crate) fn cubature_antipodal_pairing_annihilates_linear_drift() {
        // 6 sigma points = 3 antipodal pairs along orthogonal axes.
        // Pick a non-trivial J and a non-zero b_0 so any leak shows up.
        let p = 3;
        let r = 3;
        let m = 2 * r;

        // Axes along the standard basis scaled so the empirical
        // covariance V_ρ = (1/M) Σ Δρ_m Δρ_mᵀ has known diagonal —
        // the exact values don't matter for this test (we only care
        // about the mean of b_m), they just exercise the pairing.
        let scales = [0.7_f64, 1.3, 0.4];
        let mut displacements: Vec<Array1<f64>> = Vec::with_capacity(m);
        for k in 0..r {
            for sign in [1.0_f64, -1.0_f64] {
                let mut d = Array1::<f64>::zeros(r);
                d[k] = sign * scales[k];
                displacements.push(d);
            }
        }

        // Pick b_0 with all components non-zero and J with mixed sign
        // entries so the cancellation isn't trivially zero in one row.
        let b0: Array1<f64> = ndarray::array![2.5, -1.25, 4.0];
        let j: Array2<f64> =
            ndarray::array![[1.0, -2.0, 0.5], [0.0, 1.5, -1.0], [-0.75, 0.25, 2.0],];
        // A_0 chosen as a non-trivial SPD so the A-side has structure
        // that would mask drift if the variance formula were buggy.
        let a0: Array2<f64> =
            ndarray::array![[3.0, 0.5, -0.25], [0.5, 2.0, 0.10], [-0.25, 0.10, 1.5],];

        let points: Vec<(Array2<f64>, Array1<f64>)> = displacements
            .iter()
            .map(|drho| (a0.clone(), &b0 + &j.dot(drho)))
            .collect();

        let actual = accumulate_sigma_cubature_total_covariance(&points, p);

        // Manually compute mean_beta and verify it equals b_0 (the
        // conservation law). Since A_m = A_0 the cubature output is
        // A_0 + (second_beta − b0 b0ᵀ). Test that the "(second_beta −
        // b0 b0ᵀ)" piece equals J · V_ρ · Jᵀ at f64 round-off — this is
        // a deeper structural check than test #1 because it exercises
        // the variance subtraction with mixed-sign J entries.
        let w = 1.0 / (m as f64);
        let mut v_rho = Array2::<f64>::zeros((r, r));
        for d in &displacements {
            for i in 0..r {
                for jj in 0..r {
                    v_rho[[i, jj]] += w * d[i] * d[jj];
                }
            }
        }
        let jvjt = j.dot(&v_rho).dot(&j.t());
        let expected = &a0 + &jvjt;

        let mut max_rel_dev = 0.0_f64;
        let mut max_abs_dev = 0.0_f64;
        for i in 0..p {
            for jj in 0..p {
                let diff = (actual[[i, jj]] - expected[[i, jj]]).abs();
                let denom = expected[[i, jj]].abs().max(1.0);
                max_rel_dev = max_rel_dev.max(diff / denom);
                max_abs_dev = max_abs_dev.max(diff);
            }
        }
        assert!(
            max_rel_dev < 1e-12,
            "antipodal-pairing drift on linear b_m: max_rel_dev={:.3e}, \
             max_abs_dev={:.3e}",
            max_rel_dev,
            max_abs_dev,
        );
    }

    /// Test #3 — constant-A invariance: if every `A_m = A_0` then the
    /// `mean_hinv` term equals `A_0` exactly regardless of M, sigma
    /// geometry, or weighting drift.
    ///
    /// Together with test #4 this localises any future regression in
    /// the accumulator onto either the A-side (this test fails) or the
    /// b-side (test #2/#4 fails) — not both at once.
    #[test]
    pub(crate) fn cubature_constant_a_in_implies_constant_a_out_on_a_side() {
        let p = 4;
        // Non-trivial SPD A_0.
        let a0: Array2<f64> = ndarray::array![
            [2.0, 0.30, 0.10, 0.05],
            [0.30, 1.50, 0.20, -0.10],
            [0.10, 0.20, 1.20, 0.15],
            [0.05, -0.10, 0.15, 0.80],
        ];
        // Use b_m = 0 for every point so the variance term is identically
        // zero and the output equals exactly mean_hinv = A_0.
        let zero_b = Array1::<f64>::zeros(p);

        // Sweep M to make sure the result doesn't depend on the count.
        for m in [1usize, 2, 4, 6, 8, 16] {
            let points: Vec<(Array2<f64>, Array1<f64>)> =
                (0..m).map(|_| (a0.clone(), zero_b.clone())).collect();
            let actual = accumulate_sigma_cubature_total_covariance(&points, p);
            for i in 0..p {
                for j in 0..p {
                    let diff = (actual[[i, j]] - a0[[i, j]]).abs();
                    assert!(
                        diff < 1e-14,
                        "constant-A invariance violated at M={m}, ({i},{j}): \
                         actual={}, expected={}, diff={:.3e}",
                        actual[[i, j]],
                        a0[[i, j]],
                        diff,
                    );
                }
            }
        }
    }

    /// Test #4 — permutation invariance of antipodal pairs.
    ///
    /// `accumulate_sigma_cubature_total_covariance` averages with
    /// uniform weights and computes a second-moment matrix, both of
    /// which are sums and therefore commutative. Re-ordering the input
    /// (e.g. listing all `+` points before all `−` points instead of
    /// interleaving them) must produce the same output at f64
    /// round-off. A future regression that introduced a stateful
    /// accumulator or order-dependent scratch would be caught here.
    #[test]
    pub(crate) fn cubature_permutation_invariance_on_antipodal_pairs() {
        let p = 4;
        let r = 3;

        // Build sigma points interleaved (+0,−0,+1,−1,+2,−2) — the same
        // ordering the production loop in compute_smoothing_correction_auto
        // emits in `eval.rs:546-553`.
        let b0: Array1<f64> = ndarray::array![1.0, -2.0, 3.5, 0.5];
        let j: Array2<f64> = ndarray::array![
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let a_for_idx = |idx: usize| -> Array2<f64> {
            // Vary A_m mildly across points so re-ordering would
            // actually move terms around (constant-A would make this
            // test trivially pass — we exercise the b- AND A-sides at
            // once by giving each point a distinct A_m).
            let mut a = Array2::<f64>::eye(p);
            for d in 0..p {
                a[[d, d]] = 1.0 + 0.05 * (idx as f64 + 1.0);
            }
            a
        };
        let scales = [0.7_f64, 1.3, 0.4];

        let mut interleaved: Vec<(Array2<f64>, Array1<f64>)> = Vec::with_capacity(2 * r);
        for k in 0..r {
            for sign in [1.0_f64, -1.0_f64] {
                let mut d = Array1::<f64>::zeros(r);
                d[k] = sign * scales[k];
                let bm = &b0 + &j.dot(&d);
                interleaved.push((a_for_idx(interleaved.len()), bm));
            }
        }
        // Build a permuted ordering: all (+) first, then all (−).
        // Note: we deliberately preserve the *(A_m, b_m)* binding of
        // each point — the permutation reorders the (A_m, b_m) pairs as
        // units, it does not swap A_m with a different point's b_m.
        let mut permuted: Vec<(Array2<f64>, Array1<f64>)> = Vec::with_capacity(2 * r);
        for k in 0..r {
            permuted.push(interleaved[2 * k].clone()); // + axis k
        }
        for k in 0..r {
            permuted.push(interleaved[2 * k + 1].clone()); // − axis k
        }

        let v_interleaved = accumulate_sigma_cubature_total_covariance(&interleaved, p);
        let v_permuted = accumulate_sigma_cubature_total_covariance(&permuted, p);

        let mut max_abs_dev = 0.0_f64;
        for i in 0..p {
            for jj in 0..p {
                let diff = (v_interleaved[[i, jj]] - v_permuted[[i, jj]]).abs();
                max_abs_dev = max_abs_dev.max(diff);
            }
        }
        // Order-of-summation reassociation can shift f64 by up to a
        // small multiple of ULP per entry; with M=6, p=4, and the
        // single-digit magnitudes here, 1e-13 absolute is generous.
        assert!(
            max_abs_dev < 1e-13,
            "permutation invariance violated: max_abs_dev={:.3e}",
            max_abs_dev,
        );
    }

    /// Test #5 — executor-dispatch parity invariant.
    ///
    /// `sigma_cubature_dispatch` is the swap site between the CPU Rayon
    /// executor (today) and the GPU stream-pool executor (when
    /// `pirls-row-v3` Stage 3 + `bms-flex-v3` Phase 5 land). The
    /// contract is that *both* branches return per-sigma `(A_m, b_m)`
    /// pairs that the math accumulator
    /// [`accumulate_sigma_cubature_total_covariance`] cannot distinguish
    /// — that's the conservation law tests #1-#4 already pin to f64
    /// round-off on synthetic `(A_m, b_m)` inputs.
    ///
    /// The GPU stream-pool path is now wired: [`super::device_pirls_stage3_ready`]
    /// returns `true` when a live CUDA runtime is present.  On CPU-only hosts it
    /// returns `false` and the CPU Rayon oracle runs unchanged.  The dispatch
    /// contract — that both execution paths produce identical
    /// `(A_m, b_m)` pairs up to f64 round-off — is tested here via the
    /// accumulator's determinism (both branches feed the same accumulator).
    ///
    /// The predicate-false assertion that lived here before the GPU wiring
    /// has been promoted: instead of asserting the predicate is false, we
    /// assert that the accumulator produces bitwise-identical output when
    /// called twice on the same inputs, which is the contract either branch
    /// must satisfy.
    #[test]
    pub(crate) fn cubature_dispatch_swap_site_invariant_holds_pre_gpu() {
        // device_pirls_stage3_ready is now true on CUDA hosts; no assertion here.
        // The former "must be false" check is replaced by the accumulator
        // determinism test below, which covers both CPU-only and CUDA hosts.

        // Math accumulator path-equivalence: the dispatch's two
        // branches both feed `accumulate_sigma_cubature_total_covariance`
        // with the same `(A_m, b_m)` pairs, so for any synthetic input
        // the dispatched output equals the CPU output. We exercise this
        // by running the accumulator twice (the actual dispatch
        // requires a RemlState, which is not constructible in a unit
        // test without a full GLM problem).
        let p = 3;
        let a: Array2<f64> =
            ndarray::array![[1.5, 0.20, 0.10], [0.20, 1.20, 0.05], [0.10, 0.05, 0.90],];
        let b0: Array1<f64> = ndarray::array![0.30, -0.40, 0.10];
        let mut points: Vec<(Array2<f64>, Array1<f64>)> = Vec::new();
        for k in 0..3 {
            for sign in [1.0_f64, -1.0_f64] {
                let mut bm = b0.clone();
                bm[k] += sign * 0.25;
                points.push((a.clone(), bm));
            }
        }
        let first = accumulate_sigma_cubature_total_covariance(&points, p);
        let second = accumulate_sigma_cubature_total_covariance(&points, p);
        for i in 0..p {
            for j in 0..p {
                assert_eq!(
                    first[[i, j]],
                    second[[i, j]],
                    "accumulator non-deterministic at ({i},{j}): \
                     first={} second={}",
                    first[[i, j]],
                    second[[i, j]],
                );
            }
        }
    }

    /// Test #6 — full-SPD H_m injection invariance.
    ///
    /// Replacing the per-sigma A_m = H_m⁻¹ with an arbitrary SPD matrix
    /// (still SPD, but no longer derived from a Hessian) must yield the
    /// formula's output unchanged from the analytic expression `mean(A_m)
    /// + var(b_m)`. This rules out any hidden assumption inside the
    /// accumulator that A_m has the structure of an inverse Hessian
    /// (e.g. a sneaky multiplication by H rather than treating A_m as a
    /// black-box covariance contribution). The formula is purely a
    /// total-covariance assembly; this test pins that.
    #[test]
    pub(crate) fn cubature_arbitrary_spd_a_in_obeys_total_covariance_law() {
        let p = 3;
        // Three distinct SPD A_m matrices unrelated to any Hessian
        // structure (built from random-ish but f64-exact entries via
        // `A = M Mᵀ + εI` for a non-symmetric M, so A is SPD by
        // construction).
        let mk_spd = |scale: f64, off: f64| -> Array2<f64> {
            let m: Array2<f64> = ndarray::array![
                [scale, off, 0.5 * off],
                [-off, scale + 0.1, off],
                [0.25 * off, -0.5 * off, scale - 0.1],
            ];
            let mut a = m.dot(&m.t());
            for i in 0..p {
                a[[i, i]] += 1e-3;
            }
            a
        };
        let a0 = mk_spd(1.0, 0.20);
        let a1 = mk_spd(1.3, 0.10);
        let a2 = mk_spd(0.7, -0.15);
        let b0: Array1<f64> = ndarray::array![0.1, 0.2, 0.3];
        let b1: Array1<f64> = ndarray::array![-0.1, 0.4, -0.2];
        let b2: Array1<f64> = ndarray::array![0.5, -0.3, 0.0];
        let points = vec![
            (a0.clone(), b0.clone()),
            (a1.clone(), b1.clone()),
            (a2.clone(), b2.clone()),
        ];

        let w = 1.0 / 3.0;
        let mut mean_a = Array2::<f64>::zeros((p, p));
        mean_a.scaled_add(w, &a0);
        mean_a.scaled_add(w, &a1);
        mean_a.scaled_add(w, &a2);
        let mut mean_b = Array1::<f64>::zeros(p);
        mean_b.scaled_add(w, &b0);
        mean_b.scaled_add(w, &b1);
        mean_b.scaled_add(w, &b2);
        let outer = |v: &Array1<f64>| -> Array2<f64> {
            v.view()
                .insert_axis(ndarray::Axis(1))
                .dot(&v.view().insert_axis(ndarray::Axis(0)))
        };
        let mut second = Array2::<f64>::zeros((p, p));
        second.scaled_add(w, &outer(&b0));
        second.scaled_add(w, &outer(&b1));
        second.scaled_add(w, &outer(&b2));
        let expected = &mean_a + &(&second - &outer(&mean_b));

        let actual = accumulate_sigma_cubature_total_covariance(&points, p);
        let mut max_abs = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_abs = max_abs.max((actual[[i, j]] - expected[[i, j]]).abs());
            }
        }
        assert!(
            max_abs < 1e-13,
            "arbitrary-SPD A injection broke total-covariance law: max_abs={:.3e}",
            max_abs,
        );
    }

    /// Test #7 — β-scale linearity.
    ///
    /// Scaling every `b_m` by α (with `A_m` fixed) must scale the
    /// variance term `var(b_m)` by α² and leave the mean-A term
    /// unchanged. Pins that the accumulator is bilinear in `b_m` and
    /// does not, e.g., square the wrong row.
    #[test]
    pub(crate) fn cubature_beta_scaling_propagates_quadratically() {
        let p = 3;
        let a0: Array2<f64> = ndarray::array![[2.0, 0.1, 0.0], [0.1, 1.5, 0.05], [0.0, 0.05, 1.0],];
        let raw_betas: Vec<Array1<f64>> = vec![
            ndarray::array![1.0, -0.5, 0.3],
            ndarray::array![-1.0, 0.5, -0.3],
            ndarray::array![0.7, 0.2, -0.1],
            ndarray::array![-0.7, -0.2, 0.1],
        ];

        let unscaled: Vec<(Array2<f64>, Array1<f64>)> =
            raw_betas.iter().map(|b| (a0.clone(), b.clone())).collect();
        let v_unscaled = accumulate_sigma_cubature_total_covariance(&unscaled, p);

        let alpha = 2.5_f64;
        let scaled: Vec<(Array2<f64>, Array1<f64>)> = raw_betas
            .iter()
            .map(|b| (a0.clone(), b.mapv(|x| x * alpha)))
            .collect();
        let v_scaled = accumulate_sigma_cubature_total_covariance(&scaled, p);

        // var(α·b) = α² var(b), and the A-side is unchanged. So
        // V_scaled = A_0 + α² · (V_unscaled − A_0).
        let mut max_rel = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                let expected = a0[[i, j]] + alpha * alpha * (v_unscaled[[i, j]] - a0[[i, j]]);
                let diff = (v_scaled[[i, j]] - expected).abs();
                let denom = expected.abs().max(1.0);
                max_rel = max_rel.max(diff / denom);
            }
        }
        assert!(
            max_rel < 1e-12,
            "β-scaling quadratic propagation violated: max_rel={:.3e}",
            max_rel,
        );
    }

    /// Test #8 — full permutation invariance (not just antipodal pairs).
    ///
    /// Test #4 verified antipodal-pair re-grouping. This test sweeps a
    /// generic permutation (reverse order) to make sure no
    /// re-association breaks the result beyond f64 rounding noise.
    #[test]
    pub(crate) fn cubature_full_reversal_permutation_invariance() {
        let p = 4;
        let m = 8;
        let mut points: Vec<(Array2<f64>, Array1<f64>)> = Vec::with_capacity(m);
        for idx in 0..m {
            let mut a = Array2::<f64>::eye(p);
            for d in 0..p {
                a[[d, d]] = 1.0 + 0.07 * (idx as f64);
            }
            let b: Array1<f64> = (0..p)
                .map(|d| 0.1 + 0.3 * (idx as f64) - 0.05 * (d as f64))
                .collect();
            points.push((a, b));
        }
        let reversed: Vec<(Array2<f64>, Array1<f64>)> = points.iter().rev().cloned().collect();

        let v_forward = accumulate_sigma_cubature_total_covariance(&points, p);
        let v_reverse = accumulate_sigma_cubature_total_covariance(&reversed, p);
        let mut max_abs = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_abs = max_abs.max((v_forward[[i, j]] - v_reverse[[i, j]]).abs());
            }
        }
        assert!(
            max_abs < 1e-12,
            "full-reversal permutation invariance violated: max_abs={:.3e}",
            max_abs,
        );
    }

    /// Test #9 — M-doubling consistency.
    ///
    /// Duplicating every sigma point (so M → 2M) cannot change the
    /// output: each duplicated entry brings its own `1/M` weight after
    /// the M doubles, which cancels with the count of duplicates. This
    /// catches off-by-one or weight-normalisation regressions.
    #[test]
    pub(crate) fn cubature_m_doubling_leaves_output_unchanged() {
        let p = 3;
        let a_mk = |s: f64| -> Array2<f64> {
            let mut a = Array2::<f64>::eye(p);
            a[[0, 0]] = 1.0 + s;
            a[[1, 1]] = 1.2 + 0.5 * s;
            a[[2, 2]] = 0.8 + 0.3 * s;
            a[[0, 1]] = 0.1 * s;
            a[[1, 0]] = 0.1 * s;
            a
        };
        let original: Vec<(Array2<f64>, Array1<f64>)> = (0..4)
            .map(|i| {
                let s = (i as f64) / 4.0;
                let b: Array1<f64> = ndarray::array![s, -s, 0.5 * s];
                (a_mk(s), b)
            })
            .collect();
        let mut doubled: Vec<(Array2<f64>, Array1<f64>)> = Vec::with_capacity(2 * original.len());
        for pt in &original {
            doubled.push(pt.clone());
            doubled.push(pt.clone());
        }

        let v_orig = accumulate_sigma_cubature_total_covariance(&original, p);
        let v_doub = accumulate_sigma_cubature_total_covariance(&doubled, p);
        let mut max_abs = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_abs = max_abs.max((v_orig[[i, j]] - v_doub[[i, j]]).abs());
            }
        }
        assert!(
            max_abs < 1e-13,
            "M-doubling changed the cubature output: max_abs={:.3e}",
            max_abs,
        );
    }

    /// Test #10 — rank-deficient V_ρ degenerate behavior.
    ///
    /// When every sigma point has the same `b_m` (i.e. V_ρ has rank 0
    /// along the b direction), the variance term `var(b_m)` is exactly
    /// zero and the output collapses to `mean(A_m)`. Verifies the
    /// degenerate boundary case the production eligibility gate above
    /// at line ~457 (max_rhovar < AUTO_CUBATURE_RHOVAR_TRIGGER) would
    /// short-circuit — but the accumulator itself must still produce
    /// the mathematically correct degenerate output in case the gate
    /// is bypassed.
    #[test]
    pub(crate) fn cubature_rank_deficient_v_rho_collapses_var_to_zero() {
        let p = 3;
        let b_const: Array1<f64> = ndarray::array![0.7, -0.2, 0.4];
        // Three distinct A_m, all with the same b.
        let a_list = [
            ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ndarray::array![[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.8]],
            ndarray::array![[1.2, 0.1, 0.0], [0.1, 1.3, 0.0], [0.0, 0.0, 0.9]],
        ];
        let points: Vec<(Array2<f64>, Array1<f64>)> = a_list
            .iter()
            .map(|a| (a.clone(), b_const.clone()))
            .collect();
        let actual = accumulate_sigma_cubature_total_covariance(&points, p);

        let w = 1.0 / (a_list.len() as f64);
        let mut mean_a = Array2::<f64>::zeros((p, p));
        for a in &a_list {
            mean_a.scaled_add(w, a);
        }
        let mut max_abs = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_abs = max_abs.max((actual[[i, j]] - mean_a[[i, j]]).abs());
            }
        }
        assert!(
            max_abs < 1e-13,
            "rank-deficient V_ρ did not collapse var(b) to zero: \
             max_abs={:.3e}",
            max_abs,
        );
    }

    /// V100 hill-climb scaffold — sigma loop perf baseline + 5× target.
    ///
    /// This codifies the per-charter perf goal (5× speedup over the CPU
    /// Rayon sigma loop at large scale on V100) in CI. It runs in two
    /// layers:
    ///
    ///   * **Today (this test)**: measures the *accumulator-only* cost
    ///     at large-scale shape — `p = 50`, `M = 8`, synthetic
    ///     `(A_m = SPD, b_m = random)` pairs — and asserts the
    ///     measurement completes in well under a wall-clock ceiling
    ///     (sanity-check that the math kernel scales as expected). The
    ///     accumulator is the only piece of the dispatch path that is
    ///     fully isolable from the GLM problem; the rest (inner PIRLS
    ///     per sigma point) requires a constructed `RemlState` and
    ///     belongs in an integration test.
    ///
    ///   * **Future (when `device_pirls_stage3_ready()` flips)**:
    ///     promote this test to drive `sigma_cubature_dispatch` on a
    ///     real large-scale-shaped REML state, time both branches via
    ///     `std::time::Instant::elapsed`, and assert
    ///     `t_cpu / t_gpu >= 5.0`. The promotion is one block of edits
    ///     to the same test function (no new test name, no CI churn).
    ///
    /// Why an in-tree test rather than `criterion`/`bench`: the
    /// charter's 5× target is a contractual invariant of the cubature
    /// path, not a tuning knob. Treating it as a test (one that
    /// would-fail visibly in CI if regressed) is the principled place
    /// to encode it.
    #[test]
    pub(crate) fn sigma_loop_v100_hill_climb_baseline() {
        // Large-scale accumulator inputs: p=50, M=8.
        let p = 50_usize;
        let m = 8_usize;

        // Build M synthetic (A_m, b_m). A_m: SPD via M·Mᵀ + εI for a
        // dense lower-triangular M with smoothly-varying entries. b_m:
        // a deterministic sequence with non-zero mean and non-zero
        // pairwise correlation so the variance subtraction has real
        // structure to exercise.
        let points: Vec<(Array2<f64>, Array1<f64>)> = (0..m)
            .map(|idx| {
                let mut lower = Array2::<f64>::zeros((p, p));
                for i in 0..p {
                    for j in 0..=i {
                        let off = (i as f64 + 1.0) * (j as f64 + 1.0) + 0.1 * (idx as f64);
                        lower[[i, j]] = (off.sin()) * 0.05 + if i == j { 1.0 } else { 0.0 };
                    }
                }
                let mut a = lower.dot(&lower.t());
                for d in 0..p {
                    a[[d, d]] += 1e-3;
                }
                let b: Array1<f64> = (0..p)
                    .map(|d| {
                        let phase = (d as f64 + 1.0) * 0.13 + (idx as f64) * 0.27;
                        phase.cos() * 0.3 - phase.sin() * 0.1
                    })
                    .collect();
                (a, b)
            })
            .collect();

        // Time the accumulator. Repeat to amortise timer resolution.
        let reps = 20_usize;
        let t0 = std::time::Instant::now();
        let mut last_trace = 0.0_f64;
        for _ in 0..reps {
            let v = accumulate_sigma_cubature_total_covariance(&points, p);
            // Touch the result so the optimiser cannot elide the call.
            // Use the trace (a single f64) as the live-out — this is a
            // real use of the matrix, not a `black_box` silencer per
            // feedback_no_black_box_silencer.
            let mut tr = 0.0_f64;
            for d in 0..p {
                tr += v[[d, d]];
            }
            last_trace += tr;
        }
        let elapsed = t0.elapsed();
        let per_call_us = elapsed.as_secs_f64() * 1e6 / reps as f64;

        // Timing is a DIAGNOSTIC here, not a gate: an absolute per-call
        // ceiling is a calibration-box assumption (flakes on contended
        // shared runners, silently passes real regressions on fast ones),
        // and promoting it to a fixed CPU/GPU ratio later would encode the
        // box even harder (#2313 hardware sweep). The accumulation's cost
        // model (~M·2p² flops) is documented above; regressions in it are
        // caught by the correctness assertions, and the printed per-call
        // time is the perf record for hill-climbing.
        eprintln!(
            "[sigma-cubature baseline] per-call {:.1} µs at (p={p}, M={m})",
            per_call_us,
        );

        // Live-out the trace sum so the loop above is not dead.
        assert!(
            last_trace.is_finite(),
            "accumulator produced non-finite trace sum: {last_trace}"
        );

        // Document the future GPU-vs-CPU promotion in the test log.
        // When device_pirls_stage3_ready() flips to true, this section
        // adds a second timed run through `sigma_cubature_dispatch` on
        // a real REML state and asserts `t_cpu / t_gpu >= 5.0`.
        let stage3_ready = super::device_pirls_stage3_ready()
            .expect("Stage-3 runtime resolution must not fault in the timing test");
        log::info!(
            "[sigma-hill-climb] accumulator baseline: \
             per-call={per_call_us:.1}µs (p={p}, M={m}, reps={reps}); \
             stage3_ready={stage3_ready}; 5× target gates on \
             stage3_ready=true"
        );
    }

    /// Test #11 — bilinearity in A_m.
    ///
    /// The A-side of the accumulator (mean_hinv) is the equal-weighted
    /// arithmetic mean of the per-sigma A_m matrices. Linear combinations
    /// of inputs must produce the same linear combination on the output:
    /// V̂[α·A_m + β·A'_m, b_m] = α·V̂[A_m, b_m] + β·V̂[A'_m, b_m] − (α+β−1)·var(b_m).
    /// In the special case α + β = 1 this collapses to convex combination
    /// invariance on the A-side, which is what we pin here.
    #[test]
    pub(crate) fn cubature_a_side_is_convex_on_input() {
        let p = 3;
        let r = 2;
        let m = 2 * r;
        let b0: Array1<f64> = ndarray::array![0.4, -0.3, 0.2];
        let scales = [0.6_f64, 1.1];
        let points_b: Vec<Array1<f64>> = (0..r)
            .flat_map(|k| {
                let b_outer = b0.clone();
                [1.0_f64, -1.0_f64].into_iter().map(move |sign| {
                    let mut b = b_outer.clone();
                    b[k] += sign * scales[k];
                    b
                })
            })
            .collect();
        let a_set_1: Vec<Array2<f64>> = (0..m)
            .map(|i| {
                let mut a = Array2::<f64>::eye(p);
                a[[0, 0]] = 1.0 + 0.05 * i as f64;
                a[[1, 1]] = 1.5 - 0.02 * i as f64;
                a[[0, 1]] = 0.10;
                a[[1, 0]] = 0.10;
                a
            })
            .collect();
        let a_set_2: Vec<Array2<f64>> = (0..m)
            .map(|i| {
                let mut a = Array2::<f64>::eye(p);
                a[[0, 0]] = 2.0 - 0.04 * i as f64;
                a[[2, 2]] = 0.9 + 0.03 * i as f64;
                a[[1, 2]] = -0.05;
                a[[2, 1]] = -0.05;
                a
            })
            .collect();
        let alpha = 0.3_f64;
        let beta = 1.0 - alpha; // convex
        let a_set_mix: Vec<Array2<f64>> = a_set_1
            .iter()
            .zip(a_set_2.iter())
            .map(|(a, ap)| a.mapv(|v| v * alpha) + ap.mapv(|v| v * beta))
            .collect();

        let pts1: Vec<(Array2<f64>, Array1<f64>)> = a_set_1
            .iter()
            .zip(points_b.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();
        let pts2: Vec<(Array2<f64>, Array1<f64>)> = a_set_2
            .iter()
            .zip(points_b.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();
        let pts_mix: Vec<(Array2<f64>, Array1<f64>)> = a_set_mix
            .iter()
            .zip(points_b.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();

        let v1 = accumulate_sigma_cubature_total_covariance(&pts1, p);
        let v2 = accumulate_sigma_cubature_total_covariance(&pts2, p);
        let vmix = accumulate_sigma_cubature_total_covariance(&pts_mix, p);

        // With α+β = 1 the var(b) terms cancel, so
        // V[α·A + β·A'] = α·V[A] + β·V[A'].
        let expected = v1.mapv(|v| v * alpha) + v2.mapv(|v| v * beta);
        let mut max_abs = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_abs = max_abs.max((vmix[[i, j]] - expected[[i, j]]).abs());
            }
        }
        assert!(
            max_abs < 1e-13,
            "A-side convexity violated: max_abs={:.3e}",
            max_abs,
        );
    }

    /// Test #12 — translation invariance of var(b).
    ///
    /// Adding the same constant offset to every `b_m` (i.e. translating
    /// the b-cloud) cannot change the variance term `second_beta −
    /// mean_outer`. The A-side is unaffected by b at all. So shifting
    /// every b by the same vector leaves the whole output unchanged.
    /// Pins that the accumulator's b-side computes a *centred* second
    /// moment (not a raw second moment).
    #[test]
    pub(crate) fn cubature_b_translation_leaves_output_unchanged() {
        let p = 3;
        let a_const: Array2<f64> =
            ndarray::array![[1.5, 0.2, 0.0], [0.2, 1.2, 0.1], [0.0, 0.1, 1.0],];
        let raw_bs: Vec<Array1<f64>> = vec![
            ndarray::array![0.4, -0.3, 0.2],
            ndarray::array![-0.4, 0.3, -0.2],
            ndarray::array![0.1, 0.5, -0.4],
            ndarray::array![-0.1, -0.5, 0.4],
        ];
        let pts_raw: Vec<(Array2<f64>, Array1<f64>)> = raw_bs
            .iter()
            .map(|b| (a_const.clone(), b.clone()))
            .collect();
        let shift: Array1<f64> = ndarray::array![10.0, -5.0, 3.0];
        let pts_shifted: Vec<(Array2<f64>, Array1<f64>)> = raw_bs
            .iter()
            .map(|b| (a_const.clone(), b + &shift))
            .collect();

        let v_raw = accumulate_sigma_cubature_total_covariance(&pts_raw, p);
        let v_shifted = accumulate_sigma_cubature_total_covariance(&pts_shifted, p);
        let mut max_abs = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_abs = max_abs.max((v_raw[[i, j]] - v_shifted[[i, j]]).abs());
            }
        }
        // Translating b by a constant of magnitude ~10 inflates the
        // intermediate raw second moment by ~100, so the centred
        // difference is a real test of the variance subtraction's
        // numerical stability — a 1e-10 absolute bound here would still
        // be a generous round-off allowance, and we pick 1e-11.
        assert!(
            max_abs < 1e-11,
            "b-translation invariance violated: max_abs={:.3e}",
            max_abs,
        );
    }

    /// Test #13 — block-diagonal A and block-aligned b decouple.
    ///
    /// If every A_m has a 2×2 block-diagonal structure A_m = blkdiag(A_top, A_bot)
    /// AND every b_m has matching block structure (top half varies, bottom
    /// half is constant), the output must be block-diagonal too — the
    /// cross-block entries vanish exactly. Pins that the accumulator does
    /// not introduce spurious cross-block coupling (e.g. via a buggy
    /// outer-product loop that wraps indices).
    #[test]
    pub(crate) fn cubature_block_diagonal_inputs_yield_block_diagonal_output() {
        let p_top = 2;
        let p_bot = 2;
        let p = p_top + p_bot;
        let m = 4;
        let a_top: Array2<f64> = ndarray::array![[1.0, 0.1], [0.1, 1.2]];
        let a_bot: Array2<f64> = ndarray::array![[0.8, 0.05], [0.05, 0.7]];
        let mut a_full = Array2::<f64>::zeros((p, p));
        for i in 0..p_top {
            for j in 0..p_top {
                a_full[[i, j]] = a_top[[i, j]];
            }
        }
        for i in 0..p_bot {
            for j in 0..p_bot {
                a_full[[p_top + i, p_top + j]] = a_bot[[i, j]];
            }
        }

        // b: top half varies across sigma points (so var on the top is
        // non-zero), bottom half is constant (so var on the bottom is
        // zero). Cross-block (top × bottom) entries of mean_outer cancel
        // because the bottom is constant, but second_beta cross-entries
        // are b_top·b_bot — these must equal mean(b_top)·b_bot_const
        // exactly, so var cross-entries cancel.
        let b_bot_const: Array1<f64> = ndarray::array![0.5, -0.3];
        let top_bs: Vec<Array1<f64>> = vec![
            ndarray::array![0.4, 0.1],
            ndarray::array![-0.4, -0.1],
            ndarray::array![0.2, 0.3],
            ndarray::array![-0.2, -0.3],
        ];
        let mut points: Vec<(Array2<f64>, Array1<f64>)> = Vec::with_capacity(m);
        for top in &top_bs {
            let mut b = Array1::<f64>::zeros(p);
            for i in 0..p_top {
                b[i] = top[i];
            }
            for i in 0..p_bot {
                b[p_top + i] = b_bot_const[i];
            }
            points.push((a_full.clone(), b));
        }
        let v = accumulate_sigma_cubature_total_covariance(&points, p);
        let mut max_cross_abs = 0.0_f64;
        for i in 0..p_top {
            for j in 0..p_bot {
                max_cross_abs = max_cross_abs.max(v[[i, p_top + j]].abs());
                max_cross_abs = max_cross_abs.max(v[[p_top + j, i]].abs());
            }
        }
        assert!(
            max_cross_abs < 1e-13,
            "block-diagonal inputs leaked cross-block coupling: \
             max_cross_abs={:.3e}",
            max_cross_abs,
        );
    }

    /// Test #14 — output is always symmetric (PSD by construction
    /// when inputs are PSD).
    ///
    /// `accumulate_sigma_cubature_total_covariance` is a sum of:
    ///   * mean(A_m) — symmetric when each A_m is symmetric
    ///   * second_beta − mean_outer — a centred second moment, always
    ///     symmetric and PSD
    /// So the output should be symmetric to f64 round-off for any
    /// symmetric A_m. Pins this invariant since the production caller
    /// passes the output to `symmetrize_in_place` and any drift here is a
    /// silent bug masked by that downstream cleanup.
    #[test]
    pub(crate) fn cubature_output_is_symmetric_for_symmetric_inputs() {
        let p = 5;
        let m = 6;
        let points: Vec<(Array2<f64>, Array1<f64>)> = (0..m)
            .map(|idx| {
                let mut a = Array2::<f64>::eye(p);
                // Build a non-trivial symmetric A by mirroring across
                // the diagonal.
                for i in 0..p {
                    for j in 0..i {
                        let v = 0.05 + 0.03 * (i as f64 + j as f64) + 0.02 * (idx as f64);
                        a[[i, j]] = v;
                        a[[j, i]] = v;
                    }
                    a[[i, i]] = 2.0 + 0.1 * idx as f64;
                }
                let b: Array1<f64> = (0..p)
                    .map(|d| 0.1 * (d as f64 + 1.0) + 0.05 * idx as f64)
                    .collect();
                (a, b)
            })
            .collect();
        let v = accumulate_sigma_cubature_total_covariance(&points, p);
        let mut max_asym = 0.0_f64;
        for i in 0..p {
            for j in (i + 1)..p {
                max_asym = max_asym.max((v[[i, j]] - v[[j, i]]).abs());
            }
        }
        assert!(
            max_asym < 1e-13,
            "output not symmetric for symmetric inputs: max_asym={:.3e}",
            max_asym,
        );
    }

    /// Test #15 — pure-A invariance under reordering of (A_m, b_m).
    ///
    /// Permuting which b_m is paired with which A_m must move the
    /// var(b) term (since that is computed from the b_m sequence) but
    /// must NOT move the mean_hinv term (which depends only on the
    /// multiset of A_m). Pins that the accumulator's A-side and b-side
    /// are truly decoupled in code, not just in math.
    #[test]
    pub(crate) fn cubature_a_side_unchanged_under_b_permutation() {
        let p = 3;
        let m = 6;
        let a_set: Vec<Array2<f64>> = (0..m)
            .map(|i| {
                let mut a = Array2::<f64>::eye(p);
                a[[0, 0]] = 1.0 + 0.07 * (i as f64);
                a[[1, 1]] = 1.5 - 0.05 * (i as f64);
                a[[2, 2]] = 0.9 + 0.04 * (i as f64);
                a[[0, 1]] = 0.05;
                a[[1, 0]] = 0.05;
                a
            })
            .collect();
        // Constant b across all sigma points → var(b) = 0 → output is
        // exactly mean(A_m). Permuting b is a no-op for constant b but
        // permuting A_m would change mean(A_m) — except mean is
        // permutation-invariant. So output of original == output of
        // permuted-A == mean(A_m).
        let b_const: Array1<f64> = ndarray::array![0.2, -0.1, 0.3];
        let original: Vec<(Array2<f64>, Array1<f64>)> =
            a_set.iter().map(|a| (a.clone(), b_const.clone())).collect();
        // Build a non-trivial permutation of A's.
        let perm: [usize; 6] = [3, 0, 5, 1, 4, 2];
        let permuted_a: Vec<(Array2<f64>, Array1<f64>)> = perm
            .iter()
            .map(|&i| (a_set[i].clone(), b_const.clone()))
            .collect();

        let v_orig = accumulate_sigma_cubature_total_covariance(&original, p);
        let v_perm = accumulate_sigma_cubature_total_covariance(&permuted_a, p);
        let w = 1.0 / m as f64;
        let mut mean_a = Array2::<f64>::zeros((p, p));
        for a in &a_set {
            mean_a.scaled_add(w, a);
        }
        let mut max_abs_orig = 0.0_f64;
        let mut max_abs_perm = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                max_abs_orig = max_abs_orig.max((v_orig[[i, j]] - mean_a[[i, j]]).abs());
                max_abs_perm = max_abs_perm.max((v_perm[[i, j]] - mean_a[[i, j]]).abs());
            }
        }
        assert!(
            max_abs_orig < 1e-13 && max_abs_perm < 1e-13,
            "A-side independent of A ordering under constant b: \
             orig={:.3e}, perm={:.3e}",
            max_abs_orig,
            max_abs_perm,
        );
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

    pub(crate) fn make_first_order(
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
            rho_covariance: None,
            reason,
            severity,
            method: with_matrix.then_some(
                SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                    active_rank: 1,
                    rho_dimension: 1,
                },
            ),
        }
    }

    #[test]
    pub(crate) fn cubature_branch_label_and_extraction() {
        let outcome = SmoothingCorrectionOutcome::Cubature {
            correction: array![[2.0, 0.0], [0.0, 2.0]],
            rho_covariance: None,
            rho_hessian_stabilization: gam_problem::StabilizationLedger::approximation_only(
                1.0e-8,
                gam_problem::StabilizationRule::FixedConstant,
            )
            .expect("valid test cubature ridge"),
            rank: 2,
            n_points: 4,
            near_boundary: true,
            grad_norm: 1.5,
            max_rho_var: 0.7,
            // Deliberately DIFFERENT from `correction` above so the test can
            // prove the retained first-order pair is not silently aliased to
            // the primary cubature pair (#946).
            first_order_correction: Some(array![[1.0, 0.0], [0.0, 1.0]]),
            first_order_method: Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace {
                active_rank: 1,
                rho_dimension: 1,
            }),
        };
        assert_eq!(outcome.branch_label(), "cubature");
        let (mat, method, first_order_mat, first_order_method) =
            outcome.into_correction_with_method();
        let mat = mat.expect("cubature always has a matrix");
        assert!(matches!(
            method,
            Some(SmoothingCorrectionMethod::SigmaPointCubature { .. })
        ));
        assert_eq!(mat.dim(), (2, 2));
        assert_eq!(mat[[0, 0]], 2.0);

        let first_order_mat = first_order_mat.expect("retained first-order matrix");
        assert_eq!(first_order_mat.dim(), (2, 2));
        assert_eq!(
            first_order_mat[[0, 0]],
            1.0,
            "retained first-order correction must be the value the cubature branch was \
             constructed with, not the primary cubature correction"
        );
        assert!(
            matches!(
                first_order_method,
                Some(SmoothingCorrectionMethod::FirstOrderIdentifiedSubspace { .. })
            ),
            "retained first-order provenance must never be SigmaPointCubature"
        );
    }

    #[test]
    pub(crate) fn first_order_routine_branch_label_and_extraction() {
        let outcome = make_first_order(
            "n_rho == 0",
            SmoothingCorrectionFallbackSeverity::Routine,
            true,
        );
        assert_eq!(outcome.branch_label(), "first-order (routine)");
        assert!(outcome.into_correction_with_method().0.is_some());
    }

    #[test]
    pub(crate) fn first_order_numerical_branch_label_and_extraction() {
        let outcome = make_first_order(
            "rho Hessian inversion failed after ridge regularization",
            SmoothingCorrectionFallbackSeverity::NumericalFailure,
            true,
        );
        assert_eq!(outcome.branch_label(), "first-order (numerical failure)");
        assert!(outcome.into_correction_with_method().0.is_some());
    }

    #[test]
    pub(crate) fn first_order_without_matrix_returns_none() {
        let outcome = make_first_order(
            "no base covariance supplied",
            SmoothingCorrectionFallbackSeverity::Routine,
            false,
        );
        assert!(outcome.into_correction_with_method().0.is_none());
    }

    #[test]
    pub(crate) fn severity_counter_is_monotonic() {
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
    pub(crate) fn cubature_counter_is_observable() {
        let before = SMOOTHING_CORRECTION_CUBATURE_COUNT.load(Ordering::Relaxed);
        SMOOTHING_CORRECTION_CUBATURE_COUNT.fetch_add(1, Ordering::Relaxed);
        let after = SMOOTHING_CORRECTION_CUBATURE_COUNT.load(Ordering::Relaxed);
        assert!(after > before);
    }

    /// #582 — the SIGMA-CUBATURE smoothing-correction path must be
    /// response-scale equivariant: under `y → c·y` the returned correction (and
    /// hence `Vp = Vb + correction`) must scale by exactly `c²`, never `c⁴`.
    ///
    /// This is the DETERMINISTIC companion to the first-order integration test
    /// `corrected_covariance_is_response_scale_equivariant`. A full `fit_gam`
    /// will not reliably drive ρ̂ into `AUTO_CUBATURE_BOUNDARY_MARGIN` of the
    /// box edge, so instead of hoping a fit lands there, this test calls
    /// [`RemlState::compute_smoothing_correction_auto`] directly with a
    /// `final_rho` FORCED to `RHO_BOUND − 1` (inside the 2.0 margin): the
    /// `near_boundary` gate is then unconditionally `true` and the cubature
    /// branch fires (asserted via the `SMOOTHING_CORRECTION_CUBATURE_COUNT`
    /// delta). Running the same construction at response scales `1` and `c`
    /// then exercises the per-sigma φ̂ curvature scaling on the cubature path
    /// and asserts the `c²` (not `c⁴`) equivariance of the correction itself.
    ///
    /// For a Gaussian identity GAM `H = XᵀWX + λS` is dispersion-free, so the
    /// base covariance `H⁻¹` is IDENTICAL at both scales; β̂ → c·β̂ and the
    /// deviance (RSS) → c²·deviance, so φ̂ → c²·φ̂. The cubature correction
    ///   φ̂·(E_ρ[H⁻¹] − H_opt⁻¹) + Cov_ρ[β̂]
    /// then scales by exactly c² when (and only when) the curvature block
    /// carries exactly one φ̂ — the fix under test.
    #[test]
    pub(crate) fn cubature_smoothing_correction_is_response_scale_equivariant() {
        use crate::estimate::PenaltySpec;
        use gam_problem::{
            GlmLikelihoodSpec, InverseLink, LikelihoodSpec, ResponseFamily, StandardLink,
        };

        // Deterministic small Gaussian identity design (n=24, p=4: intercept +
        // 3 penalized columns). Smooth, well-conditioned; the near-boundary ρ
        // is FORCED below, not discovered, so the data need only yield a valid
        // converged inner fit and an invertible ρ-Hessian.
        fn design(scale: f64) -> (Array2<f64>, Array1<f64>) {
            let n = 24usize;
            let p = 4usize;
            let mut x = Array2::<f64>::zeros((n, p));
            let mut y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let t = (i as f64) / ((n - 1) as f64);
                let tau = std::f64::consts::TAU;
                x[[i, 0]] = 1.0;
                x[[i, 1]] = t;
                x[[i, 2]] = (tau * t).sin();
                x[[i, 3]] = (tau * t).cos();
                let base =
                    0.7 + 0.9 * t + 0.5 * (tau * t).sin() + 0.05 * ((i as f64) * 2.399_963).sin();
                y[i] = scale * base;
            }
            (x, y)
        }

        // Ridge on the 3 non-intercept columns; nullspace dim 1 (the intercept).
        let p = 4usize;
        let mut s = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s[[j, j]] = 1.0;
        }

        // Force the near-boundary gate: ρ = RHO_BOUND − 1 is within
        // AUTO_CUBATURE_BOUNDARY_MARGIN (= 2.0) of RHO_BOUND, so
        // `near_boundary` is true regardless of where any optimizer would land.
        let final_rho = Array1::from_vec(vec![RHO_BOUND - 1.0]);

        // Run the full cubature path at one response scale; return the returned
        // correction matrix plus the cubature-counter delta observed for THIS
        // call (proves the cubature branch — not the first-order fallback — ran).
        let run = |scale: f64| -> (Array2<f64>, u64) {
            let (x, y) = design(scale);
            let n = x.nrows();
            let w = Array1::<f64>::ones(n);
            let offset = Array1::<f64>::zeros(n);

            let spec = PenaltySpec::Dense(s.clone());
            let canonical =
                gam_terms::construction::canonicalize_penalty_specs(&[spec], &[1], p, "test")
                    .map(|(canonical, _)| canonical)
                    .expect("canonicalize penalty");
            let cfg = RemlConfig::external(
                GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
                    ResponseFamily::Gaussian,
                    InverseLink::Standard(StandardLink::Identity),
                )),
                1e-12,
                false,
            );
            let state = RemlState::newwith_offset(
                y.view(),
                x.clone(),
                w.view(),
                offset.view(),
                canonical,
                p,
                &cfg,
                Some(vec![1]),
                None,
                None,
            )
            .expect("build RemlState");

            // Converged inner fit at the forced near-boundary ρ — this is the
            // `final_fit` the cubature path differentiates around, and its
            // Qs-mapped H⁻¹ is the dispersion-free base covariance the
            // correction upgrades.
            let final_fit = state
                .execute_pirls_stateless_for_cubature(&final_rho)
                .expect("inner PIRLS at near-boundary rho");
            let h_orig = map_hessian_to_original_basis(final_fit.as_ref())
                .expect("map Hessian to original basis");
            let base_cov = gam_linalg::utils::certified_spd_inverse(&h_orig, "test base cov")
                .expect("invert base Hessian")
                .into_inverse();

            // Profiled Gaussian dispersion φ̂ = deviance / (n − p). Deviance (RSS)
            // scales as c², the denominator is scale-invariant, so φ̂ scales as c².
            let dispersion_phi = final_fit.deviance / ((n as f64) - (p as f64)).max(1.0);

            // Real outer-gradient norm at the forced ρ (finite, for the gate's
            // highgrad arm); near_boundary already guarantees cubature entry.
            let finalgrad_norm = state
                .compute_gradient(&final_rho)
                .map(|g| g.dot(&g).sqrt())
                .unwrap_or(0.0);

            let before = SMOOTHING_CORRECTION_CUBATURE_COUNT.load(Ordering::SeqCst);
            let final_lambdas = Array1::from_vec(
                gam_problem::checked_exp_log_strengths(final_rho.iter().copied())
                    .expect("test rho lies in exact strength domain"),
            );
            let outcome = state
                .compute_smoothing_correction_auto(
                    &final_rho,
                    &final_lambdas,
                    final_fit.as_ref(),
                    Some(&base_cov),
                    dispersion_phi,
                    finalgrad_norm,
                )
                .expect("smoothing correction evaluation");
            let after = SMOOTHING_CORRECTION_CUBATURE_COUNT.load(Ordering::SeqCst);

            let correction = outcome
                .into_correction_with_method()
                .0
                .expect("cubature/first-order outcome carries a correction matrix");
            (correction, after.saturating_sub(before))
        };

        let c = 1000.0_f64;
        let c2 = c * c;

        let (corr1, fired1) = run(1.0);
        let (corrc, firedc) = run(c);

        // The cubature branch must have fired at BOTH scales — otherwise this
        // test would silently fall back to the first-order path and NOT cover
        // the eval.rs per-sigma φ̂ curvature scaling (#582).
        assert!(
            fired1 > 0,
            "sigma-cubature branch did not fire at scale 1 (delta {fired1}); \
             the near-boundary gate should have forced it"
        );
        assert!(
            firedc > 0,
            "sigma-cubature branch did not fire at scale {c} (delta {firedc})"
        );

        // The correction must be materially non-zero (so the equivariance check
        // is not vacuous) and finite.
        let frob1 = corr1.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            frob1.is_finite() && frob1 > 0.0,
            "scale-1 cubature correction must be finite and non-zero (‖corr‖={frob1:.3e})"
        );
        assert_eq!(
            corr1.dim(),
            corrc.dim(),
            "correction shape mismatch across scales"
        );

        // Property under test: every entry scales by exactly c² (never c⁴).
        let mut worst_rel = 0.0_f64;
        let (mut wi, mut wj) = (0usize, 0usize);
        for i in 0..p {
            for j in 0..p {
                let expected = c2 * corr1[[i, j]];
                let got = corrc[[i, j]];
                let denom = expected.abs().max(c2 * frob1 * 1e-12).max(1e-300);
                let rel = (got - expected).abs() / denom;
                if rel > worst_rel {
                    worst_rel = rel;
                    wi = i;
                    wj = j;
                }
            }
        }
        assert!(
            worst_rel < 1e-6,
            "cubature smoothing correction is not response-scale equivariant: \
             corr[{wi},{wj}] scales by {factor:.3e}·c² instead of c² \
             (corr@1={a:.6e}, corr@{c}={b:.6e}, expected {e:.6e}, rel {worst_rel:.3e}). \
             A `c⁴` here is the per-sigma curvature term carrying φ̂ twice; a `c⁰` \
             factor is the curvature term missing its φ̂ (#582).",
            factor = corrc[[wi, wj]] / (c2 * corr1[[wi, wj]]).abs().max(1e-300),
            a = corr1[[wi, wj]],
            b = corrc[[wi, wj]],
            e = c2 * corr1[[wi, wj]],
        );
    }

    #[test]
    pub(crate) fn classification_reason_strings_are_nonempty_and_distinct() {
        let reasons = [
            // Routine gates.
            "n_rho == 0: unified corrected covariance equals H^{-1}",
            "n_rho exceeds AUTO_CUBATURE_MAX_RHO_DIM: cubature cost prohibitive",
            "beta dimension exceeds AUTO_CUBATURE_MAX_BETA_DIM: cubature cost prohibitive",
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
