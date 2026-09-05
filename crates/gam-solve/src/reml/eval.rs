use super::inner_strategy::GeometryBackendKind;
use super::penalty_logdet::PenaltyPseudologdet;
use super::*;
use crate::model_types::SmoothingCorrectionMethod;
use gam_linalg::matrix::symmetrize_in_place;
use std::sync::atomic::Ordering;

// Inset from RHO_BOUND when scaling a sigma-point step so the inner PIRLS
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
        rank: usize,
        n_points: usize,
        /// Worst `V(node) − V(ρ̂)` over the nodes this correction was built
        /// from. See [`SmoothingCorrectionMethod::SigmaPointCubature`].
        max_node_criterion_rise: f64,
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
        /// Why the cubature upgrade was not taken. `Cow` rather than
        /// `&'static str` because one of these reasons is not a fixed
        /// classification but a propagated numerical failure — the typed error
        /// from a sigma point's inner solve — and collapsing that to a constant
        /// would discard the only description of what actually went wrong
        /// (#2601).
        reason: std::borrow::Cow<'static, str>,
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
                max_node_criterion_rise,
                first_order_correction,
                first_order_method,
                ..
            } => (
                Some(correction),
                Some(SmoothingCorrectionMethod::SigmaPointCubature {
                    rank,
                    n_points,
                    max_node_criterion_rise,
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
///
/// `centre_fit` is the converged fit at `rho_hat`. Its coefficient vector is
/// handed to every sigma point as a shared, immutable seed — see
/// [`sigma_cubature_evaluate_cpu_rayon`] for why that is not the cross-call
/// state the stateless callee exists to avoid.
pub(crate) fn sigma_cubature_dispatch(
    state: &RemlState<'_>,
    sigma_points: &[Array1<f64>],
    centre_fit: Option<&PirlsResult>,
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

    sigma_cubature_evaluate_cpu_rayon(state, sigma_points, centre_fit)
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
///
/// ## Why the sigma points are SEEDED and still stateless
///
/// The callee threads no MUTABLE cross-call state — that is what makes the
/// sigma fits independent of the production trajectory and of each other. A
/// shared *immutable* seed is a different thing entirely: `centre_beta` is the
/// converged coefficient vector at `rho_hat`, one constant, identical for every
/// point, read-only, and computed before any sigma fit starts. It couples
/// nothing.
///
/// It matters most for the case that motivated it. `beta_hat(rho)` is
/// continuous in rho, so the centre mode is the natural seed for a perturbation
/// of rho — and for a SHAPE-CONSTRAINED fit it is also a *feasible* one. The
/// cold seed is not: `default_beta_guess_external` lands on or outside the
/// constraint cone, and for a homogeneous curvature cone the projection that
/// repairs it starts the inner active-set QP from a face with every row tight.
/// That is the #873 degenerate-vertex regime, and at an off-trajectory rho
/// (where lambda can be many decades from its fitted value) the QP does not
/// recover from it — measured as `LmStepSearchExhausted` with the LM parameter
/// pinned at its ceiling, on a fit whose own inner solves all reached
/// ‖g‖ ~ 1e-13 (#2601).
pub(crate) fn sigma_cubature_evaluate_cpu_rayon(
    state: &RemlState<'_>,
    sigma_points: &[Array1<f64>],
    centre_fit: Option<&PirlsResult>,
) -> Result<Vec<SigmaPointResult>, EstimationError> {
    // Map the centre mode back to the ORIGINAL coefficient basis once: the
    // per-rho reparameterization `Qs` differs at every sigma point, and
    // `fit_model_for_fixed_rho_with_adaptive_kkt` takes its warm start in the
    // original basis and applies each point's own transform itself.
    let centre_beta = centre_fit
        .map(|fit| Coefficients::new(fit.reparam_result.qs.dot(fit.beta_transformed.as_ref())));
    let rows: Vec<Result<SigmaPointResult, EstimationError>> = (0..sigma_points.len())
        .into_par_iter()
        .map(|idx| -> Result<SigmaPointResult, EstimationError> {
            let fit_point = state
                .execute_pirls_stateless_for_cubature(&sigma_points[idx], centre_beta.as_ref())?;
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

/// Accumulate the sigma-point cubature total covariance `V̂_p`.
///
/// Math. The estimand is the law of total covariance for the
/// smoothing-parameter-marginalised posterior,
///
/// ```text
///     V_p = E_ρ[Cov(β|ρ)] + Cov_ρ[β̂(ρ)] = φ̂·E_ρ[H(ρ)⁻¹] + Cov_ρ[β̂(ρ)].
/// ```
///
/// `paired[j]` holds the `(H(ρ)⁻¹, β̂(ρ))` pair at the `+` and `−` node of
/// upgraded ρ-eigendirection `j` (the inverse-Hessian blocks already carrying
/// their `φ̂`). The two terms are accumulated as
///
/// ```text
///     φ̂·Ê_ρ[H(ρ)⁻¹] = mean over all 2r nodes of the per-node block
///     Ĉov_ρ[β̂]      = Σ_j c_j c_jᵀ + Σ_k f_k f_kᵀ,
///     c_j = (β̂(ρ_j⁺) − β̂(ρ_j⁻))/2,     f_k = Qs·J·u_k/√σ_k.
/// ```
///
/// # Why the covariance term is a PER-DIRECTION chord and not a joint moment
///
/// The previous form took the empirical covariance of `β̂` over the whole node
/// set. That reproduces `J·V_ρ·Jᵀ` for a linear `β̂` only through a conspiracy
/// between the `√rank` node radius and the `1/(2·rank)` weight, and it breaks
/// the moment as soon as the two sides of a direction are not at the same
/// distance — which is exactly what criterion-calibrated nodes produce, and
/// what an asymmetric criterion (flat one way, a cliff the other) requires.
///
/// The per-direction chord has no such conspiracy: with `β̂` linear and the
/// nodes at `±t_j` it gives `c_j = t_j·Qs·J·u_j`, so at the uncalibrated step
/// `t_j = σ_j^{-1/2}` the sum is `Σ_j (Qs·J·u_j)(Qs·J·u_j)ᵀ/σ_j = J·V_ρ·Jᵀ`
/// restricted to the upgraded subspace — *identically*, for any node scaling.
///
/// `fallback_columns` carries the first-order column of every ACTIVE direction
/// that was not upgraded, so the assembled correction covers the same subspace
/// the first-order correction does and the cubature is a strict upgrade rather
/// than a differently-truncated estimate (#2728).
///
/// The result is PSD by construction: a mean of PSD inverse-Hessian blocks plus
/// a sum of rank-one Grams. Since the caller forms the additive correction as
/// `total − φ̂·H(ρ̂)⁻¹`, the corrected covariance `Vb + correction` telescopes
/// back to this PSD matrix.
pub(crate) fn accumulate_sigma_cubature_total_covariance(
    paired: &[(SigmaPointResult, SigmaPointResult)],
    fallback_columns: &[Array1<f64>],
    p: usize,
) -> Array2<f64> {
    let mut total = Array2::<f64>::zeros((p, p));
    let node_count = 2 * paired.len();
    if node_count > 0 {
        let weight = 1.0 / node_count as f64;
        for ((cov_plus, _), (cov_minus, _)) in paired {
            total.scaled_add(weight, cov_plus);
            total.scaled_add(weight, cov_minus);
        }
    }
    let accumulate_gram = |column: &Array1<f64>, total: &mut Array2<f64>| {
        for row in 0..p {
            let scaled = column[row];
            if scaled == 0.0 {
                continue;
            }
            for col in 0..p {
                total[[row, col]] += scaled * column[col];
            }
        }
    };
    for ((_, beta_plus), (_, beta_minus)) in paired {
        let chord = (beta_plus - beta_minus).mapv(|value| 0.5 * value);
        accumulate_gram(&chord, &mut total);
    }
    for column in fallback_columns {
        accumulate_gram(column, &mut total);
    }
    total
}

/// Criterion level a one-sigma sigma-point node sits at.
///
/// The cubature is a symmetric two-point rule for `ρ ~ N(ρ̂, V_ρ)` with each
/// node one posterior sd out along a ρ-Hessian eigendirection. Under the
/// QUADRATIC model that defines `V_ρ` in the first place, the criterion at that
/// node is above the optimum by exactly
///
/// ```text
///     V(ρ̂ + σ^{-1/2}·u) − V(ρ̂) = ½·σ·(σ^{-1/2})² = ½.
/// ```
///
/// So `½` is not a tuning parameter: it is the criterion level the rule already
/// assumes its node occupies. Measuring it instead of assuming it changes
/// nothing wherever the quadratic model holds, and moves the node wherever it
/// does not (#2728).
const PROFILE_SIGMA_RISE: f64 = 0.5;

/// Factor within which an achieved criterion rise counts as agreeing with
/// [`PROFILE_SIGMA_RISE`].
///
/// Under the quadratic model this acceptance certifies, `ΔV ∝ t²`, so admitting
/// `ΔV ∈ [rise/κ, rise·κ]` bounds the node's relative position error by
/// `√κ − 1` above and `1 − κ^{-1/2}` below — at `κ = 1.5`, +22% / −18% — on a
/// quantity that is itself a refinement to `Vb`. It also makes the common case,
/// a criterion that really is quadratic, cost exactly one criterion evaluation
/// per node.
const PROFILE_SIGMA_ACCEPT_FACTOR: f64 = 1.5;

/// Criterion evaluations one node's calibration may spend after the first.
///
/// Each is a single inner solve at fixed ρ. The bracketed power-law secant
/// below lands a quadratic criterion on its FIRST step and any clean power law
/// in two, so this budget only binds on a criterion that is neither over the
/// searched interval. When it does bind, the bracket endpoint closest to the
/// target level is returned and the level it actually achieved is reported
/// alongside it — on the outcome, in the log line, and in the serialized
/// `SmoothingCorrectionMethod` — so a poorly placed node is visible rather than
/// silently trusted.
const PROFILE_SIGMA_MAX_EVALS: usize = 12;

/// A sigma-point node whose position was calibrated against the criterion.
pub(crate) struct CalibratedSigmaNode {
    /// The node itself, `ρ̂ + step·direction`.
    pub rho: Array1<f64>,
    /// Step length actually taken along `direction`.
    pub step: f64,
    /// Step length the quadratic model asked for, `σ^{-1/2}`.
    pub wald_step: f64,
    /// `V(node) − V(ρ̂)` at the returned step.
    pub achieved_rise: f64,
    /// Criterion evaluations this node's calibration spent.
    pub evaluations: usize,
    /// The step ran into the ρ box before reaching [`PROFILE_SIGMA_RISE`].
    pub box_limited: bool,
}

/// Largest `t ≥ 0` keeping `rho + t·direction` inside the ρ box on every
/// coordinate.
///
/// The previous code clamped each COORDINATE of the displaced point
/// independently, which silently rotates the ray being sampled away from the
/// eigendirection it was supposed to sample. Scaling the whole step instead
/// keeps the node on its own direction, so the two-point rule stays a rule
/// about that eigendirection.
fn sigma_step_to_rho_box(rho: &Array1<f64>, direction: &Array1<f64>) -> f64 {
    let lo = -RHO_BOUND + AUTO_CUBATURE_RHO_CLAMP_INSET;
    let hi = RHO_BOUND - AUTO_CUBATURE_RHO_CLAMP_INSET;
    let mut limit = f64::INFINITY;
    for (centre, component) in rho.iter().zip(direction.iter()) {
        if *component > 0.0 {
            limit = limit.min((hi - centre) / component);
        } else if *component < 0.0 {
            limit = limit.min((lo - centre) / component);
        }
    }
    if limit.is_finite() { limit.max(0.0) } else { f64::INFINITY }
}

impl<'a> RemlState<'a> {
    /// Place one sigma-point node at the criterion level the cubature rule
    /// assumes it occupies, rather than at the step a possibly-inapplicable
    /// quadratic model implies.
    ///
    /// `direction` is a unit ρ-vector carrying its own sign; `wald_step` is
    /// `σ^{-1/2}` for that eigendirection. The returned node satisfies
    /// `V(ρ̂ + step·direction) − V(ρ̂) ≈ PROFILE_SIGMA_RISE`, except where the ρ
    /// box is reached first (`box_limited`) — which is the honest statement
    /// that the criterion never rises to that level inside the fit's own λ
    /// range — or where the evaluation budget runs out, in which case the
    /// bracket endpoint closest to the target level is returned and
    /// `achieved_rise` reports where it actually landed.
    ///
    /// Exact reduction: if the criterion is quadratic along `direction`, the
    /// very first evaluation lands on `PROFILE_SIGMA_RISE` and `step ==
    /// wald_step`, so the node — and therefore the whole correction — is
    /// identical to what the uncalibrated rule produced.
    pub(crate) fn calibrate_sigma_node(
        &self,
        rho_hat: &Array1<f64>,
        centre_cost: f64,
        direction: &Array1<f64>,
        wald_step: f64,
    ) -> Result<CalibratedSigmaNode, EstimationError> {
        let box_limit = sigma_step_to_rho_box(rho_hat, direction);
        let at = |step: f64| -> Array1<f64> {
            let mut point = rho_hat.clone();
            point
                .iter_mut()
                .zip(direction.iter())
                .for_each(|(coordinate, component)| *coordinate += step * component);
            point
        };
        let centre_node = |evaluations: usize| CalibratedSigmaNode {
            rho: rho_hat.clone(),
            step: 0.0,
            wald_step,
            achieved_rise: 0.0,
            evaluations,
            box_limited: true,
        };
        if !wald_step.is_finite() || wald_step <= 0.0 || box_limit <= 0.0 {
            // Either the direction has no resolvable width, or ρ̂ already sits
            // on the box face along it. Both mean the node is the centre and
            // this side of the chord is zero-length.
            return Ok(centre_node(0));
        }

        let target = PROFILE_SIGMA_RISE;
        let mut step = wald_step.min(box_limit);
        let mut rise = self.compute_cost_uncharged(&at(step))? - centre_cost;
        let mut evaluations = 1usize;
        // Bracket: `lo` is the largest step known to undershoot the target,
        // `hi` the smallest known to overshoot it. `V(ρ̂) − V(ρ̂) = 0` seeds
        // `lo` for free.
        let (mut lo, mut lo_rise) = (0.0_f64, 0.0_f64);
        let (mut hi, mut hi_rise) = (f64::INFINITY, f64::INFINITY);
        let mut best = (step, rise);
        let accepts = |value: f64| {
            value.is_finite()
                && value >= target / PROFILE_SIGMA_ACCEPT_FACTOR
                && value <= target * PROFILE_SIGMA_ACCEPT_FACTOR
        };
        // Closeness in log-ratio to the target, so an overshoot and an
        // undershoot by the same factor rank equally.
        let closeness = |value: f64| {
            if value.is_finite() && value > 0.0 {
                (value / target).ln().abs()
            } else {
                f64::INFINITY
            }
        };

        while !accepts(rise) && evaluations <= PROFILE_SIGMA_MAX_EVALS {
            if closeness(rise) < closeness(best.1) {
                best = (step, rise);
            }
            if !rise.is_finite() || rise > target {
                // A non-finite criterion is not "flat" — it is unevaluable at
                // this step, which is the same instruction as an overshoot:
                // come back in.
                hi = step;
                hi_rise = rise;
            } else {
                if step >= box_limit {
                    // The criterion never reaches one sigma inside the fit's
                    // own λ range along this direction. The box face IS the
                    // node; nothing further out exists to sample.
                    return Ok(CalibratedSigmaNode {
                        rho: at(step),
                        step,
                        wald_step,
                        achieved_rise: rise,
                        evaluations,
                        box_limited: true,
                    });
                }
                lo = step;
                lo_rise = if rise.is_finite() { rise.max(0.0) } else { 0.0 };
            }

            // Propose the next step from the power law `ΔV = c·t^q` implied by
            // the bracket. With only an overshoot in hand the exponent is the
            // quadratic model's `q = 2`, which lands a genuinely quadratic
            // criterion on the target in ONE step.
            let proposal = if hi.is_finite() && lo > 0.0 && lo_rise > 0.0 && hi_rise > lo_rise {
                let exponent = (hi_rise / lo_rise).ln() / (hi / lo).ln();
                if exponent.is_finite() && exponent > 0.0 {
                    lo * (target / lo_rise).powf(exponent.recip())
                } else {
                    (lo * hi).sqrt()
                }
            } else if hi.is_finite() && hi_rise > 0.0 {
                hi * (target / hi_rise).sqrt()
            } else if lo_rise > 0.0 {
                (lo * (target / lo_rise).sqrt()).min(box_limit)
            } else {
                // Perfectly flat so far: the only information is that the
                // criterion has not moved, so jump to the box face.
                box_limit
            };
            // With no overshoot bracketed yet the search interval runs to the
            // box face, and stepping exactly ONTO it is legal — the loop's
            // next pass then returns the box-limited node. Without this the
            // interior geometric fallback could only approach the face
            // asymptotically and a genuinely flat direction would burn the
            // whole evaluation budget never reaching it.
            let bracketed = hi.is_finite();
            let bracket_hi = if bracketed { hi } else { box_limit };
            let inside = proposal.is_finite()
                && proposal > lo
                && (proposal < bracket_hi || (!bracketed && proposal <= box_limit));
            let next = if inside {
                proposal
            } else if lo > 0.0 {
                (lo * bracket_hi).sqrt()
            } else {
                0.5 * bracket_hi
            };
            if !(next > 0.0) || (next - step).abs() <= f64::EPSILON * step.abs() {
                break;
            }
            step = next;
            rise = self.compute_cost_uncharged(&at(step))? - centre_cost;
            evaluations += 1;
        }
        if closeness(rise) < closeness(best.1) {
            best = (step, rise);
        }
        let (step, rise) = if accepts(rise) { (step, rise) } else { best };
        Ok(CalibratedSigmaNode {
            rho: at(step),
            step,
            wald_step,
            achieved_rise: rise,
            evaluations,
            box_limited: step >= box_limit,
        })
    }
}

/// Process-wide count of cubature upgrades that succeeded inside
/// `RemlState::compute_smoothing_correction_auto`. Paired with
/// `SMOOTHING_CORRECTION_NUMERICAL_FAILURE_COUNT` for visibility.
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

        let lambdas_slice = lambdas
            .as_slice()
            .expect("owned Array1 is contiguous, so as_slice always succeeds");

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
            let lambdas_slice = lambdas
                .as_slice()
                .expect("owned Array1 is contiguous, so as_slice always succeeds");
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

        let lambdas_slice = lambdas
            .as_slice()
            .expect("owned Array1 is contiguous, so as_slice always succeeds");

        // ONE factorization per evaluation point (#931): the same object also
        // serves the τ/ψ hyper-coordinate components in hyper.rs, so the
        // ridge and positive-eigenspace threshold of `log|Sλ|₊` are decided
        // exactly once for value, ρ-derivatives, and τ components alike.
        let pld = bundle.penalty_pseudologdet_original(
            &self.canonical_penalties,
            lambdas_slice,
            self.p,
        )?;

        // The derivative contraction must read the SAME penalty components the
        // factorization was built from (#2454): `∂log|S̃|₊/∂ρ_k = λ_k tr(S̃⁺S̃_k)`
        // is only the derivative of `pld.value()` when `S̃_k` is the block
        // whose weighted sum `pld` factorized.
        let applied = bundle.applied_canonical_penalties(&self.canonical_penalties)?;
        let value = pld.value();
        let rank = pld.rank();
        let (det1, det2) = pld.rho_derivatives_from_penalties(&applied, lambdas_slice);
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
                // #2450 — THE SAMPLER TARGETS A DISTRIBUTION; THE CRITERION DOES NOT.
                //
                // The tiers below sample `π(ρ|y) ∝ exp(−criterion(ρ))`, so the
                // criterion they are handed has to BE a log-density. The one the
                // optimizer minimizes is not: `evaluate_configured_rho_prior`
                // evaluates every unset coordinate directly as `Flat`, hence
                // exact zero for every finite ρ. That is the declared pure
                // REML/LAML criterion, but handing it to a sampler leaves no
                // proper prior over ρ: measured on the
                // n=600 anisotropic-Duchon fit in
                // `margslope_duchon_slowdown`, the NUTS tier doubles to maximum
                // depth and the fit does not return in 2136 s, against 1.28 s
                // once ρ carries a proper prior.
                //
                // `rho_prior_distribution_correction` provides the proper PC
                // contribution missing from the flat criterion. Adding it HERE, at the
                // sampler's own call site, is what keeps the two apart: no
                // criterion site is touched, so certification, the rail
                // certificates and every fit's λ̂ are byte-unchanged by
                // construction rather than by review.
                //
                // The Tier-0 certificate above is left on the criterion as the
                // optimizer sees it: it asks whether the PLUG-IN Gaussian is
                // adequate, which is a question about the object the fit
                // reports, and moving it is a separate decision recorded on
                // #2450.
                Some(escalator.escalate_rho_posterior(
                    final_rho,
                    &outer_hessian,
                    &mut |rho| {
                        self.without_persistent_warm_start_store(|| self.compute_cost(rho).ok())
                            .map(|cost| cost + self.rho_prior_distribution_correction(rho).0)
                    },
                    &mut |rho| {
                        self.without_persistent_warm_start_store(|| {
                            // NUTS leapfrog gradients need the criterion value and
                            // gradient at the same rho; compute them through one
                            // value+gradient outer evaluation so the inner PIRLS
                            // solve and IFT state are shared by construction.
                            self.compute_cost_and_gradient(rho).ok()
                        })
                        .map(|(cost, gradient)| {
                            let (prior_cost, prior_gradient) =
                                self.rho_prior_distribution_correction(rho);
                            (cost + prior_cost, gradient + prior_gradient)
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
        outer_gradient: &Array1<f64>,
        outer_hessian: Option<&Array2<f64>>,
        caller_measured_hessian_error: &[gam_linalg::curvature_resolution::MeasuredHessianError],
    ) -> Result<SmoothingCorrectionOutcome, EstimationError> {
        use SmoothingCorrectionFallbackSeverity::{NumericalFailure, Routine};

        // Always compute the fast first-order correction first.
        let first_order = super::compute_smoothing_correction(
            self,
            final_rho,
            final_lambdas,
            final_fit,
            outer_gradient,
            outer_hessian,
            caller_measured_hessian_error,
        );
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
        let first_order_routine =
            |correction: Option<Array2<f64>>, reason: std::borrow::Cow<'static, str>| {
                SmoothingCorrectionOutcome::FirstOrder {
                    correction,
                    rho_covariance: first_order_rho_covariance.clone(),
                    reason,
                    severity: Routine,
                    method: first_order_method,
                }
            };
        let first_order_numerical =
            |correction: Option<Array2<f64>>, reason: std::borrow::Cow<'static, str>| {
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
                "n_rho == 0: unified corrected covariance equals H^{-1}".into(),
            ));
        }
        if n_rho > AUTO_CUBATURE_MAX_RHO_DIM {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "n_rho exceeds AUTO_CUBATURE_MAX_RHO_DIM: cubature cost prohibitive".into(),
            ));
        }
        if final_fit.beta_transformed.len() > AUTO_CUBATURE_MAX_BETA_DIM {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "beta dimension exceeds AUTO_CUBATURE_MAX_BETA_DIM: cubature cost prohibitive"
                    .into(),
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

        // Reuse the certified `V_ρ` the first-order path already produced.
        //
        // This site used to build its OWN `V_ρ`, by adding a relative ridge to
        // the ρ-Hessian and inverting that. Two objects called `V_ρ` inside one
        // routine forced a blanket bail-out whenever the certified inverse was
        // rank-deficient, because the ridged inverse turns every dropped
        // direction into a `1/ridge` eigenvalue and the eigen-truncation below
        // would then have selected exactly those. With the certified spectrum
        // in hand there is nothing to bail out of: a direction that is not
        // `Active` is simply not a candidate node (#2728).
        let Some(spectrum) = first_order.spectrum.as_ref() else {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "certified rho spectrum unavailable: nothing for cubature to reuse".into(),
            ));
        };
        let active_directions = spectrum.active_directions();
        if active_directions.is_empty() {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "no active rho directions: the correction is already exactly zero".into(),
            ));
        }

        // Trigger. `max_rhovar` is the widest resolved ρ-posterior variance
        // `1/σ_j` over the ACTIVE directions — the certified ones, so a
        // structural or saturation null can no longer set it.
        let max_rhovar = active_directions
            .iter()
            .map(|&index| spectrum.eigenvalues[index].recip())
            .fold(0.0_f64, f64::max);
        if !near_boundary && !highgrad && max_rhovar < AUTO_CUBATURE_RHOVAR_TRIGGER {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "resolved rho posterior variance below trigger threshold".into(),
            ));
        }

        let Some(base_cov) = base_covariance else {
            // Caller did not supply a base covariance to upgrade. This
            // is a configuration choice (the caller has nothing to add
            // the cubature correction onto), not a numerical failure;
            // the first-order delta is the documented outcome.
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "no base covariance supplied: nothing for cubature to upgrade".into(),
            ));
        };
        let p = base_cov.nrows();
        if spectrum.sensitivity_orig.nrows() != p {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "certified sensitivities do not match the base covariance dimension".into(),
            ));
        }

        // Rank the active directions by the variance each one contributes to
        // the correction, `‖Qs·J·u_j‖²/σ_j`, and upgrade the largest.
        //
        // The previous rule ranked by `1/σ_j` — the spread of ρ — and kept
        // whichever direction the outer surface was flattest along. That is
        // exactly backwards: at a SATURATED smoothing parameter `1/σ_j` is
        // huge *because* `∂β̂/∂ρ → 0` there, so the old rule spent the whole
        // rank budget on the one direction that contributes nothing and
        // dropped every direction that does (#2728: rank=1 retained
        // `λ = 7.2e-9` and discarded the other six).
        let mut ranked: Vec<(usize, f64)> = active_directions
            .iter()
            .map(|&index| (index, spectrum.first_order_variance(index)))
            .filter(|(_, variance)| variance.is_finite() && *variance > 0.0)
            .collect();
        if ranked.is_empty() {
            return self.finalize_smoothing_outcome(first_order_routine(
                first_order_correction,
                "every active direction contributes zero first-order variance".into(),
            ));
        }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total_variance: f64 = ranked.iter().map(|(_, variance)| *variance).sum();
        let mut rank = 0usize;
        let mut captured = 0.0_f64;
        for (_, variance) in ranked
            .iter()
            .take(AUTO_CUBATURE_MAX_EIGENVECTORS.min(ranked.len()))
        {
            captured += *variance;
            rank += 1;
            if captured / total_variance >= AUTO_CUBATURE_TARGET_VAR_FRAC {
                break;
            }
        }
        let upgraded: Vec<usize> = ranked[..rank].iter().map(|(index, _)| *index).collect();
        if rank < ranked.len() {
            log::info!(
                "[sigma-cubature] upgrading {rank} of {} active rho direction(s), capturing \
                 {:.4} of the first-order correction variance; the remainder keeps its \
                 first-order column",
                ranked.len(),
                captured / total_variance,
            );
        }

        // Place each node where the criterion says one sigma is, not where the
        // curvature guessed it was.
        //
        // The rule this branch implements is a symmetric two-point quadrature
        // for `ρ ~ N(ρ̂, V_ρ)`: put the node one posterior sd out and read the
        // integrand there. `σ_j^{-1/2}` is the sd the QUADRATIC model of the
        // criterion implies, and by construction the quadratic model predicts
        // the criterion to rise by exactly `PROFILE_SIGMA_RISE` at that node.
        // That prediction is checkable — one criterion evaluation — and where
        // it fails, the node, not the prediction, is what has to move.
        //
        // #2728 is the failure: at `λ = 7.2e-9` the ρ-curvature is ~0 by the
        // chain-rule identity `H_ρ = diag(λ)H_λdiag(λ) + diag(g_ρ)` rather
        // than because the profile is flat, so `σ^{-1/2} = 308` in log-λ and
        // the node landed at `ΔV = 3309` — posterior weight `e^-3309` — while
        // carrying weight ½. Calibrating to the criterion also lets the two
        // sides differ, which that fixture needs and a `±` step cannot express:
        // the profile there is flat downwards and a cliff upwards.
        // A failure here must NOT propagate: cubature is an upgrade over a
        // correction that is already computed and already correct, and #2601
        // records what happens when a failure to refine the *uncertainty* is
        // allowed to destroy a converged point estimate.
        let centre_cost = match self.compute_cost_uncharged(final_rho) {
            Ok(cost) if cost.is_finite() => cost,
            Ok(_) => {
                return self.finalize_smoothing_outcome(first_order_numerical(
                    first_order_correction,
                    "outer criterion is not finite at the converged rho".into(),
                ));
            }
            Err(error) => {
                return self.finalize_smoothing_outcome(first_order_numerical(
                    first_order_correction,
                    format!("outer criterion unavailable at the converged rho: {error}").into(),
                ));
            }
        };
        let mut nodes: Vec<CalibratedSigmaNode> = Vec::with_capacity(2 * rank);
        for &index in &upgraded {
            let axis = spectrum.eigenvectors.column(index).to_owned();
            let wald_step = spectrum.eigenvalues[index].sqrt().recip();
            for sign in [1.0_f64, -1.0_f64] {
                let direction = axis.mapv(|value| value * sign);
                match self.calibrate_sigma_node(final_rho, centre_cost, &direction, wald_step) {
                    Ok(node) => nodes.push(node),
                    Err(error) => {
                        return self.finalize_smoothing_outcome(first_order_numerical(
                            first_order_correction,
                            format!("sigma-node calibration failed: {error}").into(),
                        ));
                    }
                }
            }
        }
        let sigma_points: Vec<Array1<f64>> = nodes.iter().map(|node| node.rho.clone()).collect();
        // Unreachable: `rank >= 1` ensures at least two sigma points
        // (one positive, one negative) per eigenvector. Treat as a
        // NumericalFailure guard so any future regression surfaces.
        if sigma_points.is_empty() {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "empty sigma-point set (unreachable guard)".into(),
            ));
        }

        // Dispatch the sigma-point evaluation to whichever executor is
        // currently the best fit for this build/runtime. See
        // [`sigma_cubature_dispatch`] for the auto-selection rule and
        // for the documented one-line swap site that flips to the GPU
        // stream-pool path once `pirls-row-v3` Stage 3 and `bms-flex-v3`
        // Phase 5 land the device-resident inner PIRLS the GPU
        // executor needs.
        // A sigma point is an OFF-TRAJECTORY rho: `final_rho ± sqrt(rank·λ_k)·v_k`
        // in log-smoothing units, and that offset is largest exactly when the
        // rho posterior is broad — which is the case cubature exists FOR. The
        // inner solve there is a genuinely different problem from the fitted
        // one, and for a shape-constrained fit it can be an intractable one (a
        // cold-started active-set QP at λ many decades from the optimum).
        //
        // Every other way this function can fail — the rho-Hessian refusing
        // inversion, the eigendecomposition failing, the ridge metadata being
        // invalid, the assembled covariance going non-finite — degrades to the
        // first-order correction with a recorded reason and severity, because
        // cubature is an UPGRADE over a correction that is already computed and
        // already correct. This one call propagated instead, so a failure to
        // refine the *uncertainty* destroyed a point estimate that had already
        // converged and certified: `gamfit.fit(frame, "y ~ s(x)",
        // constraints={"s(x)": "convex"})` returned `Parameter constraint
        // violation: KKT residuals exceed tolerance` for a fit whose own inner
        // solves all reached ‖g‖ ≈ 1e-13 (#2601).
        //
        // Fall back on the same contract as every sibling branch. The severity
        // is `NumericalFailure`, not `Routine` — this is a numerical failure of
        // the upgrade, it increments the failure counter, and it logs at WARN
        // with the propagated error, so it is recorded rather than silent.
        let point_results = match sigma_cubature_dispatch(self, &sigma_points, Some(final_fit)) {
            Ok(results) => results,
            Err(error) => {
                return self.finalize_smoothing_outcome(first_order_numerical(
                    first_order_correction,
                    format!("sigma-point inner solve failed at an off-trajectory rho: {error}")
                        .into(),
                ));
            }
        };

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
        let scaled_pairs: Vec<SigmaPointResult> = point_results
            .into_iter()
            .map(|(cov_point, beta_point)| (cov_point.mapv(|v| dispersion_phi * v), beta_point))
            .collect();
        if scaled_pairs.len() != nodes.len() {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "sigma-point executor returned a different number of results than nodes".into(),
            ));
        }
        // Per-node attribution. The two terms of the law of total covariance
        // behave very differently off the optimum: `Cov_ρ[β̂]` is bounded by the
        // range of β̂, but `E_ρ[φ̂·H(ρ)⁻¹]` is an average of inverse Hessians and
        // diverges as a penalty switches off. Recording both — together with
        // where the node was ASKED to sit and where the criterion actually put
        // it — is what makes a wide `Vp` attributable after the fact (#2728).
        if log::log_enabled!(log::Level::Info) {
            for (index, (cov_point, _)) in scaled_pairs.iter().enumerate() {
                let node = &nodes[index];
                log::info!(
                    "[sigma-cubature] node={index} step={:.4e} wald_step={:.4e} ΔV={:.6e} \
                     target={PROFILE_SIGMA_RISE} evals={} box_limited={} \
                     tr(φ̂·H(ρ)⁻¹)={:.6e} tr(φ̂·H(ρ̂)⁻¹)={:.6e}",
                    node.step,
                    node.wald_step,
                    node.achieved_rise,
                    node.evaluations,
                    node.box_limited,
                    cov_point.diag().iter().sum::<f64>(),
                    dispersion_phi * base_cov.diag().iter().sum::<f64>(),
                );
            }
        }
        // The executor returns results in node order, and nodes were pushed as
        // consecutive (+, −) pairs, one pair per upgraded direction.
        let mut paired: Vec<(SigmaPointResult, SigmaPointResult)> = Vec::with_capacity(rank);
        let mut remaining = scaled_pairs.into_iter();
        while let (Some(plus), Some(minus)) = (remaining.next(), remaining.next()) {
            paired.push((plus, minus));
        }
        // Every ACTIVE direction that was not upgraded keeps the first-order
        // column `Qs·J·u_k/√σ_k` it would have contributed to `J·V_ρ·Jᵀ`, so
        // the cubature covers the same subspace the first-order correction
        // does. Without this the truncation would silently SHRINK the
        // correction relative to the term it is supposed to upgrade.
        let fallback_columns: Vec<Array1<f64>> = ranked[rank..]
            .iter()
            .map(|&(index, _)| {
                let scale = spectrum.eigenvalues[index].sqrt().recip();
                spectrum
                    .sensitivity_orig
                    .dot(&spectrum.eigenvectors.column(index))
                    .mapv(|value| value * scale)
            })
            .collect();
        let mut total_cov =
            accumulate_sigma_cubature_total_covariance(&paired, &fallback_columns, p);
        if !total_cov.iter().all(|v| v.is_finite()) {
            return self.finalize_smoothing_outcome(first_order_numerical(
                first_order_correction,
                "assembled total covariance contains non-finite entries".into(),
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
        log::info!(
            "[sigma-cubature] tr(correction)={:.6e} tr(φ̂·H(ρ̂)⁻¹)={:.6e}",
            corr.diag().iter().sum::<f64>(),
            dispersion_phi * base_cov.diag().iter().sum::<f64>(),
        );

        // Worst criterion rise over the nodes this correction was built from.
        // `f64::max` returns the non-NaN operand, so a node whose criterion was
        // unevaluable does not silently become the maximum; a run in which
        // EVERY node was unevaluable reports `-inf`, which is as visibly wrong
        // as it should be.
        let max_node_criterion_rise = nodes
            .iter()
            .map(|node| node.achieved_rise)
            .fold(f64::NEG_INFINITY, f64::max);
        self.finalize_smoothing_outcome(SmoothingCorrectionOutcome::Cubature {
            correction: corr,
            rho_covariance: first_order_rho_covariance.clone(),
            rank,
            n_points: sigma_points.len(),
            max_node_criterion_rise,
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
                max_node_criterion_rise,
                near_boundary,
                grad_norm,
                max_rho_var,
                ..
            } => {
                SMOOTHING_CORRECTION_CUBATURE_COUNT.fetch_add(1, Ordering::Relaxed);
                log::info!(
                    "[smoothing-correction] branch={} rank={} points={} near_boundary={} \
                     grad_norm={:.3e} max_rho_var={:.3e} max_node_criterion_rise={:.3e} \
                     (target {PROFILE_SIGMA_RISE})",
                    branch_label,
                    rank,
                    n_points,
                    near_boundary,
                    grad_norm,
                    max_rho_var,
                    max_node_criterion_rise,
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
            SmoothingCorrectionOutcome::Unavailable {
                reason: SmoothingCorrectionUnavailable::OuterHessianNotAnalytic { error },
                ..
            } => {
                // Structural, not numerical: no analytic outer Hessian exists
                // for this fit, so the counter of numerical failures does not
                // move.
                log::info!(
                    "[smoothing-correction] branch=unavailable reason=outer-hessian-not-analytic \
                     ({error})"
                );
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
    //! Math-spec validation tests for the sigma-cubature accumulation formula.
    //!
    //! These pin the math of [`accumulate_sigma_cubature_total_covariance`]
    //! independently of the execution model that produced the per-node
    //! `(A_m, b_m)` pairs, so the same oracle covers both the CPU Rayon sigma
    //! loop and the GPU stream-pool sigma executor.
    //!
    //! # What the accumulator computes
    //!
    //! ```text
    //!     V̂_p = mean over all 2r nodes of A_m
    //!         + Σ_j c_j c_jᵀ,   c_j = (b_j⁺ − b_j⁻)/2
    //!         + Σ_k f_k f_kᵀ    (first-order columns of directions not upgraded)
    //! ```
    //!
    //! The covariance term is a PER-DIRECTION chord Gram, not a joint empirical
    //! moment over the node set. The joint moment reproduced `J·V_ρ·Jᵀ` only
    //! through a conspiracy between the `√rank` node radius and the `1/(2·rank)`
    //! weight, and that conspiracy breaks as soon as the two sides of a
    //! direction sit at different distances — which is what criterion-calibrated
    //! nodes are (#2728). The chord form is exact for ANY per-direction node
    //! scaling, which is what `cubature_linear_exactness_recovers_jvjt` below
    //! now pins.
    use super::{SigmaPointResult, accumulate_sigma_cubature_total_covariance};
    use ndarray::{Array1, Array2};

    /// Group a flat node list — emitted by the production loop as consecutive
    /// `(+, −)` pairs, one pair per upgraded ρ-eigendirection — into the pairs
    /// the accumulator consumes.
    fn paired(points: &[(Array2<f64>, Array1<f64>)]) -> Vec<(SigmaPointResult, SigmaPointResult)> {
        points
            .chunks_exact(2)
            .map(|pair| (pair[0].clone(), pair[1].clone()))
            .collect()
    }

    /// Accumulate a flat node list with no first-order fallback columns.
    fn accumulate(points: &[(Array2<f64>, Array1<f64>)], p: usize) -> Array2<f64> {
        accumulate_sigma_cubature_total_covariance(&paired(points), &[], p)
    }

    fn max_abs_deviation(actual: &Array2<f64>, expected: &Array2<f64>) -> f64 {
        let mut worst = 0.0_f64;
        for (a, e) in actual.iter().zip(expected.iter()) {
            worst = worst.max((a - e).abs());
        }
        worst
    }

    fn outer(v: &Array1<f64>) -> Array2<f64> {
        v.view()
            .insert_axis(ndarray::Axis(1))
            .dot(&v.view().insert_axis(ndarray::Axis(0)))
    }

    /// Cubature linear exactness: if `b_m = b_0 + J·(ρ_m − ρ̂)` is linear in `ρ`
    /// and `A_m = A_0` is constant, the accumulator must return
    /// `A_0 + J·V_ρ,r·Jᵀ` exactly, where `V_ρ,r = Σ_j t_j² u_j u_jᵀ` is the
    /// covariance the node set represents.
    ///
    /// The claim is stronger than the one the joint-moment form could make: it
    /// holds for ANY per-direction node scaling `t_j`, not only the `√(r·d_j)`
    /// radius the old `1/(2r)`-weighted rule needed. That is exactly the
    /// property criterion calibration requires, since it moves each `t_j`
    /// independently (#2728). Here the nodes sit at `t_j = √d_j`, the one-sigma
    /// step, so `V_ρ,r` is the ρ-covariance itself.
    ///
    /// Any drift from this is a math bug, not a numerics issue, so the
    /// tolerance is at f64 round-off.
    #[test]
    pub(crate) fn cubature_linear_exactness_recovers_jvjt() {
        // p = 4 outputs, d_ρ = 3 inputs, r = 3 upgraded eigendirections → 6
        // nodes. Three distinct eigenvalues so off-diagonal covariance entries
        // are genuinely exercised.
        let p = 4;
        let d_rho = 3;
        let r = 3;

        let eigenvalues = [0.25_f64, 0.49, 0.81];
        // A hand-built orthonormal U (no RNG): columns are the eigenvectors of
        // V_ρ,r.
        let u: Array2<f64> = ndarray::array![
            [1.0 / 3f64.sqrt(), 1.0 / 2f64.sqrt(), 1.0 / 6f64.sqrt()],
            [1.0 / 3f64.sqrt(), -1.0 / 2f64.sqrt(), 1.0 / 6f64.sqrt()],
            [1.0 / 3f64.sqrt(), 0.0, -2.0 / 6f64.sqrt()],
        ];
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

        // Nodes at ρ_m − ρ̂ = ±√d_j · u_j, in the (+, −) order the production
        // loop emits.
        let mut sigma_displacements: Vec<Array1<f64>> = Vec::with_capacity(2 * r);
        for k in 0..r {
            let scale = eigenvalues[k].sqrt();
            let axis = u.column(k).to_owned();
            for sign in [1.0_f64, -1.0_f64] {
                sigma_displacements.push(axis.mapv(|v| v * sign * scale));
            }
        }

        // Plain integers so the synthetic data is exactly representable.
        let b0: Array1<f64> = ndarray::array![1.0, -2.0, 3.5, 0.5];
        let jacobian: Array2<f64> = ndarray::array![
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        // A_0 SPD so the output is a real covariance matrix, with off-diagonal
        // structure so the A-term assertion is not vacuous.
        let mut a0 = Array2::<f64>::eye(p);
        a0[[0, 1]] = 0.25;
        a0[[1, 0]] = 0.25;
        a0[[2, 3]] = -0.10;
        a0[[3, 2]] = -0.10;

        let points: Vec<(Array2<f64>, Array1<f64>)> = sigma_displacements
            .iter()
            .map(|drho| (a0.clone(), &b0 + &jacobian.dot(drho)))
            .collect();

        let jvjt = jacobian.dot(&v_rho_r).dot(&jacobian.t());
        let expected = &a0 + &jvjt;
        let actual = accumulate(&points, p);

        let mut max_rel_dev = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                let diff = (actual[[i, j]] - expected[[i, j]]).abs();
                max_rel_dev = max_rel_dev.max(diff / expected[[i, j]].abs().max(1.0));
            }
        }
        assert!(
            max_rel_dev < 1e-12,
            "cubature linear-exactness violation: max_rel_dev={max_rel_dev:.3e}",
        );
    }

    /// Asymmetric nodes stay exact. The `+` and `−` node of a direction sit at
    /// different distances whenever the criterion is asymmetric along it — flat
    /// one way, a cliff the other — which is exactly the #2728 fixture. For a
    /// linear `b`, the chord over `(+a, −b)` is `((a+b)/2)·J·u`, so the
    /// accumulator must return `A_0 + ((a+b)/2)²·(Ju)(Ju)ᵀ`.
    ///
    /// The joint-moment form could not state this: its `1/(2r)` weighting ties
    /// the answer to a symmetric `√r` radius.
    #[test]
    pub(crate) fn cubature_asymmetric_nodes_use_the_half_chord() {
        let p = 3;
        let axis: Array1<f64> = ndarray::array![1.0, 0.0];
        let jacobian: Array2<f64> = ndarray::array![[2.0, -1.0], [0.5, 1.0], [-1.5, 0.25]];
        let b0: Array1<f64> = ndarray::array![0.3, -0.7, 1.1];
        let a0: Array2<f64> =
            ndarray::array![[1.5, 0.2, 0.0], [0.2, 1.2, 0.1], [0.0, 0.1, 1.0]];

        let plus_step = 0.25_f64;
        let minus_step = 1.75_f64;
        let points = vec![
            (
                a0.clone(),
                &b0 + &jacobian.dot(&axis.mapv(|v| v * plus_step)),
            ),
            (
                a0.clone(),
                &b0 + &jacobian.dot(&axis.mapv(|v| v * -minus_step)),
            ),
        ];

        let half_chord = 0.5 * (plus_step + minus_step);
        let column = jacobian.dot(&axis).mapv(|v| v * half_chord);
        let expected = &a0 + &outer(&column);
        let actual = accumulate(&points, p);
        assert!(
            max_abs_deviation(&actual, &expected) < 1e-13,
            "asymmetric half-chord violated: max_abs={:.3e}",
            max_abs_deviation(&actual, &expected),
        );
    }

    /// Degenerate sanity: a pair whose two nodes carry the same `β̂` has a
    /// zero-length chord, so the output is exactly `mean(A_m)`. Guards the
    /// chord formula against a stray sign or an off-by-one in the pairing.
    #[test]
    pub(crate) fn cubature_zero_chord_collapses_to_mean_a() {
        let p = 3;
        let a0: Array2<f64> = ndarray::array![[2.0, 0.5, 0.0], [0.5, 1.5, 0.25], [0.0, 0.25, 1.0]];
        let b0: Array1<f64> = ndarray::array![0.1, -0.2, 0.3];
        let points = vec![(a0.clone(), b0.clone()), (a0.clone(), b0.clone())];
        let actual = accumulate(&points, p);
        assert!(
            max_abs_deviation(&actual, &a0) < 1e-14,
            "zero-chord pair did not collapse to mean(A_m): max_abs={:.3e}",
            max_abs_deviation(&actual, &a0),
        );
    }

    /// The chord depends on the DIFFERENCE across a pair, so a linear `b` gives
    /// exactly `Σ_k J·d_k·d_kᵀ·Jᵀ` for pair displacements `±d_k` — with no
    /// `1/M` normaliser anywhere. This is the deeper structural check: it
    /// exercises the per-direction accumulation with mixed-sign `J` entries and
    /// unequal per-direction scales, where a residual `1/M` would show up as a
    /// uniform 1/3 shortfall.
    #[test]
    pub(crate) fn cubature_pair_differences_carry_the_whole_b_side() {
        let p = 3;
        let r = 3;
        let scales = [0.7_f64, 1.3, 0.4];
        let mut displacements: Vec<Array1<f64>> = Vec::with_capacity(2 * r);
        for k in 0..r {
            for sign in [1.0_f64, -1.0_f64] {
                let mut d = Array1::<f64>::zeros(r);
                d[k] = sign * scales[k];
                displacements.push(d);
            }
        }

        let b0: Array1<f64> = ndarray::array![2.5, -1.25, 4.0];
        let j: Array2<f64> =
            ndarray::array![[1.0, -2.0, 0.5], [0.0, 1.5, -1.0], [-0.75, 0.25, 2.0]];
        let a0: Array2<f64> =
            ndarray::array![[3.0, 0.5, -0.25], [0.5, 2.0, 0.10], [-0.25, 0.10, 1.5]];

        let points: Vec<(Array2<f64>, Array1<f64>)> = displacements
            .iter()
            .map(|drho| (a0.clone(), &b0 + &j.dot(drho)))
            .collect();

        // V_ρ,r = Σ_k d_k d_kᵀ over the `+` displacement of each pair.
        let mut v_rho = Array2::<f64>::zeros((r, r));
        for k in 0..r {
            let d = &displacements[2 * k];
            for i in 0..r {
                for jj in 0..r {
                    v_rho[[i, jj]] += d[i] * d[jj];
                }
            }
        }
        let expected = &a0 + &j.dot(&v_rho).dot(&j.t());
        let actual = accumulate(&points, p);
        assert!(
            max_abs_deviation(&actual, &expected) < 1e-12,
            "per-direction b-side accumulation violated: max_abs={:.3e}",
            max_abs_deviation(&actual, &expected),
        );
    }

    /// Constant-A invariance: if every `A_m = A_0` then the A-term equals `A_0`
    /// exactly regardless of how many pairs there are. Together with the b-side
    /// tests this localises a future regression onto one side or the other, not
    /// both at once.
    #[test]
    pub(crate) fn cubature_constant_a_in_implies_constant_a_out_on_a_side() {
        let p = 4;
        let a0: Array2<f64> = ndarray::array![
            [2.0, 0.30, 0.10, 0.05],
            [0.30, 1.50, 0.20, -0.10],
            [0.10, 0.20, 1.20, 0.15],
            [0.05, -0.10, 0.15, 0.80],
        ];
        // Identical β̂ at every node, so every chord is zero and the output is
        // exactly the A-term.
        let zero_b = Array1::<f64>::zeros(p);
        for pairs in [1usize, 2, 3, 4, 8] {
            let points: Vec<(Array2<f64>, Array1<f64>)> = (0..2 * pairs)
                .map(|_| (a0.clone(), zero_b.clone()))
                .collect();
            let actual = accumulate(&points, p);
            assert!(
                max_abs_deviation(&actual, &a0) < 1e-14,
                "constant-A invariance violated at {pairs} pair(s): max_abs={:.3e}",
                max_abs_deviation(&actual, &a0),
            );
        }
    }

    /// Permutation invariance. Re-ordering the PAIRS is a re-ordering of two
    /// sums, and swapping the two nodes WITHIN a pair only flips the sign of
    /// that pair's chord, which its Gram is blind to. Both must leave the
    /// output unchanged to f64 round-off; a stateful accumulator or an
    /// order-dependent scratch buffer would be caught here.
    #[test]
    pub(crate) fn cubature_permutation_invariance_on_antipodal_pairs() {
        let p = 4;
        let r = 3;
        let b0: Array1<f64> = ndarray::array![1.0, -2.0, 3.5, 0.5];
        let j: Array2<f64> = ndarray::array![
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        // Vary A_m across nodes so a re-ordering would actually move terms.
        let a_for_idx = |idx: usize| -> Array2<f64> {
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
        // Re-order the pairs (2, 0, 1) and flip the node order inside each.
        let mut reordered: Vec<(Array2<f64>, Array1<f64>)> = Vec::with_capacity(2 * r);
        for &k in &[2usize, 0, 1] {
            reordered.push(interleaved[2 * k + 1].clone());
            reordered.push(interleaved[2 * k].clone());
        }

        let v_interleaved = accumulate(&interleaved, p);
        let v_reordered = accumulate(&reordered, p);
        assert!(
            max_abs_deviation(&v_interleaved, &v_reordered) < 1e-13,
            "permutation invariance violated: max_abs={:.3e}",
            max_abs_deviation(&v_interleaved, &v_reordered),
        );
    }

    /// Executor-dispatch parity invariant.
    ///
    /// `sigma_cubature_dispatch` is the swap site between the CPU Rayon
    /// executor and the GPU stream-pool executor. The contract is that both
    /// branches return per-node `(A_m, b_m)` pairs the accumulator cannot
    /// distinguish. Since the accumulator is the only piece of that path
    /// isolable from a constructed `RemlState`, what is pinned here is its
    /// bitwise determinism on a fixed input — the property either branch must
    /// satisfy for the swap to be observationally neutral.
    #[test]
    pub(crate) fn cubature_dispatch_swap_site_invariant_holds_pre_gpu() {
        let p = 3;
        let a: Array2<f64> =
            ndarray::array![[1.5, 0.20, 0.10], [0.20, 1.20, 0.05], [0.10, 0.05, 0.90]];
        let b0: Array1<f64> = ndarray::array![0.30, -0.40, 0.10];
        let mut points: Vec<(Array2<f64>, Array1<f64>)> = Vec::new();
        for k in 0..3 {
            for sign in [1.0_f64, -1.0_f64] {
                let mut bm = b0.clone();
                bm[k] += sign * 0.25;
                points.push((a.clone(), bm));
            }
        }
        let first = accumulate(&points, p);
        let second = accumulate(&points, p);
        for i in 0..p {
            for j in 0..p {
                assert_eq!(
                    first[[i, j]],
                    second[[i, j]],
                    "accumulator non-deterministic at ({i},{j}): first={} second={}",
                    first[[i, j]],
                    second[[i, j]],
                );
            }
        }
    }

    /// Full-SPD `A_m` injection invariance.
    ///
    /// Replacing the per-node `A_m = H_m⁻¹` with an arbitrary SPD matrix — still
    /// SPD, no longer derived from a Hessian — must leave the output equal to
    /// the analytic expression `mean(A_m) + Σ_j c_j c_jᵀ`. This rules out any
    /// hidden assumption that `A_m` has inverse-Hessian structure; the formula
    /// is a total-covariance assembly and nothing more.
    #[test]
    pub(crate) fn cubature_arbitrary_spd_a_in_obeys_total_covariance_law() {
        let p = 3;
        // SPD by construction via `A = M Mᵀ + εI` for a non-symmetric M.
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
        let a_list = [
            mk_spd(1.0, 0.20),
            mk_spd(1.3, 0.10),
            mk_spd(0.7, -0.15),
            mk_spd(1.1, 0.05),
        ];
        let b_list: [Array1<f64>; 4] = [
            ndarray::array![0.1, 0.2, 0.3],
            ndarray::array![-0.1, 0.4, -0.2],
            ndarray::array![0.5, -0.3, 0.0],
            ndarray::array![0.2, 0.1, -0.4],
        ];
        let points: Vec<(Array2<f64>, Array1<f64>)> = a_list
            .iter()
            .zip(b_list.iter())
            .map(|(a, b)| (a.clone(), b.clone()))
            .collect();

        let w = 1.0 / a_list.len() as f64;
        let mut expected = Array2::<f64>::zeros((p, p));
        for a in &a_list {
            expected.scaled_add(w, a);
        }
        for pair in 0..2 {
            let chord = (&b_list[2 * pair] - &b_list[2 * pair + 1]).mapv(|v| 0.5 * v);
            expected = expected + outer(&chord);
        }

        let actual = accumulate(&points, p);
        assert!(
            max_abs_deviation(&actual, &expected) < 1e-13,
            "arbitrary-SPD A injection broke the total-covariance law: max_abs={:.3e}",
            max_abs_deviation(&actual, &expected),
        );
    }

    /// β-scale linearity: scaling every `b_m` by α (with `A_m` fixed) scales
    /// every chord by α and therefore the b-side by α², leaving the A-term
    /// untouched. Pins that the accumulator is bilinear in `b_m` and does not
    /// square the wrong row.
    #[test]
    pub(crate) fn cubature_beta_scaling_propagates_quadratically() {
        let p = 3;
        let a0: Array2<f64> = ndarray::array![[2.0, 0.1, 0.0], [0.1, 1.5, 0.05], [0.0, 0.05, 1.0]];
        let raw_betas: Vec<Array1<f64>> = vec![
            ndarray::array![1.0, -0.5, 0.3],
            ndarray::array![-1.0, 0.5, -0.3],
            ndarray::array![0.7, 0.2, -0.1],
            ndarray::array![-0.7, -0.2, 0.1],
        ];

        let unscaled: Vec<(Array2<f64>, Array1<f64>)> =
            raw_betas.iter().map(|b| (a0.clone(), b.clone())).collect();
        let v_unscaled = accumulate(&unscaled, p);

        let alpha = 2.5_f64;
        let scaled: Vec<(Array2<f64>, Array1<f64>)> = raw_betas
            .iter()
            .map(|b| (a0.clone(), b.mapv(|x| x * alpha)))
            .collect();
        let v_scaled = accumulate(&scaled, p);

        let mut max_rel = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                let expected = a0[[i, j]] + alpha * alpha * (v_unscaled[[i, j]] - a0[[i, j]]);
                let diff = (v_scaled[[i, j]] - expected).abs();
                max_rel = max_rel.max(diff / expected.abs().max(1.0));
            }
        }
        assert!(
            max_rel < 1e-12,
            "β-scaling quadratic propagation violated: max_rel={max_rel:.3e}",
        );
    }

    /// Full-reversal invariance. Reversing the flat node list reverses the pair
    /// order AND swaps the two nodes inside each pair; the A-mean is blind to
    /// the first and each chord only changes sign under the second, so the
    /// output must not move beyond f64 rounding noise.
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

        let v_forward = accumulate(&points, p);
        let v_reverse = accumulate(&reversed, p);
        assert!(
            max_abs_deviation(&v_forward, &v_reverse) < 1e-12,
            "full-reversal permutation invariance violated: max_abs={:.3e}",
            max_abs_deviation(&v_forward, &v_reverse),
        );
    }

    /// Duplicating every pair leaves the A-term unchanged and DOUBLES the
    /// b-term.
    ///
    /// This is the law that distinguishes the per-direction rule from the joint
    /// moment it replaced. Under the old `1/M`-weighted joint form, duplicating
    /// the node set was a no-op on both terms. Under the chord form a duplicated
    /// pair is a second direction contributing its own chord, so the b-side is
    /// additive over pairs — which is precisely why the correction no longer
    /// depends on how many directions happen to be upgraded (#2728).
    #[test]
    pub(crate) fn cubature_pair_duplication_doubles_the_chord_gram() {
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
        for chunk in original.chunks_exact(2) {
            doubled.extend_from_slice(chunk);
            doubled.extend_from_slice(chunk);
        }

        // The A-term is the mean over nodes, so duplicating every pair leaves
        // it alone.
        let weight = 1.0 / original.len() as f64;
        let mut mean_a = Array2::<f64>::zeros((p, p));
        for (a, _) in &original {
            mean_a.scaled_add(weight, a);
        }
        let v_orig = accumulate(&original, p);
        let v_doub = accumulate(&doubled, p);
        let expected = &mean_a + &(&v_orig - &mean_a).mapv(|v| 2.0 * v);
        assert!(
            max_abs_deviation(&v_doub, &expected) < 1e-13,
            "pair duplication did not double the chord Gram: max_abs={:.3e}",
            max_abs_deviation(&v_doub, &expected),
        );
    }

    /// Rank-deficient `V_ρ` degenerate behaviour: when both nodes of every pair
    /// carry the same `β̂` — `V_ρ` has rank 0 along the β direction — every
    /// chord is zero and the output collapses to `mean(A_m)`. The production
    /// trigger short-circuits this case, but the accumulator must still be
    /// correct on it in case the trigger is bypassed.
    #[test]
    pub(crate) fn cubature_rank_deficient_v_rho_collapses_var_to_zero() {
        let p = 3;
        let b_const: Array1<f64> = ndarray::array![0.7, -0.2, 0.4];
        let a_list = [
            ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ndarray::array![[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 0.8]],
            ndarray::array![[1.2, 0.1, 0.0], [0.1, 1.3, 0.0], [0.0, 0.0, 0.9]],
            ndarray::array![[0.9, 0.0, 0.2], [0.0, 1.1, 0.0], [0.2, 0.0, 1.4]],
        ];
        let points: Vec<(Array2<f64>, Array1<f64>)> = a_list
            .iter()
            .map(|a| (a.clone(), b_const.clone()))
            .collect();
        let actual = accumulate(&points, p);

        let w = 1.0 / (a_list.len() as f64);
        let mut mean_a = Array2::<f64>::zeros((p, p));
        for a in &a_list {
            mean_a.scaled_add(w, a);
        }
        assert!(
            max_abs_deviation(&actual, &mean_a) < 1e-13,
            "rank-deficient V_ρ did not collapse the chord to zero: max_abs={:.3e}",
            max_abs_deviation(&actual, &mean_a),
        );
    }

    /// A direction that was NOT upgraded keeps the first-order column it would
    /// have contributed to `J·V_ρ·Jᵀ`, so the assembled correction covers the
    /// same subspace the first-order correction does.
    ///
    /// Without this the eigen-truncation would silently ship a correction over
    /// a SMALLER subspace than the term it exists to upgrade — which is what a
    /// rank-1 truncation did on the #2728 fixture, where it also picked the
    /// wrong direction to keep. The claim: upgrading direction 0 and falling
    /// back on direction 1 must reproduce the same matrix as upgrading BOTH,
    /// whenever `β̂` is linear over the node interval.
    #[test]
    pub(crate) fn cubature_fallback_columns_restore_the_untreated_subspace() {
        let p = 3;
        let a0: Array2<f64> = ndarray::array![[1.4, 0.1, 0.0], [0.1, 1.1, 0.2], [0.0, 0.2, 0.9]];
        let b0: Array1<f64> = ndarray::array![0.5, -0.25, 0.75];
        let j: Array2<f64> = ndarray::array![[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]];
        let steps = [0.6_f64, 1.4];

        let node = |direction: usize, sign: f64| -> (Array2<f64>, Array1<f64>) {
            let mut drho = Array1::<f64>::zeros(2);
            drho[direction] = sign * steps[direction];
            (a0.clone(), &b0 + &j.dot(&drho))
        };

        // Both directions upgraded.
        let both = vec![node(0, 1.0), node(0, -1.0), node(1, 1.0), node(1, -1.0)];
        let upgraded_both = accumulate(&both, p);

        // Direction 0 upgraded, direction 1 kept as a first-order column
        // `J·u_1·t_1` — the same column the first-order Gram would build.
        let only_first = vec![node(0, 1.0), node(0, -1.0)];
        let mut axis = Array1::<f64>::zeros(2);
        axis[1] = steps[1];
        let fallback = vec![j.dot(&axis)];
        let upgraded_one =
            accumulate_sigma_cubature_total_covariance(&paired(&only_first), &fallback, p);

        assert!(
            max_abs_deviation(&upgraded_both, &upgraded_one) < 1e-13,
            "fallback column did not reproduce the upgraded direction on a \
             linear β̂: max_abs={:.3e}",
            max_abs_deviation(&upgraded_both, &upgraded_one),
        );
    }

    /// The A-side is convex in its input: with `α + β = 1` the chord terms are
    /// identical across the three runs and cancel, so
    /// `V[α·A + β·A'] = α·V[A] + β·V[A']`.
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
        let beta = 1.0 - alpha;
        let a_set_mix: Vec<Array2<f64>> = a_set_1
            .iter()
            .zip(a_set_2.iter())
            .map(|(a, ap)| a.mapv(|v| v * alpha) + ap.mapv(|v| v * beta))
            .collect();

        let build = |a_set: &[Array2<f64>]| -> Vec<(Array2<f64>, Array1<f64>)> {
            a_set
                .iter()
                .zip(points_b.iter())
                .map(|(a, b)| (a.clone(), b.clone()))
                .collect()
        };
        let v1 = accumulate(&build(&a_set_1), p);
        let v2 = accumulate(&build(&a_set_2), p);
        let vmix = accumulate(&build(&a_set_mix), p);

        let expected = v1.mapv(|v| v * alpha) + v2.mapv(|v| v * beta);
        assert!(
            max_abs_deviation(&vmix, &expected) < 1e-13,
            "A-side convexity violated: max_abs={:.3e}",
            max_abs_deviation(&vmix, &expected),
        );
    }

    /// Translation invariance of the b-side. Adding the same offset to every
    /// `b_m` cannot change a chord, which is a difference; the A-side never
    /// sees `b` at all. Under the joint-moment form this held only up to the
    /// cancellation of a raw second moment against `mean·meanᵀ`, so a shift of
    /// magnitude 10 inflated the intermediates by ~100 and the tolerance had to
    /// absorb it. The chord form has no such intermediate: the invariance is
    /// EXACT, and the tolerance says so.
    #[test]
    pub(crate) fn cubature_b_translation_leaves_output_unchanged() {
        let p = 3;
        let a_const: Array2<f64> =
            ndarray::array![[1.5, 0.2, 0.0], [0.2, 1.2, 0.1], [0.0, 0.1, 1.0]];
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

        let v_raw = accumulate(&pts_raw, p);
        let v_shifted = accumulate(&pts_shifted, p);
        assert!(
            max_abs_deviation(&v_raw, &v_shifted) < 1e-14,
            "b-translation invariance violated: max_abs={:.3e}",
            max_abs_deviation(&v_raw, &v_shifted),
        );
    }

    /// Block-diagonal `A_m` with block-aligned `b_m` must decouple: when the
    /// bottom half of `β̂` is constant across a pair its chord entries are zero
    /// there, so the Gram has no cross-block entries and the output stays
    /// block-diagonal. Pins that the outer-product loop does not wrap indices.
    #[test]
    pub(crate) fn cubature_block_diagonal_inputs_yield_block_diagonal_output() {
        let p_top = 2;
        let p_bot = 2;
        let p = p_top + p_bot;
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

        let b_bot_const: Array1<f64> = ndarray::array![0.5, -0.3];
        let top_bs: Vec<Array1<f64>> = vec![
            ndarray::array![0.4, 0.1],
            ndarray::array![-0.4, -0.1],
            ndarray::array![0.2, 0.3],
            ndarray::array![-0.2, -0.3],
        ];
        let mut points: Vec<(Array2<f64>, Array1<f64>)> = Vec::with_capacity(top_bs.len());
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
        let v = accumulate(&points, p);
        let mut max_cross_abs = 0.0_f64;
        for i in 0..p_top {
            for j in 0..p_bot {
                max_cross_abs = max_cross_abs.max(v[[i, p_top + j]].abs());
                max_cross_abs = max_cross_abs.max(v[[p_top + j, i]].abs());
            }
        }
        assert!(
            max_cross_abs < 1e-13,
            "block-diagonal inputs leaked cross-block coupling: max_cross_abs={max_cross_abs:.3e}",
        );
    }

    /// The output is symmetric for symmetric inputs — a mean of symmetric
    /// matrices plus a sum of rank-one Grams. The production caller runs
    /// `symmetrize_in_place` afterwards, which would mask drift here, so the
    /// invariant is pinned upstream of that cleanup.
    #[test]
    pub(crate) fn cubature_output_is_symmetric_for_symmetric_inputs() {
        let p = 5;
        let m = 6;
        let points: Vec<(Array2<f64>, Array1<f64>)> = (0..m)
            .map(|idx| {
                let mut a = Array2::<f64>::eye(p);
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
        let v = accumulate(&points, p);
        let mut max_asym = 0.0_f64;
        for i in 0..p {
            for j in (i + 1)..p {
                max_asym = max_asym.max((v[[i, j]] - v[[j, i]]).abs());
            }
        }
        assert!(
            max_asym < 1e-13,
            "output not symmetric for symmetric inputs: max_asym={max_asym:.3e}",
        );
    }

    /// The A-side depends only on the multiset of `A_m`, not on their order or
    /// on which `b_m` they are paired with. With a constant `β̂` every chord is
    /// zero, so both orderings must return exactly `mean(A_m)`.
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
        let b_const: Array1<f64> = ndarray::array![0.2, -0.1, 0.3];
        let original: Vec<(Array2<f64>, Array1<f64>)> =
            a_set.iter().map(|a| (a.clone(), b_const.clone())).collect();
        let perm: [usize; 6] = [3, 0, 5, 1, 4, 2];
        let permuted_a: Vec<(Array2<f64>, Array1<f64>)> = perm
            .iter()
            .map(|&i| (a_set[i].clone(), b_const.clone()))
            .collect();

        let v_orig = accumulate(&original, p);
        let v_perm = accumulate(&permuted_a, p);
        let w = 1.0 / m as f64;
        let mut mean_a = Array2::<f64>::zeros((p, p));
        for a in &a_set {
            mean_a.scaled_add(w, a);
        }
        assert!(
            max_abs_deviation(&v_orig, &mean_a) < 1e-13
                && max_abs_deviation(&v_perm, &mean_a) < 1e-13,
            "A-side depended on A ordering under constant β̂: orig={:.3e}, perm={:.3e}",
            max_abs_deviation(&v_orig, &mean_a),
            max_abs_deviation(&v_perm, &mean_a),
        );
    }

    /// V100 hill-climb scaffold — sigma-loop accumulator perf record.
    ///
    /// The per-charter goal is a 5x speedup over the CPU Rayon sigma loop at
    /// large scale on V100. The accumulator is the only piece of the dispatch
    /// path fully isolable from a constructed `RemlState` (the rest is an inner
    /// PIRLS per node), so what runs here is the accumulator at large-scale
    /// shape — `p = 50`, 4 upgraded directions — and the per-call time is
    /// PRINTED as the hill-climbing record.
    ///
    /// Timing is deliberately not a gate: an absolute per-call ceiling is a
    /// calibration-box assumption that flakes on contended shared runners and
    /// silently passes real regressions on fast ones, and promoting it to a
    /// fixed CPU/GPU ratio would encode the box harder (#2313). Correctness
    /// regressions are caught by the assertions above.
    #[test]
    pub(crate) fn sigma_loop_v100_hill_climb_baseline() {
        let p = 50_usize;
        let m = 8_usize;

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

        let reps = 20_usize;
        let t0 = std::time::Instant::now();
        let mut last_trace = 0.0_f64;
        for _ in 0..reps {
            let v = accumulate(&points, p);
            // Touch the result so the optimiser cannot elide the call. The
            // trace is a real use of the matrix, not a `black_box` silencer.
            let mut tr = 0.0_f64;
            for d in 0..p {
                tr += v[[d, d]];
            }
            last_trace += tr;
        }
        let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;

        assert!(
            last_trace.is_finite(),
            "accumulator produced non-finite trace sum: {last_trace}"
        );

        let stage3_ready = super::device_pirls_stage3_ready()
            .expect("Stage-3 runtime resolution must not fault in the timing test");
        log::info!(
            "[sigma-hill-climb] accumulator baseline: per-call={per_call_us:.1}us \
             (p={p}, M={m}, reps={reps}); stage3_ready={stage3_ready}; the 5x \
             target gates on stage3_ready=true"
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
        reason: std::borrow::Cow<'static, str>,
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
            rank: 2,
            n_points: 4,
            max_node_criterion_rise: 0.51,
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
        // The node-calibration provenance must survive extraction: it is what
        // tells a consumer whether the quadrature nodes sat where the posterior
        // has mass, and it replaced the perturbation ledger (#2728).
        let Some(SmoothingCorrectionMethod::SigmaPointCubature {
            max_node_criterion_rise,
            ..
        }) = method
        else {
            panic!("cubature outcome must extract as SigmaPointCubature");
        };
        assert_eq!(max_node_criterion_rise, 0.51);
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
            "n_rho == 0".into(),
            SmoothingCorrectionFallbackSeverity::Routine,
            true,
        );
        assert_eq!(outcome.branch_label(), "first-order (routine)");
        assert!(outcome.into_correction_with_method().0.is_some());
    }

    #[test]
    pub(crate) fn first_order_numerical_branch_label_and_extraction() {
        let outcome = make_first_order(
            "rho Hessian inversion failed after ridge regularization".into(),
            SmoothingCorrectionFallbackSeverity::NumericalFailure,
            true,
        );
        assert_eq!(outcome.branch_label(), "first-order (numerical failure)");
        assert!(outcome.into_correction_with_method().0.is_some());
    }

    #[test]
    pub(crate) fn first_order_without_matrix_returns_none() {
        let outcome = make_first_order(
            "no base covariance supplied".into(),
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
    /// will not reliably land ρ̂ anywhere in particular, so instead of hoping,
    /// this test calls [`RemlState::compute_smoothing_correction_auto`]
    /// directly with a fixed `final_rho`.
    ///
    /// That ρ used to be `RHO_BOUND − 1`, chosen so the `near_boundary` arm of
    /// the gate would be unconditionally true. It is now an INTERIOR ρ, because
    /// the boundary arm and the cubature precondition turned out to be mutually
    /// exclusive on this design: inside the 2.0 margin `λ ≳ e²⁸`, the ridge has
    /// collapsed to its null space, ρ is unidentified (`active_rank = 0`), and
    /// `compute_smoothing_correction_auto` correctly refuses to escalate rather
    /// than impute variance the geometry does not support. An interior ρ keeps
    /// ρ identified and reaches cubature through `max_rho_var` or the
    /// high-gradient certificate — the arm the gate's own comment describes for
    /// "broad-but-well-converged posteriors". The branch is still asserted to
    /// have fired, via the `SMOOTHING_CORRECTION_CUBATURE_COUNT` delta, so the
    /// coverage this test exists for cannot be lost silently.
    ///
    /// Running the same construction at response scales `1` and `c`
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

            // ρ̂ by root-finding the outer stationarity condition, NOT a forced value.
            //
            // This fixture used to hand `compute_smoothing_correction_auto` a
            // ρ of its own choosing — first `RHO_BOUND - 1`, then an interior
            // `0.0` — and both produced the identical refusal
            // `first-order V_rho rank-deficient`. The reason the two agreed is
            // that neither is a stationary point, and the rank rule is not a
            // statement about ρ's value at all. Measured at `ρ = 0` (#2614):
            //
            //     grad = [4.341019]   |g| = 4.341019e0
            //     rho-Hessian eigenvalue = -2.125902     class = BelowGradientFloor
            //     active=0/1  structural_zero=0  below_gradient_floor=1
            //     eigensolver backward error = 3.021089e-14
            //
            // The curvature is NEGATIVE. This ρ is not a minimum, and the only
            // thing that stopped `invert_identified_rho_hessian` from calling
            // that the contradiction it is, is its own resolution floor
            // `floor = Σ_k |g_k|·v_k² = 4.34` (#2428), which is larger than
            // `|σ| = 2.13` and so classifies the direction as unresolvable
            // instead. That floor exists because the production caller
            // (`estimate/optimizer.rs:2628`) passes
            // `outer_result.final_gradient` — the RESIDUAL gradient the outer
            // certificate accepted ρ̂ with, which is tiny by construction. Feed
            // it a full non-stationary gradient and it masks everything,
            // uniformly in ρ. That is exactly the ρ-independence measured.
            //
            // So the fixture has to satisfy the precondition the whole
            // correction path is written for, rather than pick a ρ and hope.
            // `n_rho == 1`, so stationarity is a scalar root and bisection
            // brackets it with no derivative and no tuning; the bracket is the
            // ρ domain itself. What makes this a fixture rather than a
            // reimplemented optimizer is that the result is CERTIFIED below:
            // the gradient at the returned ρ̂ is asserted small, so a
            // mis-converged root fails loudly instead of silently reproducing
            // the defect this comment describes.
            //
            // The root is scale-invariant, which is what lets the equivariance
            // comparison stay exact across the two runs: for profiled Gaussian
            // REML, `y -> c·y` sends `rss -> c²·rss` and the score picks up
            // `dof·ln(c²)`, an additive constant in ρ. The ρ-gradient is
            // therefore identical at both scales and both runs bisect to the
            // same ρ̂.
            let outer_gradient_at = |candidate: f64| -> f64 {
                let probe = Array1::from_vec(vec![candidate]);
                state
                    .compute_gradient(&probe)
                    .unwrap_or_else(|err| {
                        panic!("outer gradient at rho={candidate}: {err}")
                    })[0]
            };
            let mut lo_rho = 1.0 - RHO_BOUND;
            let mut hi_rho = RHO_BOUND - 1.0;
            let g_lo = outer_gradient_at(lo_rho);
            let g_hi = outer_gradient_at(hi_rho);
            assert!(
                g_lo < 0.0 && g_hi > 0.0,
                "the REML profile has no interior stationary ρ on this design at                  scale {scale}: g({lo_rho}) = {g_lo:.6e}, g({hi_rho}) = {g_hi:.6e}.                  A monotone profile means the optimum is a rail, where the outer                  gradient does not vanish and the correction's identification rule                  has no converged ρ̂ to be applied at — the fixture's data would                  need to favour some smoothing, not the ρ search."
            );
            let mut bisections = 0usize;
            while hi_rho - lo_rho > 1e-13 * (1.0 + hi_rho.abs()) && bisections < 200 {
                let mid = 0.5 * (lo_rho + hi_rho);
                if outer_gradient_at(mid) > 0.0 {
                    hi_rho = mid;
                } else {
                    lo_rho = mid;
                }
                bisections += 1;
            }
            let final_rho = Array1::from_vec(vec![0.5 * (lo_rho + hi_rho)]);

            // Converged inner fit at the certified stationary ρ — this is the
            // `final_fit` the cubature path differentiates around, and its
            // Qs-mapped H⁻¹ is the dispersion-free base covariance the
            // correction upgrades.
            let final_fit = state
                .execute_pirls_stateless_for_cubature(&final_rho, None)
                .expect("inner PIRLS at the converged rho");
            let h_orig = map_hessian_to_original_basis(final_fit.as_ref())
                .expect("map Hessian to original basis");
            let base_cov = gam_linalg::utils::certified_spd_inverse(&h_orig, "test base cov")
                .expect("invert base Hessian")
                .into_inverse();

            // Profiled Gaussian dispersion φ̂ = deviance / (n − p). Deviance (RSS)
            // scales as c², the denominator is scale-invariant, so φ̂ scales as c².
            let dispersion_phi = final_fit.deviance / ((n as f64) - (p as f64)).max(1.0);

            // The residual gradient at ρ̂, which is what the production caller
            // passes and what the identification floor is calibrated for.
            //
            // Do NOT swallow a failed outer gradient into a zero-LENGTH array.
            //
            // This previously fell back to `Array1::zeros(0)` behind a
            // `log::debug!`, which no test harness in this crate has a backend
            // for. A zero-length outer gradient is not a small gradient: it is
            // an EMPTY identified subspace, so `first_order.active_rank` is 0,
            // `V_ρ` comes back as `[[0.0]]`, and
            // `compute_smoothing_correction_auto` correctly declines to
            // escalate with "first-order V_rho rank-deficient". The test then
            // reports only that a correction matrix was absent.
            //
            // Measured: moving `final_rho` from `RHO_BOUND − 1` to an interior
            // `0.0` changed nothing — same `[[0.0]]`, same reason — which rules
            // out the near-boundary saturation I first blamed and leaves this
            // silent substitution as the candidate. If the gradient really is
            // unavailable on this route, that is the finding and it must be
            // said out loud rather than converted into a degenerate input.
            let finalgrad = state
                .compute_gradient(&final_rho)
                .unwrap_or_else(|err| panic!(
                    "outer gradient unavailable at rho={final_rho:?}: {err}. \
                     A zero-length substitute would empty the identified subspace \
                     and make the cubature precondition fail for a reason that has \
                     nothing to do with this test's subject."
                ));
            let finalgrad_norm = finalgrad.dot(&finalgrad).sqrt();

            // Certify the bracket actually converged. Without this the fixture
            // could hand a mis-converged ρ straight back into the defect above,
            // and the failure would again present as an inscrutable
            // rank-deficiency rather than as "the root search did not finish".
            let stationarity_tol = 1e-6 * (1.0 + final_fit.deviance.abs());
            assert!(
                finalgrad_norm <= stationarity_tol,
                "the ρ bracket did not reach stationarity in {bisections} bisections:                  ρ̂ = {final_rho:?}, |g| = {finalgrad_norm:.6e} exceeds                  {stationarity_tol:.6e}"
            );

            // Kept for the failure path below; `Ok`/`Err` is itself a finding,
            // so the `Err` is CARRIED rather than flattened away. `.ok()` here
            // would have thrown out the one string that distinguishes "the
            // Hessian says the direction is unresolvable" from "the Hessian
            // could not be computed at all" -- the same discard this whole
            // diagnostic exists to undo.
            let self_hessian_for_diagnosis = state.compute_lamlhessian_consistent(&final_rho);

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
                    &finalgrad,
                    // This harness has no outer solver behind it, so there is no
                    // second assembly of the rho-Hessian to compare against: an
                    // absent measurement, not a zero (#2748).
                    None,
                    &[],
                )
                .expect("smoothing correction evaluation");
            let after = SMOOTHING_CORRECTION_CUBATURE_COUNT.load(Ordering::SeqCst);

            // Name the outcome that failed to carry a correction.
            //
            // `SmoothingCorrectionOutcome` records WHY it has no matrix -- the
            // `FirstOrder` variant carries a `reason` explaining why the
            // cubature upgrade was not taken, and `Unavailable` carries a typed
            // `SmoothingCorrectionUnavailable`. `.0.expect(..)` threw all of it
            // away and reported only that a matrix was absent, which is the
            // same defect (#2465) that made #2614's spline-scan refusal cost
            // two exact-but-misdirected repairs: a verdict has to carry the
            // quantity it was decided against. The type derives `Debug`, so
            // this costs one formatted string on the failure path only.
            let outcome_description = format!("{outcome:?}");
            let correction = outcome.into_correction_with_method().0.unwrap_or_else(|| {
                // Name the SPECTRUM, not just the verdict.
                //
                // Two hypotheses for `active_rank = 0` have now been refuted by
                // measurement: the near-boundary ρ (moving to an interior 0.0
                // changed nothing) and a swallowed outer gradient (the panic
                // above never fired, so `compute_gradient` succeeded). What is
                // left is the classification rule itself, and it splits the one
                // direction three ways —
                //
                //   Active             σ > floor
                //   StructuralZero     |σ| <= eigensolver backward error
                //   BelowGradientFloor otherwise
                //
                // with `floor = Σ_k |g_k|·v_k²` (#2428). Those three have
                // different causes and different fixes, and `V_ρ = [[0.0]]`
                // looks identical under all of them. `invert_identified_rho_hessian`
                // already computes the discriminating numbers; the failure path
                // just never asked for them.
                let spectrum = match self_hessian_for_diagnosis.as_ref() {
                    Ok(h) => {
                        match crate::estimate::smoothing_correction::invert_identified_rho_hessian(
                            h, 0, &finalgrad, None, &[],
                        ) {
                            Ok(inv) => format!(
                                "active={}/{} structural_zero={} unresolvable_curvature={} \
                                 below_gradient_floor={} eigenvalues={:?} classes={:?} \
                                 curvature_resolution={:.6e}",
                                inv.active_rank,
                                h.nrows(),
                                inv.structural_zero,
                                inv.unresolvable_curvature,
                                inv.below_gradient_floor,
                                inv.eigenvalues,
                                inv.classifications,
                                inv.curvature_resolution,
                            ),
                            Err(err) => format!("inverter refused: {err}"),
                        }
                    }
                    Err(err) => format!("rho Hessian unavailable at this rho: {err}"),
                };
                panic!(
                    "cubature/first-order outcome carries a correction matrix; \
                     got: {outcome_description}; rho={final_rho:?} \
                     grad={finalgrad:?} |g|={finalgrad_norm:.6e} \
                     deviance={:.6e} rho-Hessian spectrum: {spectrum}",
                    final_fit.deviance,
                )
            });
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
             an interior identified rho should reach it via max_rho_var or the \
             high-gradient certificate"
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
            // A sigma point's inner solve failing is a NUMERICAL-severity
            // fallback carrying the propagated typed error, not a fixed
            // classification string — hence the `Cow` (#2601).
            "sigma-point inner solve failed at an off-trajectory rho: <typed error>",
            "assembled total covariance contains non-finite entries",
        ];
        for r in reasons.iter() {
            assert!(!r.is_empty(), "classification reason must not be empty");
            let routine = make_first_order(
                std::borrow::Cow::Borrowed(r),
                SmoothingCorrectionFallbackSeverity::Routine,
                true,
            );
            let numerical = make_first_order(
                std::borrow::Cow::Borrowed(r),
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
