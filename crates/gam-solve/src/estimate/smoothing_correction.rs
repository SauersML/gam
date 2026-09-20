use super::*;

/// Default inner P-IRLS tolerance floor.
///
/// The inner Newton iteration certifies the coefficient mode against this
/// (scale-aware) tolerance independently of the outer REML tolerance. Coupling
/// the two collapses two unrelated convergence concepts: when a user dials the
/// outer tolerance up to e.g. 1e-3 to make the smoothing-parameter search
/// coarser, the inner solve becomes coarse too, returning betas whose
/// stationarity residual is ~1e-3·scale rather than the floating-point noise
/// floor. Outer derivatives then read those imprecise betas as if they were
/// the true mode and accumulate error. Keeping the inner floor at 1e-6 lets
/// the outer loop relax without contaminating the coefficient certificate.
pub(crate) const PIRLS_INNER_TOLERANCE_FLOOR: f64 = 1e-6;

#[derive(Clone)]
pub(crate) struct RemlConfig {
    pub(crate) likelihood: GlmLikelihoodSpec,
    pub(crate) link_kind: InverseLink,
    pub(crate) pirls_convergence_tolerance: f64,
    pub(crate) max_iterations: usize,
    pub(crate) reml_convergence_tolerance: f64,
    pub(crate) firth_bias_reduction: bool,
}

impl RemlConfig {
    pub(crate) fn external(
        likelihood: GlmLikelihoodSpec,
        reml_tol: f64,
        firth_bias_reduction: bool,
    ) -> Self {
        // Inner P-IRLS certifies the coefficient mode against
        // `pirls_convergence_tolerance`; the outer REML iteration certifies
        // the smoothing-parameter optimum against `reml_convergence_tolerance`.
        // These are different concepts and must not be coupled. The inner
        // tolerance is at most the outer tolerance (so a user who *tightens*
        // the outer also tightens the inner), but never coarser than the
        // floor — a coarse outer must not silently pollute the inner mode.
        let pirls_tol = reml_tol.min(PIRLS_INNER_TOLERANCE_FLOOR);
        let link_kind = likelihood.spec.link.clone();
        Self {
            likelihood,
            link_kind,
            pirls_convergence_tolerance: pirls_tol,
            max_iterations: 0,
            reml_convergence_tolerance: reml_tol,
            firth_bias_reduction,
        }
        .with_max_iterations(300)
    }

    pub(crate) fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub(crate) fn link_function(&self) -> LinkFunction {
        self.link_kind.link_function()
    }

    pub(crate) fn as_pirls_config(&self) -> pirls::PirlsConfig {
        pirls::PirlsConfig {
            likelihood: self.likelihood.clone(),
            link_kind: self.link_kind.clone(),
            max_iterations: self.max_iterations,
            convergence_tolerance: self.pirls_convergence_tolerance,
            firth_bias_reduction: self.firth_bias_reduction,
            // Caller (the REML runtime) populates this hint just before
            // each `execute_pirls_if_needed` call from the cached final
            // λ of the previous successful PIRLS solve.
            initial_lm_lambda: None,
        }
    }
}
/// Small ridge added to the rho-space LAML Hessian before inversion, for
/// numerical stability when smoothing parameters are weakly identified.
///
/// **Stabilization semantics:** this ridge is a numerical perturbation, not an
/// explicit prior. It enters only the inverse used to build `V_rho` for
/// the smoothing-correction propagation step. It does NOT enter the LAML
/// objective, its gradient, the saved coefficients, or any user-visible
/// summary — the rho-Hessian itself is recomputed from first principles
/// in every place that consults it. The perturbation never escapes the
/// local `V_rho` inverse (it touches no saved coefficient, objective, or
/// user-visible summary).
/// Minimum penalized-deviance floor, expressed as a fraction of the
/// problem's own deviance scale (the weighted null deviance `D₀`, see
/// [`smooth_floor_dp`]). The floor exists only to keep the profiled
/// dispersion `φ̂ = D_p/(n−M_p)` strictly positive when a smooth fits the
/// data essentially perfectly (`D_p ↓ 0`), so it must trigger on the
/// *relative* smallness `D_p/D₀`, never on an absolute magnitude — an
/// absolute floor silently breaks the exact scale-equivariance of the
/// Gaussian REML fit under a response rescale `y → a·y` (#1127).
pub(crate) const DP_FLOOR: f64 = 1e-12;
/// Width of the smooth transition region for the deviance floor, also as a
/// fraction of the deviance scale `D₀`.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;

/// Smooth, differentiable approximation of `max(dp, floor)` where the floor
/// and the width of the smoothing band are taken **relative to the supplied
/// deviance `scale`** (the weighted null deviance `D₀` of the response).
///
/// Returns the smoothed value, first derivative, and second derivative with
/// respect to `dp`.
///
/// # Why the floor must be relative (issue #1127)
///
/// The penalized deviance `D_p = Σ wᵢ(yᵢ−μ̂ᵢ)² + β̂ᵀSβ̂` is exactly quadratic
/// in the response, so under a multiplicative rescale `y → a·y` it scales as
/// `D_p → a²·D_p`. The profiled Gaussian REML criterion depends on `D_p` only
/// through `log D_p` (the `(ν/2)·log(2πφ̂)` term, `φ̂ = D_p/ν`), so the rescale
/// shifts the cost by the *additive constant* `ν·log a` and leaves the
/// ρ-gradient — hence the selected `λ̂`, the EDF, and `ŝ(x)/a` — exactly
/// invariant. An **absolute** floor destroys this: when `a` is small enough
/// that `D_p` enters the fixed band (e.g. `D_p ≈ 3.6e-11` at `a = 1e-6` with
/// a band of width `1e-8`), `dp_c` is spuriously inflated toward the absolute
/// floor, `log dp_c` stops tracking `2·log a + const`, and the optimizer
/// converges at an over-smoothed `λ̂` — reshaping, not merely rescaling, the
/// smooth. Scaling both the floor and its width by `D₀ ∝ a²` makes the band a
/// fixed *fraction* of the deviance, so `smooth_floor_dp(a²·dp, a²·D₀) =
/// a²·smooth_floor_dp(dp, D₀)` exactly and equivariance is restored.
///
/// `scale = 1.0` recovers the historical absolute floor byte-for-byte, which
/// is the correct default for callers without a Gaussian response scale in
/// hand (the floor is consumed only on the profiled-Gaussian path).
pub(crate) fn smooth_floor_dp(dp: f64, scale: f64) -> (f64, f64, f64) {
    let scale = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    };
    let floor = DP_FLOOR * scale;
    let tau = (DP_FLOOR_SMOOTH_WIDTH * scale).max(f64::MIN_POSITIVE);
    let scaled = (dp - floor) / tau;

    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = floor + tau * softplus;
    let dp_cgrad2 = sigma * (1.0 - sigma) / tau;
    (dp_c, sigma, dp_cgrad2)
}

/// Compute the smoothing parameter uncertainty correction matrix `V_corr = J * V_rho * J^T`.
///
/// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty.
/// The corrected covariance for `beta` is: `V*_beta = V_beta + J * V_rho * J^T`.
/// where:
/// - `V_beta = H^{-1}` (conditional covariance treating `lambda` as fixed)
/// - `J = d(beta)/d(rho)` (Jacobian wrt log-smoothing parameters)
/// - `V_rho = (d^2 LAML / d rho^2)^{-1}` (outer covariance)
///
/// Returns the correction matrix in the ORIGINAL coefficient basis.
///
/// Full correction reference.
/// Let `rho ~ N(mu, Sigma)` with `mu = rho_hat`, `Sigma = V_rho`,
/// and define:
/// - `A(rho) = H_rho^{-1}`
/// - `b(rho) = beta_hat_rho`
///
/// The exact Gaussian-mixture identity is:
///   `Var(beta) = E[A(rho)] + Var(b(rho))`.
///
/// Around `mu`, this routine keeps the first-order terms:
///
///   `E[A(rho)]      ~= A(mu) = Hmu^{-1}`
///   `Var(b(rho))    ~= J Sigma J^T`
///   `Var(beta)      ~= Hmu^{-1} + J V_rho J^T`.
///
/// Equivalent first-order propagation around the outer optimum `rho*`:
///
///   `Var(beta_hat) ~= Var(beta_hat | rho_hat) + (d beta_hat / d rho) Var(rho_hat) (d beta_hat / d rho)^T`
///                  `= V_beta + J V_rho J^T`.
///
/// Components:
///   `J[:,k] = d(beta_hat)/d(rho_k) = -H^{-1}(A_k beta_hat),  A_k = exp(rho_k) S_k`
///   `V_rho  = (d^2 V / d rho^2 at rho*)^{-1}`
///
/// Exact non-Gaussian V_ρ^{-1} requires the full Hessian with:
///   - tr(H^{-1}H_{kℓ})
///   - tr(H^{-1}H_k H^{-1}H_ℓ)
///   - pseudo-det second derivatives in S
///   - and H_{kℓ} terms containing fourth-likelihood derivatives.
///
/// This routine obtains V_ρ^{-1} from the analytic rho-space Hessian selected
/// by `compute_lamlhessian_consistent`, then inverts its explicitly identified
/// subspace without perturbing the matrix. If exact geometry is unavailable,
/// the typed status records why; no substitute Hessian is used.
///
/// Notes on omitted higher-order terms:
/// - The exact `E[A(rho)]` and `Var(b(rho))` can be written with the Gaussian
///   smoothing/heat operator `exp(0.5 * Delta_Sigma)` (equivalently Wick/Isserlis
///   contractions of high-order derivatives).
/// - Those infinite-series corrections are not expanded in this routine.
pub(crate) struct SmoothingCorrectionComputation {
    /// The correction's square-root factor `B` in the original coefficient
    /// frame, `C = J·V_ρ·Jᵀ = B·Bᵀ` ([`smoothing_correction_factor`]). A route
    /// that publishes a dense covariance forms `C` from it with
    /// [`smoothing_correction_gram`]; the factorized route keeps `B` (#3283).
    pub factor: Option<Array2<f64>>,
    /// Regularized inverse outer Hessian `Cov(rho_hat)` in the same rho ordering
    /// as the fitted smoothing-parameter vector. This exposes the #740 quantity
    /// to LR Bartlett inference without changing the production algebra that
    /// computes it.
    pub rho_covariance: Option<Array2<f64>>,
    /// Identified-subspace rank of the rho-Hessian inverse used to build
    /// `correction`. `Some(n)` if the matrix was SPD and fully inverted;
    /// `Some(r)` with `r < n` if the pseudo-inverse dropped non-identified
    /// directions; `None` when no inversion was attempted or it failed before
    /// producing a usable V_ρ.
    pub active_rank: Option<usize>,
    pub status: SmoothingCorrectionStatus,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SmoothingCorrectionStatus {
    Computed,
    NotApplicableNoSmoothingParameters,
    ZeroNoIdentifiedOuterDirections,
    Unavailable(SmoothingCorrectionUnavailable),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SmoothingCorrectionUnavailable {
    ObjectiveInnerHessian {
        error: String,
    },
    InnerHessianDimension {
        rows: usize,
        cols: usize,
        coefficients: usize,
    },
    InnerHessianNotPositiveDefinite,
    SensitivitySolve,
    OuterHessian {
        error: String,
    },
    OuterHessianInverse { error: String },
    PenaltyDimension {
        rho: usize,
        lambdas: usize,
        canonical_penalties: usize,
    },
    PenaltyStructure { error: String },
    NonFiniteCorrection,
}

/// Certified inverse of the rho-space LAML Hessian. A pseudoinverse is admitted
/// only for zero directions whose count is independently certified by the
/// structural penalty map; positive curvature is never truncated and negative
/// curvature is never salvaged as covariance.
#[derive(Debug)]
pub struct InvertedRhoHessian {
    pub inverse: Array2<f64>,
    pub active_rank: usize,
    pub structural_zero: usize,
    /// Directions dropped because their curvature sits under the outer loop's
    /// own gradient noise floor (#2428). Distinct from `structural_zero`.
    pub below_gradient_floor: usize,
    /// Directions dropped because their curvature is smaller than the MEASURED
    /// `‖δH‖₂` of the matrix that reported it (#2748). Distinct from both of
    /// the above: not certified flat, not a chain-rule term — just not
    /// resolved.
    pub unresolvable_curvature: usize,
    /// Coordinate axes the outer certificate railed and therefore never judged
    /// (see [`EigenClassification::Railed`]). Zero whenever no rail is passed.
    pub railed: usize,
    pub used_structural_pseudoinverse: bool,
    /// The MEASURED curvature resolution the classification above was taken
    /// at: `‖δH‖₂` under Weyl, as the largest of this site's measured
    /// components (#2690, #2748).
    ///
    /// It was called `eigenvalue_backward_error_bound` while the eigensolver's
    /// residual was the only component there. It is not that any more — a
    /// caller-supplied assembly measurement can exceed it by eleven orders —
    /// and a field named for one component while carrying the maximum over
    /// several is a claim the code does not enforce.
    pub curvature_resolution: f64,
    pub eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    pub classifications: Vec<EigenClassification>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigenClassification {
    Active,
    StructuralZero,
    /// Curvature indistinguishable from zero at the accuracy the OUTER LOOP
    /// itself demonstrated when it accepted this ρ̂ (#2428).
    ///
    /// Not a structural zero of the penalty map: it is created dynamically by
    /// a smoothing parameter running to its saturation limit, where the term
    /// is numerically pinned to its own null space and the REML profile goes
    /// flat in that coordinate. The count is therefore NOT knowable a priori
    /// and is excluded from the structural-nullity identity below.
    BelowGradientFloor,
    /// Curvature indistinguishable from zero at the MEASURED accuracy of the
    /// Hessian itself, `‖δH‖₂` under Weyl (#2748).
    ///
    /// Distinct from both siblings, and the distinction is the whole point of
    /// having three names. `StructuralZero` is excused by STRUCTURE — the
    /// penalty map certifies the criterion is exactly flat there, and no
    /// measurement can change that. `BelowGradientFloor` is excused by the
    /// chain rule — the ρ-curvature there is `Σ_k g_k v_k²` and carries no
    /// second-order content. This one is excused by RESOLUTION: the curvature
    /// may be perfectly real, but the matrix that reports it is not known
    /// accurately enough for its magnitude — or its sign — to be a
    /// measurement. Merging it into `StructuralZero`, as this site did while
    /// `zero_bound` was only ever an eigensolver backward error, made the name
    /// assert a certificate from the penalty map that the penalty map had not
    /// issued.
    UnresolvableCurvature,
    /// A coordinate axis `e_k` the outer certificate RAILED: `ρ̂_k` sits on a
    /// box face, so it is a boundary estimate, not an interior mode. The
    /// certificate judged definiteness off these coordinates, and this site
    /// does the same.
    ///
    /// It carries zero variance, and that is exact rather than a convenience.
    /// A rail face is where the term's effective degrees of freedom sit at their
    /// `λ → ∞` or `λ → 0` limit to within the gradient's resolution, so
    /// `∂β̂/∂ρ_k = −H⁻¹ λ_k S_k β̂` vanishes there: `J e_k = 0`, and no value of
    /// `V_ρ` along `e_k` changes `J V_ρ Jᵀ`. The curvature `H_kk` is reported,
    /// never judged: it is the Hessian of a criterion that is flat toward the
    /// limit, and its sign at roundoff scale is not a statement about the fit.
    Railed,
}

/// The rectangular square-root factor of `Q J Vρ Jᵀ Qᵀ`.
///
/// `invert_identified_rho_hessian` has already classified every retained
/// eigendirection as strictly positive and resolved. Therefore
///
/// ```text
/// Vρ = U_active diag(1 / σ_active) U_activeᵀ
/// Q J Vρ Jᵀ Qᵀ = B Bᵀ,
/// B = Q J U_active diag(1 / sqrt(σ_active)).
/// ```
///
/// Forming the covariance as the Gram matrix `B Bᵀ` preserves its defining
/// positive-semidefinite structure in floating-point arithmetic. In
/// particular, each diagonal is accumulated as a sum of squares. The previous
/// `(J Vρ) Jᵀ` followed by `Q (…) Qᵀ` route accumulated signed cancellation in
/// two generic matrix products. When a true correction diagonal was zero or
/// much smaller than the matrix's signed off-diagonal scale, that route could
/// emit a negative diagonal of order ε; a small conditional covariance need
/// not dominate that error when the corrected covariance is assembled
/// ([`smoothing_correction_gram`]).
///
/// `B` is `p × r`, `r` the number of active directions, so a route that cannot
/// hold a `p × p` matrix carries the correction as `B` itself (#3283).
fn smoothing_correction_factor(
    jacobian_trans: &Array2<f64>,
    qs: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
    classifications: &[EigenClassification],
) -> Array2<f64> {
    let rho_dimension = jacobian_trans.ncols();
    assert_eq!(eigenvalues.len(), rho_dimension);
    assert_eq!(eigenvectors.dim(), (rho_dimension, rho_dimension));
    assert_eq!(classifications.len(), rho_dimension);
    assert_eq!(qs.ncols(), jacobian_trans.nrows());

    let active_directions: Vec<usize> = classifications
        .iter()
        .enumerate()
        .filter_map(|(index, class)| {
            matches!(class, EigenClassification::Active).then_some(index)
        })
        .collect();
    let mut factor_trans =
        Array2::<f64>::zeros((jacobian_trans.nrows(), active_directions.len()));
    for (factor_column, &eigen_index) in active_directions.iter().enumerate() {
        let eigenvalue = eigenvalues[eigen_index];
        assert!(
            eigenvalue.is_finite() && eigenvalue > 0.0,
            "an active rho-Hessian eigendirection must have finite positive curvature"
        );
        let scale = eigenvalue.sqrt().recip();
        let direction = jacobian_trans.dot(&eigenvectors.column(eigen_index));
        factor_trans
            .column_mut(factor_column)
            .assign(&direction.mapv(|value| value * scale));
    }

    qs.dot(&factor_trans)
}

/// `C = B Bᵀ` from the factor [`smoothing_correction_factor`] returns, with
/// every diagonal entry accumulated as the sum of squares `‖b_i‖²`.
pub(crate) fn smoothing_correction_gram(factor: &Array2<f64>) -> Array2<f64> {
    let mut correction = factor.dot(&factor.t());
    gam_linalg::matrix::symmetrize_in_place(&mut correction);
    // Make the Gram diagonal's non-negativity explicit rather than relying on
    // a backend-specific GEMM diagonal accumulation path.
    for index in 0..factor.nrows() {
        let row = factor.row(index);
        correction[[index, index]] = row.dot(&row);
    }
    correction
}

/// The eigensolver's own backward error, expressed as a **curvature
/// resolution** under Weyl's law (#2690).
///
/// `matrix` here is an ANALYTICALLY formed symmetric matrix, so the applicable
/// law is [`gam_linalg::curvature_resolution::CurvatureLaw::AnalyticWeyl`]:
/// there is no step and no `1/h²` amplification, and the resolution of an
/// eigenvalue is `‖δH‖₂`, the perturbation the returned eigenpairs are exact
/// for. The finite-difference law `(2/√3)·√(ε_f·M₄)` — the other half of #2690
/// — governs a curvature differenced from criterion VALUES and must never be
/// applied here; routing this through the shared constructor is what records
/// that choice at every comparison site.
///
/// The measured quantity is `max_i ‖H v_i − σ_i v_i‖₂`, a certified `‖δH‖₂` for
/// the eigenpairs as returned — exactly Weyl's perturbation, computed rather
/// than assumed — with each pair's residual certified against its own
/// evaluation error by [`eigenpair_residual_bounds`].
///
/// ⚠ This bounds *"given this matrix, how wrong is σ?"*. It says nothing about
/// *"how wrong is this matrix?"* — the assembly error of `H` itself, which is
/// the `‖δH‖₂` a criterion-level resolution question needs and which is nine
/// orders larger on the fixtures #2690 measured. Do not read one for the other.
pub(crate) fn eigenpair_backward_error_bound(
    matrix: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> Result<gam_linalg::curvature_resolution::CurvatureResolution, String> {
    let resolution = eigenpair_residual_bounds(matrix, eigenvalues, eigenvectors)?
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    gam_linalg::curvature_resolution::CurvatureResolution::analytic_weyl(resolution)
        .map_err(|error| error.to_string())
}

/// Per eigenpair, a certified bound on the exact residual `‖H v_i − σ_i v_i‖₂`:
/// the computed residual norm plus the norm of its own evaluation error.
///
/// The residual is computed in floating point, so where it rounds to zero (a
/// diagonal input, say) it certifies only down to that evaluation error.
/// Component `j` is the `n`-term inner product `Σ_k H_jk v_k`, one product
/// `σ_i v_j` and one subtraction, so Higham's model (`gam_linalg::roundoff`)
/// puts the exact component within `γ_{n+1}·(Σ_k |H_jk||v_k| + |σ_i||v_j|)` of
/// the computed one. No coefficient is chosen.
pub(crate) fn eigenpair_residual_bounds(
    matrix: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    eigenvectors: &Array2<f64>,
) -> Result<Array1<f64>, String> {
    let n = matrix.nrows();
    if matrix.ncols() != n || eigenvalues.len() != n || eigenvectors.dim() != (n, n) {
        return Err("eigendecomposition dimensions do not match the symmetric matrix".into());
    }
    if !matrix.iter().all(|value| value.is_finite())
        || !eigenvalues.iter().all(|value| value.is_finite())
        || !eigenvectors.iter().all(|value| value.is_finite())
    {
        return Err("eigendecomposition contains a non-finite value".into());
    }
    let magnitudes = matrix.mapv(f64::abs);
    let evaluation_growth = gam_linalg::roundoff::accumulation_growth(n + 1);
    let mut bounds = Array1::<f64>::zeros(n);
    for column in 0..n {
        let vector = eigenvectors.column(column);
        let eigenvalue = eigenvalues[column];
        let residual = matrix.dot(&vector) - &vector.mapv(|value| value * eigenvalue);
        let evaluation_scale = magnitudes.dot(&vector.mapv(f64::abs))
            + &vector.mapv(|value| (value * eigenvalue).abs());
        bounds[column] = residual.dot(&residual).sqrt()
            + evaluation_growth * evaluation_scale.dot(&evaluation_scale).sqrt();
    }
    Ok(bounds)
}

/// Invert the ρ-Hessian on the subspace where its curvature is actually
/// resolvable, judged by the SAME standard the outer certificate used to accept
/// this ρ̂ (#2428).
///
/// `outer_gradient` is the outer loop's residual gradient at the certified
/// point, per ρ coordinate; pass an empty array when it is unavailable. The
/// outer certificate admits a point as a minimum by testing that
/// `H + diag(|g|)` is PSD off the railed coordinates
/// (`certificate_hessian_is_psd_off_railed_above_gradient_floor`): a curvature
/// smaller than the residual gradient in that coordinate is below the noise
/// floor of the very machinery that produced both numbers, so it cannot be
/// called negative. Along eigenvector `v` that test is exactly
/// `σ + Σ_k |g_k| v_k² ≥ 0`, so the per-direction floor is `Σ_k |g_k| v_k²`,
/// evaluated here on the eigenvectors we already have.
///
/// Applying any WEAKER standard here than the certificate applied there lets the
/// two subsystems reach opposite verdicts on one matrix at one converged point —
/// which is exactly #2428: the outer loop certified the fit, and this inversion
/// then destroyed it over an eigenvalue 19x below the residual gradient.
///
/// Directions under that floor are dropped from the inverse rather than
/// regularized. That is the correct limit, not a convenience: they arise when a
/// smoothing parameter saturates, and there `∂β̂/∂ρ → 0`, so the direction
/// contributes nothing to `J·V_ρ·Jᵀ` however large `1/σ` would have been.
///
/// # ⚠ The floor is an EXACT IDENTITY on these directions, not a noise floor (#2676)
///
/// `ρ = log λ` is a nonlinear reparameterisation, so for any smooth criterion
///
/// ```text
///     H_ρ = diag(λ)·H_λ·diag(λ) + diag(g_ρ)
/// ```
///
/// holds exactly. The second term is pure chain rule. So for any direction `t`
/// on which the λ-space curvature vanishes — a saturated rail (`∂β̂/∂ρ → 0`,
/// which is the case this floor was written for) or a penalty-map structural
/// null (`S_i ∝ S_j` ⟹ `H_λ·diag(λ)t = 0`) — the ρ-curvature is
///
/// ```text
///     t'H_ρ t = Σ_k g_k t_k²        EXACTLY
/// ```
///
/// and `Σ_k |g_k| t_k²` is `direction_floor` verbatim. Such a direction
/// therefore does not sit *near* the boundary of the tests below; it sits **on
/// it, by identity**, and which side it lands on is decided by the disagreement
/// between the gradient evaluation and the Hessian evaluation. This is measured:
/// `geo_disease_eas_matern` refuses at `|σ|/floor = 1.005540`, i.e. the identity
/// holding to 0.55%, and the sign of that 0.55% residual is the whole verdict.
///
/// Every branch would have its boundary there:
///
/// * `σ < −floor` fires when the residual is negative and `g < 0`;
/// * `σ > floor` (⟹ `Active`) fires when the residual is positive and `g > 0`,
///   which drops `identified_null` and then trips the nullity check instead —
///   the `geo_disease_matern` refusal.
///
/// **The repair was not to widen `floor`.** The quantity is not under-resolved;
/// it was being compared against itself. `invariance` is the penalty map's
/// certified null subspace, lifted to ρ by `diag(λ)⁻¹`
/// ([`crate::penalty_invariance`]), and it is DEFLATED before any of the tests
/// below run: those directions are accounted as the structural zeros the
/// penalty map says they are, and only the orthogonal complement is judged, by
/// the existing rule, unchanged. Passing `None` reproduces the pre-#2676
/// behaviour bit for bit, which is what every model without a redundant penalty
/// map gets.
///
/// # ⚠ This is NOT literally the same standard the certificate applied
///
/// The doc above says this mirrors `run.rs`'s certificate. One difference
/// remains, recorded so nobody re-derives the equivalence:
///
/// * `σ_i + Σ_k|g_k|v_k² ≥ 0` on `H`'s own eigenvectors is the Rayleigh
///   quotient of `H + diag(|g|)` at particular vectors — it is IMPLIED by that
///   matrix being PSD, not equivalent to it. So with an empty exclusion set,
///   and given the SAME `H` and `g`, this test cannot fail in exact arithmetic
///   on a point the certificate passed. When it does, what disagreed is the two
///   subsystems' evaluations, not the fit.
///
/// Both sites deflate the same invariance through
/// `crate::penalty_invariance::judged_subspace_basis` (#2676). This entry point
/// excludes no coordinate; [`invert_identified_rho_hessian_off_railed`] takes
/// the certificate's railed set and excludes it through that same call, which
/// is what the fitted-model correction does.
pub fn invert_identified_rho_hessian(
    hessian_rho: &Array2<f64>,
    expected_structural_nullity: usize,
    outer_gradient: &Array1<f64>,
    invariance: Option<&Array2<f64>>,
    caller_measured_hessian_error: &[gam_linalg::curvature_resolution::MeasuredHessianError],
) -> Result<InvertedRhoHessian, String> {
    invert_identified_rho_hessian_off_railed(
        hessian_rho,
        expected_structural_nullity,
        outer_gradient,
        invariance,
        &[],
        caller_measured_hessian_error,
    )
}

/// [`invert_identified_rho_hessian`] judged off the coordinates the outer
/// certificate railed, the face the certificate itself judged on.
///
/// The certificate accepts ρ̂ by testing `H + diag(|g|)` for PSD on
/// `judged_subspace_basis(n, railed, invariance)`: a railed `ρ̂_k` is a
/// boundary estimate and its axis is not part of the curvature verdict. Judging
/// the full matrix here instead re-decides those axes on a curvature the
/// certificate never looked at, and a roundoff-negative `H_kk` on a coordinate
/// saturated at `λ → ∞` then refused the correction for EVERY direction — the
/// fit shipped with no smoothing correction at all although only the railed
/// axes were in question. Each railed axis is accounted as
/// [`EigenClassification::Railed`] with zero variance, which is exact because
/// `J e_k = 0` on a rail face; every other direction is judged by the rule
/// below, unchanged. An empty `railed` is [`invert_identified_rho_hessian`]
/// bit for bit.
pub fn invert_identified_rho_hessian_off_railed(
    hessian_rho: &Array2<f64>,
    expected_structural_nullity: usize,
    outer_gradient: &Array1<f64>,
    invariance: Option<&Array2<f64>>,
    railed: &[usize],
    caller_measured_hessian_error: &[gam_linalg::curvature_resolution::MeasuredHessianError],
) -> Result<InvertedRhoHessian, String> {
    let n = hessian_rho.nrows();
    if expected_structural_nullity > n {
        return Err(format!(
            "structural nullity {expected_structural_nullity} exceeds rho dimension {n}"
        ));
    }
    if !outer_gradient.is_empty() && outer_gradient.len() != n {
        return Err(format!(
            "outer gradient has {} coordinate(s) but the rho Hessian is {n}x{n}",
            outer_gradient.len()
        ));
    }
    // Reject a non-finite Hessian before the eigensolver sees it: a NaN spectrum
    // would otherwise classify as "unresolvable" rather than as the hard input
    // defect it is.
    gam_linalg::utils::validate_finite_symmetric_matrix(hessian_rho, "rho Hessian")
        .map_err(|error| error.to_string())?;

    // #2676: the criterion is EXACTLY constant along `diag(lambda)^{-1} null(G)`,
    // so the curvature there is `sum_k g_k t_k^2` — the chain-rule term, not a
    // measurement. Judge its orthogonal complement and account those directions
    // as the structural zeros the penalty map certifies. With no invariance
    // (`deflation = None`) every line below runs on `hessian_rho` itself, so a
    // model without a redundant penalty map does not move by an ulp.
    let railed: Vec<usize> = railed
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<usize>>()
        .into_iter()
        .collect();
    if let Some(&index) = railed.iter().find(|&&index| index >= n) {
        return Err(format!(
            "railed coordinate {index} is outside the {n}-dimensional rho Hessian"
        ));
    }
    let railed_dimension = railed.len();

    let deflation = invariance.filter(|basis| basis.nrows() == n && basis.ncols() > 0);
    let judged = if railed.is_empty() {
        deflation
            .and_then(|basis| crate::penalty_invariance::judged_subspace_basis(n, &[], Some(basis)))
            // A basis that spans everything deflates nothing, and compressing
            // against it would only re-symmetrize `hessian_rho` in the last
            // bits. Drop back to the untouched path instead.
            .filter(|basis| basis.ncols() < n)
    } else {
        // `None` here means nothing is left to judge: every coordinate is
        // railed, or the invariance spans the whole interior face.
        Some(
            crate::penalty_invariance::judged_subspace_basis(n, &railed, deflation)
                .unwrap_or_else(|| Array2::<f64>::zeros((n, 0))),
        )
    };
    let compressed = judged
        .as_ref()
        .map(|basis| crate::penalty_invariance::compress_to_judged_subspace(hessian_rho, basis));
    let work = compressed.as_ref().unwrap_or(hessian_rho);
    let judged_dimension = work.nrows();

    let (judged_eigenvalues, judged_eigenvectors) = if judged_dimension == 0 {
        (Array1::<f64>::zeros(0), Array2::<f64>::zeros((0, 0)))
    } else {
        work.eigh(faer::Side::Lower)
            .map_err(|error| format!("rho-Hessian eigendecomposition failed: {error}"))?
    };

    // Every direction, in ρ coordinates and in one order: the deflated
    // invariance first (its Rayleigh quotient reported as measured, never
    // judged), then the railed axes (likewise reported, never judged), then
    // the judged complement.
    //
    // The deflated block is read back OUT of the judged basis rather than taken
    // from `deflation`'s own columns. Those are not the same set in general:
    // `judged_subspace_basis` can shrink the deflation (the railed-face
    // restriction, the dependence drop), and reporting the input's columns here
    // would name a subspace the verdict was not taken on. With rails it is the
    // complement of the judged basis AND the railed axes — the judged basis is
    // zero on every railed row, so the two stack to an orthonormal set — so
    // the invariance residual below is measured on invariance directions only:
    // its identity `T'H_ρT = T'diag(g_ρ)T` does not hold on a railed axis.
    let railed_axes = {
        let mut axes = Array2::<f64>::zeros((n, railed_dimension));
        for (column, &index) in railed.iter().enumerate() {
            axes[[index, column]] = 1.0;
        }
        axes
    };
    let removed = judged.as_ref().and_then(|basis| {
        let mut kept = Array2::<f64>::zeros((n, basis.ncols() + railed_dimension));
        kept.slice_mut(ndarray::s![.., ..basis.ncols()]).assign(basis);
        kept.slice_mut(ndarray::s![.., basis.ncols()..])
            .assign(&railed_axes);
        crate::penalty_invariance::deflated_directions(n, &kept)
    });
    let deflated_dimension = removed.as_ref().map_or(0, |directions| directions.ncols());
    if deflated_dimension + railed_dimension + judged_dimension != n {
        return Err(format!(
            "rho-space split does not partition R^{n}: {deflated_dimension} deflated + \
             {railed_dimension} railed + {judged_dimension} judged"
        ));
    }
    let unjudged_dimension = deflated_dimension + railed_dimension;

    // `hessian_rho` is an ANALYTIC Hessian, so the applicable curvature law is
    // Weyl's `‖δH‖₂` (#2690), not the finite-difference `(2/√3)·√(ε_f·M₄)`.
    // The distinction is load-bearing: the two are orders apart on this
    // criterion, and `ε_f` — the value-level evaluation error the FD law needs
    // — bounds the error of a separately-coded second derivative not at all.
    //
    // ‖δH‖₂ is MEASURED, per site, and this site has two measurements of it
    // that answer different questions (#2748):
    //
    // * the eigensolver's residual — *"given this matrix, how wrong is σ?"*.
    //   It is a statement about the decomposition and says nothing about the
    //   assembly;
    // * the penalty map's invariance residual — *"how wrong is this matrix?"*.
    //   On the certified invariance `T` the identity
    //   `T'H_ρT = T'diag(g_ρ)T` holds EXACTLY, so its residual is error and
    //   only error, measured in situ on this ρ, on this (H_ρ, g_ρ) pair, and
    //   in exactly the currency the tests below spend (a Hessian eigenvalue
    //   against a gradient-built floor).
    //
    // Measured on `geo_disease_eas_matern_k6`: `8.342e-19` and `9.872e-8`.
    // Refusing a `-2.010e-8` curvature on the first is refusing on the
    // eigensolver while the assembly is demonstrably eleven orders worse.
    let assembly_inconsistency = removed.as_ref().and_then(|directions| {
        crate::penalty_invariance::invariance_residual_2norm(
            hessian_rho,
            outer_gradient,
            directions,
        )
    });
    // Nothing judged, nothing decomposed: the eigensolver made no error.
    let eigensolver_backward_error = if judged_dimension == 0 {
        0.0
    } else {
        eigenpair_backward_error_bound(work, &judged_eigenvalues, &judged_eigenvectors)?
            .resolution()
    };
    let mut measured_hessian_error = vec![gam_linalg::curvature_resolution::MeasuredHessianError::new(
        "eigensolver backward error",
        eigensolver_backward_error,
    )];
    // Whatever the caller measured on the SAME matrix at the same point. The
    // assembly's own error is not visible from inside this function -- it only
    // ever sees the finished matrix -- so a caller that holds an exactly-zero
    // identity about the assembly (the symmetrization defect, say) hands it in
    // here rather than throwing it away.
    measured_hessian_error.extend_from_slice(caller_measured_hessian_error);
    if let Some(value) = assembly_inconsistency {
        measured_hessian_error.push(
            gam_linalg::curvature_resolution::MeasuredHessianError::new(
                "penalty-map invariance residual |T'(H_rho - diag(g_rho))T|_2",
                value,
            ),
        );
    }
    let curvature_resolution =
        gam_linalg::curvature_resolution::CurvatureResolution::analytic_weyl_from_components(
            &measured_hessian_error,
        )
        .map_err(|error| error.to_string())?;
    let zero_bound = curvature_resolution.resolution();

    let mut eigenvalues = Array1::<f64>::zeros(n);
    let mut eigenvectors = Array2::<f64>::zeros((n, n));
    for column in 0..deflated_dimension {
        let direction = removed
            .as_ref()
            .expect("a deflated dimension implies a removed subspace")
            .column(column)
            .to_owned();
        eigenvalues[column] = direction.dot(&hessian_rho.dot(&direction));
        eigenvectors.column_mut(column).assign(&direction);
    }
    for (offset, &index) in railed.iter().enumerate() {
        let column = deflated_dimension + offset;
        eigenvalues[column] = hessian_rho[[index, index]];
        eigenvectors[[index, column]] = 1.0;
    }
    for index in 0..judged_dimension {
        let target = unjudged_dimension + index;
        eigenvalues[target] = judged_eigenvalues[index];
        match judged.as_ref() {
            Some(basis) => eigenvectors
                .column_mut(target)
                .assign(&basis.dot(&judged_eigenvectors.column(index))),
            None => eigenvectors
                .column_mut(target)
                .assign(&judged_eigenvectors.column(index)),
        }
    }

    // The outer certificate's EXACT chain-rule term along eigenvector `v`:
    // `Σ_k |g_k| v_k²`, the magnitude of `diag(g_rho)`'s Rayleigh quotient
    // there. Not a resolution — one of the two exact terms of
    // `H_rho = diag(lambda) H_lambda diag(lambda) + diag(g_rho)`.
    let chain_rule_term = |i: usize| -> f64 {
        if outer_gradient.is_empty() {
            return 0.0;
        }
        let v = eigenvectors.column(i);
        let mut term = 0.0_f64;
        for k in 0..n {
            term += outer_gradient[k].abs() * v[k] * v[k];
        }
        if term.is_finite() { term } else { 0.0 }
    };

    // Identification floor: a direction counts as ACTIVE only if its curvature
    // clears BOTH the chain-rule scale and the resolution, so `max` is the
    // right combination here — this asks "is this positive curvature a
    // measurement?", and either bar failing means it is not.
    let direction_floor = |i: usize| -> f64 {
        let term = chain_rule_term(i);
        if term > zero_bound { term } else { zero_bound }
    };

    // Refusal bar: the SUM, not the max, and this is what makes this site agree
    // with the outer certificate instead of applying a strictly stronger
    // standard than it (#2748).
    //
    // The certificate accepts rho-hat by testing that `H + diag(|g|)` is PSD to
    // its own curvature resolution `r`, i.e. `λ_min(H + diag|g|) ≥ −r`. Along
    // an eigenvector of `H` that reads
    //
    //     σ  ≥  −( Σ_k |g_k| v_k²  +  r )
    //
    // — the two are ADDED, because they are different mechanisms: the chain
    // rule contributes exactly `Σ_k g_k v_k²` (bounded in magnitude by the
    // first term) and the assembly error contributes up to `‖δH‖₂`. Taking
    // their maximum was not derived from anything, and it made this gate refuse
    // points the certificate accepts — precisely the #2428 defect this site
    // exists to have fixed. Measured on `papuan_oce_matern_k12`:
    // `σ = −4.746e-7` against `Σ|g_k|v_k² = 4.407e-7` and a measured
    // `‖δH‖₂ = 1.013e-7`; the max refuses it and the sum, which is what the
    // certificate itself applied, does not.
    let refusal_bar = |i: usize| -> f64 { chain_rule_term(i) + zero_bound };

    for i in unjudged_dimension..n {
        let sigma = eigenvalues[i];
        let floor = refusal_bar(i);
        if sigma < -floor {
            let components = measured_hessian_error
                .iter()
                .map(|component| component.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "rho Hessian has negative curvature {sigma:.3e} below the outer certificate's own \
                 bar {floor:.3e} on that direction -- the sum of the EXACT chain-rule term \
                 {:.3e} of `H_rho = diag(lambda) H_lambda diag(lambda) + diag(g_rho)` (not a \
                 resolution: one of the two exact terms) and the curvature resolution of an \
                 analytically-formed eigenvalue, which is Weyl's ||dH||_2, measured here as \
                 {curvature_resolution} from [{components}] \
                 -- #2690, #2748); the penalty map certified {expected_structural_nullity} null \
                 direction(s) and {deflated_dimension} were deflated and {railed_dimension} \
                 railed axis/axes excluded before judging, so the direction above was judged; \
                 the outer loop certified this point as a minimum, \
                 and the curvature is larger than everything this assembly's own exactly-zero \
                 identities say it could have got wrong, so this is a genuine contradiction \
                 rather than an unresolvable direction",
                chain_rule_term(i),
            ));
        }
    }

    let mut inverse = Array2::<f64>::zeros((n, n));
    let mut projector = Array2::<f64>::zeros((n, n));
    let mut classifications = Vec::with_capacity(n);
    let mut active_rank = 0usize;
    let mut structural_zero = 0usize;
    let mut below_gradient_floor = 0usize;
    let mut unresolvable_curvature = 0usize;
    let mut railed_count = 0usize;

    for i in 0..n {
        let sigma = eigenvalues[i];
        let floor = direction_floor(i);
        let class = if i < deflated_dimension {
            // #2676: this direction is not judged at all. Its curvature is
            // `sum_k g_k t_k^2` by the chain rule and by nothing else, so
            // `sigma > floor`, `|sigma| <= zero_bound` and `sigma < -floor`
            // are three readings of one rounding residual. It IS the penalty
            // map's certified null, so it is accounted as one.
            EigenClassification::StructuralZero
        } else if i < unjudged_dimension {
            // A railed axis: a boundary estimate with `J e_k = 0`, excluded
            // from the verdict exactly as the certificate excluded it.
            EigenClassification::Railed
        } else if sigma > floor {
            EigenClassification::Active
        } else if sigma.abs() <= zero_bound {
            // Under the MEASURED resolution of this matrix. That used to be the
            // eigensolver's backward error and nothing else, which is why this
            // branch could be called a structural zero without anyone
            // noticing: at `1e-16` the two are the same population. Now that
            // `zero_bound` can carry a measured assembly error (#2748) they are
            // not, and a direction excused by resolution must not be reported
            // as one the penalty map certified.
            EigenClassification::UnresolvableCurvature
        } else {
            EigenClassification::BelowGradientFloor
        };
        classifications.push(class);
        match class {
            EigenClassification::Active => {
                active_rank += 1;
                let inv_lambda = 1.0 / sigma;
                if !inv_lambda.is_finite() {
                    return Err(format!(
                        "positive rho curvature {sigma:.3e} has an unrepresentable reciprocal"
                    ));
                }
                let v = eigenvectors.column(i);
                for row in 0..n {
                    for col in 0..n {
                        inverse[[row, col]] += inv_lambda * v[row] * v[col];
                        projector[[row, col]] += v[row] * v[col];
                    }
                }
            }
            EigenClassification::StructuralZero => structural_zero += 1,
            EigenClassification::BelowGradientFloor => below_gradient_floor += 1,
            EigenClassification::UnresolvableCurvature => unresolvable_curvature += 1,
            EigenClassification::Railed => railed_count += 1,
        }
    }
    // The penalty map certifies HOW MANY directions must be null; it cannot say
    // which eigenpair each one lands on, and once a rail saturates, a structural
    // zero and a saturation null are indistinguishable by magnitude (both sit
    // under the assembly noise, which is far above eigensolver backward error).
    // So the identity to enforce is that the Hessian exhibits AT LEAST the
    // certified number of null directions — finding fewer contradicts the
    // penalty map and is a real defect. Finding more is a saturated rail, which
    // is a property of this ρ̂, not of the penalty map, and is expected.
    // Every non-active direction is a candidate for the penalty map's certified
    // null: the map says HOW MANY must be null, never which eigenpair each
    // lands on, and a direction can fail to be active for any of the three
    // reasons above. Excluding the resolution-excused ones would make the
    // identity refuse a fit for having MORE evidence about its own Hessian,
    // which is backwards. A railed axis is non-active for the same reason: it
    // carries no variance, and a certified null can land on it.
    let identified_null =
        structural_zero + below_gradient_floor + unresolvable_curvature + railed_count;
    if identified_null < expected_structural_nullity {
        return Err(format!(
            "rho Hessian has only {identified_null} null direction(s) ({structural_zero} certified by the penalty map, {unresolvable_curvature} under the measured curvature resolution {curvature_resolution}, {below_gradient_floor} under the outer gradient floor, {railed_count} railed), but the penalty map certifies {expected_structural_nullity}"
        ));
    }

    // Every direction resolvable: return the Cholesky-certified inverse, which
    // is what this function has always returned for a strictly positive
    // definite ρ-Hessian. Keeping that path verbatim means no fit that succeeds
    // today moves by a single ulp — the eigen route below engages only where the
    // old code aborted.
    if active_rank == n {
        let certified =
            gam_linalg::utils::certified_spd_inverse(hessian_rho, "unperturbed rho Hessian")
                .map_err(|error| error.to_string())?;
        return Ok(InvertedRhoHessian {
            inverse: certified.into_inverse(),
            active_rank: n,
            structural_zero: 0,
            below_gradient_floor: 0,
            unresolvable_curvature: 0,
            railed: 0,
            used_structural_pseudoinverse: false,
            curvature_resolution: zero_bound,
            eigenvalues,
            eigenvectors,
            classifications,
        });
    }

    gam_linalg::matrix::symmetrize_in_place(&mut inverse);
    // Nothing was judged, so nothing was inverted: `V_ρ = 0` exactly and there
    // is no product to certify.
    if judged_dimension == 0 {
        return Ok(InvertedRhoHessian {
            inverse,
            active_rank,
            structural_zero,
            below_gradient_floor,
            unresolvable_curvature,
            railed: railed_count,
            used_structural_pseudoinverse: true,
            curvature_resolution: zero_bound,
            eigenvalues,
            eigenvectors,
            classifications,
        });
    }
    // Certify `H V = P` in the coordinates the spectrum was actually taken in.
    // Without a deflation those are the ρ coordinates themselves, verbatim.
    // With one they are the judged complement, and they have to be: the lifted
    // eigenvectors satisfy `(Z'HZ)u = σu`, so `H(Zu)` retains an invariance-
    // block component of size `‖Q'HZ‖ = ‖Q' diag(g) Z‖` — the same chain-rule
    // term again, since `Q'·diag(λ)H_λ = (diag(λ)Q)'H_λ = 0` on the certified
    // null. Certifying the ρ-space product would therefore be re-testing the
    // gradient, at the one site whose whole purpose is to stop doing that.
    let (certify_matrix, certify_inverse, certify_projector) = match judged.as_ref() {
        Some(basis) => (
            compressed.clone().expect("a judged basis implies a compression"),
            basis.t().dot(&inverse).dot(basis),
            basis.t().dot(&projector).dot(basis),
        ),
        None => (hessian_rho.clone(), inverse.clone(), projector.clone()),
    };
    let matrix_max_abs = gam_linalg::utils::validate_finite_symmetric_matrix(
        &certify_matrix,
        "structurally singular rho Hessian",
    )
    .map_err(|error| error.to_string())?;
    let residual = certify_matrix.dot(&certify_inverse) - &certify_projector;
    gam_linalg::utils::certify_linear_system_residual(
        certify_matrix.nrows(),
        matrix_max_abs,
        &certify_projector,
        &certify_inverse,
        &residual,
        "rho-Hessian structural pseudoinverse",
    )
    .map_err(|error| error.to_string())?;

    Ok(InvertedRhoHessian {
        inverse,
        active_rank,
        structural_zero,
        below_gradient_floor,
        unresolvable_curvature,
        railed: railed_count,
        used_structural_pseudoinverse: true,
        curvature_resolution: zero_bound,
        eigenvalues,
        eigenvectors,
        classifications,
    })
}

/// Relative-defect ceiling under which a pair is reported as a NEAR degeneracy
/// in `[INDEF-HESS]` diagnostics — the `near_degenerate_not_an_invariance`
/// line (#2676).
///
/// Purely a reporting bound: no verdict is taken on it, and the line it gates
/// exists to say that the penalty map certifies NOTHING for this pair. The
/// EXACT case is decided separately, at the residual norm's own arithmetic
/// floor `sqrt(block_dim^2) * EPSILON`, which is derived rather than chosen.
///
/// This replaces a `cos > 0.999` bar, which is `delta > 4.5e-2` — a pair four
/// per cent apart triggering a line that said "structural_redundancy_detected".
const INDEF_HESS_NEAR_DEGENERACY_DEFECT: f64 = 1e-1;

/// How much of the dominant-negative eigenvector must lie on a pair's
/// antisymmetric direction before the pair is named as its cause. `1/sqrt(2)`
/// is the value that direction takes when the eigenvector IS the pair's
/// antisymmetric direction and nothing else, so this admits any eigenvector at
/// least half-aligned with it in energy.
const INDEF_HESS_ANTISYMMETRIC_ALIGNMENT: f64 = 0.5;

/// Penalty-count crossover at which the [INDEF-HESS] pair dump switches from
/// the full O(k²) grid to top-3 pairs only. Bounds log volume on large-scale
/// rho_dim while keeping the per-pair detail useful for small models.
const INDEF_HESS_PAIR_DUMP_GRID_MAX_K: usize = 16;

/// Number of smallest-defect pairs to dump when
/// `n_pen > INDEF_HESS_PAIR_DUMP_GRID_MAX_K`.
const INDEF_HESS_PAIR_DUMP_TOP_N: usize = 3;

/// Diagnostic emitted whenever the post-fit rho-Hessian has at least one
/// non-identified direction (active_rank < n). Reports the eigendecomposition,
/// the dominant-negative eigenvector, per-eigenpair classification, and the
/// pairwise proportionality DEFECT
/// `delta_ij = min_c ||S_j - c S_i||_F / ||S_i||_F`.
///
/// # Why the defect and not the cosine (#2676)
///
/// `1 - cos = delta^2 / 2`, so the cosine is the defect squared and loses every
/// digit of a defect under `sqrt(EPSILON)`. This dump printed
/// `cos = 1.000000  one_minus_cos = 2.42e-9` on the `geo_disease_matern`
/// fixture, which is `delta = 7.0e-5` — an operator pair measurably NOT
/// proportional, reported under a headline that said
/// `structural_redundancy_detected`. That reading is the whole of #2676's
/// history. The dump now separates the two cases and names them:
///
/// * `structural_redundancy_detected` — `delta` at the residual norm's own
///   arithmetic floor `sqrt(block_dim^2) * EPSILON`. The criterion IS exactly
///   constant along the pair's antisymmetric direction and the certificate
///   deflates it.
/// * `near_degenerate_not_an_invariance` — `delta` measurably above that floor.
///   The outer Hessian is ill-conditioned along the direction, the criterion
///   carries genuine curvature of order `delta^2` there, the penalty map
///   certifies nothing, and a negative curvature is a resolution question
///   rather than a structure one.
///
/// Output is capped: when the penalty count exceeds 16, only the three
/// smallest-defect pairs are dumped instead of the full O(k²) grid.
///
/// # The reparameterisation split (#2676)
///
/// `rho = log(lambda)` is a nonlinear reparameterisation, so for ANY smooth
/// criterion `V`,
///
/// ```text
///     H_rho = diag(lambda) * H_lambda * diag(lambda) + diag(g_rho)
/// ```
///
/// exactly, where `g_rho = dV/drho`. The second term is pure chain rule and
/// carries no curvature information. Consequently a direction `t` that the
/// penalty map certifies as structurally null — i.e. `H_lambda w = 0` for
/// `w = diag(lambda) t`, which is what `S_i ∝ S_j` produces — has
///
/// ```text
///     t' H_rho t = sum_k g_k t_k^2      (EXACTLY, not approximately)
/// ```
///
/// and `sum_k |g_k| t_k^2` is precisely the per-direction gradient floor this
/// gate compares against. So the certified null direction sits ON the decision
/// boundary of every test in `invert_identified_rho_hessian` by identity, and
/// which side it lands on is decided by the disagreement between the gradient
/// evaluation and the Hessian evaluation — not by any property of the fit.
///
/// `outer_gradient` (empty when unavailable) is therefore dumped alongside the
/// spectrum as the split `sigma_i` vs `reparam_i = sum_k g_k v_k^2` vs
/// `intrinsic_i = sigma_i - reparam_i`. `intrinsic_i` is the Rayleigh quotient
/// of `diag(lambda) H_lambda diag(lambda)`, the only part whose sign is a
/// statement about the criterion. A direction whose `intrinsic_i` is a
/// vanishing fraction of `|sigma_i|` is on the identity, and a refusal decided
/// there is measuring the two evaluations against each other.
fn dump_indefinite_rho_hessian_diagnostic(
    hessian_rho: &Array2<f64>,
    final_rho: &Array1<f64>,
    canonical: &[gam_terms::construction::CanonicalPenalty],
    inverted: Option<&InvertedRhoHessian>,
    outer_gradient: &Array1<f64>,
) {
    // Everything below exists only to be logged at debug level: an
    // eigendecomposition and `O(K² block²)` pair traces nobody reads otherwise.
    let k = hessian_rho.nrows();
    if k == 0 || !log::log_enabled!(log::Level::Debug) {
        return;
    }

    // Reuse the eigendecomposition already computed by the inverter when present
    // (the slow path always populates it). Only recompute on the rare paths
    // where the diagnostic is called without an `InvertedRhoHessian` (e.g. the
    // eigendecomposition-failed bail in `compute_smoothing_correction`).
    let (eigenvalues_owned, eigenvectors_owned);
    let (eigenvalues_ref, eigenvectors_ref) = match inverted {
        Some(inv) if !inv.eigenvalues.is_empty() && !inv.eigenvectors.is_empty() => {
            (&inv.eigenvalues, &inv.eigenvectors)
        }
        _ => match hessian_rho.eigh(faer::Side::Lower) {
            Ok((evals, evecs)) => {
                eigenvalues_owned = evals;
                eigenvectors_owned = evecs;
                (&eigenvalues_owned, &eigenvectors_owned)
            }
            Err(err) => {
                log::debug!("[INDEF-HESS] eigendecomposition failed: {err}");
                return;
            }
        },
    };

    log::debug!("[INDEF-HESS] rho={:?}", final_rho.as_slice().unwrap_or(&[]),);
    if let Some(inv) = inverted {
        let deflated = inv
            .classifications
            .iter()
            .take_while(|class| matches!(class, EigenClassification::StructuralZero))
            .count();
        if deflated > 0 {
            log::debug!(
                "[INDEF-HESS] the leading {deflated} direction(s) below may be the penalty map's \
                 certified invariance, DEFLATED before any test (#2676): for those, the reported \
                 value is the Rayleigh quotient `t' H t = sum_k g_k t_k^2` -- a chain-rule term, \
                 not a curvature measurement"
            );
        }
    }
    log::debug!(
        "[INDEF-HESS] eigenvalues={:?}",
        eigenvalues_ref.as_slice().unwrap_or(&[]),
    );
    if let Some(inv) = inverted {
        log::debug!(
            "[INDEF-HESS] active_rank={}/{} structural_zero={} unresolvable_curvature={} below_gradient_floor={} curvature_resolution={:.3e}",
            inv.active_rank,
            k,
            inv.structural_zero,
            inv.unresolvable_curvature,
            inv.below_gradient_floor,
            inv.curvature_resolution,
        );
        if !inv.classifications.is_empty() {
            let labels: Vec<&'static str> = inv
                .classifications
                .iter()
                .map(|c| match c {
                    EigenClassification::Active => "A",
                    EigenClassification::UnresolvableCurvature => "R",
                    EigenClassification::StructuralZero => "Z",
                    EigenClassification::BelowGradientFloor => "G",
                    EigenClassification::Railed => "B",
                })
                .collect();
            log::debug!(
                "[INDEF-HESS] classifications={:?} (A=active; Z=certified null of the penalty \
                 map, excused by STRUCTURE; R=under the measured curvature resolution ||dH||_2, \
                 excused by RESOLUTION; G=under the outer loop's own gradient floor, i.e. a \
                 saturation null, excused by the CHAIN RULE; B=a railed coordinate axis, a \
                 boundary estimate the certificate did not judge)",
                labels,
            );
        }
    }

    let mut neg_idx = 0usize;
    let mut min_eig = f64::INFINITY;
    for (i, &v) in eigenvalues_ref.iter().enumerate() {
        if v < min_eig {
            min_eig = v;
            neg_idx = i;
        }
    }
    let v_neg = eigenvectors_ref.column(neg_idx);
    // `.to_vec()` not `.as_slice()`: an eigenvector is a COLUMN of a row-major
    // `Array2`, so it is never contiguous for k > 1 and `as_slice()` returns
    // `None` — every `[INDEF-HESS]` line ever logged printed `eigenvector=[]`
    // for exactly that reason. And the field is the MINIMUM eigenvalue, which
    // is routinely positive (measured `+9.56e-2` on `geo_disease_matern`);
    // naming it `negative_eigenvalue` asserted a sign it does not carry.
    log::debug!(
        "[INDEF-HESS] min_eigenvalue={:.4e} min_eigenvector={:?}",
        min_eig,
        v_neg.to_vec(),
    );

    // Split each eigenvalue into the chain-rule term and the intrinsic
    // curvature (see this function's doc, #2676). Without the gradient the
    // split cannot be formed, and the gate's decision boundary is invisible.
    if outer_gradient.len() == k {
        let mut sigma = Vec::with_capacity(k);
        let mut reparam = Vec::with_capacity(k);
        let mut intrinsic = Vec::with_capacity(k);
        let mut floor = Vec::with_capacity(k);
        for i in 0..k {
            let v = eigenvectors_ref.column(i);
            let mut signed = 0.0_f64;
            let mut absolute = 0.0_f64;
            for c in 0..k {
                let w = v[c] * v[c];
                signed += outer_gradient[c] * w;
                absolute += outer_gradient[c].abs() * w;
            }
            sigma.push(eigenvalues_ref[i]);
            reparam.push(signed);
            intrinsic.push(eigenvalues_ref[i] - signed);
            floor.push(absolute);
        }
        log::debug!(
            "[INDEF-HESS] outer_gradient={:?}",
            outer_gradient.to_vec(),
        );
        log::debug!(
            "[INDEF-HESS] reparam_split sigma={sigma:?} reparam=sum_k g_k v_k^2={reparam:?} \
             intrinsic=sigma-reparam={intrinsic:?} gradient_floor=sum_k |g_k| v_k^2={floor:?} \
             (a certified-null direction has intrinsic == 0 EXACTLY, so |sigma| == floor and \
             the gate's boundary runs through it; see #2676)"
        );
    } else {
        log::debug!(
            "[INDEF-HESS] outer_gradient unavailable ({} of {k} coordinate(s)); the \
             reparameterisation split sigma = intrinsic + sum_k g_k v_k^2 cannot be formed, \
             and every per-direction floor collapsed to the eigensolver backward error (#2676)",
            outer_gradient.len(),
        );
    }

    let n_pen = canonical.len();
    let mut tr_aa = vec![0.0_f64; n_pen];
    for i in 0..n_pen {
        let local = &canonical[i].local;
        let mut s = 0.0;
        for r in 0..local.nrows() {
            for c in 0..local.ncols() {
                s += local[[r, c]] * local[[r, c]];
            }
        }
        tr_aa[i] = s;
    }
    log::debug!(
        "[INDEF-HESS] penalty_count={} ranges={:?} ranks={:?}",
        n_pen,
        (0..n_pen)
            .map(|i| (canonical[i].col_range.start, canonical[i].col_range.end))
            .collect::<Vec<_>>(),
        (0..n_pen).map(|i| canonical[i].rank()).collect::<Vec<_>>(),
    );

    // Collect compatible pairs with their proportionality DEFECT.
    //
    // #2676: this dump used to carry `cos` and `1 - cos`, and that is the
    // coordinate the whole issue got lost in. With
    // `delta = min_c ||S_j - c S_i||_F / ||S_i||_F` the relation is
    // `1 - cos = delta^2 / 2`, so a pair `1.9e-5` apart prints as
    // `cos = 1.000000` / `one_minus_cos = 1.8e-10` and reads as an identity.
    // `delta` is formed directly from the residual, never through `1 - cos`,
    // and it is what the penalty map's Gram is judging when it certifies a
    // nullity.
    struct PairDefect {
        i: usize,
        j: usize,
        defect: f64,
        scale: f64,
        antisym_proj: f64,
    }
    let mut pairs: Vec<PairDefect> = Vec::new();
    for i in 0..n_pen {
        for j in (i + 1)..n_pen {
            let ci = &canonical[i];
            let cj = &canonical[j];
            if ci.col_range != cj.col_range {
                continue;
            }
            let local_i = &ci.local;
            let local_j = &cj.local;
            let mut dot = 0.0;
            for r in 0..local_i.nrows() {
                for c in 0..local_i.ncols() {
                    dot += local_i[[r, c]] * local_j[[r, c]];
                }
            }
            let (defect, scale) = if tr_aa[i] > 0.0 && tr_aa[j] > 0.0 {
                let scale = dot / tr_aa[i];
                let mut residual_sq = 0.0;
                for r in 0..local_i.nrows() {
                    for c in 0..local_i.ncols() {
                        let residual = local_j[[r, c]] - scale * local_i[[r, c]];
                        residual_sq += residual * residual;
                    }
                }
                ((residual_sq / tr_aa[i]).sqrt(), scale)
            } else {
                (f64::NAN, f64::NAN)
            };
            let antisym_proj = if v_neg.len() == n_pen {
                (v_neg[i] - v_neg[j]) / std::f64::consts::SQRT_2
            } else {
                f64::NAN
            };
            pairs.push(PairDefect {
                i,
                j,
                defect,
                scale,
                antisym_proj,
            });
        }
    }

    // Headline: structural redundancy detection. Pair defect at the residual
    // norm's own arithmetic floor AND the dominant-negative eigenvector's
    // top-2 absolute components on indices (i, j) with opposite signs.
    if min_eig < 0.0 && v_neg.len() == n_pen {
        for p in &pairs {
            let entries = canonical[p.i].local.len() as f64;
            if !(p.defect <= entries.sqrt() * f64::EPSILON) {
                continue;
            }
            let mut indexed: Vec<(usize, f64)> = v_neg
                .iter()
                .enumerate()
                .map(|(idx, &val)| (idx, val))
                .collect();
            indexed.sort_by(|a, b| {
                b.1.abs()
                    .partial_cmp(&a.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if indexed.len() < 2 {
                continue;
            }
            let top0 = indexed[0].0;
            let top1 = indexed[1].0;
            let (a, b) = if top0 == p.i && top1 == p.j {
                (indexed[0].1, indexed[1].1)
            } else if top0 == p.j && top1 == p.i {
                (indexed[1].1, indexed[0].1)
            } else {
                continue;
            };
            if a * b >= 0.0 {
                continue;
            }
            log::debug!(
                "[INDEF-HESS] structural_redundancy_detected pair=({},{}) relative_defect={:.6e} \
                 best_scale={:.6e} antisym_proj={:.4e} (proportional to the arithmetic that \
                 formed them, so the criterion IS exactly constant along this direction)",
                p.i,
                p.j,
                p.defect,
                p.scale,
                p.antisym_proj,
            );
            break;
        }
    }

    // A near-degenerate pair that is NOT an invariance, said as such (#2676).
    //
    // Without this line the dump was silent about the case that actually
    // produced every `geo_disease_*_matern` refusal: a pair whose defect is
    // small enough to make the outer Hessian near-singular along the
    // antisymmetric direction, and large enough that the criterion carries
    // genuine curvature there and nothing is deflated. The old dump printed
    // `cos = 1.000000` for exactly this case and let the reader conclude the
    // opposite.
    if min_eig < 0.0 && v_neg.len() == n_pen {
        for p in &pairs {
            let entries = canonical[p.i].local.len() as f64;
            let arithmetic_floor = entries.sqrt() * f64::EPSILON;
            if !(p.defect > arithmetic_floor)
                || !(p.defect <= INDEF_HESS_NEAR_DEGENERACY_DEFECT)
                || p.antisym_proj.abs() < INDEF_HESS_ANTISYMMETRIC_ALIGNMENT
            {
                continue;
            }
            log::debug!(
                "[INDEF-HESS] near_degenerate_not_an_invariance pair=({},{}) \
                 relative_defect={:.6e} arithmetic_floor={arithmetic_floor:.6e} \
                 antisym_proj={:.4e} — this pair is MEASURABLY distinct ({:.3e}x the floor), so \
                 the criterion is NOT constant along their antisymmetric direction, the penalty \
                 map certifies NOTHING here, and no direction is deflated. A negative curvature \
                 on this direction is a resolution question, not a structure one (#2676)",
                p.i,
                p.j,
                p.defect,
                p.antisym_proj,
                p.defect / arithmetic_floor,
            );
            break;
        }
    }

    // Cap output: dump the full grid when small, otherwise only the N smallest
    // defects — the near-proportional end is the informative one.
    if n_pen <= INDEF_HESS_PAIR_DUMP_GRID_MAX_K {
        for p in &pairs {
            log::debug!(
                "[INDEF-HESS] pair=({},{}) relative_defect={:.6e} best_scale={:.6e} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.defect,
                p.scale,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
        // Note: we no longer log a "ranges_differ" line per skipped pair to
        // keep the diagnostic O(k). The headline pair already captures intent.
    } else {
        let mut top: Vec<&PairDefect> = pairs.iter().filter(|p| p.defect.is_finite()).collect();
        top.sort_by(|a, b| {
            a.defect
                .partial_cmp(&b.defect)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for p in top.iter().take(INDEF_HESS_PAIR_DUMP_TOP_N) {
            log::debug!(
                "[INDEF-HESS] top_pair=({},{}) relative_defect={:.6e} best_scale={:.6e} tr_ii={:.4e} tr_jj={:.4e} v_neg[i]-v_neg[j]/sqrt2={:.4e}",
                p.i,
                p.j,
                p.defect,
                p.scale,
                tr_aa[p.i],
                tr_aa[p.j],
                p.antisym_proj,
            );
        }
    }
}

/// `outer_gradient` is the outer loop's residual gradient at the certified ρ̂,
/// per coordinate (empty when unavailable). It is the resolution floor the
/// ρ-Hessian's definiteness must be judged against — see
/// [`invert_identified_rho_hessian`] and #2428.
///
/// `railed` names the ρ coordinates the outer certificate railed (empty when
/// none): they are boundary estimates with `∂β̂/∂ρ_k = 0`, excluded from the
/// ρ-Hessian's verdict exactly as the certificate excluded them — see
/// [`invert_identified_rho_hessian_off_railed`].
pub(crate) fn compute_smoothing_correction(
    reml_state: &RemlState<'_>,
    final_rho: &Array1<f64>,
    lambdas: &Array1<f64>,
    final_fit: &pirls::PirlsResult,
    outer_gradient: &Array1<f64>,
    railed: &[usize],
    outer_hessian: Option<&Array2<f64>>,
    caller_measured_hessian_error: &[gam_linalg::curvature_resolution::MeasuredHessianError],
) -> SmoothingCorrectionComputation {
    use gam_linalg::faer_ndarray::FaerCholesky;

    let n_rho = final_rho.len();
    if n_rho == 0 {
        return SmoothingCorrectionComputation {
            factor: None,
            rho_covariance: None,
            active_rank: None,
            status: SmoothingCorrectionStatus::NotApplicableNoSmoothingParameters,
        };
    }

    let n_coeffs_trans = final_fit.beta_transformed.len();
    // The mode response `dβ̂/dρ_k = −H⁻¹λ_k S̃_k β̂` and the penalty map's
    // invariance read the penalties `H` carries, `S̃_k = Π S_k Π` (#2454, #2901).
    // On `y ~ s(x) + s(x, g, bs='fs')` (seed 0) the raw roots moved the fs
    // coordinate's response by 1.79e4 against a response of 1.09e-6.
    let applied_penalties = match final_fit.reparam_result.applied_penalties() {
        Ok(penalties) => penalties,
        Err(error) => {
            return SmoothingCorrectionComputation {
                factor: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::PenaltyStructure {
                        error: error.to_string(),
                    },
                ),
            };
        }
    };
    let ct = &applied_penalties;
    if lambdas.len() != n_rho || ct.len() != n_rho {
        return SmoothingCorrectionComputation {
            factor: None,
            rho_covariance: None,
            active_rank: None,
            status: SmoothingCorrectionStatus::Unavailable(
                SmoothingCorrectionUnavailable::PenaltyDimension {
                    rho: n_rho,
                    lambdas: lambdas.len(),
                    canonical_penalties: ct.len(),
                },
            ),
        };
    }
    // #2676: the SAME object supplies the certified nullity COUNT and the
    // subspace itself. They used to be split — the count was kept, the
    // eigenvectors thrown away — which is why the gate could only ask "how
    // many directions must be null?" and never "is THIS direction one of
    // them?". Nothing else about the count changes: the Gram is the
    // `tr(S_i S_j)` one it always was.
    //
    // The invariance is read in the ORIGINAL frame, from the block-local
    // `S̃_k = Π S_k Π` the criterion applies there. `null(w ↦ Σ w_k A_k)` and the
    // Gram `⟨A_i, A_j⟩` are invariant under the orthogonal congruence by `Qs`
    // (`S_k ↦ QsᵀS_kQs`), so this is the same subspace; but the
    // rotated penalties are dense, and their double-double Gram costs
    // `O(K² p²)` against `O(Σ_overlapping block²)`. On forty coordinates at
    // `p = 221` that was two thirds of the post-fit correction.
    let original_applied = match crate::estimate::reml::applied_canonical_penalties_for(
        &final_fit.reparam_result,
        &reml_state.canonical_penalties,
    ) {
        Ok(penalties) => penalties,
        Err(error) => {
            return SmoothingCorrectionComputation {
                factor: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::PenaltyStructure {
                        error: error.to_string(),
                    },
                ),
            };
        }
    };
    if original_applied.len() != n_rho {
        return SmoothingCorrectionComputation {
            factor: None,
            rho_covariance: None,
            active_rank: None,
            status: SmoothingCorrectionStatus::Unavailable(
                SmoothingCorrectionUnavailable::PenaltyDimension {
                    rho: n_rho,
                    lambdas: lambdas.len(),
                    canonical_penalties: original_applied.len(),
                },
            ),
        };
    }
    let invariance =
        match crate::penalty_invariance::PenaltyMapInvariance::from_canonical_penalties(
            &original_applied,
            reml_state.p,
        ) {
            Ok(invariance) => invariance,
            Err(error) => {
                return SmoothingCorrectionComputation {
                    factor: None,
                    rho_covariance: None,
                    active_rank: None,
                    status: SmoothingCorrectionStatus::Unavailable(
                        SmoothingCorrectionUnavailable::PenaltyStructure { error },
                    ),
                };
            }
        };
    let structural_nullity = invariance.dimension();
    let lifted_invariance = invariance.theta_directions(lambdas, n_rho, 0);

    // Step 1: Compute the Jacobian J = d(beta)/d(rho) in transformed space.
    //
    // Exact implicit-function identity at the inner optimum:
    //   dβ̂/dρ_k = -H^{-1}(S_k^ρ β̂),   S_k^ρ = λ_k S_k,
    //   λ_k = exp(ρ_k).
    //
    // In transformed coordinates with root penalties S_k = R_kᵀR_k:
    //   S_k β̂ = R_kᵀ(R_k β̂),
    // so each Jacobian column is one linear solve with H.

    // Use the same objective-consistent inner Hessian surface used by REML:
    // - non-Firth: H = X'W_HX + S (+ stabilization if present)
    // - Firth logit: H_total = H - d²Phi/dβ²
    // Conclusion:
    //   J[:,k] = dβ̂/dρ_k must use the Jacobian of the actual stationarity
    //   system G*(β,ρ)=0, i.e. H_total for Firth-adjusted fits. Using only
    //   X'W_HX+S here would be inconsistent with the fitted objective and would
    //   misstate smoothing-parameter uncertainty propagation.
    let h_trans = match reml_state.objective_innerhessian(final_rho) {
        Ok(hessian) => hessian,
        Err(error) => {
            return SmoothingCorrectionComputation {
                factor: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::ObjectiveInnerHessian {
                        error: error.to_string(),
                    },
                ),
            };
        }
    };

    // The IFT solve below feeds length-`n_coeffs_trans` right-hand sides into
    // the Cholesky factor of `h_trans`, and faer asserts `rhs.len() == factor.n()`.
    // A Hessian that does not match the coefficient dimension (e.g. a degenerate
    // 0×0 placeholder from a geometry backend that failed to materialize a real
    // dense inner Hessian) would otherwise abort the whole fit inside the solve.
    if h_trans.nrows() != n_coeffs_trans || h_trans.ncols() != n_coeffs_trans {
        log::debug!(
            "smoothing-correction inner Hessian shape {}x{} does not match coefficient dimension {}; skipping.",
            h_trans.nrows(),
            h_trans.ncols(),
            n_coeffs_trans
        );
        return SmoothingCorrectionComputation {
            factor: None,
            rho_covariance: None,
            active_rank: None,
            status: SmoothingCorrectionStatus::Unavailable(
                SmoothingCorrectionUnavailable::InnerHessianDimension {
                    rows: h_trans.nrows(),
                    cols: h_trans.ncols(),
                    coefficients: n_coeffs_trans,
                },
            ),
        };
    }

    // Factor the Hessian for solving. When its strict Cholesky refuses, the IFT
    // solve is taken on the identified subspace post-fit inference uses: each
    // column of G_ρ is λ_k·S_k β, which lies in range(S_k) and so is
    // orthogonal to null(H), where the min-norm solve is exact (#2901 V22).
    let h_chol;
    let identified_inverse;
    let sensitivity = match h_trans.cholesky(faer::Side::Lower) {
        Ok(factor) => {
            h_chol = factor;
            crate::sensitivity::FitSensitivity::from_faer_cholesky(&h_chol, n_coeffs_trans)
        }
        Err(cholesky_error) => {
            match super::identified_hessian::IdentifiedHessianInverse::from_dense(
                &h_trans,
                final_fit.reparam_result.e_transformed.nrows(),
            ) {
                Ok(inverse) => {
                    identified_inverse = inverse;
                    identified_inverse.sensitivity()
                }
                Err(identified_error) => {
                    log::debug!(
                        "smoothing-correction inner Hessian refused its strict Cholesky \
                         ({cholesky_error:?}) and its identified subspace ({identified_error}); \
                         skipping."
                    );
                    return SmoothingCorrectionComputation {
                        factor: None,
                        rho_covariance: None,
                        active_rank: None,
                        status: SmoothingCorrectionStatus::Unavailable(
                            SmoothingCorrectionUnavailable::InnerHessianNotPositiveDefinite,
                        ),
                    };
                }
            }
        }
    };

    let beta_trans = final_fit.beta_transformed.as_ref();
    // Build the stationarity-gradient derivative matrix G_ρ where column k is
    // ∂g(β,ρ)/∂ρ_k = λ_k S_k β, then delegate the IFT solve
    // dβ/dρ = -H⁻¹G_ρ to the canonical evidence helper. This keeps the
    // coefficient-space prediction correction and the joint-evidence
    // Arrow-Schur path on the same hand-derived IFT identity.
    let mut dg_drho_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    // Per-ρ_k support: the coefficient range its stationarity-gradient
    // derivative ∂g/∂ρ_k is nonzero on. Each column is block-local (only the
    // k-th penalty block), so this is exactly cp.col_range; structurally
    // inactive columns keep an empty support and the cone-of-influence solve
    // skips them entirely (their sensitivity is identically zero). See #779.
    let mut col_supports: Vec<std::ops::Range<usize>> = vec![0..0; n_rho];
    for k in 0..n_rho {
        let cp = &ct[k];
        if cp.rank() == 0 {
            continue;
        }
        // S_k β — block-local: R^T (R β[block]), embedded into p-vector.
        let r = &cp.col_range;
        col_supports[k] = r.start..r.end;
        let beta_block = beta_trans.slice(s![r.start..r.end]);
        let r_beta = cp.root.dot(&beta_block);
        for a in 0..cp.block_dim() {
            dg_drho_trans[[r.start + a, k]] = lambdas[k]
                * (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum::<f64>();
        }
    }
    // Lazy/local cone-of-influence propagation (#779): confine each column's
    // sensitivity to the coupling component of `h_trans` containing the moved
    // penalty block, and skip structurally inactive columns. Exact on a
    // block-decoupled Hessian (entries outside the cone are identically zero)
    // and identical to the full joint solve on a fully coupled Hessian.
    let jacobian_trans =
        match sensitivity.mode_response_coned(h_trans.view(), dg_drho_trans.view(), &col_supports) {
            Some(jacobian) => jacobian,
            None => {
                log::debug!(
                    "IFT beta-rho sensitivity solve failed for smoothing correction; skipping."
                );
                return SmoothingCorrectionComputation {
                    factor: None,
                    rho_covariance: None,
                    active_rank: None,
                    status: SmoothingCorrectionStatus::Unavailable(
                        SmoothingCorrectionUnavailable::SensitivitySolve,
                    ),
                };
            }
        };

    // Step 2: Build V_rho by inverting the LAML Hessian in rho-space.
    // The authoritative inner-strategy path chooses the rho-space Hessian
    // evaluation policy here. Unified may still perform local numerical
    // salvage inside the exact branch, but the branch choice itself no longer
    // lives inline at the call site.
    let mut hessian_rho = match reml_state.compute_lamlhessian_consistent(final_rho) {
        Ok(h) => h,
        Err(err) => {
            log::debug!(
                "LAML Hessian unavailable ({}); skipping smoothing correction.",
                err
            );
            let reason = SmoothingCorrectionUnavailable::OuterHessian {
                error: err.to_string(),
            };
            return SmoothingCorrectionComputation {
                factor: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(reason),
            };
        }
    };

    // The skew part this symmetrization is about to discard is EXACTLY zero in
    // exact arithmetic (a Hessian is symmetric, Clairaut), and `H[i,j]` and
    // `H[j,i]` are separate accumulations of the same mixed partial through the
    // same implicit-function assembly. So whatever survives is that assembly's
    // error, measured in situ on this very matrix -- a certified `‖δH‖₂`
    // component by Weyl, obtained for free at a site that was already averaging
    // the transpose and throwing the difference away (#2748).
    let symmetrization_defect = gam_linalg::matrix::symmetrization_defect_2norm(&hessian_rho);
    gam_linalg::matrix::symmetrize_in_place(&mut hessian_rho);

    // The THIRD exactly-zero identity available here, and the one that measures
    // the pair rather than either half (#2748).
    //
    // The gate below compares an eigenvalue of `hessian_rho` against a floor
    // built from `outer_gradient`, but those two are not one evaluation:
    // `outer_gradient` is the residual gradient the OUTER SEARCH carried out of
    // its own last step, and `hessian_rho` was just assembled here from a fresh
    // evaluation bundle at the same rho. Re-evaluating the gradient against the
    // same objective at the same point must return the same vector -- it is the
    // same function of the same argument -- so any difference is the
    // inconsistency between the two evaluations, which is exactly the currency
    // a "sigma versus a gradient-built floor" comparison spends.
    //
    // `max_k |dg_k|` bounds the per-direction term `sum_k |dg_k| v_k^2` the
    // floor would have moved by, so it enters as one more measured component of
    // the resolution rather than as a change to the floor itself.
    //
    // Absent when the gradient cannot be re-evaluated, or when the outer
    // gradient was not supplied at all: an absent measurement stays absent.
    // The FOURTH exactly-zero identity available here, and the one that measures
    // the pair of ASSEMBLIES rather than either half (#2748).
    //
    // The outer certificate judged a ρ-Hessian at this ρ, from its own
    // evaluation bundle, and accepted the point. `hessian_rho` was just
    // assembled HERE, from a fresh bundle at the same ρ. Two assemblies of one
    // mixed-partial object at one argument must return the same matrix — it is
    // the same function of the same argument, exactly as for the gradient
    // above — so whatever separates them is the inconsistency between the two
    // assemblies. And that inconsistency is precisely the currency this site
    // then spends: the gate below re-judges a direction the certificate has
    // already cleared, so its resolution has to include how far the two
    // assemblies of the judged matrix can be apart.
    //
    // Without it the gate judges against an eigensolver's backward error, which
    // bounds the DECOMPOSITION and says nothing about the ASSEMBLY. Measured on
    // #2748's `geo_disease` k=12 cell: the certificate reports `hessian_psd=NO`
    // and returns `stationary` (its gradient-residue floor cleared the negative
    // direction), and this site refuses the same point at
    // `σ = −2.571e-4` against a bar of `2.571e-4` — an excess of order `1e-8`
    // in a resolution ledger whose every other component measured exactly zero.
    //
    // The spectral norm is exact rather than bounded: both matrices are
    // symmetric and `n_rho` is small, so the difference's largest-magnitude
    // eigenvalue is a decomposition, not an estimate. An absent or
    // differently-shaped outer Hessian yields an absent measurement, never a
    // zero — the same discipline the gradient identity above follows.
    let assembly_reevaluation_defect = outer_hessian
        .filter(|outer| outer.nrows() == n_rho && outer.ncols() == n_rho)
        .and_then(|outer| {
            use faer::Side;
            use gam_linalg::faer_ndarray::FaerEigh;
            let mut difference = Array2::<f64>::zeros((n_rho, n_rho));
            for i in 0..n_rho {
                for j in 0..n_rho {
                    difference[[i, j]] = 0.5
                        * ((hessian_rho[[i, j]] - outer[[i, j]])
                            + (hessian_rho[[j, i]] - outer[[j, i]]));
                }
            }
            if difference.iter().any(|value| !value.is_finite()) {
                return None;
            }
            difference
                .eigh(Side::Lower)
                .ok()
                .map(|(values, _)| values.iter().fold(0.0_f64, |worst, v| worst.max(v.abs())))
        })
        .filter(|value| value.is_finite());
    if let Some(defect) = assembly_reevaluation_defect {
        log::debug!(
            "[RHO-HESSIAN] outer-vs-correction re-assembly defect ‖H_correction − H_outer‖₂ = {defect:.6e} \
             (n_rho={n_rho}); this is a measured component of the curvature resolution the \
             definiteness gate below judges against (#2748)"
        );
    }
    let gradient_reevaluation_defect = (!outer_gradient.is_empty())
        .then(|| reml_state.compute_gradient(final_rho).ok())
        .flatten()
        .filter(|fresh| fresh.len() == outer_gradient.len())
        .map(|fresh| {
            (0..fresh.len()).fold(0.0_f64, |worst, k| {
                worst.max((fresh[k] - outer_gradient[k]).abs())
            })
        })
        .filter(|value| value.is_finite());

    // Step 3: invert the exact, unperturbed Hessian on its explicitly
    // identified spectral subspace. A diagonal ridge would change V_rho and
    // therefore the covariance estimand while being invisible in the result.
    let inverted = match invert_identified_rho_hessian_off_railed(
        &hessian_rho,
        structural_nullity,
        outer_gradient,
        lifted_invariance.as_ref(),
        railed,
        &{
            let mut components =
                vec![gam_linalg::curvature_resolution::MeasuredHessianError::new(
                    "rho-Hessian symmetrization defect |(H - H')/2|_2",
                    symmetrization_defect,
                )];
            if let Some(defect) = assembly_reevaluation_defect {
                components.push(
                    gam_linalg::curvature_resolution::MeasuredHessianError::new(
                        "outer-vs-correction rho-Hessian re-assembly defect |H_correction - H_outer|_2",
                        defect,
                    ),
                );
            }
            if let Some(defect) = gradient_reevaluation_defect {
                components.push(
                    gam_linalg::curvature_resolution::MeasuredHessianError::new(
                        "outer-gradient re-evaluation defect max_k |g_fresh - g_outer|",
                        defect,
                    ),
                );
            }
            // #2748: whatever standard the OUTER certificate judged this same
            // matrix at this same point against, so this gate does not re-decide
            // a direction the certificate already cleared at a coarser
            // resolution.
            components.extend_from_slice(caller_measured_hessian_error);
            components
        },
    ) {
        Ok(inverse) => inverse,
        Err(error) => {
            log::debug!("Exact LAML rho-Hessian inversion failed: {error}");
            dump_indefinite_rho_hessian_diagnostic(
                &hessian_rho,
                final_rho,
                ct,
                None,
                outer_gradient,
            );
            return SmoothingCorrectionComputation {
                factor: None,
                rho_covariance: None,
                active_rank: None,
                status: SmoothingCorrectionStatus::Unavailable(
                    SmoothingCorrectionUnavailable::OuterHessianInverse { error },
                ),
            };
        }
    };

    let n_rho_total = hessian_rho.nrows();
    if inverted.active_rank == 0 {
        // Every direction is a structural zero of the penalty map, a railed
        // axis (`J e_k = 0`), or carries no resolvable curvature, so
        // J·V_ρ·Jᵀ is zero.
        log::debug!(
            "LAML rho Hessian has no identified directions (active_rank=0/{}, structural_zero={}, railed={}, curvature_resolution={:.3e}); smoothing correction is exactly zero.",
            n_rho_total,
            inverted.structural_zero,
            inverted.railed,
            inverted.curvature_resolution,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            ct,
            Some(&inverted),
            outer_gradient,
        );
        // Publish that zero, not an absence: `V_c = V_b` exactly at identified rank 0, the same
        // identity the custom-family lane mints when every outer coordinate is railed (#2677).
        let p_original = final_fit.reparam_result.qs.nrows();
        return SmoothingCorrectionComputation {
            factor: Some(Array2::<f64>::zeros((p_original, 0))),
            rho_covariance: Some(inverted.inverse),
            active_rank: Some(0),
            status: SmoothingCorrectionStatus::ZeroNoIdentifiedOuterDirections,
        };
    }

    if inverted.active_rank < n_rho_total {
        log::debug!(
            "LAML rho Hessian is not fully identified (active_rank={}/{}, structural_zero={}, below_gradient_floor={}, unresolvable_curvature={}, railed={}, curvature_resolution={:.3e}); using its certified structural pseudoinverse. `below_gradient_floor` counts saturation nulls: directions whose curvature is under the outer loop\'s own residual gradient, so they carry no resolvable rho-variance (#2428).",
            inverted.active_rank,
            n_rho_total,
            inverted.structural_zero,
            inverted.below_gradient_floor,
            inverted.unresolvable_curvature,
            inverted.railed,
            inverted.curvature_resolution,
        );
        dump_indefinite_rho_hessian_diagnostic(
            &hessian_rho,
            final_rho,
            ct,
            Some(&inverted),
            outer_gradient,
        );
    }

    let used_structural_pseudoinverse = inverted.used_structural_pseudoinverse;
    let active_rank_used = inverted.active_rank;
    if used_structural_pseudoinverse {
        log::trace!(
            "Applied rank-deficient pseudo-inverse on identified rho-Hessian subspace before smoothing correction."
        );
    }

    // Step 4: Compute V_corr through the identified square-root factor of V_rho.
    //
    // This is the first-order smoothing-parameter uncertainty inflation:
    //   Var(β̂) ≈ Var(β̂|ρ̂) + (dβ̂/dρ) Var(ρ̂) (dβ̂/dρ)ᵀ.
    //
    // Here:
    //   J = dβ̂/dρ,  J[:,k] = -H^{-1}(A_k β̂),
    //   V_ρ = (∇²_{ρρ}V)^{-1} evaluated at the final ρ.
    let qs = &final_fit.reparam_result.qs;
    let factor_orig = smoothing_correction_factor(
        &jacobian_trans,
        qs,
        &inverted.eigenvalues,
        &inverted.eigenvectors,
        &inverted.classifications,
    );
    let rho_covariance = inverted.inverse;

    // Validate the result: every diagonal entry `‖b_i‖²` of `C = B Bᵀ` finite,
    // which bounds every off-diagonal entry `|b_iᵀ b_j| ≤ ‖b_i‖‖b_j‖` as well.
    if !factor_orig.rows().into_iter().all(|row| row.dot(&row).is_finite()) {
        log::debug!("Non-finite values in smoothing correction matrix; skipping.");
        return SmoothingCorrectionComputation {
            factor: None,
            rho_covariance: Some(rho_covariance),
            active_rank: Some(active_rank_used),
            status: SmoothingCorrectionStatus::Unavailable(
                SmoothingCorrectionUnavailable::NonFiniteCorrection,
            ),
        };
    }
    SmoothingCorrectionComputation {
        factor: Some(factor_orig),
        rho_covariance: Some(rho_covariance),
        active_rank: Some(active_rank_used),
        status: SmoothingCorrectionStatus::Computed,
    }
}

#[cfg(test)]
mod smoothing_correction_gram_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn zero_mode_response_produces_bit_exact_zero_covariance_2490() {
        let jacobian = Array2::<f64>::zeros((3, 2));
        let qs = array![
            [1.0_f64, 0.25, -0.5],
            [0.0, 0.75, 0.125],
            [0.5, -0.25, 1.0],
        ];
        let eigenvalues = array![0.5_f64, 2.0];
        let eigenvectors = Array2::<f64>::eye(2);
        let classifications = vec![
            EigenClassification::Active,
            EigenClassification::Active,
        ];

        let correction = smoothing_correction_gram(&smoothing_correction_factor(
            &jacobian,
            &qs,
            &eigenvalues,
            &eigenvectors,
            &classifications,
        ));

        assert!(
            correction.iter().all(|&value| value == 0.0),
            "a zero coefficient response to rho must produce an exact zero covariance, got {correction:?}"
        );
        assert_eq!(
            gam_problem::se_from_covariance(&correction).unwrap(),
            Array1::<f64>::zeros(3),
        );
    }

    #[test]
    fn factor_gram_matches_dense_congruence_with_nonnegative_diagonal_2490() {
        let jacobian = array![
            [1.0_f64, -2.0],
            [3.0, 4.0],
            [-5.0, 6.0],
        ];
        let qs = array![
            [1.0_f64, 0.0, 0.0],
            [0.0, 0.8, -0.6],
            [0.0, 0.6, 0.8],
        ];
        let inv_sqrt_two = 2.0_f64.sqrt().recip();
        let eigenvectors = array![
            [inv_sqrt_two, -inv_sqrt_two],
            [inv_sqrt_two, inv_sqrt_two],
        ];
        let eigenvalues = array![0.25_f64, 4.0];
        let classifications = vec![
            EigenClassification::Active,
            EigenClassification::Active,
        ];

        let correction = smoothing_correction_gram(&smoothing_correction_factor(
            &jacobian,
            &qs,
            &eigenvalues,
            &eigenvectors,
            &classifications,
        ));
        let inverse = eigenvectors
            .dot(&Array2::from_diag(&eigenvalues.mapv(f64::recip)))
            .dot(&eigenvectors.t());
        let expected = qs
            .dot(&jacobian)
            .dot(&inverse)
            .dot(&jacobian.t())
            .dot(&qs.t());

        let covariance_scale = expected
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let arithmetic_bound =
            64.0 * correction.nrows().max(1) as f64 * f64::EPSILON * covariance_scale;
        for (&actual, &reference) in correction.iter().zip(expected.iter()) {
            assert!((actual - reference).abs() <= arithmetic_bound);
        }
        assert!(correction.diag().iter().all(|&value| value >= 0.0));
    }
}
