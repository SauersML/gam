use super::*;
use crate::estimate::evaluation::{
    materialize_link_outer_hessian, sas_effective_epsilon, sas_effective_epsilon_second,
    sas_log_delta_edge_barriercostgrad, sas_log_delta_edge_barriercostgradhess,
    sas_log_deltaridgeweight,
};
use crate::estimate::edf_accounting::penalized_edf_bundle_within_bands;
use crate::estimate::penalty::scaled_covariance;
use crate::estimate::prefit::{
    arm_jeffreys_on_prefit_binomial_separation, reject_prefit_unidentifiable_unpenalized_space,
    reject_prefit_unpenalized_rank_deficiency,
};
use gam_linalg::matrix::FactorizedSystem;
use gam_linalg::utils::KahanSum;
use gam_problem::dispersion_cov::se_from_covariance;
use gam_problem::OrderedRhoBounds;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

fn certify_factorized_inference_solve(
    hessian: &gam_linalg::matrix::SymmetricMatrix,
    rhs: &Array2<f64>,
    solution: &Array2<f64>,
    label: &str,
) -> Result<(), EstimationError> {
    let residual = hessian.dot_matrix(solution) - rhs;
    gam_linalg::utils::certify_linear_system_residual(
        hessian.nrows(),
        hessian.max_abs_entry(),
        rhs,
        solution,
        &residual,
        label,
    )
    .map_err(|error| {
        EstimationError::RemlOptimizationFailed(format!(
            "exact factorized inference solve did not certify: {error}"
        ))
    })?;
    Ok(())
}

fn certify_factorized_inference_vector_solve(
    hessian: &gam_linalg::matrix::SymmetricMatrix,
    rhs: &Array1<f64>,
    solution: &Array1<f64>,
    label: &str,
) -> Result<(), EstimationError> {
    let rhs_matrix = rhs.view().insert_axis(Axis(1)).to_owned();
    let solution_matrix = solution.view().insert_axis(Axis(1)).to_owned();
    certify_factorized_inference_solve(hessian, &rhs_matrix, &solution_matrix, label)
}

/// The accepted inference factor of the transformed penalized Hessian: its
/// strict Cholesky, or, when that refuses, its inverse on the identified
/// subspace PIRLS solved it on and the criterion scored (#2901 V22).
enum InferenceHessianFactor {
    Strict(Box<dyn FactorizedSystem>),
    Identified(super::identified_hessian::IdentifiedHessianInverse),
}

impl InferenceHessianFactor {
    /// Solve `H·X = B` and certify it: against `B` for the strict factor, and
    /// against its identified part `U·Uᵀ·B` for the identified inverse, whose
    /// solution is the min-norm one.
    fn certified_solve(
        &self,
        hessian: &gam_linalg::matrix::SymmetricMatrix,
        rhs: &Array2<f64>,
        label: &str,
    ) -> Result<Array2<f64>, EstimationError> {
        match self {
            Self::Strict(factor) => {
                let solution = factor.solvemulti(rhs).map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "{label}: exact factorized solve failed: {reason}"
                    ))
                })?;
                certify_factorized_inference_solve(hessian, rhs, &solution, label)?;
                Ok(solution)
            }
            Self::Identified(inverse) => {
                let solution = inverse.apply(rhs);
                certify_factorized_inference_solve(
                    hessian,
                    &inverse.project(rhs),
                    &solution,
                    label,
                )?;
                Ok(solution)
            }
        }
    }

    fn certified_vector_solve(
        &self,
        hessian: &gam_linalg::matrix::SymmetricMatrix,
        rhs: &Array1<f64>,
        label: &str,
    ) -> Result<Array1<f64>, EstimationError> {
        match self {
            Self::Strict(factor) => {
                let solution = factor.solve(rhs).map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "{label}: exact factorized solve failed: {reason}"
                    ))
                })?;
                certify_factorized_inference_vector_solve(hessian, rhs, &solution, label)?;
                Ok(solution)
            }
            Self::Identified(_) => {
                let rhs_matrix = rhs.view().insert_axis(Axis(1)).to_owned();
                let solution = self.certified_solve(hessian, &rhs_matrix, label)?;
                Ok(solution.column(0).to_owned())
            }
        }
    }

    /// `(coefficients, penalty nullity)` for the EDF bundle. An identified
    /// inverse counts only its identified directions, and every unidentified
    /// direction lies in the penalty's null space, so it leaves that nullity too.
    fn edf_dimensions(&self, coefficients: usize, penalty_nullity: f64) -> (usize, f64) {
        match self {
            Self::Strict(_) => (coefficients, penalty_nullity),
            Self::Identified(inverse) => {
                let unidentified = coefficients.saturating_sub(inverse.rank());
                (
                    inverse.rank(),
                    (penalty_nullity - unidentified as f64).max(0.0),
                )
            }
        }
    }

    fn identified(&self) -> Option<&super::identified_hessian::IdentifiedHessianInverse> {
        match self {
            Self::Strict(_) => None,
            Self::Identified(inverse) => Some(inverse),
        }
    }

    /// The right-hand side a solve is certified against: `B` for the strict
    /// factor, `U·Uᵀ·B` for the identified inverse.
    fn solved_rhs(&self, rhs: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Strict(_) => rhs.clone(),
            Self::Identified(inverse) => inverse.project(rhs),
        }
    }

    /// Hager–Higham estimate of `‖H⁻¹‖₁` from this factor's own solves, for the
    /// EDF trace bands (#2901). It is a lower bound, so the bands' second-order
    /// term is an estimate.
    fn inverse_one_norm_estimate(&self, dimension: usize) -> Result<f64, EstimationError> {
        let solve = |values: &mut [f64]| -> Result<(), EstimationError> {
            let rhs = Array2::from_shape_fn((dimension, 1), |(row, _)| values[row]);
            let solution = match self {
                Self::Strict(factor) => factor.solvemulti(&rhs).map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "EDF trace band: exact factorized solve failed: {reason}"
                    ))
                })?,
                Self::Identified(inverse) => inverse.apply(&rhs),
            };
            for (slot, value) in values.iter_mut().zip(solution.iter()) {
                *slot = *value;
            }
            Ok(())
        };
        gam_linalg::condition::estimate_inverse_one_norm(dimension, solve, solve)
    }
}

/// The original-basis penalized Hessian's factor for the dense inference
/// bundle: a certified strict Cholesky, or, when the transformed factor was
/// identified, the same identified inverse rotated through `Qs`.
enum OriginalBasisHessianFactor<'a> {
    Strict(gam_linalg::utils::CertifiedSpdFactor<'a>),
    Identified {
        hessian: &'a Array2<f64>,
        inverse: super::identified_hessian::IdentifiedHessianInverse,
        label: &'static str,
    },
}

impl<'a> OriginalBasisHessianFactor<'a> {
    fn new(
        hessian: &'a Array2<f64>,
        transformed: Option<&InferenceHessianFactor>,
        qs: &Array2<f64>,
        label: &'static str,
    ) -> Result<Self, gam_linalg::utils::CertifiedSymmetricSolveError> {
        match transformed.and_then(InferenceHessianFactor::identified) {
            Some(inverse) => Ok(Self::Identified {
                hessian,
                inverse: inverse.rotated(qs),
                label,
            }),
            None => gam_linalg::utils::certified_spd_factorize(hessian, label).map(Self::Strict),
        }
    }

    fn solve_matrix(
        &self,
        rhs: &Array2<f64>,
    ) -> Result<Array2<f64>, gam_linalg::utils::CertifiedSymmetricSolveError> {
        match self {
            Self::Strict(factor) => factor.solve_matrix(rhs).map(|solved| solved.0),
            Self::Identified {
                hessian,
                inverse,
                label,
            } => inverse.certified_solve(hessian, rhs, label),
        }
    }

    fn inverse(&self) -> Result<Array2<f64>, gam_linalg::utils::CertifiedSymmetricSolveError> {
        match self {
            Self::Strict(factor) => factor
                .inverse()
                .map(gam_linalg::utils::CertifiedSpdInverse::into_inverse),
            Self::Identified {
                hessian,
                inverse,
                label,
            } => inverse.certified_inverse(hessian, label),
        }
    }

    /// The right-hand side a solve is certified against: `B` for the strict
    /// factor, `U·Uᵀ·B` for the identified inverse.
    fn solved_rhs(&self, rhs: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Strict(_) => rhs.clone(),
            Self::Identified { inverse, .. } => inverse.project(rhs),
        }
    }

    /// Hager–Higham estimate of `‖H⁻¹‖₁` from this factor's own solves, for the
    /// EDF reconciliation's trace bands (#2901). It is a lower bound, so the bands'
    /// second-order term is an estimate.
    fn inverse_one_norm_estimate(
        &self,
        dimension: usize,
    ) -> Result<f64, gam_linalg::utils::CertifiedSymmetricSolveError> {
        let solve = |values: &mut [f64]| -> Result<(), gam_linalg::utils::CertifiedSymmetricSolveError> {
            let rhs = Array2::from_shape_fn((dimension, 1), |(row, _)| values[row]);
            let solution = self.solve_matrix(&rhs)?;
            for (slot, value) in values.iter_mut().zip(solution.iter()) {
                *slot = *value;
            }
            Ok(())
        };
        gam_linalg::condition::estimate_inverse_one_norm(dimension, solve, solve)
    }
}

/// Scale-free stationarity residual for the Negative-Binomial conditional ML
/// problem in `tau = log(theta)`. The score is `d log L / d theta`, so the
/// minimization gradient in `tau` is `-theta * score`. A score inside its own
/// rounding band is zero to the arithmetic's resolution; that is the residual of
/// a root and of the Poisson limit alike, and there is no profiling box whose
/// faces would need a multiplier. Otherwise the residual is normalized by the
/// observed log-theta curvature, so it is the Newton displacement still required
/// for theta stationarity rather than an arbitrary percent drift.
fn negbin_theta_stationarity_residual(theta: f64, profile: &pirls::NegbinThetaScore) -> f64 {
    let pirls::NegbinThetaScore { score, info, band } = *profile;
    if !theta.is_finite()
        || theta <= 0.0
        || !score.is_finite()
        || !info.is_finite()
        || !band.is_finite()
    {
        return f64::INFINITY;
    }
    if score.abs() <= band {
        return 0.0;
    }
    let log_theta_gradient = -theta * score;
    let log_theta_curvature = theta * theta * info - theta * score;
    if !log_theta_curvature.is_finite() || log_theta_curvature <= 0.0 {
        return f64::INFINITY;
    }
    // Both numerator and curvature scale linearly with case weights, so their
    // ratio is invariant to objective rescaling. An absolute denominator floor
    // would instead certify flat theta coordinates whenever raw weights happen
    // to be small.
    (log_theta_gradient / log_theta_curvature).abs()
}

/// Whether the point a fit is about to ship IS the point the outer certificate
/// was minted at, compared over EVERY optimized coordinate.
///
/// The outer optimizer for a flexible-link fit searches the joint coordinate
/// `theta = [rho (k entries), link-shape coordinates]`, so `certified` is that
/// whole joint vector while the shipped `rho` is only its leading block — the
/// link coordinates are shipped separately, inside the link state. Comparing
/// the two vectors directly therefore compares a `K`-vector against a
/// `K + link_dim` one and can never agree (#2727): it refused fits that had
/// converged, whose certificate certified, at `|Pg|` as low as `1.446e-6`.
///
/// Comparing only the rho prefix would turn that over-strict gate into an
/// under-strict one — it would stop checking the link coordinates entirely,
/// and those are exactly the coordinates the flexible-link lane exists to
/// optimize. So the shipped point is reassembled in the optimizer's OWN
/// coordinate system and compared whole. It has to be the raw `theta`
/// coordinates rather than the shipped link state, because the state stores
/// values that have been through the smooth-bound maps
/// (`sas_effective_epsilon`) while the certificate holds the pre-image.
///
/// The comparison stays BITWISE for the reason the rho-only one did: point
/// identity is decided exactly by bit equality, and re-judging a gradient here
/// would refuse honest noise-band certificates with coin-flip probability.
///
/// Reassembling one vector only to compare it against the vector it was sliced
/// from looks circular, and is not: the check is TEMPORAL. `final_rho`,
/// `final_link_coords` and `outer_result` are `let mut` bindings reassigned
/// inside the alternation `loop`, so this asks whether the point being shipped
/// at the END is still the certificate being held at the END. Seeds and
/// nuisance refinements may initialize work between those two moments; they
/// must never promote a different point under the old certificate. Comparing
/// only the rho block would leave the link coordinates free to move across
/// exactly that window.
fn shipped_joint_point_is_certified(
    rho: &Array1<f64>,
    link_coords: &Array1<f64>,
    certified: &Array1<f64>,
) -> bool {
    if certified.len() != rho.len() + link_coords.len() {
        return false;
    }
    rho.iter()
        .chain(link_coords.iter())
        .zip(certified.iter())
        .all(|(shipped, certified)| shipped.to_bits() == certified.to_bits())
}

#[derive(Clone)]
struct NegbinJointCheckpoint {
    merit: f64,
    theta: f64,
    rho: Array1<f64>,
    rho_residual: f64,
    rho_bound: f64,
    theta_residual: f64,
    theta_bound: f64,
}

/// The square matrices the first-order smoothing correction's assembly holds
/// live: the dense bundle charges them inside its atomic peak, and a fit whose
/// inference stayed factorized charges the same owner on its own
/// ([`reserve_smoothing_correction_workspace`], #3283).
const FIRST_ORDER_SMOOTHING_WORKSPACES: usize = 8;

/// Reserve the complete peak live set of the optional dense inference path.
///
/// The count is assembled from named algorithmic owners rather than a
/// dimension cliff: ten square matrices can survive into/alongside the fit
/// payload, six are base factorization/GEMM workspaces, and eight belong to the
/// first-order smoothing correction. Charging the whole set atomically prevents several individually acceptable p×p allocations from
/// jointly exceeding the process-wide memory ledger.
fn reserve_dense_covariance_bundle(p: usize) -> Option<gam_runtime::resource::MemoryReservation> {
    const STORED_SQUARE_MATRICES: usize = 10;
    const BASE_FACTORIZATION_AND_GEMM_WORKSPACES: usize = 6;
    /// A constrained fit assembles its truncated covariance as a sum of Grams
    /// (`ConstrainedPosteriorCorrection::truncated_covariance_psd`, #2705 group
    /// A), which holds three `p × p` blocks live at once: the Cholesky factor of
    /// `Σ`, the projected factor `P L`, and the Gram it accumulates into. Plus
    /// the untruncated conditional covariance the marginal composition keeps so
    /// the corrected estimand can be built from `Vb`, not from `Σ_π`.
    ///
    /// Priced here rather than left to slack, because the whole point of this
    /// reservation is that several individually acceptable `p × p` allocations
    /// must not jointly exceed the ledger — and #2724 is on record for what an
    /// allocating route costs when the pricing does not follow it.
    const CONSTRAINED_TRUNCATION_WORKSPACES: usize = 4;
    const PEAK_SQUARE_MATRIX_EQUIVALENTS: usize = STORED_SQUARE_MATRICES
        + BASE_FACTORIZATION_AND_GEMM_WORKSPACES
        + FIRST_ORDER_SMOOTHING_WORKSPACES
        + CONSTRAINED_TRUNCATION_WORKSPACES;

    let policy = gam_runtime::resource::ResourcePolicy::for_problem(
        gam_runtime::resource::ProblemHints::default(),
    );
    if !policy.material_policy().allow_operator_materialization {
        return None;
    }
    match gam_runtime::resource::MemoryGovernor::global().try_reserve_dense_f64_copies(
        p,
        p,
        PEAK_SQUARE_MATRIX_EQUIVALENTS,
        "standard GAM dense covariance/influence bundle",
    ) {
        Ok(reservation) => Some(reservation),
        Err(error) => {
            log::debug!(
                "Dense covariance/influence bundle not reserved; using factorized inference: {error}"
            );
            None
        }
    }
}

/// Truncate the ρ-MARGINAL posterior covariance to the fit's feasible set,
/// in place (#2705 group A).
///
/// `covariance` arrives as `Vp = Vb + J·V_ρ·Jᵀ` built from the UNTRUNCATED
/// conditional `Vb`, and leaves as the covariance of `N(β_unc, Vp)` restricted
/// to `{β : Aβ ≥ b}`. The truncation is rebuilt AT `Vp` — its own lift
/// `G_p = Vp·Aᵀ·W_p⁻¹` and its own orthant moments at `W_p = A·Vp·Aᵀ` — because
/// a lift derived from a different covariance is not a projector for this one,
/// and subtracting it is not a truncation of anything.
///
/// A geometry whose moments were DECLINED never truncated the conditional
/// covariance either, so there is nothing here to keep consistent with and the
/// marginal is published untruncated, exactly as the conditional one is.
///
/// A MOMENT failure declines rather than propagating. The corrected covariance
/// is a refinement of an already-published conditional one, and #2601 is on
/// record for what happens when a failure to refine the uncertainty is allowed
/// to destroy a converged point estimate; the honest degradation is the typed
/// absence every consumer of `beta_covariance_corrected` already handles. A
/// STRUCTURAL failure — a geometry whose constraint width disagrees with the
/// covariance it is supposed to constrain — is still fatal, because that is a
/// wiring defect and no absence describes it.
pub(crate) fn apply_marginal_constraint_truncation(
    geometry: &crate::constrained_posterior::ConstrainedPosteriorGeometry,
    covariance: &mut Array2<f64>,
) -> Result<Result<(), String>, EstimationError> {
    if geometry.decline().is_some() {
        return Ok(Ok(()));
    }
    let p = covariance.nrows();
    if geometry.constraints.a.ncols() != p {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "constrained posterior geometry has {} constraint columns against a {p}x{p} \
             corrected covariance",
            geometry.constraints.a.ncols(),
        )));
    }
    let center = match geometry.unconstrained_center() {
        Ok(center) if center.len() == p => center,
        Ok(center) => {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "constrained posterior geometry carries a length-{} centre against a {p}x{p} \
                 corrected covariance",
                center.len(),
            )));
        }
        Err(reason) => return Ok(Err(reason)),
    };
    let marginal_correction =
        match crate::constrained_posterior::constrained_posterior_correction_from_covariance(
            covariance,
            center,
            &geometry.constraints,
        ) {
            Ok(correction) => correction,
            Err(reason) => return Ok(Err(reason)),
        };
    if let Some(correction) = marginal_correction {
        // Same positive-semidefinite assembly the conditional covariance uses:
        // the marginal is read for standard errors too, and a pinned coordinate
        // cancels there for exactly the same reason.
        match correction.truncated_covariance_psd(covariance, &geometry.constraints) {
            Ok(truncated) => *covariance = truncated,
            Err(reason) => return Ok(Err(reason)),
        }
    }
    Ok(Ok(()))
}

/// Reserve the first-order smoothing correction's workspace on a fit whose
/// inference stayed factorized (#3283).
///
/// The correction is assembled exactly as on the dense branch, whose bundle
/// charges this owner inside its atomic peak; only its result differs, the
/// `p × r` factor instead of the `p × p` Gram. A refusal is the typed absence
/// [`crate::model_types::SmoothingCorrectionAbsence::CorrectionWorkspaceRefused`].
fn reserve_smoothing_correction_workspace(
    p: usize,
) -> Result<gam_runtime::resource::MemoryReservation, String> {
    gam_runtime::resource::MemoryGovernor::global()
        .try_reserve_dense_f64_copies(
            p,
            p,
            FIRST_ORDER_SMOOTHING_WORKSPACES,
            "factorized-branch first-order smoothing correction workspace",
        )
        .map_err(|error| error.to_string())
}

/// The truncation of the ρ-marginal `Vp = Vb + B·Bᵀ` to the fit's feasible set
/// on the factorized branch (#3283), where neither covariance is formed.
///
/// It is [`apply_marginal_constraint_truncation`]'s construction: `Vp`'s own
/// lift and orthant moments at `W_p = A·Vp·Aᵀ`, built from the only block the
/// decomposition reads, `Vp·Aᵀ = Vb·Aᵀ + B·(Bᵀ·Aᵀ)`, with
/// `conditional_times_constraints = Vb·Aᵀ` solved through the Hessian factor.
/// `Ok(Ok(None))` means no retained face moves the answer; a MOMENT failure is
/// `Ok(Err(reason))`, which the caller publishes as the typed absence, and a
/// STRUCTURAL mismatch is fatal, both exactly as on the dense branch. A declined
/// geometry never truncated the conditional law, and the caller does not call
/// this for one.
fn factorized_marginal_constraint_truncation(
    geometry: &crate::constrained_posterior::ConstrainedPosteriorGeometry,
    conditional_times_constraints: &Array2<f64>,
    factor: &Array2<f64>,
) -> Result<Result<Option<crate::constrained_posterior::ConstrainedPosteriorCorrection>, String>, EstimationError>
{
    let constraints = &geometry.constraints;
    let p = factor.nrows();
    if constraints.a.ncols() != p
        || conditional_times_constraints.dim() != (p, constraints.a.nrows())
    {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "constrained posterior geometry has {}x{} constraints against a {p}-row smoothing \
             correction factor and a {:?} conditional normal block",
            constraints.a.nrows(),
            constraints.a.ncols(),
            conditional_times_constraints.dim(),
        )));
    }
    let center = match geometry.unconstrained_center() {
        Ok(center) if center.len() == p => center,
        Ok(center) => {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "constrained posterior geometry carries a length-{} centre against a {p}-row \
                 smoothing correction factor",
                center.len(),
            )));
        }
        Err(reason) => return Ok(Err(reason)),
    };
    let marginal_times_constraints =
        conditional_times_constraints + &factor.dot(&factor.t().dot(&constraints.a.t()));
    Ok(crate::constrained_posterior::constrained_posterior_correction(
        marginal_times_constraints.view(),
        center,
        constraints,
    ))
}

/// Reserve the square matrices that remain live even when inference stays
/// factorized: the two PIRLS Hessian surfaces, the fitted reparameterization,
/// its exported copy, the reusable factor, the exported original-basis
/// precision, and the transformed penalty surface retained by the fit.
fn reserve_factorized_inference_state(
    p: usize,
) -> Option<gam_runtime::resource::MemoryReservation> {
    const RETAINED_FACTOR_AND_PRECISION_MATRICES: usize = 7;
    let policy = gam_runtime::resource::ResourcePolicy::for_problem(
        gam_runtime::resource::ProblemHints::default(),
    );
    if !policy.material_policy().allow_operator_materialization {
        return None;
    }
    match gam_runtime::resource::MemoryGovernor::global().try_reserve_dense_f64_copies(
        p,
        p,
        RETAINED_FACTOR_AND_PRECISION_MATRICES,
        "standard GAM factorized inference state",
    ) {
        Ok(reservation) => Some(reservation),
        Err(error) => {
            log::debug!("Factorized inference state could not be fully reserved: {error}");
            None
        }
    }
}

/// Fit an external design, allowing heuristic λ warm-start seeds
/// for the outer smoothing search.
pub fn optimize_external_designwith_heuristic_log_lambdas<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<BlockwisePenalty>,
    heuristic_log_lambdas: Option<&[f64]>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let specs: Vec<PenaltySpec> = s_list
        .into_iter()
        .map(PenaltySpec::from_blockwise)
        .collect();
    optimize_external_designwith_heuristic_log_lambdas_andwarm_start(
        y,
        w,
        x,
        offset,
        specs,
        heuristic_log_lambdas,
        None,
        opts,
    )
}

/// The resolution the outer certificate's curvature verdict was decided at,
/// when that verdict admitted the point (#2748, #1561).
///
/// The certificate's definiteness tests add a shift before they factor: the
/// larger of the measured `‖δH‖₂` and `√ε·max(max|H_ii|, 1)`. A `psd = true`
/// verdict and a `psd = false` verdict cleared by the gradient-residue floor
/// therefore both admit every direction inside that shift. The smoothing
/// correction re-judges the same ρ-Hessian at the same point and has to judge
/// it at the standard the verdict was taken at, or the fit dies between two
/// layers that disagree only about resolution. Only a floor that cleared
/// publishes its shift: a refusal admits nothing, and unmeasured curvature has
/// no verdict to honour.
pub(crate) fn certificate_curvature_verdict_resolution(
    certificate: Option<&crate::model_types::OuterCriterionCertificate>,
) -> Option<f64> {
    certificate
        .filter(|certificate| {
            matches!(
                certificate.curvature,
                crate::rho_optimizer::CurvatureEvidence::Measured { .. }
            )
        })
        .and_then(|certificate| certificate.curvature_floor)
        .filter(|clearance| clearance.cleared)
        .map(|clearance| clearance.decided_at_resolution)
        .filter(|value| value.is_finite() && *value > 0.0)
}

/// The measured error of a ρ-Hessian whose negative direction the criterion
/// contradicted (#2612, #1561).
///
/// Standard REML certifies on stationarity. A measured negative direction the
/// gradient floor does not clear is adjudicated against the criterion: the
/// escape steps `ρ ± αv` along the judged block's most negative eigenvector,
/// from one e-fold down to the step at which `½|λ_min|α²` reaches the
/// criterion's resolution. When the objective never falls, the verdict is
/// withdrawn as `CriterionContradicted` together with its floor clearance, and
/// the point ships. Along `v` the matrix predicted a decrease the criterion does
/// not have, so the matrix is wrong there by at least `|λ_min|`. That is a
/// MEASURED lower bound on `‖δH‖₂`, the currency the smoothing correction's
/// definiteness gate spends.
///
/// `certificate_curvature_verdict_resolution` forwards nothing for a withdrawn
/// verdict, so the correction re-judged that direction on the matrix's word and
/// refused the fit the outer loop had just accepted.
/// `quality_vs_inla_binomial_smooth_probability` at 7ebbacd3d (MSI job 555236)
/// refused `σ = −1.651e-6` against a bar of `1.333e-7`, and the prostate EBM case
/// at fa0e33bb4 (CI run 34702231507) refused `σ = −1.755e-6` against `1.330e-7`.
/// In both, the only resolution components were the eigensolver's backward error
/// and three exactly-zero identities.
///
/// `hessian` is the matrix the certificate judged and `invariance` the
/// criterion's exact invariance at the same ρ. The block is taken by the owners
/// the adjudication used, off the certificate's railed face. A verdict withdrawn
/// as `CriterionUnresolvable` (#3036) publishes the same `|λ_min|`: its claim
/// predicts no decrease the criterion resolves at any step the adjudication may
/// take, so the matrix is unconfirmed along `v` by that much. Every other
/// verdict publishes nothing here.
pub(crate) fn certificate_contradicted_curvature_error(
    certificate: Option<&crate::model_types::OuterCriterionCertificate>,
    hessian: Option<&Array2<f64>>,
    invariance: Option<&Array2<f64>>,
) -> Option<f64> {
    use gam_linalg::faer_ndarray::FaerEigh;

    let certificate =
        certificate.filter(|certificate| certificate.curvature.withdrawn_by_criterion())?;
    let hessian = hessian?;
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || hessian.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let railed: Vec<usize> = certificate
        .railed_facts
        .iter()
        .map(|fact| fact.index)
        .collect();
    let deflate = invariance.filter(|basis| basis.nrows() == n && basis.ncols() > 0);
    let judged = crate::penalty_invariance::judged_subspace_basis(n, &railed, deflate)?;
    let block = crate::penalty_invariance::compress_to_judged_subspace(hessian, &judged);
    let (eigenvalues, _) = block.eigh(faer::Side::Lower).ok()?;
    let lambda_min = eigenvalues
        .iter()
        .fold(f64::INFINITY, |smallest, value| smallest.min(*value));
    (lambda_min.is_finite() && lambda_min < 0.0).then_some(-lambda_min)
}

fn reml_inner_progress_feedback(
    state: &crate::estimate::reml::RemlState<'_>,
) -> crate::rho_optimizer::InnerProgressFeedback {
    crate::rho_optimizer::InnerProgressFeedback {
        cap: Arc::clone(&state.outer_inner_cap),
        accepted_iter: Arc::new(AtomicUsize::new(0)),
        last_iters: Arc::clone(&state.last_inner_iters),
        last_converged: Arc::clone(&state.last_inner_converged),
        ift_residual: Arc::clone(&state.last_ift_prediction_residual),
        accept_rho: Arc::clone(&state.last_pirls_accept_rho),
        // The standard REML path does not consume the cold-reeval pulse
        // (#2349); give it an inert, unshared flag so the guard's writes go
        // nowhere and behavior is unchanged.
        force_cold: Arc::new(AtomicBool::new(false)),
    }
}

fn with_reml_beta_seed_hook<'state, 'data>() -> impl FnMut(
    &mut &'state mut crate::estimate::reml::RemlState<'data>,
    &Array1<f64>,
) -> Result<
    crate::rho_optimizer::SeedOutcome,
    EstimationError,
> {
    |state, beta| {
        // The REML state stores β as a starting-iterate HINT and validates
        // its width against the design (`self.p`) at store time, silently
        // dropping a mismatched or non-finite hint rather than faulting
        // (see `setwarm_start_original_beta`). A wrong-length seed is
        // therefore never an error: a row-relaxed cross-fold prefix seed
        // degrades to a ρ-only resume, exactly the desired warm-start
        // behaviour. The slot's post-call state (the supplied β if it fit,
        // else the prior state) is what the next eval warm-starts from, so
        // `Installed` is the correct contract reply.
        state.setwarm_start_original_beta(Some(beta.view()));
        Ok(crate::rho_optimizer::SeedOutcome::Installed)
    }
}

/// The weighted-mean response level an unpenalized intercept would absorb, used
/// to center the response during outer REML λ-selection (issue #1000).
///
/// For an identity-link Gaussian fit, adding a constant to the response only
/// shifts the intercept, so λ̂ and the smooth shape must be invariant to the
/// response mean. The outer score/gradient nonetheless accumulate
/// `yᵀy`-magnitude sufficient statistics, so a large response mean costs
/// precision and drifts λ̂. Returns `Some(m)` with
/// `m = Σ wᵢ (yᵢ − offsetᵢ) / Σ wᵢ` — the constant a pure offset relabeling
/// moves into the intercept — so the caller can subtract it and keep the working
/// response `O(σ)` regardless of the mean.
///
/// Returns `None` (do not center, exact previous behaviour) unless the fit is
/// identity-link Gaussian and carries an unpenalized intercept column to absorb
/// the shift, and has no linear constraints that could pin the intercept. A zero
/// or non-finite mean also returns `None` — there is nothing to gain.
///
/// # It was a CORRECTNESS requirement while PIRLS carried a ridge (#2671)
///
/// PIRLS used to charge a fixed stabilization ridge `delta * ||beta||^2` against
/// a target of zero, so the outer criterion was a function of WHERE THE ORIGIN
/// OF `y` SAT: shifting `y` by `m` moved the intercept by `m` and moved the
/// criterion by `(n/2)/D_p * delta * ((beta0 + m)^2 - beta0^2)`. Centering here
/// pinned that origin at the weighted response mean. That ridge is removed
/// (#2901 V22), so an unpenalized intercept absorbs the shift exactly and
/// centering is a precision requirement. The scalar and joint routes must still
/// share this gate, so that they grade criteria formed on the same response.
///
/// MEASURED at `517b6303f` on `mk_1d(15, t^2, 0.05, 7)`, `y ~ matern(x,nu=5/2)`,
/// three arms of one run: the route that DOES center moved `4.085e-14` under a
/// `+10` shift; the route that did not moved `5.047e-5` — a separation of
/// `1.24e9`, and the un-centered route's fit went from ACCEPTED (pre-centered
/// response) to REFUSED (`+10`). Any outer λ-search over this family must
/// therefore condition through this gate and
/// [`conditioned_outer_response`]; while the ridge existed, a route that skipped
/// it selected λ̂/ψ̂ from the user's choice of response units.
pub(crate) fn gaussian_identity_response_center(
    cfg: &RemlConfig,
    conditioning: &ParametricColumnConditioning,
    has_linear_constraints: bool,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Option<f64> {
    if has_linear_constraints
        || conditioning.intercept_idx.is_none()
        || !matches!(cfg.likelihood.spec.response, ResponseFamily::Gaussian)
        || !matches!(cfg.link_function(), LinkFunction::Identity)
    {
        return None;
    }
    let mut weight_sum = 0.0_f64;
    let mut weighted = KahanSum::default();
    for ((&yi, &wi), &oi) in y.iter().zip(w.iter()).zip(offset.iter()) {
        if wi > 0.0 {
            weight_sum += wi;
            weighted.add(wi * (yi - oi));
        }
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let m = weighted.sum() / weight_sum;
    (m.is_finite() && m != 0.0).then_some(m)
}

/// The multiplicative scale an identity-link Gaussian outer REML λ-search should
/// divide the (already centered) response by so its magnitude is `O(1)` for the
/// duration of the search (issue #1127).
///
/// Replacing the response `y` by `a·y` (`a > 0`) for an identity-link Gaussian
/// fit must rescale the entire fit by `a` and leave `λ̂` / EDF unchanged: the
/// penalized normal equations are exactly linear in `y`, so `β̂(a·y)=a·β̂(y)`
/// at any fixed `λ`, and the profiled REML criterion is `a`-invariant up to the
/// additive constant `−(n−p)·ln a` (the dispersion `σ̂²` absorbs the `a²`).
/// Numerically, though, the outer λ-selection's convergence band is keyed to an
/// *absolute* objective scale (the inner-solve `objective_scale.max(1.0)` floor
/// and the outer `1e-6` gradient floor): when the whole Gaussian objective is
/// `O(a²) ≪ 1` those floors swamp the real signal and the optimizer declares
/// premature convergence at an over-smoothed `λ` — silently over-smoothing
/// small-magnitude responses (strains, volts, mole fractions, returns;
/// `a ≈ 1e-6`). Normalizing the working response to `O(1)` makes the absolute
/// floors track the true signal, restoring scale equivariance.
///
/// Returns `Some(s)` with `s = √(Σ wᵢ (yᵢ − mean)² / Σ wᵢ)` — the weighted RMS
/// of the centered response — so the caller can divide by it and keep the outer
/// working response `O(1)` regardless of magnitude. The same gate as
/// [`gaussian_identity_response_center`] applies (identity-link Gaussian with an
/// unpenalized intercept and no linear constraints); a non-finite, zero, or
/// already-`O(1)` RMS returns `None` (do not scale, exact previous behaviour) —
/// scaling near unity buys nothing and only risks a needless allocation.
pub(crate) fn gaussian_identity_response_scale(
    cfg: &RemlConfig,
    conditioning: &ParametricColumnConditioning,
    has_linear_constraints: bool,
    center: f64,
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
) -> Option<f64> {
    if has_linear_constraints
        || conditioning.intercept_idx.is_none()
        || !matches!(cfg.likelihood.spec.response, ResponseFamily::Gaussian)
        || !matches!(cfg.link_function(), LinkFunction::Identity)
    {
        return None;
    }
    // A multiplicative response rescale `y → y/s` must be matched by `η → η/s`
    // for the residual to scale cleanly. The intercept and smooth coefficients
    // scale freely, but a *fixed* offset column does not — scaling the working
    // response while leaving the offset on its original scale would change the
    // residual geometry, not just its magnitude. The offset is shared verbatim
    // into the outer state and reused by the accept-fit, so rather than thread a
    // separately scaled copy everywhere, restrict the (rare) offset case to the
    // exact previous path: only normalize when there is no nonzero offset.
    if offset.iter().any(|&o| o != 0.0) {
        return None;
    }
    let mut weight_sum = 0.0_f64;
    let mut weighted_sq = KahanSum::default();
    for ((&yi, &wi), &oi) in y.iter().zip(w.iter()).zip(offset.iter()) {
        if wi > 0.0 {
            weight_sum += wi;
            let centered = (yi - oi) - center;
            weighted_sq.add(wi * centered * centered);
        }
    }
    if weight_sum <= 0.0 {
        return None;
    }
    let rms = (weighted_sq.sum() / weight_sum).sqrt();
    // Only normalize when the magnitude is far enough from `O(1)` to matter; a
    // factor within ~one order of magnitude of unity cannot push the objective
    // through the absolute floors, so leave the exact previous path untouched.
    (rms.is_finite() && rms > 0.0 && !(0.1..=10.0).contains(&rms)).then_some(rms)
}

/// Apply the outer-λ-search response conditioning `(y − center)/scale`.
///
/// The ONLY place this arithmetic is written. Both routes that run an outer
/// λ-search over an identity-link Gaussian response must condition through this
/// function, or they minimize two different penalized problems and their
/// criteria are not comparable — see the module note on
/// [`gaussian_identity_response_center`] and #2671. `(None, None)` returns
/// `None` so the caller keeps borrowing the original response with no
/// allocation and no behavioural change.
pub(crate) fn conditioned_outer_response(
    center: Option<f64>,
    scale: Option<f64>,
    y: ArrayView1<'_, f64>,
) -> Option<Array1<f64>> {
    match (center, scale) {
        (None, None) => None,
        (center, scale) => {
            let c = center.unwrap_or(0.0);
            let s = scale.unwrap_or(1.0);
            Some(y.mapv(|value| (value - c) / s))
        }
    }
}

/// Pin the λ-search nuisance freeze to a canonical, cache-independent anchor
/// (#2363).
///
/// The λ-search optimizes `F(ρ) = REML(ρ, ψ)` with the estimated nuisance ψ
/// (Gamma shape, Tweedie φ, Beta precision) held FIXED across ρ. Without that
/// freeze ψ is re-profiled from every trial's warm-start η, the analytic outer
/// gradient — which holds ψ fixed — can never match the cost's ψ(ρ) motion, the
/// projected gradient floors above tolerance and the search stalls or rails
/// (#1074 / #1477 / #2369). The freeze is therefore load-bearing; what was not
/// load-bearing, and was wrong, is WHERE ψ got captured.
///
/// Until this call existed, ψ was captured opportunistically at the first
/// inner solve that happened to converge — and the persistent warm-start
/// cache decides which solve that is. It donates `initial_rho`, which
/// `run_outer_with_plan` inserts as seed 0, and it donates the warm β that
/// decides whether the pre-search reference solve at ρ = 0 converges at all. So
/// a cold machine and a warm machine froze ψ at different fits, i.e. they
/// minimized DIFFERENT criteria and legitimately landed at different optima:
/// measured on the #2363 fixture, the same Beta fit reported REML −6.382e2 cold
/// and −7.408e2 warm. That is the invariant violation at its root — the
/// objective was a function of the search path, not of the problem.
///
/// The repair is to define ψ, once, by a computation that depends on nothing but
/// `(data, model spec)`: solve P-IRLS at the symmetric reference ρ = 0 (every
/// λ = 1 — the same order-invariant reference point `canonical_rho_keys` uses to
/// label ρ-coordinates) on a state that has no warm start attached and no
/// persistent session open, and let the ordinary capture path freeze ψ at THAT
/// solve's converged η. If the reference point's inner solve does not converge,
/// the caller's full-length heuristic ρ (clamped into the design's
/// resolvability envelope) is the one other anchor tried, so the anchor stays a
/// pure function of the problem in every branch.
///
/// The warm-start slots are emptied on the way IN, so the anchor solve cannot
/// inherit a caller-supplied β. They are deliberately NOT emptied on the way out:
/// the β the anchor leaves behind is itself a function of (data, model spec), so
/// letting the search start from it keeps both a cold and a warm run entering the
/// search from the same predictor — and it is the same β the ρ = 0 reference solve
/// used to leave there before this function existed, so the cold path is
/// unchanged. It does mean `load_persistent_warm_start_once` finds an occupied
/// slot and skips its restore, which is the point: that restore would hand the
/// warm run a different starting β and a different set of adaptive LM/tolerance
/// signals. Nothing is lost — the cached β still reaches the inner solve at the
/// cached ρ through `OuterConfig::initial_inner_seed`, which installs it at
/// exactly the outer coordinate that owns it.
///
/// The on-disk session must not be active while this computation runs (the inner
/// solve would reload the cached β mid-anchor), so a direct call on an attached
/// state is refused. The design-moving evaluator satisfies this precondition
/// with `RemlState::without_persistent_warm_start_store`; the standard evaluator
/// calls before attaching persistence. The ρ-keyed eval/P-IRLS memos are kept:
/// they cache a deterministic computation at a ρ the caller is about to evaluate
/// again.
///
/// Negative-Binomial θ is deliberately not handled here. It is seeded from the
/// resolved family spec before the search and then driven to a certified joint
/// (θ, ρ) fixed point by the alternation loop below, which is already a function
/// of the data alone.
pub(crate) fn freeze_lambda_search_nuisance_at_canonical_anchor(
    reml_state: &RemlState<'_>,
    resolved_likelihood_scale: &gam_problem::ResolvedLikelihoodScale,
    k: usize,
    heuristic_log_lambdas: Option<&[f64]>,
) -> Result<(), EstimationError> {
    freeze_lambda_search_nuisance_at_canonical_anchor_with_ext_count(
        reml_state,
        resolved_likelihood_scale,
        k,
        heuristic_log_lambdas,
        0,
    )
}

/// Joint-design counterpart of
/// [`freeze_lambda_search_nuisance_at_canonical_anchor`].
///
/// `external_hyper_count` preserves the objective-completeness policy of the
/// joint surface while the anchor evaluates `rho = 0`. In particular, an
/// anchor must not memoize a value under the fixed-design correction policy and
/// then let a joint `[rho, psi]` evaluation reuse it.
pub(crate) fn freeze_lambda_search_nuisance_at_canonical_anchor_with_ext_count(
    reml_state: &RemlState<'_>,
    resolved_likelihood_scale: &gam_problem::ResolvedLikelihoodScale,
    k: usize,
    heuristic_log_lambdas: Option<&[f64]>,
    external_hyper_count: usize,
) -> Result<(), EstimationError> {
    let (frozen, family) = match resolved_likelihood_scale {
        gam_problem::ResolvedLikelihoodScale::Gamma {
            estimated: true, ..
        } => (&reml_state.frozen_gamma_shape, "gamma shape"),
        gam_problem::ResolvedLikelihoodScale::Tweedie {
            estimated: true, ..
        } => (&reml_state.frozen_tweedie_phi, "tweedie dispersion"),
        gam_problem::ResolvedLikelihoodScale::BetaPrecision {
            estimated: true, ..
        } => (&reml_state.frozen_beta_phi, "beta precision"),
        gam_problem::ResolvedLikelihoodScale::Dispersion {
            estimated: true, ..
        } => (&reml_state.frozen_dispersion_phi, "dispersion"),
        _ => return Ok(()),
    };
    if k == 0 || frozen.load(Ordering::Relaxed) != 0 {
        return Ok(());
    }
    if reml_state.persistent_warm_start_store().is_some() {
        crate::bail_invalid_estim!(
            "the {family} λ-search freeze must be anchored before the persistent warm-start \
             layer is attached, or while it is scoped off (#2363/#2426); with the on-disk \
             session open the anchor solve would reload the cached β and the outer criterion \
             would again depend on cache state"
        );
    }
    // The anchor must see the same starting predictor on every machine, so an
    // externally supplied seed is not admissible input to it.
    reml_state.clear_warm_start_predictor_state();
    reml_state.clear_warm_start_adaptive_signals();

    // The anchors are clamped into the envelope of the design's own #2812
    // resolvability domain, the domain the λ search then runs on (#2902 row 9).
    // Past the ρ = 0 anchor they are tried only when its inner solve refused,
    // and the search domain is then read at these same prior weights.
    let (domain_lower, domain_upper) =
        crate::estimate::rho_domain::resolvability_domain_from_design(
            reml_state.weights,
            &reml_state.x,
            &reml_state.canonical_penalties,
        )
        .map_err(EstimationError::LayoutError)?;
    let envelope = gam_problem::OrderedRhoBounds::envelope(
        domain_lower.iter().copied(),
        domain_upper.iter().copied(),
    )?;
    let mut anchors = vec![Array1::<f64>::zeros(k)];
    if let Some(heuristic) = heuristic_log_lambdas.filter(|h| h.len() == k) {
        let clamped = Array1::from_iter(heuristic.iter().map(|&value| envelope.clamp(value)));
        if clamped.iter().any(|value| *value != 0.0) {
            anchors.push(clamped);
        }
    }
    for anchor in &anchors {
        if let Err(error) =
            reml_state.compute_cost_with_ext_count(anchor, external_hyper_count)
        {
            log::trace!("[OUTER] nuisance anchor candidate rejected: {error:?}");
            continue;
        }
        let bits = frozen.load(Ordering::Relaxed);
        if bits != 0 {
            log::debug!(
                "[OUTER] {family} λ-search freeze anchored at ρ=[{}] before any warm start (#2363): \
                 value {:.6e}; the outer criterion is now a function of the data and the model spec alone",
                anchor
                    .iter()
                    .take(4)
                    .map(|value| format!("{value:.3}"))
                    .collect::<Vec<_>>()
                    .join(","),
                f64::from_bits(bits),
            );
            break;
        }
    }
    if frozen.load(Ordering::Relaxed) == 0 {
        // No deterministic anchor produced a converged inner solve. The search
        // is about to try the same points and will report its own refusal;
        // leaving the freeze unset keeps the pre-existing capture path rather
        // than converting an outer-search failure into a different error here.
        log::debug!(
            "[OUTER] no deterministic anchor converged for the {family} λ-search freeze; \
             the outer criterion cannot be pinned before the outer search"
        );
    }
    Ok(())
}

/// The Student-t `(σ, ν)` at the outer coordinates `(ln(σ/s₀), ln ν)`, where
/// `s₀` is [`pirls::student_t_reference_scale`]. The shift by `ln s₀` leaves
/// every derivative of the criterion in `ln σ` unchanged and makes the
/// coordinate, its box, and the zero seed equivariant under `y ↦ a·y`.
fn student_t_outer_point(reference_scale: f64, log_relative_sigma: f64, log_nu: f64) -> (f64, f64) {
    (reference_scale * log_relative_sigma.exp(), log_nu.exp())
}

pub(crate) fn optimize_external_designwith_heuristic_log_lambdas_andwarm_start<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<PenaltySpec>,
    heuristic_log_lambdas: Option<&[f64]>,
    warm_start_beta: Option<ArrayView1<'_, f64>>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    if opts.family.is_binomial_mixture() && opts.mixture_link.is_none() {
        crate::bail_invalid_estim!("BinomialMixture requires mixture_link specification");
    }
    let x = x.into();
    if let Some(message) = row_mismatch_message(y.len(), w.len(), x.nrows(), offset.len()) {
        crate::bail_invalid_estim!("{}", message);
    }

    let p = x.ncols();
    validate_penalty_specs(&s_list, p, "external-design fit")?;
    let (canonical, active_nullspace_dims) = gam_terms::construction::canonicalize_penalty_specs(
        &s_list,
        &opts.nullspace_dims,
        p,
        "external-design fit",
    )?;
    let conditioning = ParametricColumnConditioning::infer_from_penalty_specs(&x, &s_list);
    let x_fit = conditioning.apply_to_design(&x);
    let fit_linear_constraints =
        conditioning.transform_linear_constraints_to_internal(opts.linear_constraints.clone());
    let k = canonical.len();
    if active_nullspace_dims.len() != k {
        crate::bail_invalid_estim!(
            "nullspace_dims length mismatch: expected {k} entries for active penalties, got {}",
            active_nullspace_dims.len()
        );
    }
    let (cfg, effective_sas_link) = resolved_external_config(opts)?;
    reject_prefit_unidentifiable_unpenalized_space(w, p, &canonical)?;
    // Student-t `(σ, ν)` are outer LAML hyperparameters searched jointly with
    // ρ in the coordinates `(ln(σ/s₀), ln ν)` (see `student_t_outer_point`).
    // Install the zero seed before any evaluation reads the family.
    let student_t_reference_scale = match cfg.likelihood.student_t_parameters() {
        Some(_) => Some(pirls::student_t_reference_scale(y, offset, w)?),
        None => None,
    };
    let mut cfg = cfg;
    if let Some(scale) = student_t_reference_scale {
        let (sigma, nu) = student_t_outer_point(scale, 0.0, 0.0);
        cfg.likelihood = cfg.likelihood.clone().with_student_t(sigma, nu);
    }
    reject_prefit_unpenalized_rank_deficiency(w, &x_fit, &canonical)?;
    let jeffreys_arming_evidence =
        arm_jeffreys_on_prefit_binomial_separation(&mut cfg, opts, y, w, &x_fit, &canonical)?;

    let design_kind = match &x {
        DesignMatrix::Dense(_) => "dense",
        DesignMatrix::Sparse(_) => "sparse",
    };
    log::debug!(
        "[GAM fit] n={} p={} k={} fam={:?} link={:?} X={} reml_iter={} firth={}",
        y.len(),
        p,
        k,
        opts.family,
        cfg.link_function(),
        design_kind,
        opts.max_iter,
        cfg.firth_bias_reduction
    );

    // Own the external arrays once; the conditioned design is shared through `reml_state`.
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x;
    let offset_o = offset.to_owned();
    let canonical_shared = Arc::new(canonical);
    let cfg_shared = Arc::new(cfg.clone());

    // Issue #1000: for an identity-link Gaussian fit with an unpenalized
    // intercept, adding a constant `c` to the response is a *pure relabeling of
    // the intercept* — the hat matrix annihilates the constant column, so the
    // residuals, the profiled REML criterion, λ̂, and the smooth shape are all
    // invariant to `c`. Numerically, though, the outer REML score/gradient
    // accumulate `yᵀy`-magnitude sufficient statistics (e.g. the cached
    // `XᵀW(y−offset)`), so an uncentered large-mean response injects a `c²`
    // term that loses precision and drifts λ̂ — silently over-smoothing
    // large-mean responses (Kelvin temperatures, financial levels, calendar
    // years). Center the response by the (weighted) mean the intercept would
    // absorb for the duration of the outer λ-search only: the constant lands in
    // the intercept, which the final accept-fit below recovers *exactly* by
    // re-fitting the original (uncentered) response at the REML-selected λ̂.
    // This mirrors the existing column conditioning, which centers the design
    // columns into the intercept for the same numerical reason.
    let response_center = gaussian_identity_response_center(
        &cfg,
        &conditioning,
        opts.linear_constraints.is_some(),
        y_o.view(),
        w_o.view(),
        offset_o.view(),
    );
    // Issue #1127 (down-scale sibling of #1000): replacing the response `y` by
    // `a·y` must rescale the whole fit by `a` and leave `λ̂`/EDF unchanged (the
    // normal equations are exactly linear in `y`; the profiled REML criterion is
    // `a`-invariant up to the additive `−(n−p)·ln a` the dispersion absorbs).
    // But the outer λ-selection's convergence band is keyed to an *absolute*
    // objective scale (an inner `objective_scale.max(1.0)` floor and a `1e-6`
    // outer gradient floor); when the Gaussian objective is `O(a²) ≪ 1` those
    // floors swamp the signal and the optimizer stops early at an over-smoothed
    // `λ`. Normalize the (centered) working response to `O(1)` for the outer
    // λ-search only, mirroring the #1000 centering: the final accept-fit below
    // re-fits the *original* response at the REML-selected λ̂, so β, μ̂, σ̂² and
    // every reported quantity stay exactly on the user's scale. `center` here is
    // the constant the intercept already absorbs (so the scale is measured on the
    // residual signal, not on the offset).
    let response_scale = gaussian_identity_response_scale(
        &cfg,
        &conditioning,
        opts.linear_constraints.is_some(),
        response_center.unwrap_or(0.0),
        y_o.view(),
        w_o.view(),
        offset_o.view(),
    );
    // The outer loop borrows the response for the lifetime of `reml_state`;
    // the conditioned copy (when any) is owned at function scope so the borrow
    // outlives the state. Off the Gaussian-identity path both `response_center`
    // and `response_scale` are `None` and the outer loop borrows the original
    // response verbatim — no allocation, no behavioural change. When only one is
    // active we still apply just that transform. Both are exactly invertible by
    // the accept-fit, which re-fits the original `y_o` at the selected λ̂.
    let reml_y_conditioned: Option<Array1<f64>> =
        conditioned_outer_response(response_center, response_scale, y_o.view());
    let reml_y_view = reml_y_conditioned
        .as_ref()
        .map_or_else(|| y_o.view(), |conditioned| conditioned.view());

    let mut reml_state = RemlState::newwith_offset_shared(
        reml_y_view,
        x_fit,
        w_o.view(),
        offset_o.view(),
        Arc::clone(&canonical_shared),
        p,
        Arc::clone(&cfg_shared),
        Some(active_nullspace_dims.clone()),
        None,
        fit_linear_constraints.clone(),
    )?;
    reml_state.set_rho_prior(opts.rho_prior.clone());
    // #1082: this search decides the #784 block-local correction's admission
    // once, at its certified Laplace optimum, so no evaluation before that (the
    // canonical-key and nuisance anchors at ρ = 0, the prepass, every seed) can
    // latch it.
    reml_state.defer_block_correction_admission();
    let resolved_likelihood_scale = cfg
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let estimates_negbin_theta = matches!(
        resolved_likelihood_scale,
        gam_problem::ResolvedLikelihoodScale::NegativeBinomial {
            estimated: true,
            ..
        }
    );
    if let gam_problem::ResolvedLikelihoodScale::NegativeBinomial {
        theta,
        estimated: true,
    } = resolved_likelihood_scale
    {
        let theta_seed = theta.value();
        if !(theta_seed.is_finite() && theta_seed > 0.0) {
            return Err(EstimationError::InvalidInput(format!(
                "estimated Negative-Binomial theta seed must be finite and positive, got {theta_seed}"
            )));
        }
        // Treat the estimated family value as a warm-start coordinate. This
        // makes an exhaustion checkpoint resumable by reconstructing the same
        // estimated-NB family with the carried theta and passing the carried rho
        // through the ordinary smoothing warm-start input.
        reml_state
            .frozen_negbin_theta
            .store(theta_seed.to_bits(), Ordering::Relaxed);
    }

    // #2363: pin the λ-search nuisance BEFORE any warm start — external, in
    // memory, or on disk — can reach this state. `freeze_lambda_search_nuisance_at_canonical_anchor`
    // documents why the criterion is otherwise a function of the search path.
    freeze_lambda_search_nuisance_at_canonical_anchor(
        &reml_state,
        &resolved_likelihood_scale,
        k,
        heuristic_log_lambdas,
    )?;
    // #2812 / #2902 row 8: the λ-selection domain of each coordinate is derived
    // from the conditioned design's Gram on that penalty's columns and the
    // penalty's spectrum, not the picked ±RHO_BOUND box (SPEC rule 20). The
    // Gram is the data curvature `XᵀWX` of the penalized Hessian, so `W` is the
    // Fisher working weight of the canonical anchor's inner solve at ρ = 0 (the
    // solve the nuisance freeze above ran, before any warm start): its scale
    // follows the response's units (`μ³/4` for the inverse-Gaussian `1/μ²`
    // link), and a prior-weight Gram leaves the domain fixed while `λ̂` moves
    // with those units, off the domain's lower face in small units. When that
    // solve returns a typed per-rho refusal (`is_trial_point_infeasible`) there
    // is no fitted working weight at ρ = 0, and the Gram is read at the prior
    // weights. Every other failure is not about ρ = 0 and is propagated.
    let domain_weights = if k == 0 {
        w_o.to_owned()
    } else {
        match reml_state.data_curvature_weights(&Array1::zeros(k)) {
            Ok(weights) => weights,
            Err(error) if !error.is_trial_point_infeasible() => return Err(error),
            Err(error) => {
                log::debug!(
                    "[OUTER] ρ-domain Gram read at the prior weights: the canonical anchor's \
                     inner solve at ρ = 0 refused ({error})"
                );
                w_o.to_owned()
            }
        }
    };
    let rho_resolvability =
        crate::estimate::rho_domain::resolvability_domain_and_limit_faces_from_design(
            domain_weights.view(),
            &reml_state.x,
            canonical_shared.as_slice(),
        )
        .map_err(EstimationError::LayoutError)?;
    // The domain is a numerical device, not the prior: the ρ-posterior
    // integrates over all of ℝ^K and continues the criterion past each
    // saturated face (#2812).
    let rho_continuation = rho_resolvability.continuation();
    let crate::estimate::rho_domain::ResolvabilityDomain {
        lower: rho_domain_lower,
        upper: rho_domain_upper,
        lower_is_limit: rho_lower_is_limit,
        upper_is_limit: rho_upper_is_limit,
        ..
    } = rho_resolvability;
    if let Some(store) = opts.persistent_warm_start_store.clone() {
        // Attach only after the canonical nuisance anchor so cache history
        // cannot influence the criterion frame.
        reml_state.attach_persistent_warm_start_store(store);
    }
    reml_state.setwarm_start_original_beta(warm_start_beta);

    // Term/margin-order invariance (#1538/#1539). The per-ρ-coordinate canonical
    // keys label each coordinate by its placement-independent (penalty + data)
    // content, letting the outer optimizer operate in an identical canonical
    // coordinate layout for every term order (attached via
    // `with_rho_canonical_keys` below). `None` when the coordinate count does not
    // match the ρ-dimension (legacy native-order path, unchanged).
    let canon_keys = reml_state.canonical_rho_keys(k);

    let reml_tol = cfg.reml_convergence_tolerance;
    let reml_max_iter = opts.max_iter;
    let outer_eval_idx = AtomicUsize::new(0usize);
    let mixture_optspec = if opts.optimize_mixture {
        opts.mixture_link.clone()
    } else {
        None
    };
    let sas_optspec = if opts.optimize_sas {
        effective_sas_link
    } else {
        None
    };
    let mixture_dim = mixture_optspec
        .as_ref()
        .map(|s| s.initial_rho.len())
        .unwrap_or(0);
    let sas_dim = if sas_optspec.is_some() { 2 } else { 0 };
    let student_t_dim = if student_t_reference_scale.is_some() { 2 } else { 0 };
    let sasridgeweight = if sas_dim > 0 {
        sas_log_deltaridgeweight()
    } else {
        0.0
    };
    // Estimated Negative-Binomial theta and smoothing rho are solved by block
    // coordinate optimization, but acceptance is JOINT: both analytic partials
    // are measured at the identical fixed-theta PIRLS solution. The outer
    // iteration budget is also the alternation budget, so exhaustion is not a
    // second hidden tuning parameter; it returns a typed error carrying the best
    // measured checkpoint instead of minting the final iterate as a fit.
    let mut final_rho;
    // #2727: the link-shape coordinates of the shipped point, in the OUTER
    // optimizer's own coordinate system (raw `theta`, before the link state's
    // smooth-bound maps). Empty on every rho-only arm. Carried separately
    // because the shipped link STATE stores transformed values
    // (`sas_effective_epsilon`), so it cannot be compared against the
    // certificate's raw coordinates.
    let mut final_link_coords: Array1<f64>;
    let mut final_mixture_state;
    let mut final_sas_state;
    let mut final_mixture_param_covariance;
    let mut final_sas_param_covariance;
    let mut outer_result;
    let mut pirls_res;
    let mut negbin_alternation_round: usize = 0;
    let mut negbin_rho_seed: Option<Array1<f64>> = None;
    // Set once the #784 correction is admitted at the Laplace optimum: every
    // later run of the standard arm is the corrected search continued from that
    // optimum, alone (#1082).
    let mut corrected_continuation = false;
    // The Laplace optimum's exact analytic outer Hessian, bound to that
    // optimum: the corrected continuation's BFGS starts from its inverse.
    let mut continuation_curvature: Option<Array2<f64>> = None;
    let mut negbin_best_checkpoint: Option<NegbinJointCheckpoint> = None;
    // The box every outer arm searches the ρ block in, and so the box its
    // certificate judges rails against: the #2812 resolvability domain (#2902
    // row 8). Every post-fit projection below reads this one box (#2412), so a
    // coordinate railed on a derived face is scored as railed there too.
    let rho_model_domain: (Array1<f64>, Array1<f64>) = (rho_domain_lower, rho_domain_upper);
    loop {
        (
            final_rho,
            final_link_coords,
            final_mixture_state,
            final_sas_state,
            final_mixture_param_covariance,
            final_sas_param_covariance,
            outer_result,
        ) = if mixture_dim > 0 && sas_dim > 0 {
            crate::bail_invalid_estim!(
                "simultaneous mixture and SAS optimization is not supported"
            );
        } else if mixture_dim == 0 && sas_dim == 0 && student_t_dim == 0 {
            use crate::rho_optimizer::{OuterEvalOrder, OuterProblem};
            use gam_problem::{DeclaredHessianForm, Derivative};

            let rho_warm_start = negbin_rho_seed
                .as_ref()
                .and_then(|rho| rho.as_slice())
                .or(heuristic_log_lambdas);
            let analytic_outer_hessian_available = reml_state.analytic_outer_hessian_enabled();
            // Every family's search consumes the declared exact outer Hessian
            // (ARC), as profiled Gaussian identity always has. The #2359 split
            // that held non-Gaussian links to gradient-only BFGS, paying the
            // order-four family tower only at the mint audit, made binomial and
            // Poisson fits 6-10x slower than the Gaussian path: BFGS rebuilds
            // from secant pairs the curvature the evaluator already has exactly
            // (48-61 outer iterations on a one-smooth n=1000 logistic fit).
            let n_obs = y_o.len();
            let problem = OuterProblem::new(k)
                .with_gradient(Derivative::Analytic)
                .with_hessian(if analytic_outer_hessian_available {
                    DeclaredHessianForm::Either
                } else {
                    DeclaredHessianForm::Unavailable
                })
                .with_barrier(
                    crate::estimate::reml::reml_outer_engine::BarrierConfig::from_constraints(
                        fit_linear_constraints.as_ref(),
                    ),
                )
                .with_tolerance(reml_tol)
                .with_outer_inner_cap(reml_inner_progress_feedback(&reml_state))
                .with_problem_size(n_obs, x_o.ncols())
                .with_bounds(rho_model_domain.0.clone(), rho_model_domain.1.clone())
                // #2954: which of those faces are the terms' limit models, so a
                // mint may rail a coordinate there and nowhere else.
                .with_limit_faces(rho_lower_is_limit.clone(), rho_upper_is_limit.clone())
                // Make the outer smoothing-parameter search invariant to the order
                // the smooth terms / tensor margins were written (#1538/#1539). The
                // structural keys label each ρ-coordinate by its placement-
                // independent penalty content, so the optimizer canonicalizes the
                // coordinate layout and resolves the flat double-penalty REML valley
                // identically for `s(x)+s(z)` vs `s(z)+s(x)` and `te(x,z)` vs
                // `te(z,x)`. `None` (coordinate count not matching ρ-dim) leaves the
                // native-order path unchanged.
                .with_rho_canonical_keys(canon_keys.clone());
            let problem = if let Some(h) = rho_warm_start {
                problem.with_heuristic_log_lambdas(h.to_vec())
            } else {
                problem
            };
            let problem = if let Some(h) = rho_warm_start.filter(|h| h.len() == k) {
                problem.with_initial_rho(Array1::from_iter(h.iter().copied()))
            } else {
                problem
            };
            let problem = match (corrected_continuation, negbin_rho_seed.as_ref(), continuation_curvature.as_ref()) {
                (true, Some(seed), Some(hessian)) => {
                    problem.with_initial_curvature(seed.clone(), hessian.clone())
                }
                _ => problem,
            };

            // Geometric-mean log prior-weight anchor `log g(w) = (1/n₊)·Σ log wᵢ`
            // over the positive-weight rows. The pure-REML optimum for a *profiled*
            // (Gaussian-identity) fit drifts by `ρ̂ → ρ̂ + log c` under a global
            // prior-weight rescale `w → c·w` (`H = XᵀWX + λS`, so λ → c·λ keeps the
            // penalised curvature proportional to the data curvature, β̂ / EDF /
            // predictions fixed). The outer ρ-search seed and the relative-from-seed
            // convergence test would otherwise be referenced to a weight-independent
            // origin (0), so a heavily up-weighted fit starts `log c` further from
            // its (shifted) optimum and the optimiser stops short — exactly the
            // weight-scale non-invariance of λ̂ reported in issue #877. Anchoring the
            // seed at `log g(w)` makes the search start the SAME relative distance
            // from the optimum regardless of the weight magnitude.
            //
            // This is the SAME gated anchor the outer ρ-prior uses
            // ([`RemlState::rho_weight_anchor`]): it is the geometric-mean
            // log-weight for a profiled-dispersion family and *exactly 0* for a
            // fixed-dispersion family (Poisson, binomial, …). For fixed dispersion
            // `w = c` is exact `c`-fold replication: the two encodings share an
            // identical LAML objective and optimum, so anchoring the seed by their
            // (differing) per-row log-weight mean would seed the weighted encoding
            // `log c` above its true optimum and the relative-convergence test would
            // stop it short — over-smoothing vs replication (issue #893). With all
            // weights 1 (or any fixed-dispersion family) the anchor is exactly 0, so
            // those fits stay byte-identical.
            let weight_log_geom_mean: f64 = reml_state.rho_weight_anchor();
            // The outer search enters from ONE deterministic, data-derived start
            // and the certified second-order search owns everything after it.
            // No candidate is scored, ranked or restarted: a start-point cost is
            // not a basin certificate, and a lattice of starts is a grid search.
            //
            // A caller-supplied full-length ρ (a warm start, a cached optimum, or
            // the Negative-Binomial alternation's previous ρ̂) is that start as
            // given, clamped into the search box. Otherwise the start is the
            // mgcv-style commensurate-curvature `initial.sp` point
            // `ρ_j = ln(tr(XᵀWX_j)/tr(S_j))` (#2069/#1575), which balances each
            // penalty block against the data curvature it regularizes, so a
            // block with little data support starts at a large λ_j by
            // construction (#1266/#1464). Coordinates it cannot place fall back
            // to the weight-scale anchor `rho_weight_anchor` (exactly 0 for unit
            // weights and every fixed-dispersion family; #877/#893).
            //
            // The window is the domain the outer optimizer itself searches: the
            // envelope of the #2812 resolvability domain (#2902 row 9). With no
            // penalty there is no coordinate to place, and the precision box
            // stands in for the empty envelope. `OrderedRhoBounds::new` refuses a
            // non-finite or inverted interval (#2379).
            let (envelope_lower, envelope_upper) = if rho_model_domain.0.is_empty() {
                crate::estimate::rho_domain::coordinate_domain(None, None)
            } else {
                (
                    rho_model_domain.0.iter().copied().fold(f64::INFINITY, f64::min),
                    rho_model_domain.1.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                )
            };
            let start_bounds = OrderedRhoBounds::new(envelope_lower, envelope_upper)?;
            let outer_start = if let Some(h) = rho_warm_start.filter(|h| h.len() == k) {
                Array1::from_iter(h.iter().map(|&v| start_bounds.clamp(v)))
            } else {
                let anchor = Array1::from_elem(k, start_bounds.clamp(weight_log_geom_mean));
                // The pilot P-IRLS solve behind the `initial.sp` point runs at
                // `anchor`. A typed per-rho refusal there
                // (`is_trial_point_infeasible`) leaves no working weight to
                // balance against, so the search enters at `anchor` and steps
                // past it exactly as it steps past any infeasible trial point.
                // Every other failure is not about `anchor` and is propagated.
                match reml_state.analytic_initial_sp_rho(&anchor, start_bounds) {
                    Ok(Some(start)) => start,
                    Ok(None) => anchor,
                    Err(error) if error.is_trial_point_infeasible() => anchor,
                    Err(error) => return Err(error),
                }
            };
            log::debug!(
                "[OUTER] standard REML single start: {:?} (bounds {:.3}..{:.3})",
                outer_start.as_slice().unwrap_or(&[]),
                start_bounds.lower(),
                start_bounds.upper(),
            );
            let problem = problem.with_initial_rho(outer_start);
            // Attach the outer-loop cache session. The session shares its
            // realized-fit-context key with the inner beta record (different
            // payload namespace), so a SIGKILL mid-outer-iter leaves both the
            // last accepted β (inner record) and the best rho seen so far
            // (outer iterate) on disk for the next run.
            let problem = match reml_state.outer_cache_session() {
                Some(session) => problem.with_cache_session(session),
                None => problem,
            };

            let obj = problem.build_objective_with_eval_order(
                &mut reml_state,
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.compute_cost(rho)
                },
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                    state.compute_outer_eval_with_order(
                        rho,
                        if analytic_outer_hessian_available {
                            OuterEvalOrder::ValueGradientHessian
                        } else {
                            OuterEvalOrder::ValueAndGradient
                        },
                    )
                },
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>,
                 order: OuterEvalOrder| {
                    outer_eval_idx.fetch_add(1, Ordering::Relaxed);
                    state.compute_outer_eval_with_order(rho, order)
                },
                Some(|state: &mut &mut crate::estimate::reml::RemlState<'_>| {
                    state.reset_outer_seed_state()
                }),
                // The EFS map is the fixed point of the Laplace trace identity
                // alone. Once the #784 block correction is latched the criterion
                // also carries Delta_b(rho), whose rho-gradient that map never
                // sees, so its fixed point is not a stationary point of the
                // corrected criterion: the corrected continuation has no
                // fixed-point map and walks on the criterion's own derivatives.
                (!corrected_continuation).then_some(
                    |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                        state.compute_efs_steps(rho)
                    },
                ),
            );
            // #2348 Inc 5: standard REML can form its own λ→∞ face limit
            // exactly (the null-space-restricted fit plus the analytic
            // first-order form of the logdet/trace terms there), so the outer
            // certificate can PROVE an infinite-smoothing face instead of
            // measuring a tail beside the box.
            let obj = obj.with_rail_face_limit(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 rho: &Array1<f64>,
                 face: &[usize]| { state.rail_face_limit(rho, face) },
            );
            // #2676: publish the criterion's EXACT invariance — the directions
            // of rho along which the penalty map, and therefore the criterion,
            // does not move at all. The outer certificate deflates them instead
            // of judging a chain-rule term against its own absolute value. The
            // closure speaks rho, and `ClosureObjective` applies the theta
            // embedding from the declared layout.
            let obj = obj.with_criterion_invariance(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.criterion_invariant_directions(rho)
                },
            );
            // Standard REML publishes its current original-basis coefficients
            // and consumes a cached coefficient vector through the symmetric
            // hook below. The runner calls it only after reset and only for the
            // bitwise-matching outer seed that owns the cached vector.
            let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());

            let strategy_result = problem.run(&mut obj, "standard REML")?;
            drop(obj);
            let accepted_rho = strategy_result.rho.clone();
            (
                accepted_rho,
                // Rho-only arm: the outer coordinate IS rho, so there are no
                // link coordinates to carry and the joint point is the rho one.
                Array1::zeros(0),
                cfg.link_kind.mixture_state().cloned(),
                cfg.link_kind.sas_state().copied(),
                None,
                None,
                strategy_result,
            )
        } else {
            let use_mixture = mixture_dim > 0;
            let use_sas = sas_dim > 0;
            let use_beta_logistic =
                use_sas && matches!(cfg.link_function(), LinkFunction::BetaLogistic);
            let theta_dim = k + mixture_dim + sas_dim + student_t_dim;
            // The Student-t block trails every link-shape coordinate.
            let student_t_offset = k + mixture_dim + sas_dim;
            let sasspec = sas_optspec;
            let mixspec = mixture_optspec
                .clone()
                .or_else(|| {
                    if use_mixture {
                        None
                    } else {
                        Some(MixtureLinkSpec {
                            components: Vec::new(),
                            initial_rho: Array1::zeros(0),
                        })
                    }
                })
                .ok_or_else(|| EstimationError::InvalidInput("missing mixture spec".to_string()))?;
            let mut heuristic_theta = Vec::new();
            if let Some(hvals) = heuristic_log_lambdas
                && hvals.len() == k
            {
                heuristic_theta.extend_from_slice(hvals);
                if use_mixture {
                    heuristic_theta
                        .extend_from_slice(mixspec.initial_rho.as_slice().unwrap_or(&[]));
                }
                if let Some(spec) = sasspec {
                    heuristic_theta.push(spec.initial_epsilon);
                    heuristic_theta.push(spec.initial_log_delta);
                }
                if student_t_dim > 0 {
                    heuristic_theta.extend_from_slice(&[0.0, 0.0]);
                }
            }
            let heuristic_theta_ref = if heuristic_theta.len() == theta_dim {
                Some(heuristic_theta.as_slice())
            } else {
                None
            };
            use crate::rho_optimizer::OuterProblem;
            use gam_problem::{DeclaredHessianForm, Derivative, HessianValue, OuterEval};
            let initial_link_kind = cfg.link_kind.clone();
            let initial_likelihood = cfg.likelihood.clone();
            // Same criterion, same declaration as the profiled-REML arm above
            // (#1082): this is the location-scale / SAS-mixture LAML score, a
            // sum over the same n rows, so its d/d-theta inherits the same O(n)
            // scale. Declaring it is what keeps the outer stationarity band a
            // property of the data since #2613 -- an undeclared route falls
            // back to the bare absolute tolerance, which at large n is orders
            // below the residual a converged fit floors at.
            // #2902 row 8: the θ box is derived per coordinate, not the ±RHO_BOUND
            // box. ρ is searched in the #2812 resolvability domain. A link
            // coordinate has no penalty spectrum, so it takes the range its own
            // chart resolves: a mixture free logit is a log-scale coordinate
            // (`precision_box`), SAS raw ε is a tanh chart (`sas_epsilon_domain`),
            // SAS raw log δ passes through `smooth_bound_jet`, which stops moving at
            // the edge of its support (`smooth_bound_support`), and the standardized
            // beta-logistic `[ε, log δ]` are log-shape coordinates like the mixture
            // logit (#2902 row 34).
            let (mut link_lower, mut link_upper): (Vec<f64>, Vec<f64>) = if use_mixture {
                let (lower, upper) = crate::estimate::rho_domain::precision_box();
                (vec![lower; mixture_dim], vec![upper; mixture_dim])
            } else if use_beta_logistic {
                // The standardized beta-logistic link's `[ε, log δ]` are log-shape
                // coordinates with no penalty spectrum, so they take the precision
                // box, as a mixture free logit does (#2902 row 34).
                let (lower, upper) = crate::estimate::rho_domain::precision_box();
                (vec![lower; sas_dim], vec![upper; sas_dim])
            } else if use_sas {
                let (epsilon_lower, epsilon_upper) =
                    crate::estimate::evaluation::sas_epsilon_domain();
                let (log_delta_lower, log_delta_upper) =
                    crate::mixture_link::smooth_bound_support(crate::mixture_link::SAS_LOG_DELTA_BOUND);
                (
                    vec![epsilon_lower, log_delta_lower],
                    vec![epsilon_upper, log_delta_upper],
                )
            } else {
                (Vec::new(), Vec::new())
            };
            // Student-t `ln(σ/s₀)` and `ln ν` are log-scale coordinates with no
            // penalty spectrum, so they take the precision box too.
            if student_t_dim > 0 {
                let (lower, upper) = crate::estimate::rho_domain::precision_box();
                link_lower.extend_from_slice(&[lower; 2]);
                link_upper.extend_from_slice(&[upper; 2]);
            }
            let theta_lower =
                Array1::from_iter(rho_model_domain.0.iter().copied().chain(link_lower));
            let theta_upper =
                Array1::from_iter(rho_model_domain.1.iter().copied().chain(link_upper));
            let n_obs = y_o.len();
            let problem = OuterProblem::new(theta_dim)
                .with_gradient(Derivative::Analytic)
                .with_hessian(DeclaredHessianForm::Either)
                // The joint link evaluator already assembles exact curvature
                // on every evaluation. Use it: BFGS can lose the changing
                // link/scale coupling and stall with a nonstationary shape.
                .with_prefer_gradient_only(false)
                .with_problem_size(n_obs, x_o.ncols())
                .with_psi_dim(mixture_dim + sas_dim + student_t_dim)
                .with_barrier(
                    crate::estimate::reml::reml_outer_engine::BarrierConfig::from_constraints(
                        fit_linear_constraints.as_ref(),
                    ),
                )
                .with_tolerance(reml_tol)
                .with_outer_inner_cap(reml_inner_progress_feedback(&reml_state))
                .with_bounds(theta_lower, theta_upper);
            let problem = if let Some(h) = heuristic_theta_ref {
                problem.with_heuristic_log_lambdas(h.to_vec())
            } else {
                problem
            };
            let problem = if let Some(h) = heuristic_theta_ref {
                problem.with_initial_rho(Array1::from_iter(h.iter().copied()))
            } else {
                problem
            };
            let problem = match reml_state.outer_cache_session() {
                Some(session) => problem.with_cache_session(session),
                None => problem,
            };
            // Shared helper: parse theta into rho + link params, update link state.
            let apply_link_theta = |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                                    theta: &Array1<f64>|
             -> Result<Array1<f64>, EstimationError> {
                let rho = theta.slice(s![..k]).to_owned();
                let mut cfg_eval = cfg.clone();
                if use_mixture {
                    let mix_rho = theta.slice(s![k..(k + mixture_dim)]).to_owned();
                    cfg_eval.link_kind = InverseLink::Mixture(
                        state_fromspec(&MixtureLinkSpec {
                            components: mixspec.components.clone(),
                            initial_rho: mix_rho,
                        })
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!(
                                "invalid blended inverse link: {e}"
                            ))
                        })?,
                    );
                }
                if use_sas {
                    let epsilon = if use_beta_logistic {
                        theta[k]
                    } else {
                        let (v, _) = sas_effective_epsilon(theta[k]);
                        v
                    };
                    let delta_like = theta[k + 1];
                    cfg_eval.link_kind = if use_beta_logistic {
                        InverseLink::BetaLogistic(
                            state_from_beta_logisticspec(SasLinkSpec {
                                initial_epsilon: epsilon,
                                initial_log_delta: delta_like,
                            })
                            .map_err(|e| {
                                EstimationError::InvalidInput(format!(
                                    "invalid Beta-Logistic link: {e}"
                                ))
                            })?,
                        )
                    } else {
                        InverseLink::Sas(
                            state_from_sasspec(SasLinkSpec {
                                initial_epsilon: epsilon,
                                initial_log_delta: delta_like,
                            })
                            .map_err(|e| {
                                EstimationError::InvalidInput(format!("invalid SAS link: {e}"))
                            })?,
                        )
                    };
                }
                state.set_link_states(
                    cfg_eval.link_kind.mixture_state().cloned(),
                    cfg_eval.link_kind.sas_state().copied(),
                );
                if let Some(scale) = student_t_reference_scale {
                    let (sigma, nu) = student_t_outer_point(
                        scale,
                        theta[student_t_offset],
                        theta[student_t_offset + 1],
                    );
                    state.set_student_t_state(sigma, nu)?;
                }
                Ok(rho)
            };

            // SAS ridge/barrier cost correction (shared between cost_fn, eval_fn, efs_fn).
            // SAS only.
            //
            // #2902 row 34: #2685 had given the beta-logistic block this weak ridge on
            // both coordinates, because its shapes and `β` shared the scale of `η` and
            // nothing opposed the drift of `log δ` toward −∞. The link now standardizes
            // `logit(U)` to logit's location and scale, which removes that gauge, so
            // `[ε, log δ]` carry no counter-term.
            let sas_ridge_cost = |theta: &Array1<f64>| -> f64 {
                if use_sas && !use_beta_logistic && sasridgeweight > 0.0 {
                    let log_delta = theta[k + 1];
                    let (barriercost, _) = sas_log_delta_edge_barriercostgrad(log_delta);
                    0.5 * sasridgeweight * log_delta * log_delta + barriercost
                } else {
                    0.0
                }
            };

            let obj = problem.build_objective(
            &mut reml_state,
            |state: &mut &mut crate::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let rho = apply_link_theta(state, theta)?;
                // Route the cost through the SAME link-ext evaluator the gradient
                // closure uses (value-only), so both see the #1876 inner-KKT
                // envelope correction `Ṽ = V − ½·rᵀH⁻¹r`. Using the plain
                // `compute_cost` here would report the raw capped-β̂ value `V`
                // while the gradient closure reports `∇Ṽ`, desyncing the outer
                // trust-region ratio test on any first-order-capped inner solve.
                let value_mode =
                    crate::estimate::reml::reml_outer_engine::EvalMode::ValueOnly;
                let result = state.evaluate_unified_with_link_ext(&rho, value_mode)?;
                let cost = result.cost + sas_ridge_cost(theta);
                Ok(cost)
            },
            |state: &mut &mut crate::estimate::reml::RemlState<'_>,
             theta: &Array1<f64>| {
                let eval_idx = outer_eval_idx.fetch_add(1, Ordering::Relaxed) + 1;
                let rho = apply_link_theta(state, theta)?;
                let tcost = Instant::now();

                // Use the unified REML evaluator with link ext_coords.
                // This computes ρ gradient AND link parameter gradient jointly
                // through the same HyperCoord infrastructure used for aniso ψ.
                let eval_mode =
                    crate::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian;
                let mut result = state.evaluate_unified_with_link_ext(&rho, eval_mode)?;

                let cost = result.cost + sas_ridge_cost(theta);
                let mut grad = result
                    .gradient_for_mode(eval_mode, theta_dim)
                    .map_err(|reason| EstimationError::TrialPointRefused { reason })?;

                assert_eq!(
                    grad.len(),
                    theta_dim,
                    "unified evaluator gradient length {} != theta_dim {}",
                    grad.len(),
                    theta_dim
                );

                let grad_effective = grad.clone();
                let mut hessian = materialize_link_outer_hessian(result.hessian, theta_dim)?;

                // SAS epsilon reparameterization chain rule.
                if use_sas && !use_beta_logistic {
                    let (_, d_eps_d_raw, d2_eps_d_raw2) = sas_effective_epsilon_second(theta[k]);
                    for j in 0..theta_dim {
                        hessian[[k, j]] *= d_eps_d_raw;
                        hessian[[j, k]] *= d_eps_d_raw;
                    }
                    hessian[[k, k]] += grad_effective[k] * d2_eps_d_raw2;
                    grad[k] *= d_eps_d_raw;
                }
                // Link-block ridge (+ the SAS-only edge barrier) gradient and
                // Hessian, matching `sas_ridge_cost` term for term (#2685).
                if use_sas && !use_beta_logistic && sasridgeweight > 0.0 {
                    let log_delta = theta[k + 1];
                    grad[k + 1] += sasridgeweight * log_delta;
                    hessian[[k + 1, k + 1]] += sasridgeweight;
                    let (_, barriergrad, barrierhess) =
                        sas_log_delta_edge_barriercostgradhess(log_delta);
                    grad[k + 1] += barriergrad;
                    hessian[[k + 1, k + 1]] += barrierhess;
                }

                let cost_sec = tcost.elapsed().as_secs_f64();
                let aux_dim = mixture_dim + sas_dim + student_t_dim;
                log::trace!(
                    "[outer-eval {eval_idx}] theta_dim={} aux_dim={} unified_link_ext time_sec={:.3}",
                    theta_dim,
                    aux_dim,
                    cost_sec,
                );
                Ok(OuterEval {
                    cost,
                    gradient: grad,
                    hessian: HessianValue::Dense(hessian),
                    inner_beta_hint: state.current_original_basis_beta(),
                })
            },
            Some(|state: &mut &mut crate::estimate::reml::RemlState<'_>| {
                state.reset_outer_seed_state();
                state.set_link_states(
                    initial_link_kind.mixture_state().cloned(),
                    initial_link_kind.sas_state().copied(),
                );
                state.restore_student_t_state(&initial_likelihood);
            }),
            Some(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>,
                 theta: &Array1<f64>| {
                    let rho = apply_link_theta(state, theta)?;
                    let mut efs_eval = state.compute_efs_steps_with_link_ext(&rho)?;

                    // SAS reparameterization chain rule on ψ steps.
                    if use_sas && !use_beta_logistic {
                        let (_, d_eps_d_raw) = sas_effective_epsilon(theta[k]);
                        if efs_eval.steps.len() > k {
                            efs_eval.steps[k] *= d_eps_d_raw;
                        }
                        if let Some(ref mut pg) = efs_eval.psi_gradient
                            && !pg.is_empty() {
                                pg[0] *= d_eps_d_raw;
                            }
                    }

                    // SAS log-δ ridge + edge barrier: their gradients enter
                    // `result.gradient` from the unified evaluator (estimate.rs
                    // 2170+), and `compute_efs_steps_with_link_ext` runs the
                    // universal-form EFS step `Δρ = log(1 − 2·g_full/q_eff)`
                    // which absorbs them automatically. We only need to
                    // mirror that contribution into the *cost* slot here so
                    // the outer fixed-point bridge's line search compares
                    // augmented-cost trial points consistently.
                    efs_eval.cost += sas_ridge_cost(theta);
                    Ok(efs_eval)
                },
            ),
        );
            // #2629: this objective is built on the SAME `&mut RemlState` as the
            // standard-REML arm above and evaluates through
            // #2676: publish the criterion's EXACT invariance — the directions
            // of rho along which the penalty map, and therefore the criterion,
            // does not move at all. The outer certificate deflates them instead
            // of judging a chain-rule term against its own absolute value. The
            // closure speaks rho, and
            // `ClosureObjective` applies the theta embedding from the declared
            // layout.
            let obj = obj.with_criterion_invariance(
                |state: &mut &mut crate::estimate::reml::RemlState<'_>, rho: &Array1<f64>| {
                    state.criterion_invariant_directions(rho)
                },
            );
            // Same exact-seed cache publish/consume symmetry as the standard
            // REML arm above (issue #236).
            let mut obj = obj.with_seed_inner_state(with_reml_beta_seed_hook());
            let context = if student_t_dim > 0 {
                "Student-t scale and degrees of freedom"
            } else {
                "mixture/SAS flexible link"
            };
            let outer_result = problem.run(&mut obj, context)?;
            drop(obj);
            let final_rho = outer_result.rho.slice(s![..k]).to_owned();
            // #2727: the remainder of the joint outer coordinate. `final_rho`
            // above is only its leading rho block; these are the link-shape
            // coordinates the same certificate covers, kept raw so the shipped
            // point can be reassembled and compared whole.
            let final_link_coords = outer_result.rho.slice(s![k..]).to_owned();
            let final_mix_state = if use_mixture {
                let final_mix_rho = outer_result.rho.slice(s![k..(k + mixture_dim)]).to_owned();
                Some(
                    state_fromspec(&MixtureLinkSpec {
                        components: mixspec.components.clone(),
                        initial_rho: final_mix_rho,
                    })
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid blended inverse link: {e}"))
                    })?,
                )
            } else {
                None
            };
            let final_sas_state = if use_sas {
                let epsilon_eff = if use_beta_logistic {
                    outer_result.rho[k]
                } else {
                    let (v, _) = sas_effective_epsilon(outer_result.rho[k]);
                    v
                };
                Some(if use_beta_logistic {
                    state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: epsilon_eff,
                        initial_log_delta: outer_result.rho[k + 1],
                    })
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!("invalid Beta-Logistic link: {e}"))
                    })?
                } else {
                    state_from_sasspec(SasLinkSpec {
                        initial_epsilon: epsilon_eff,
                        initial_log_delta: outer_result.rho[k + 1],
                    })
                    .map_err(|e| EstimationError::InvalidInput(format!("invalid SAS link: {e}")))?
                })
            } else {
                cfg.link_kind.sas_state().copied()
            };
            let aux_param_covariance = None;
            let (mix_cov, sas_cov) = if use_mixture {
                (aux_param_covariance, None)
            } else if use_sas {
                (None, aux_param_covariance)
            } else {
                (None, None)
            };
            (
                final_rho,
                final_link_coords,
                final_mix_state,
                final_sas_state,
                mix_cov,
                sas_cov,
                outer_result,
            )
        };
        // The shipped Student-t `(σ̂, ν̂)`: the trailing block of the certified
        // joint coordinate, installed on the state every post-fit quantity is
        // read from and on the family the final fit and the report carry.
        if let Some(scale) = student_t_reference_scale {
            let offset_in_link = mixture_dim + sas_dim;
            let (sigma, nu) = student_t_outer_point(
                scale,
                final_link_coords[offset_in_link],
                final_link_coords[offset_in_link + 1],
            );
            reml_state.set_student_t_state(sigma, nu)?;
            cfg.likelihood = cfg.likelihood.clone().with_student_t(sigma, nu);
        }
        if estimates_negbin_theta {
            let frozen_bits = reml_state.frozen_negbin_theta.load(Ordering::Relaxed);
            if frozen_bits == 0 {
                return Err(EstimationError::InvalidInput(
                    "estimated Negative-Binomial joint solve lost its frozen theta state"
                        .to_string(),
                ));
            }
            let theta = f64::from_bits(frozen_bits);
            if !(theta.is_finite() && theta > 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "estimated Negative-Binomial joint solve has invalid theta checkpoint {theta}"
                )));
            }

            // Re-evaluate value, rho gradient, and the fixed-theta PIRLS mode
            // through one cache generation. Both partial stationarity checks below
            // therefore refer to the identical (rho, theta, beta) point.
            reml_state.reset_outer_seed_state();
            let (joint_cost, rho_gradient) = reml_state.compute_cost_and_gradient(&final_rho)?;
            let joint_bundle = reml_state.obtain_eval_bundle(&final_rho)?;
            pirls_res = joint_bundle.pirls_result.as_ref().clone();
            pirls_res.likelihood = cfg.likelihood.clone().with_negbin_theta(theta);

            let final_eta = pirls_res.final_eta.to_owned();
            let theta_profile =
                pirls::negbin_theta_score_and_info(y_o.view(), &final_eta, w_o.view(), theta)?;
            let theta_residual = negbin_theta_stationarity_residual(theta, &theta_profile);
            // This residual is a Newton displacement in the outer log-theta
            // coordinate, so it shares the outer REML tolerance. The beta
            // PIRLS tolerance certifies a different coordinate system and must
            // not silently set the theta fixed-point threshold.
            let theta_bound = reml_tol;

            let rho_lower = rho_model_domain.0.clone();
            let rho_upper = rho_model_domain.1.clone();
            // Judged against `certificate.stationarity.bound()` just below, so
            // it must be projected against the box that certificate used
            // (#2412) — otherwise a railed coordinate's outward pull is scored
            // against a bound derived without it.
            let rail_bounds = (rho_lower, rho_upper);
            let rho_residual = crate::rho_optimizer::rail_projected_gradient_norm(
                &final_rho,
                &rho_gradient,
                Some(&rail_bounds),
            );
            let rho_bound = outer_result
                .criterion_certificate
                .as_ref()
                .map(|certificate| certificate.stationarity.bound())
                .unwrap_or(reml_tol)
                .max(f64::EPSILON);
            let rho_certificate_ok = final_rho.is_empty()
                || (outer_result.converged()
                    && outer_result
                        .criterion_certificate
                        .as_ref()
                        .is_some_and(|certificate| certificate.certifies())
                    && rho_residual.is_finite()
                    && rho_residual <= rho_bound);
            // The three coordinates of the joint (θ, ρ, β) optimum are certified
            // independently. The β coordinate must be strictly converged; a
            // near-stationary stalled checkpoint is not a completed joint fit.
            let pirls_certificate_ok = pirls_res.status.is_converged();
            let theta_certificate_ok = theta_residual.is_finite() && theta_residual <= theta_bound;

            let merit = (rho_residual / rho_bound)
                .max(theta_residual / theta_bound)
                .max(if pirls_certificate_ok {
                    0.0
                } else {
                    f64::INFINITY
                });
            let checkpoint = NegbinJointCheckpoint {
                merit,
                theta,
                rho: final_rho.clone(),
                rho_residual,
                rho_bound,
                theta_residual,
                theta_bound,
            };
            if negbin_best_checkpoint
                .as_ref()
                .is_none_or(|best| checkpoint.merit <= best.merit)
            {
                negbin_best_checkpoint = Some(checkpoint);
            }

            if rho_certificate_ok && theta_certificate_ok && pirls_certificate_ok {
                outer_result.final_value = joint_cost;
                outer_result.final_measurement =
                    Some(crate::rho_optimizer::OuterFirstOrderMeasurement::new(
                        final_rho.clone(),
                        joint_cost,
                        rho_gradient,
                    ));
                outer_result.final_grad_norm = Some(rho_residual);
                log::trace!(
                    "[OUTER] negative-binomial joint optimum certified after {} round(s): \
                     rho KKT residual {:.3e} <= {:.3e}, theta residual {:.3e} <= {:.3e}",
                    negbin_alternation_round + 1,
                    rho_residual,
                    rho_bound,
                    theta_residual,
                    theta_bound,
                );
                // #1082: the certified joint Laplace optimum decides the #784
                // correction's admission, as for the rho-only search below.
                if mixture_dim == 0
                    && sas_dim == 0
                    && reml_state.block_correction_admission_deferred()
                    && reml_state.decide_block_correction_admission(&final_rho)?
                {
                    // The corrected search continues from that optimum, its one
                    // start (#1082).
                    negbin_rho_seed = Some(final_rho.clone());
                    corrected_continuation = true;
                    continuation_curvature = outer_result.final_hessian.clone();
                    continue;
                }
                break;
            }

            if negbin_alternation_round + 1 >= reml_max_iter.max(1) {
                let best = negbin_best_checkpoint
                    .as_ref()
                    .expect("the current joint checkpoint was just recorded");
                return Err(EstimationError::NegativeBinomialAlternationDidNotConverge {
                    rounds: negbin_alternation_round + 1,
                    theta_checkpoint: best.theta,
                    rho_projected_grad_norm: best.rho_residual,
                    rho_stationarity_bound: best.rho_bound,
                    theta_score_residual: best.theta_residual,
                    theta_stationarity_bound: best.theta_bound,
                    rho_checkpoint: best.rho.to_vec(),
                });
            }

            // Exact block update: maximize the conditional NB likelihood in
            // theta at the current converged eta, then re-optimize rho with theta
            // fixed. No secant/grid extrapolation and no unreported answer cap.
            let theta_next =
                pirls::estimate_negbin_theta_from_eta(y_o.view(), &final_eta, w_o.view())?;
            log::debug!(
                "[OUTER] negative-binomial joint round {} not yet certified: \
                 rho residual {:.3e}/{:.3e}, theta residual {:.3e}/{:.3e}; \
                 updating theta {:.6e} -> {:.6e} and resuming from rho checkpoint",
                negbin_alternation_round + 1,
                rho_residual,
                rho_bound,
                theta_residual,
                theta_bound,
                theta,
                theta_next,
            );
            reml_state
                .frozen_negbin_theta
                .store(theta_next.to_bits(), Ordering::Relaxed);
            negbin_rho_seed = Some(final_rho.clone());
            reml_state.reset_outer_seed_state();
            negbin_alternation_round += 1;
            continue;
        }

        // #1082: the certified Laplace optimum decides the #784 correction's
        // admission. An admitted correction changes the criterion, so the search
        // continues from this optimum under it. A search that did not certify
        // never decides, and the certificate gate after the loop refuses it typed.
        if mixture_dim == 0
            && sas_dim == 0
            && student_t_dim == 0
            && outer_result.converged()
            && outer_result
                .criterion_certificate
                .as_ref()
                .is_some_and(|certificate| certificate.certifies())
            && reml_state.block_correction_admission_deferred()
            && reml_state.decide_block_correction_admission(&final_rho)?
        {
            // The corrected search continues from that optimum, its one start
            // (#1082).
            negbin_rho_seed = Some(final_rho.clone());
            corrected_continuation = true;
            continuation_curvature = outer_result.final_hessian.clone();
            continue;
        }

        // Reuse the Gaussian-Identity XᵀWX cache the outer loop already populated,
        // so the final accept-fit skips the streaming GEMM as well.
        //
        // When the outer loop conditioned the response (centering for #1000, scaling
        // for #1127), that cache holds `XᵀW((y−center)/scale)`; the accept-fit runs
        // on the *original* response `y_o`, so reusing the conditioned `XᵀWy` would
        // solve on the shifted/rescaled scale and report every fitted value, residual
        // and dispersion off the user's scale. Rebuild the cross-product from the
        // original response in that case — the constant `XᵀWX` block is the only part
        // the cache would have saved, a one-off cost paid only on the rare
        // large-mean / small-magnitude responses that trigger conditioning.
        let final_cache_handle = if response_center.is_some() || response_scale.is_some() {
            None
        } else {
            reml_state.gaussian_fixed_cache_if_eligible()
        };
        let pirls_res_pair = pirls::fit_model_for_fixed_rho_with_adaptive_kkt(
            LogSmoothingParamsView::new(final_rho.view())?,
            pirls::PirlsProblem {
                x: reml_state.x(),
                offset: offset_o.view(),
                y: y_o.view(),
                priorweights: w_o.view(),
                covariate_se: None,
                gaussian_fixed_cache: final_cache_handle.as_deref(),
                // The final reported fit must be exact at the converged ρ/ψ — never
                // serve the frozen-W first-step approximation here.
                glm_first_step_gram: None,
            },
            pirls::PenaltyConfig {
                canonical_penalties: reml_state.canonical_penalties(),
                reparam_invariant: None,
                p,
                coefficient_lower_bounds: None,
                linear_constraints_original: fit_linear_constraints.as_ref(),
            },
            &pirls::PirlsConfig {
                link_kind: if let Some(state) = final_mixture_state.clone() {
                    InverseLink::Mixture(state)
                } else if let Some(state) = final_sas_state {
                    if matches!(cfg.link_function(), LinkFunction::BetaLogistic) {
                        InverseLink::BetaLogistic(state)
                    } else {
                        InverseLink::Sas(state)
                    }
                } else {
                    cfg.link_kind.clone()
                },
                ..cfg.as_pirls_config()
            },
            None,
            None,
            // Final, reported fit at the REML-selected λ: refine the family's
            // estimated dispersion nuisance at the converged η. For Gamma this
            // re-estimates the shape so `dispersion_phi()` and every SE / interval
            // reflect the conditional noise, not the spread of μ (#678); for Beta
            // it drives the precision φ and the mean β̂ to their joint fixed point,
            // undoing the slope attenuation from a φ frozen at the null predictor
            // (#769). λ is fixed here, so there is no scale↔λ feedback.
            true,
            None,
        )?;
        pirls_res = pirls_res_pair.0;

        break;
    } // negative-binomial joint-coordinate loop
    // Report the outer iteration count that was MEASURED, including a genuine
    // zero. A seed that is a prior fit's terminal certificate and is still
    // stationary here is accepted without iterating
    // (`claim_prior_terminal_certificate`), so zero is a reachable,
    // meaningful outcome; flooring it to one made the reported count a claim no
    // measurement supports, and every consumer asking "did a fit happen" then
    // read a fabricated pass (#2622).
    //
    // Dropping the floor cannot let a zero-iteration fit slip past the #934
    // certificate obligation from this entry point. The `certificate_valid`
    // gate below refuses to ship at all unless the outer result converged AND
    // carries a certifying analytic certificate, for every fit with a smoothing
    // coordinate; assembly then takes its `Analytic` arm on the certificate's
    // presence, never the iteration count, and its `outer_iterations == 0`
    // fixed-λ arm stays unreachable from here.
    let iters = outer_result.iterations;

    // Map beta back to original basis
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());
    let beta_orig = conditioning.backtransform_beta(&beta_orig_internal);

    // Effective sample size for dispersion/REML accounting.
    //
    // A prior weight of exactly 0 makes a row contribute nothing to any weighted
    // cross-product (XᵀWX, XᵀWy) or to the weighted RSS (w_i·r_i² = 0), so such a
    // row is statistically equivalent to an absent row. The *only* channel left by
    // which it could still perturb the fit is an explicit observation count. To
    // keep zero-weight rows exactly equivalent to absent rows (R's `n.ok =
    // nobs − Σ[w==0]`, mgcv's dropped zero-weight observations), the dispersion
    // sample size must be the count of positive-weight rows, not the raw row
    // count. Otherwise the Gaussian scale φ̂ = weighted_rss / (n − edf) puts a
    // numerator that already excludes zero-weight rows over a denominator that
    // counts them, biasing φ̂ low and shrinking every SE (#584). The REML
    // criterion's own observation count (which drives λ selection) lives in the
    // inner-solution assembly and must apply the same positive-weight count.
    let n = w_o.iter().filter(|&&wi| wi > 0.0).count() as f64;
    let mut identity_fit_is_exact = false;
    let weighted_rss = if cfg.likelihood.spec.is_gaussian_identity() {
        let fitted = {
            let mut eta = offset_o.clone();
            eta += &x_o.matrixvectormultiply(&beta_orig);
            eta
        };
        let resid = y_o.to_owned() - &fitted;
        let raw: f64 = w_o
            .iter()
            .zip(resid.iter())
            .map(|(&wi, &ri)| wi * ri * ri)
            .sum();
        // An identity-link fit whose residual sits at the arithmetic's own
        // resolution has NOT estimated a small residual variance — it has
        // reproduced the response, and the number left over is the rounding in
        // `η = Σ_j x_ij β_j`. Reporting `σ̂ ≈ 4e-16` there hands the caller
        // standard errors and a criterion that move by orders of magnitude if
        // the rows are permuted.
        //
        // Snapping it to an exact zero is the SAME decision the formula path's
        // deterministic-Gaussian dispatch already makes one level up; making it
        // here, where the dispersion is actually estimated, is what stops the
        // two entry points from reporting different inference for identical
        // data (#2595). `weighted_residual_is_at_roundoff_floor` is the shared
        // certificate, and `|y_i| + |η_i|` is a LOWER bound on the operand scale
        // that formed each residual, so this fires conservatively — never on a
        // fit that genuinely misses.
        //
        // The term count is the full design width, matching what
        // `exact_unpenalized_gaussian_beta` counts. On a sparse row that
        // overstates the operations actually summed, widening the bound by at
        // most a factor of `p` — still `p·ε` RELATIVE to the row's own scale
        // (≈2e-13 at p = 1000), orders below any signal a fit could be missing.
        let at_floor = gam_problem::weighted_residual_is_at_roundoff_floor(
            raw,
            w_o.iter().copied(),
            y_o.iter()
                .zip(fitted.iter())
                .map(|(&yi, &fi)| yi.abs() + fi.abs()),
            beta_orig.len() + 1,
        );
        identity_fit_is_exact = at_floor;
        if at_floor { 0.0 } else { raw }
    } else {
        0.0
    };

    // Default solver policy stays on the REML/Laplace path. Joint HMC remains
    // available through explicit sampling flows, but fitting does not
    // automatically densify the Hessian or escalate into NUTS.
    let (final_rho, pirls_res) = (final_rho, pirls_res);

    // Recompute beta in the finalized basis/parameterization.
    let beta_orig_internal = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());

    let log_lambdas = final_rho.clone();
    let lambdas = LogSmoothingParamsView::new(log_lambdas.view())?.exact_exp();
    let p_dim = pirls_res.beta_transformed.len();
    let penalty_rank_total = pirls_res.reparam_result.e_transformed.nrows();
    let mp = (p_dim as f64 - penalty_rank_total as f64).max(0.0);
    let mut edf_by_block = vec![0.0; k];
    // Raw per-block penalty trace tr_kk = λ_kk·tr(H⁻¹S_kk), retained so per-term
    // EDF can be assembled as |coeff_range| − Σ tr_kk (issue #1219).
    let mut penalty_block_trace = vec![0.0; k];
    // Each block's rank-bound status beside its trace (#2901).
    let mut edf_rank_bound: Vec<crate::estimate::EdfRankBound> = Vec::new();
    let mut edf_total = 0.0;
    let mut smoothing_correction = None;
    let mut smoothing_correction_method = None;
    // The exact first-order IFT correction the corrected-EDF/AIC channel reads
    // (#946). The primary pair above IS the first-order correction, so the two
    // pairs carry the same matrix.
    let mut smoothing_correction_first_order = None;
    let mut smoothing_correction_method_first_order = None;
    let mut smoothing_correction_absence = None;
    let mut rho_covariance = None;
    let mut penalized_hessian = Array2::<f64>::zeros((0, 0));
    let mut beta_covariance = None;
    let mut factorized_standard_errors = None;
    // #3283: the factorized branch's smoothing correction, as its square-root
    // factor, and the corrected standard errors solved beside the conditional
    // ones.
    let mut smoothing_correction_factorized = None;
    let mut beta_covariance_corrected = None;
    // #2705 group A: carried from where the constrained-posterior correction is
    // APPLIED to where the corrected covariance is READ, so the refusal below
    // can say which producer's budget the negative diagonal is inside.
    let mut constrained_diagonal_uncertainty: Option<Array1<f64>> = None;
    let mut constrained_removed_variance: Option<Array1<f64>> = None;
    // The ρ̂-conditional covariance `Vb = φ·H⁻¹` BEFORE the feasible set
    // truncates it. Present only when a truncation was actually applied, i.e.
    // exactly when `beta_covariance` is no longer that matrix.
    //
    // The corrected covariance is a different estimand from the conditional one
    // and needs the untruncated matrix to build: see the composition argument at
    // the `beta_covariance_corrected` assembly below (#2705 group A).
    let mut untruncated_conditional_covariance: Option<Array2<f64>> = None;
    let mut beta_covariance_frequentist = None;
    let mut coefficient_influence = None;
    let mut weighted_gram = None;
    // Factorization of stabilized Hessian in transformed basis, reused for
    // SE computation via solve-on-demand after dispersion is determined.
    let mut edf_factor: Option<InferenceHessianFactor> = None;
    // The Tier-0 seam runs only inside the inference pass below; a fit run without
    // inference keeps this typed reason instead of an unexplained absence (#2627).
    let mut rho_posterior = gam_problem::rho_posterior::RhoPosteriorOutcome::NotComputed(
        gam_problem::rho_posterior::RhoPosteriorNotComputed::InferenceNotRequested,
    );
    let mut rho_posterior_escalation = None;
    // Hold the governor charge across every dense inference allocation in this
    // fit. A refusal selects the factorized/diagonal path before any optional
    // covariance, influence, or smoothing-correction matrix is built.
    let dense_covariance_reservation = opts
        .compute_inference
        .then(|| reserve_dense_covariance_bundle(pirls_res.reparam_result.qs.nrows()))
        .flatten();
    let factorized_inference_reservation =
        if opts.compute_inference && dense_covariance_reservation.is_none() {
            reserve_factorized_inference_state(pirls_res.reparam_result.qs.nrows())
        } else {
            None
        };

    let needs_constrained_posterior = fit_linear_constraints.is_some();
    if opts.compute_inference || needs_constrained_posterior {
        // EDF by block using stabilized H and penalty roots in transformed basis.
        let h = &pirls_res.stabilizedhessian_transformed;
        let p_dim = h.nrows();
        // Factor the exact Hessian already minted by PIRLS. This inference layer
        // is not allowed to add an unaccounted diagonal to it. When the strict
        // factor refuses, a dense H is taken on its identified subspace, the one
        // PIRLS solved it min-norm on and the criterion scored (#2901 V22).
        let factor = match h.factorize_spd() {
            Ok(factor) => InferenceHessianFactor::Strict(factor),
            Err(reason) => match h {
                gam_linalg::matrix::SymmetricMatrix::Dense(dense) => {
                    let inverse = super::identified_hessian::IdentifiedHessianInverse::from_dense(
                        dense,
                        penalty_rank_total,
                    )?;
                    log::debug!(
                        "[#2901 V22] the penalized Hessian is singular on {} of {p_dim} \
                         coefficient directions (strict factorization: {reason}); inference is \
                         taken on its identified {}-dimensional subspace",
                        p_dim.saturating_sub(inverse.rank()),
                        inverse.rank(),
                    );
                    InferenceHessianFactor::Identified(inverse)
                }
                gam_linalg::matrix::SymmetricMatrix::Sparse(_) => {
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "exact inference Hessian factorization failed: {reason}"
                    )));
                }
            },
        };
        // The per-block traces read the penalties `H` carries, `λ_k S̃_k` with
        // `S̃_k = Π S_k Π` (#2454, #2901). The raw rotated roots leak onto the
        // structural null coordinates: on `y ~ s(x) + s(x, g, bs='fs')` (n=120,
        // seed 0) the fs block's raw trace was 6.09e4 against its rank of 22, the
        // admission clamped it to 22, and `edf_total` read 7.322 where
        // `tr(H⁻¹(H − S̃))` is 9.309.
        let applied_penalties = pirls_res.reparam_result.applied_penalties().map_err(|error| {
            EstimationError::LayoutError(format!(
                "projecting the EDF penalty blocks onto the reparameterization's penalized \
                 subspace failed: {error}"
            ))
        })?;
        let mut traces = vec![0.0f64; k];
        let mut trace_bands = vec![0.0f64; k];
        let inverse_one_norm = factor.inverse_one_norm_estimate(p_dim)?;
        // #2901: `tr_k ≤ rank_k` needs `H ⪰ λ_k S̃_k`. Nonnegative working weights and
        // no Firth term give `XᵀWX ⪰ 0`, which certifies every block without a
        // factorization. An observed-information weight can be negative (non-canonical
        // links, gamma-log, NB-log) and the Firth curvature is not sign-definite, so
        // otherwise each block is certified from the inertia of `H − λ_k S̃_k` shifted by
        // its rounding band, factored dense or sparse as `H` is stored.
        let structural_rank_bound = !cfg.firth_bias_reduction
            && pirls_res
                .finalweights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0);
        let mut rank_bounds: Vec<crate::estimate::EdfRankBound> = Vec::with_capacity(k);
        for (kk, cp) in applied_penalties.iter().enumerate() {
            // Build the p × rank RHS with nonzeros only in [start..end] rows.
            let r = &cp.col_range;
            let rank = cp.rank();
            let mut rhs = Array2::<f64>::zeros((p_dim, rank));
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    rhs[[r.start + row, col]] = cp.root[[col, row]];
                }
            }
            let sol = factor.certified_solve(h, &rhs, "penalty-block EDF trace")?;
            // Frobenius inner product: only the block rows of rhs are nonzero.
            let mut frob = 0.0f64;
            for col in 0..rank {
                for row in 0..cp.block_dim() {
                    frob += sol[[r.start + row, col]] * rhs[[r.start + row, col]];
                }
            }
            // The per-block penalty trace `tr_kk = λ_kk·tr(H⁻¹ S_kk)` is admitted within
            // the rounding band of this solve (#2901). A non-finite one (a `+∞`
            // overflow of a ceiling-λ block, gam#1379) refuses by name; outside
            // `[−band, rank + band]` it refuses only on a block whose rank bound is
            // certified, and an uncertified block publishes it unclamped.
            let solved_rhs = factor.solved_rhs(&rhs);
            let residual = h.dot_matrix(&sol) - &solved_rhs;
            trace_bands[kk] = gam_linalg::roundoff::solved_penalty_trace_band(
                lambdas[kk],
                solved_rhs.view(),
                sol.view(),
                residual.view(),
                h.max_abs_entry(),
                inverse_one_norm,
            )
            .map_err(EstimationError::InvalidInput)?;
            rank_bounds.push(if structural_rank_bound {
                crate::estimate::EdfRankBound::Certified(
                    crate::estimate::EdfRankCertificate::Structural,
                )
            } else {
                let scaled_penalty_block = cp.root.t().dot(&cp.root) * lambdas[kk];
                let governor = gam_runtime::resource::MemoryGovernor::global();
                match h {
                    gam_linalg::matrix::SymmetricMatrix::Dense(dense) => {
                        crate::estimate::numerical_rank_bound(
                            dense.view(),
                            scaled_penalty_block.view(),
                            r.start,
                            governor,
                        )?
                    }
                    gam_linalg::matrix::SymmetricMatrix::Sparse(sparse) => {
                        crate::estimate::sparse_numerical_rank_bound(
                            sparse,
                            scaled_penalty_block.view(),
                            r.start,
                            governor,
                        )?
                    }
                }
            });
            traces[kk] = lambdas[kk] * frob;
        }
        let block_ranks: Vec<usize> = applied_penalties.iter().map(|cp| cp.rank()).collect();
        let (edf_coefficients, edf_penalty_nullity) = factor.edf_dimensions(p_dim, mp);
        let bundle = penalized_edf_bundle_within_bands(
            &traces,
            &trace_bands,
            &rank_bounds,
            &block_ranks,
            edf_coefficients,
            edf_penalty_nullity,
        )?;
        edf_total = bundle.edf_total;
        penalty_block_trace.clone_from(&bundle.penalty_block_trace);
        edf_by_block.clone_from(&bundle.edf_by_block);
        edf_rank_bound.clone_from(&bundle.rank_bound);
        traces.clone_from(&bundle.penalty_block_trace);

        // Reconcile the EDF accounting with the influence matrix F = H⁻¹X'WX.
        //
        // The authoritative model definition of EDF is the influence-matrix
        // trace; the per-term EDF (`FitResult::per_term_edf`) reads `tr(F)` over
        // each block. Recompute the per-block penalty traces from the SAME exact
        // inverse `F` uses, so
        // `edf_total = p − Σ tr_kk = tr(F)`, `Σ edf_by_block = edf_total`, and the
        // total can never fall below a single term's own EDF. Done before the
        // dispersion `σ̂² = RSS/(n − edf_total)` is formed so it, too, uses the
        // honest effective d.f. (the trace-channel collapse otherwise biased
        // σ̂² high → inflated SEs on the same seeds).
        //
        // Per-block traces `tr_kk = λ_kk·tr(H⁻¹ S_kk)` are basis-invariant; map
        // each canonical block's penalty root into the original coefficient basis
        // (`root_orig = Qs · root_t`) and contract against the original-basis
        // inverse. Gated by the SAME resource-policy check as the dense
        // covariance bundle below, so this reconciliation and the influence
        // matrix `F` are formed in exactly the same regime, from the same
        // `map_hessian_to_original_basis(&pirls_res)` matrix, through the same
        // strict-Cholesky solve route; beyond the policy budget both switch off
        // together and the trace-channel value stands.
        {
            let p_orig = pirls_res.reparam_result.qs.nrows();
            if dense_covariance_reservation.is_some() {
                let h_orig = map_hessian_to_original_basis(&pirls_res)?;
                // Solve against the strict Cholesky rather than contracting
                // against a materialized `H⁻¹`. Both carry the same *backward*
                // error certificate, but a product `H⁻¹·R` inherits a FORWARD
                // error of order `cond(H)` times it, while `H·sol = R` does not
                // (#2668 measured `cond(H) = 2.099e8` on an ordinary
                // `y ~ s(x)` fit, which amplified the sibling influence matrix
                // by 3.9%). Both consumers of `H⁻¹` — these block traces and
                // the influence matrix below — now take the solve route, so
                // neither is amplified and the two stay consistent. That
                // consistency is what `influence_trace_matches_conditional_edf`
                // pins: `edf_total` comes from THESE traces while `tr(F)` comes
                // from the influence matrix, so moving only one would make them
                // disagree. Failure is not a request to silently change rank or
                // add a diagonal perturbation.
                let h_factor = OriginalBasisHessianFactor::new(
                    &h_orig,
                    Some(&factor),
                    &pirls_res.reparam_result.qs,
                    "edf reconciliation",
                )
                .map_err(|error| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "EDF reconciliation requires an exact SPD Hessian factorization: {error}"
                    ))
                })?;
                {
                    let qs = &pirls_res.reparam_result.qs;
                    let p_t = qs.ncols();
                    let mut traces_f = vec![0.0f64; k];
                    let mut trace_bands_f = vec![0.0f64; k];
                    let inverse_one_norm_f =
                        h_factor.inverse_one_norm_estimate(p_orig).map_err(|error| {
                            EstimationError::RemlOptimizationFailed(format!(
                                "EDF reconciliation trace band solve did not certify: {error}"
                            ))
                        })?;
                    for (kk, cp) in applied_penalties.iter().enumerate() {
                        if kk >= lambdas.len() {
                            continue;
                        }
                        let r = &cp.col_range;
                        let rank = cp.rank();
                        let mut root_t = Array2::<f64>::zeros((p_t, rank));
                        for col in 0..rank {
                            for row in 0..cp.block_dim() {
                                root_t[[r.start + row, col]] = cp.root[[col, row]];
                            }
                        }
                        // S_kk = Rᵀ R; λ_kk·tr(H⁻¹ S_kk) = λ_kk·Σ_col (R_col)ᵀ H⁻¹ R_col.
                        let root_orig = qs.dot(&root_t); // p_orig × rank
                        let sol = h_factor.solve_matrix(&root_orig).map_err(|error| {
                            EstimationError::RemlOptimizationFailed(format!(
                                "EDF reconciliation block solve did not certify: {error}"
                            ))
                        })?; // H⁻¹ R
                        let mut frob = 0.0f64;
                        for col in 0..rank {
                            for row in 0..p_orig {
                                frob += sol[[row, col]] * root_orig[[row, col]];
                            }
                        }
                        // Admitted by the shared accounting within this solve's own
                        // rounding band, exactly as the trace-channel path above
                        // (#2470, #2901).
                        let solved_rhs = h_factor.solved_rhs(&root_orig);
                        let residual = h_orig.dot(&sol) - &solved_rhs;
                        let h_orig_max_abs =
                            h_orig.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
                        trace_bands_f[kk] = gam_linalg::roundoff::solved_penalty_trace_band(
                            lambdas[kk],
                            solved_rhs.view(),
                            sol.view(),
                            residual.view(),
                            h_orig_max_abs,
                            inverse_one_norm_f,
                        )
                        .map_err(EstimationError::InvalidInput)?;
                        traces_f[kk] = lambdas[kk] * frob;
                    }
                    let block_ranks_f: Vec<usize> =
                        applied_penalties.iter().map(|cp| cp.rank()).collect();
                    let (edf_coefficients_f, edf_penalty_nullity_f) =
                        factor.edf_dimensions(p_orig, mp);
                    // The rank bounds of the trace channel carry over: rotating by
                    // `Qs` leaves the spectrum of `H − λ_k S̃_k` unchanged.
                    let bundle_f = penalized_edf_bundle_within_bands(
                        &traces_f,
                        &trace_bands_f,
                        &rank_bounds,
                        &block_ranks_f,
                        edf_coefficients_f,
                        edf_penalty_nullity_f,
                    )?;
                    edf_total = bundle_f.edf_total;
                    penalty_block_trace.clone_from(&bundle_f.penalty_block_trace);
                    edf_by_block.clone_from(&bundle_f.edf_by_block);
                    edf_rank_bound.clone_from(&bundle_f.rank_bound);
                }
            }
        }

        // Preserve the factorization for solve-on-demand SE and covariance
        // computation below, after dispersion has been determined.
        edf_factor = Some(factor);
    }

    // Persist residual-based scale for Gaussian identity models.
    // Contract: residual standard deviation sigma, not variance.
    //
    // Gaussian REML scale: σ̂² = RSS / (n − edf_total), matching mgcv's gam.scale.
    // Using the null-space dim (mp = p − rank(Σ_k S_k)) here was wrong: mp is the
    // minimum possible edf (all smooths fully penalized to their null space), so
    // n − mp ≥ n − edf_total, and σ̂² was systematically biased low whenever any
    // smooth/random-effect spent real edf. edf_total ∈ [mp, p_dim] is the effective
    // df computed just above from tr(λ_k · H⁻¹ S_k), and is exactly the residual
    // df mgcv uses. An inference-off unconstrained fit keeps the MLE RSS/n path;
    // a constrained fit computes EDF regardless because its posterior mean and
    // covariance scale are part of the fitted estimand, not optional inference.
    let resolved_likelihood_scale = pirls_res
        .likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let profiled_gaussian_standard_deviation = match resolved_likelihood_scale {
        gam_problem::ResolvedLikelihoodScale::ProfiledGaussian => {
            let denom = if opts.compute_inference || needs_constrained_posterior {
                n - edf_total
            } else {
                n
            };
            if !(denom.is_finite() && denom > 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "profiled Gaussian residual degrees of freedom must be finite and positive, got {denom:?}"
                )));
            }
            if !(weighted_rss.is_finite() && weighted_rss >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "profiled Gaussian weighted RSS must be finite and non-negative, got {weighted_rss:?}"
                )));
            }
            let variance = weighted_rss / denom;
            if !variance.is_finite() {
                return Err(EstimationError::InvalidInput(format!(
                    "profiled Gaussian residual variance is not representable: {weighted_rss:?}/{denom:?}"
                )));
            }
            Some(variance.sqrt())
        }
        _ => None,
    };
    let dispersion =
        dispersion_from_likelihood(&pirls_res.likelihood, profiled_gaussian_standard_deviation)?;
    // Persist the square root of the resolved response dispersion for every
    // scalar-scale family. It is never an overloaded Gamma shape or an inert
    // unit placeholder; family-specific inference consumes the typed metadata.
    let standard_deviation = dispersion.phi().sqrt();

    // Explicit dispersion contract for coefficient covariance matrices:
    // Vb = H⁻¹ · cov_scale, where the stored penalized Hessian is always
    // H = XᵀWX + S_λ with the penalty added UNSCALED. The multiplier therefore
    // restores ONLY the dispersion the working weight W does not already carry:
    //
    //   * Profiled Gaussian keeps W scale-free (W = priorweights), so the data
    //     term has unit implicit scale and Vb = H⁻¹·σ̂².
    //   * Every other family folds its reciprocal dispersion / full Fisher
    //     information into W (Gamma W = prior/φ, Tweedie W = prior·μ^{2−p}/φ,
    //     Beta/NB the complete fixed-scale Fisher info, Poisson/Binomial φ ≡ 1),
    //     so H already equals the true penalized Hessian (identical to mgcv's
    //     XᵀW_sfX/φ + S_λ) and Vb = H⁻¹ with NO extra dispersion factor. A
    //     post-hoc ×φ here would double-count the dispersion and shrink every SE
    //     by √φ (= 1/√shape for Gamma); see #679.
    //
    // The single source of truth for this invariant is
    // `GlmLikelihoodSpec::coefficient_covariance_scale`; the response-level
    // observation noise used by predictive intervals stays in `dispersion`
    // above (a deliberately distinct quantity, e.g. 1/shape for Gamma).
    let cov_scale = pirls_res
        .likelihood
        .coefficient_covariance_scale(standard_deviation * standard_deviation)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let zero_covariance_boundary = dispersion.is_zero_estimate()
        && matches!(
            &pirls_res.likelihood.spec.response,
            ResponseFamily::Gaussian
        )
        && matches!(
            &pirls_res.likelihood.scale,
            LikelihoodScaleMetadata::ProfiledGaussian
        );
    if !cov_scale.is_finite()
        || cov_scale < 0.0
        || (cov_scale == 0.0 && !zero_covariance_boundary)
        || (zero_covariance_boundary && cov_scale != 0.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "coefficient covariance scale {cov_scale:?} is inconsistent with dispersion {dispersion:?}"
        )));
    }

    // A fit carrying inequality constraints reports the mean of its truncated
    // Laplace posterior, never the boundary MAP. Build the posterior identity
    // in the transformed PIRLS frame, where the accepted Hessian factor,
    // constraint rows, score, and mode are exactly aligned, then lift its two
    // locations and low-rank covariance factor through Qs together.
    //
    // This work is independent of `compute_inference`: requesting standard
    // errors cannot change the fitted coefficient vector. It needs q+1 solves,
    // not a dense p×p inverse.
    let constrained_posterior = match pirls_res.linear_constraints_transformed.as_ref() {
        Some(constraints) => {
            let factor = edf_factor.as_ref().ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "constrained posterior geometry requires the accepted Hessian factor"
                        .to_string(),
                )
            })?;
            let h = &pirls_res.stabilizedhessian_transformed;
            let constraint_rhs = constraints.a.t().to_owned();
            let sigma_at_unscaled = factor.certified_solve(
                h,
                &constraint_rhs,
                "constrained posterior normal geometry",
            )?;
            let sigma_at = sigma_at_unscaled * cov_scale;

            let score_t = &pirls_res.penalized_gradient_transformed;
            let center_step_unscaled = factor.certified_vector_solve(
                h,
                score_t,
                "constrained posterior unconstrained centre",
            )?;
            // `penalized_gradient_transformed` and H share the solver's
            // objective scale. For profiled Gaussian both omit the common
            // 1/φ factor, which cancels in H⁻¹g; multiplying this displacement
            // by `cov_scale=φ` would move the Gaussian centre by an extra φ.
            let center_t = pirls_res.beta_transformed.as_ref() - &center_step_unscaled;
            let mut correction =
                crate::constrained_posterior::constrained_posterior_correction(
                    sigma_at.view(),
                    &center_t,
                    constraints,
                )
                .map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "constrained posterior moments failed: {reason}"
                    ))
                })?;
            let qs = &pirls_res.reparam_result.qs;
            if let Some(value) = correction.as_mut() {
                value.lift = qs.dot(&value.lift);
            }
            let constraints_internal = fit_linear_constraints.as_ref().ok_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "PIRLS exported transformed inequalities without their pre-reparameterization geometry"
                        .to_string(),
                )
            })?;
            Some(crate::constrained_posterior::ConstrainedPosteriorGeometry::with_moments(
                constraints_internal.clone(),
                beta_orig_internal.clone(),
                qs.dot(&center_t),
                correction,
            ))
        }
        None if needs_constrained_posterior => {
            return Err(EstimationError::RemlOptimizationFailed(
                "fit accepted linear inequalities but PIRLS did not export their transformed geometry"
                    .to_string(),
            ));
        }
        None => None,
    };
    let reported_beta_orig_internal = match constrained_posterior.as_ref() {
        Some(posterior) => posterior.posterior_mean().map_err(|reason| {
            EstimationError::RemlOptimizationFailed(format!(
                "constrained posterior mean is unavailable: {reason}"
            ))
        })?,
        None => beta_orig_internal.clone(),
    };

    // Re-install the exact rho point and inner state that will be shipped, and
    // verify it IS the certified optimum. Seeds and nuisance refinements may
    // initialize work, but they can never promote a different point under the
    // optimizer's old certificate.
    //
    // The identity check is BITWISE on ρ, not a re-judged gradient norm: the
    // retained certificate is the analytic stationarity authority minted at
    // `outer_result.rho` by the full certification machinery (noise-floor
    // widenings, flatness probes, asymptote rails). In the deep-smoothing
    // regime the analytic gradient is a noise instrument (|Pg| redraws across
    // evaluations of the SAME point — the reproducibility floor exists because
    // of it), so re-drawing it once here and comparing against the certified
    // band refuses honest noise-band certificates with coin-flip probability
    // while adding nothing to point-identity (which bit equality decides
    // exactly). The evaluation itself is kept: it installs the inner state at
    // the shipped point and supplies the shipped value/gradient fields.
    let (final_value, finalgrad, finalgrad_norm) = if final_rho.is_empty() {
        (outer_result.final_value, Array1::zeros(0), 0.0)
    } else {
        let (value, gradient) = reml_state.compute_cost_and_gradient(&final_rho)?;
        let lower = rho_model_domain.0.clone();
        let upper = rho_model_domain.1.clone();
        // Shipped as the result's `final_grad_norm` and reported in the
        // refusal below, so it uses the certificate's rail-relaxed box (#2412)
        // -- the same projection the certified |Pg| was measured with, even
        // though this gate never weighs one against the other.
        let bounds = (lower, upper);
        let projected =
            crate::rho_optimizer::rail_projected_gradient_norm(&final_rho, &gradient, Some(&bounds));
        (value, gradient, projected)
    };
    let shipped_point_is_certified = shipped_joint_point_is_certified(
        &final_rho,
        &final_link_coords,
        &outer_result.rho,
    );
    let certificate_valid = final_rho.is_empty()
        || (outer_result.converged()
            && outer_result
                .criterion_certificate
                .as_ref()
                .is_some_and(|certificate| certificate.certifies())
            && shipped_point_is_certified
            && finalgrad_norm.is_finite());
    if !certificate_valid {
        return Err(EstimationError::RemlDidNotConverge {
            context: "standard REML final shipped point".to_string(),
            reason: format!(
                "post-fit certificate identity check failed: shipped point {:?} vs \
                 certified point {:?} (converged={}, certifies={}, |Pg| at shipped point {:.3e})",
                final_rho
                    .iter()
                    .chain(final_link_coords.iter())
                    .copied()
                    .collect::<Vec<_>>(),
                outer_result.rho.to_vec(),
                outer_result.converged(),
                outer_result
                    .criterion_certificate
                    .as_ref()
                    .is_some_and(|certificate| certificate.certifies()),
                finalgrad_norm,
            ),
            iterations: outer_result.iterations,
            final_value,
            projected_grad_norm: finalgrad_norm.is_finite().then_some(finalgrad_norm),
            // This gate deliberately does NOT weigh the re-drawn gradient
            // against the certified band — see the comment above the
            // re-evaluation: in the deep-smoothing regime that comparison
            // refuses honest noise-band certificates by coin flip. What it
            // checks is bitwise point identity plus the certificate's own
            // verdict. Printing a bound beside "against stationarity bound"
            // therefore named a comparison this route does not make
            // (#2458/#2465).
            stationarity_standard: gam_problem::StationarityStandard::NoComparison,
            rho_checkpoint: final_rho.to_vec(),
        });
    }
    outer_result.final_value = final_value;
    outer_result.final_measurement = Some(crate::rho_optimizer::OuterFirstOrderMeasurement::new(
        final_rho.clone(),
        final_value,
        finalgrad,
    ));
    outer_result.final_grad_norm = Some(finalgrad_norm);
    let outer_converged = true;

    // #2901 V22: the criterion priced `½log|H|₊` on H's identified subspace, and
    // `½log|H|₊` jumps by `½ln σ` where a direction crosses the rounding band.
    // The derivative certificate at ρ̂ describes a smooth criterion only if that
    // rank is the same over the certificate's own Newton step. The rank and the
    // unidentified directions are published beside that verdict. A Firth fit
    // prices a structural rank and a sparse Hessian a strict factorization, so
    // neither has a band to cross and neither publishes a band-identified
    // subspace.
    //
    // The rank certified is the one the criterion's builder published at ρ̂, on
    // the eigenpairs it priced, judged at the PIRLS state it priced them at
    // (#2959 D1). A re-rank of the shipped fit's stabilized Hessian certified a
    // rank the criterion never used wherever the builder priced another: the root
    // pricing a mode the assembled band masks. The step bounds are taken on the
    // assembled matrix, so a root-priced rank they certify is certified, and one
    // they refuse is published as not evaluated: the root can price modes below
    // the assembled rounding band, which those bounds cannot resolve.
    let identified_subspace = match &pirls_res.stabilizedhessian_transformed {
        gam_linalg::matrix::SymmetricMatrix::Dense(dense) if !cfg.firth_bias_reduction => 'subspace: {
            let penalty_rank = pirls_res.reparam_result.e_transformed.nrows();
            let qs = &pirls_res.reparam_result.qs;
            let criterion = if final_rho.is_empty() {
                None
            } else {
                Some(reml_state.criterion_rank_decision_at(&final_rho)?)
            };
            let (spectrum, priced_pirls, decision_reason, root_priced) = match criterion
                .as_ref()
                .and_then(|(bundle, decision)| decision.as_ref().map(|decision| (bundle, decision)))
            {
                Some((bundle, decision)) => {
                    let (hessian, eigenvectors) = match decision.frame {
                        super::reml::CriterionFrame::Transformed => (
                            decision.hessian.as_ref().clone(),
                            decision.operator.eigenvectors.clone(),
                        ),
                        super::reml::CriterionFrame::Original => (
                            qs.t().dot(decision.hessian.as_ref()).dot(qs),
                            qs.t().dot(&decision.operator.eigenvectors),
                        ),
                    };
                    let root_priced = match decision.predicate {
                        super::reml::CriterionRankPredicate::IdentifiedSubspace => false,
                        super::reml::CriterionRankPredicate::RootScale => true,
                        // Only a Firth term supplies a structural rank, and a Firth
                        // fit publishes no band-identified subspace.
                        super::reml::CriterionRankPredicate::StructuralRank => break 'subspace None,
                    };
                    (
                        super::identified_hessian::FittedHessianSpectrum::from_eigensystem(
                            hessian,
                            decision.operator.raw_eigenvalues.clone(),
                            eigenvectors,
                            decision.penalty_rank,
                            decision.priced_rank(),
                        ),
                        bundle.pirls_result.as_ref(),
                        None,
                        root_priced,
                    )
                }
                None => (
                    super::identified_hessian::FittedHessianSpectrum::of(dense, penalty_rank)?,
                    &pirls_res,
                    criterion.as_ref().map(|_| {
                        crate::model_types::RankConstancyNotEvaluated::NoPublishedRankDecision
                    }),
                    false,
                ),
            };
            let rows = reml_state.x().nrows();
            let not_evaluated = if final_rho.is_empty() {
                Some(crate::model_types::RankConstancyNotEvaluated::NoSmoothingParameters)
            } else if !final_link_coords.is_empty() {
                Some(crate::model_types::RankConstancyNotEvaluated::LinkCoordinates)
            } else if reml_state.active_constraint_free_basis(priced_pirls).is_some() {
                Some(crate::model_types::RankConstancyNotEvaluated::ActiveConstraintFace)
            } else if priced_pirls.finalweights.len() != rows
                || (priced_pirls.solve_c_nontrivial
                    && (priced_pirls.derivatives_unsupported
                        || priced_pirls.solve_c_array.len() != rows))
            {
                Some(crate::model_types::RankConstancyNotEvaluated::NoRowCurvatureDerivative)
            } else {
                decision_reason
            };
            let rank_constancy = match (
                not_evaluated,
                outer_result.final_hessian.as_ref(),
                outer_result.final_gradient(),
            ) {
                (None, Some(hessian_rho), Some(gradient))
                    if hessian_rho.dim() == (gradient.len(), gradient.len()) =>
                {
                    let railed: Vec<usize> = outer_result
                        .criterion_certificate
                        .as_ref()
                        .map(|certificate| {
                            certificate
                                .lambdas_railed
                                .iter()
                                .copied()
                                .chain(
                                    certificate
                                        .stationarity
                                        .rails()
                                        .iter()
                                        .map(|rail| rail.index),
                                )
                                .collect()
                        })
                        .unwrap_or_default();
                    match super::identified_hessian::certify_fitted_identified_rank(
                        priced_pirls,
                        &spectrum,
                        &lambdas,
                        reml_state.x(),
                        super::identified_hessian::OuterCertificatePoint {
                            hessian_rho,
                            gradient,
                            railed: &railed,
                            rho: &final_rho,
                            lower: &rho_model_domain.0,
                            upper: &rho_model_domain.1,
                        },
                    ) {
                        Ok((certificate, step_radius)) => {
                            log::debug!(
                                "[#2901 V22] identified rank {} of {} is certified constant over \
                                 the certificate's Newton step {step_radius:.3e}: smallest \
                                 identified eigenvalue {:.3e}, rounding band {:.3e}",
                                certificate.rank,
                                dense.nrows(),
                                certificate.smallest_identified,
                                certificate.band,
                            );
                            crate::model_types::IdentifiedRankConstancy::Certified {
                                step_radius,
                                smallest_identified: certificate.smallest_identified,
                                largest_unidentified: certificate.largest_unidentified,
                                band: certificate.band,
                            }
                        }
                        // The step bounds are taken on the assembled matrix, whose
                        // eigensolve resolves eigenvalues only to its own rounding
                        // band. A root-priced mode can sit below that band, so their
                        // refusal of a root-priced rank is not evidence that the rank
                        // moves; a certification by them is still a certification.
                        Err(EstimationError::IdentifiedRankNotLocallyConstant { .. })
                            if root_priced =>
                        {
                            let reason =
                                crate::model_types::RankConstancyNotEvaluated::RootScalePricedRank;
                            log::debug!(
                                "[#2959 D1] root-priced rank {} of {}; its constancy over the \
                                 certificate's step was not evaluated: {}",
                                spectrum.rank(),
                                dense.nrows(),
                                reason.description(),
                            );
                            crate::model_types::IdentifiedRankConstancy::NotEvaluated { reason }
                        }
                        Err(error) => return Err(error),
                    }
                }
                (reason, ..) => {
                    let reason = reason
                        .unwrap_or(crate::model_types::RankConstancyNotEvaluated::NoOuterHessian);
                    log::debug!(
                        "[#2901 V22] identified rank {} of {}; its constancy over the \
                         certificate's step was not evaluated: {}",
                        spectrum.rank(),
                        dense.nrows(),
                        reason.description(),
                    );
                    crate::model_types::IdentifiedRankConstancy::NotEvaluated { reason }
                }
            };
            Some(crate::model_types::IdentifiedCoefficientSubspace {
                rank: spectrum.rank(),
                unidentified_basis: qs.dot(&spectrum.unidentified_basis()),
                rank_constancy,
            })
        }
        _ => None,
    };

    if opts.compute_inference || needs_constrained_posterior {
        penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
    }
    if opts.compute_inference {
        let qs = &pirls_res.reparam_result.qs;

        // Auto-select covariance strategy from the runtime resource policy.
        //
        // When the WHOLE simultaneous dense bundle fits the policy's
        // process-wide reservation (`reserve_dense_covariance_bundle`) we can
        // afford the full p×p inverse: O(p³) compute, O(p²) memory. The full
        // matrix is needed for the frequentist covariance Ve = H⁻¹ X'WX H⁻¹ φ,
        // the influence matrix F = H⁻¹ X'WX, and the smoothing-parameter
        // correction.
        //
        // For large models we use solve-on-demand against the Cholesky factor
        // already computed for EDF traces above. We solve H_t Z_t = Qs^T in
        // policy-sized column chunks, then extract the diagonal of
        // Qs · Z_t = H_orig⁻¹ to get exact posterior SEs without ever
        // materialising the p×p inverse. Prediction bands continue to work via
        // the factorised-Hessian path in PredictionCovarianceBackend::Factorized.

        // Attempt the full inverse when the bundle fits the policy budget.
        //
        // ONE strict Cholesky serves the whole bundle. `H⁻¹` itself is still
        // materialized — it IS the posterior covariance `Vb = φ·H⁻¹`, an
        // estimand rather than an intermediate — but every DERIVED quantity
        // (`F`, `Ve`, the bias-correction Jacobian) is obtained by solving
        // against this factor instead of by multiplying against `H⁻¹`. The
        // certificate on `H⁻¹` is a *backward* error bound, so a product
        // `H⁻¹·M` carries a forward error of order `cond(H)` times it; a solve
        // `H·X = M` carries the backward error only. #2668 measured
        // `cond(H) = 2.099e8` on an ordinary Gaussian `y ~ s(x)` fit, where
        // that amplification put `H·F` a measured 3.9% away from the `X'WX`
        // it is definitionally equal to.
        let posterior_factor = if dense_covariance_reservation.is_some() {
            Some(
                OriginalBasisHessianFactor::new(
                    &penalized_hessian,
                    edf_factor.as_ref(),
                    qs,
                    "posterior covariance",
                )
                .map_err(|error| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "posterior covariance requires an exact SPD Hessian factorization: {error}"
                    ))
                })?,
            )
        } else {
            None
        };
        let beta_covariance_unscaled: Option<Array2<f64>> = match posterior_factor.as_ref() {
            Some(factor) => Some(factor.inverse().map_err(|error| {
                EstimationError::RemlOptimizationFailed(format!(
                    "posterior covariance requires an exact SPD Hessian inverse: {error}"
                ))
            })?),
            None => None,
        };

        if let (Some(h_inv), Some(posterior_factor)) =
            (beta_covariance_unscaled.as_ref(), posterior_factor.as_ref())
        {
            // Full inverse available: wrap as phi-scaled covariance, compute
            // frequentist quantities, and form the smoothing correction.
            let mut posterior_covariance = scaled_covariance(h_inv.clone(), cov_scale);
            let constrained_correction = constrained_posterior
                .as_ref()
                .map(crate::constrained_posterior::ConstrainedPosteriorGeometry::correction)
                .transpose()
                .map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "constrained posterior covariance correction is unavailable: {reason}"
                    ))
                })?
                .flatten();
            if let Some(correction) = constrained_correction {
                constrained_removed_variance = Some(correction.removed_variance_diagonal());
                constrained_diagonal_uncertainty = Some(correction.diagonal_uncertainty());
                // `Σ_π` is read for its DIAGONAL immediately below, and on a
                // pinned coordinate the subtractive form `Σ − GΔGᵀ` is a
                // cancellation whose residue carries a sign — measured at
                // `−3.09e-15` on `y ~ s(x, shape=convex)`, which the strict
                // `variance > 0` gate then refuses. Assemble the identical
                // quantity as `(P L)(P L)ᵀ + (G L_C)(G L_C)ᵀ`, where the
                // diagonal is a sum of squares (#2705 group A).
                let constraints = constrained_posterior
                    .as_ref()
                    .map(|geometry| &geometry.constraints)
                    .ok_or_else(|| {
                        EstimationError::RemlOptimizationFailed(
                            "a constrained posterior correction exists without the geometry that \
                             owns its constraint system"
                                .to_string(),
                        )
                    })?;
                let truncated = correction
                    .truncated_covariance_psd(&posterior_covariance, constraints)
                    .map_err(|reason| {
                        EstimationError::RemlOptimizationFailed(format!(
                            "constrained posterior covariance could not be assembled in its \
                             positive-semidefinite form: {reason}"
                        ))
                    })?;
                untruncated_conditional_covariance = Some(posterior_covariance);
                posterior_covariance = truncated;
            }
            beta_covariance = Some(gam_problem::dispersion_cov::PhiScaledCovariance::wrap(
                posterior_covariance,
            ));

            // Frequentist covariance Ve = F H⁻¹ φ and influence matrix F = H⁻¹ X'WX.
            // Both require the full unscaled inverse; computed in original basis.
            //
            // The canonical penalties live in the TRANSFORMED frame, while
            // `h_inv` is the ORIGINAL-basis inverse — assemble S(λ) in the
            // transformed frame and map it through the same congruence as the
            // Hessian (`S_orig = Qs·S_t·Qsᵀ`, issue #1027). Pairing the
            // transformed-frame S directly with the original-frame inverse made
            // `F` (and everything reconstructed from it) frame-inconsistent.
            // `S(λ)` is assembled from the penalties `H` carries, `λ_k S̃_k`
            // (#2454, #2901): the raw rotated roots made `H − S` indefinite on
            // `y ~ s(x) + s(x, g, bs='fs')`, spectrum [−1.12e5, 7.94e4] against
            // [−1.2e-15, 121] for the engine's `S̃`.
            let p_t = qs.ncols();
            let applied_penalties = pirls_res.reparam_result.applied_penalties().map_err(|error| {
                EstimationError::LayoutError(format!(
                    "projecting the influence-matrix penalty blocks onto the \
                     reparameterization's penalized subspace failed: {error}"
                ))
            })?;
            let mut s_t = Array2::<f64>::zeros((p_t, p_t));
            for (kk, cp) in applied_penalties.iter().enumerate() {
                if kk >= lambdas.len() {
                    continue;
                }
                let r = &cp.col_range;
                let local = cp.local_ref();
                let lam = lambdas[kk];
                for i in 0..cp.block_dim() {
                    for j in 0..cp.block_dim() {
                        s_t[[r.start + i, r.start + j]] += lam * local[[i, j]];
                    }
                }
            }
            let mut s_mat = qs.dot(&s_t).dot(&qs.t());
            gam_linalg::matrix::symmetrize_in_place(&mut s_mat);

            // X'WX = H − S(λ) in the original basis — the genuine PSD weighted
            // Gram, reconstructed from the same `penalized_hessian` and `s_mat`
            // that define `F = H⁻¹X'WX` (issue #1027). Stored directly so the
            // WPS corrected-EDF correction never has to recover it from an
            // inconsistent `H·F` product.
            let mut xwx = &penalized_hessian - &s_mat;
            // `H·F = H(I − H⁻¹S) = H − S` is the RAW difference, but the gram
            // stored below is `sym(H − S)`. When `H` and `S` are both symmetric
            // those coincide and `H·F = X'WX` exactly; when they are not, this
            // `symmetrize_in_place` silently absorbs the difference and the
            // identity fails downstream with no way to see which operand caused
            // it (#2668 measures a 3.9% gap on `y ~ s(x)` and could only rule
            // out `H`: `max|H − Hᵀ| = 0.000e0` there). Report both asymmetries
            // at the one place that holds `s_mat`. `debug!` so it costs nothing
            // without a backend installed, and O(p²) beside the O(p³) work above.
            if log::log_enabled!(log::Level::Trace) {
                let asym = |m: &ndarray::Array2<f64>| {
                    let mut worst = 0.0_f64;
                    for i in 0..m.nrows() {
                        for j in 0..m.ncols() {
                            worst = worst.max((m[[i, j]] - m[[j, i]]).abs());
                        }
                    }
                    worst
                };
                let scale = xwx.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
                log::trace!(
                    "[WPS-GRAM #2668] max|H-H^T|={:.3e} max|S-S^T|={:.3e} \
                     max|H-S|={:.3e} (the stored gram is symmetrize(H-S); a \
                     non-zero S asymmetry is absorbed here and surfaces as \
                     H*F != X'WX)",
                    asym(&penalized_hessian),
                    asym(&s_mat),
                    scale
                );
            }
            gam_linalg::matrix::symmetrize_in_place(&mut xwx);

            // Influence matrix F = H⁻¹·X'WX, obtained by SOLVING `H·F = X'WX`
            // against the factor above rather than by forming `I − H⁻¹·S`.
            // The two are equal in real arithmetic; in floating point the solve
            // makes `H·F = X'WX` hold to the factorization's backward error
            // *by construction*, which is exactly the identity
            // `penalized_hessian_times_influence_equals_weighted_gram` asserts
            // and which the explicit-inverse form violated by 3.9% at
            // `cond(H) = 2.099e8` (#2668). Note the right-hand side is the
            // stored, symmetrized `xwx`, so the identity is asserted against
            // the matrix that is actually persisted.
            //
            // `F` is a product of two symmetric matrices and is therefore
            // generally NOT symmetric; it must not be symmetrized —
            // `gam_linalg::matrix::symmetrize_in_place(F)` both breaks the
            // H·F = X'WX consistency identity (so any downstream code that
            // reconstructs X'WX from H·F lands on an asymmetric/indefinite
            // matrix) AND corrupts the frequentist covariance `Ve = F·H⁻¹·φ`
            // (since (F_sym)·H⁻¹ ≠ H⁻¹·X'WX·H⁻¹) AND distorts the
            // Wood-corrected reference d.f. `tr(F_jj)² / tr(F_jj²)` consumed
            // by `smooth_test::reference_df` (tr(F²) ≠ tr(F_sym²) in general).
            // See issue #1027.
            let f_mat = posterior_factor.solve_matrix(&xwx).map_err(|error| {
                EstimationError::RemlOptimizationFailed(format!(
                    "influence matrix solve H·F = X'WX did not certify: {error}"
                ))
            })?;

            // Frequentist covariance Ve = H⁻¹·X'WX·H⁻¹·φ = φ·H⁻¹·Fᵀ (the
            // sandwich is symmetric, so `F·H⁻¹ = (H⁻¹·Fᵀ)`). Solving
            // `H·Z = Fᵀ` instead of multiplying `F·H⁻¹` gives the companion
            // identity `H·Ve·H = φ·X'WX` to backward error: `H·Z·H = Fᵀ·H =
            // (H·F)ᵀ = X'WX`. The explicit-inverse form carried the same
            // `cond(H)` amplification as `F` did, with no identity anywhere
            // that looked at it.
            let f_transpose = f_mat.t().to_owned();
            let mut ve = posterior_factor
                .solve_matrix(&f_transpose)
                .map_err(|error| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "frequentist covariance solve H·Ve/φ = Fᵀ did not certify: {error}"
                    ))
                })?;
            ve *= cov_scale;
            gam_linalg::matrix::symmetrize_in_place(&mut ve);

            weighted_gram = Some(xwx);
            coefficient_influence = Some(f_mat);
            beta_covariance_frequentist = Some(ve);
        }

        // Smoothing-parameter correction `J·V_ρ·Jᵀ` (Wood, Pya & Säfken 2016,
        // mgcv's Vc1), analytic for any number of smoothing parameters, on BOTH
        // inference branches. It is assembled as its square-root factor `B`
        // (`p × r`, `C = B·Bᵀ`). The dense branch forms `C` for `Vp = Vb + C`.
        // The factorized branch keeps `B` and solves the corrected standard
        // errors beside the conditional ones below, because a `p × p` matrix is
        // what its governor refused (#3283); it charges the correction's
        // workspace here, where the dense bundle already holds it.
        let smoothing_workspace = if beta_covariance_unscaled.is_some() {
            Ok(None)
        } else {
            reserve_smoothing_correction_workspace(qs.nrows()).map(Some)
        };
        let mut smoothing_correction_factor: Option<Array2<f64>> = None;
        if let Err(detail) = smoothing_workspace.as_ref() {
            if !final_rho.is_empty() {
                log::debug!(
                    "[SMOOTHING-CORRECTION] factorized branch could not reserve the correction's \
                     workspace ({detail}); publishing the typed absence"
                );
                smoothing_correction_absence = Some(
                    crate::model_types::SmoothingCorrectionAbsence::CorrectionWorkspaceRefused {
                        detail: detail.clone(),
                    },
                );
            }
        }
        if let Ok(smoothing_workspace_reservation) = smoothing_workspace {
            let no_outer_gradient = Array1::<f64>::zeros(0);
            // #2748 -- THE RESOLUTION THE CERTIFICATE'S VERDICT WAS TAKEN AT.
            //
            // When the certificate admits a `psd = false` it does so by its
            // gradient-residue floor, and the definiteness test behind that
            // clearance is taken at a SHIFT: the larger of the measured
            // `‖δH‖₂` and the arithmetic `√ε·max(max|H_ii|, 1)`. The second is
            // what decides whenever no identity could be measured, and on a
            // ρ-Hessian whose largest diagonal is under 1 it is a flat
            // `1.490116e-8`.
            //
            // `invert_identified_rho_hessian` is about to re-judge the SAME
            // direction at the SAME point, and derives its own resolution from
            // its eigensolver's backward error -- `2.191651e-16` on the
            // measured `geo_disease` k=12 cell, eight orders tighter. It then
            // refused a direction (`λ_min(H+diag|g|) = -1.129942e-8`) the
            // certificate had cleared, and the fit died between the two layers
            // with "this is a genuine contradiction". It is not a contradiction;
            // it is two standards. A verdict and the standard it was taken at
            // have to travel together.
            //
            // Only a clearance that actually CLEARED contributes: a refusal
            // carries no admission for this site to honour.
            //
            // A `psd = true` verdict travels as well (#1561). "PSD outright" is
            // decided by the same shifted Cholesky test, so it too admits every
            // direction inside its shift, and the floor clearance recorded beside
            // it clears whenever the raw matrix passed. Withholding it made this
            // site refuse points the certificate accepted: focused proof run
            // 34668941743 at 7d3b11307, `quality_vs_sklearn_binomial_logit`
            // (binomial `s(pc1, k=5) + s(pc2, k=5)` on prostate), where the outer
            // loop certified a minimum and this site refused sigma = -1.755e-6
            // against a bar of 1.330e-7 whose resolution components were the
            // eigensolver's 2.737740e-14 and three exactly-zero identities.
            //
            // #2612 -- A WITHDRAWN VERDICT TRAVELS AS WHAT IT MEASURED (#1561).
            //
            // Standard REML certifies on stationarity, and a negative direction
            // the floor did not clear is adjudicated against the criterion. When
            // the objective never falls along it, the certificate withdraws its
            // verdict together with its floor clearance, so the shift above
            // forwards nothing, and this site then refused that same direction
            // on the matrix's word: `quality_vs_inla_binomial_smooth_probability`
            // at 7ebbacd3d (job 555236), sigma = -1.651e-6 against a bar of
            // 1.333e-7. The adjudication measured the matrix wrong along that
            // direction by |lambda_min|, which is a component of ||dH||_2 this
            // gate is entitled to spend. The invariance is formed only for a
            // contradicted certificate, the one verdict that reads it.
            let contradicted_curvature_error = if outer_result
                .criterion_certificate
                .as_ref()
                .is_some_and(|certificate| certificate.curvature.withdrawn_by_criterion())
            {
                certificate_contradicted_curvature_error(
                    outer_result.criterion_certificate.as_ref(),
                    outer_result.final_hessian.as_ref(),
                    reml_state.criterion_invariant_directions(&final_rho).as_ref(),
                )
            } else {
                None
            };
            let measured_hessian_error: Vec<
                gam_linalg::curvature_resolution::MeasuredHessianError,
            > = certificate_curvature_verdict_resolution(
                outer_result.criterion_certificate.as_ref(),
            )
            .into_iter()
            .map(|value| {
                gam_linalg::curvature_resolution::MeasuredHessianError::new(
                    "outer-certificate curvature-verdict shift (the resolution its own PSD test was decided at)",
                    value,
                )
            })
            .chain(contradicted_curvature_error.into_iter().map(|value| {
                gam_linalg::curvature_resolution::MeasuredHessianError::new(
                    "criterion-contradicted negative curvature |lambda_min| of the certificate's judged rho-Hessian (the objective did not fall along its eigenvector)",
                    value,
                )
            }))
            .collect();
            // The ρ-block rails, not the theta-wide `railed_facts`: the Hessian
            // judged below is the ρ-Hessian, and a railed link-shape
            // coordinate is not one of its axes.
            let certified_railed_rho: Vec<usize> = outer_result
                .criterion_certificate
                .as_ref()
                .map(|certificate| {
                    certificate
                        .lambdas_railed
                        .iter()
                        .copied()
                        .chain(certificate.stationarity.rails().iter().map(|rail| rail.index))
                        .filter(|&index| index < final_rho.len())
                        .collect()
                })
                .unwrap_or_default();
            let smoothing_outcome = reml_state.compute_smoothing_correction_outcome(
                &final_rho,
                &lambdas,
                &pirls_res,
                // #2428: the residual gradient the outer certificate itself
                // used to accept this ρ̂ is the resolution floor the ρ-Hessian's
                // definiteness must be judged against. Without it the
                // correction applies a strictly stronger standard than the
                // certificate did and can reject a fit the outer loop passed.
                outer_result
                    .final_gradient()
                    .unwrap_or(&no_outer_gradient),
                // #2748: the rho-Hessian the CERTIFICATE judged, so the
                // correction can measure how far its own fresh assembly of the
                // same object at the same point is from it. The gate inside
                // re-judges a direction the certificate has already cleared;
                // two assemblies of one mixed-partial object must agree, and
                // the amount by which they do not is a measured component of
                // the resolution that re-judgement is entitled to spend. Absent
                // on a solver that tracks no Hessian, which is an absent
                // measurement rather than a zero.
                outer_result.final_hessian.as_ref(),
                // #2748: the resolution the outer certificate's own PSD test
                // cleared a negative direction at. `invert_identified_rho_hessian`
                // is about to judge the SAME matrix at the SAME point, and
                // without it would do so against an eigensolver's backward
                // error, refusing fits this certificate accepted (#2428). Empty
                // when the certificate cleared nothing.
                &measured_hessian_error,
                // The coordinates the certificate held at a rail and excluded
                // from its own PSD test. They are boundary estimates with
                // `∂β̂/∂ρ_k = 0`, so the correction gives their axes zero
                // variance instead of re-judging a curvature the certificate
                // never looked at.
                &certified_railed_rho,
            );
            match smoothing_outcome {
                super::reml::eval::SmoothingCorrectionOutcome::Unavailable { reason, .. } => {
                    // The only typed absence is an outer Hessian with no
                    // analytic form for this fit at all (a non-canonical Firth
                    // link, routed to BFGS): nothing about the optimum is
                    // suspect, the correction simply cannot be formed, and the
                    // fit was accepted with that link on purpose (#2158).
                    // Railed coordinates are not a reason: the correction
                    // excludes them exactly as the certificate did, so a
                    // refusal on a railed fit is a real defect like any other.
                    if !matches!(
                        reason,
                        crate::estimate::smoothing_correction::SmoothingCorrectionUnavailable::OuterHessianNotAnalytic { .. }
                    ) {
                        return Err(EstimationError::InvalidInput(format!(
                            "exact smoothing-corrected covariance unavailable: {reason:?}"
                        )));
                    }
                    log::info!(
                        "[SMOOTHING-CORRECTION] typed-unavailable on a non-analytic-outer-Hessian \
                         fit ({reason:?}); shipping the plug-in covariance without a smoothing correction"
                    );
                    smoothing_correction_absence = Some(
                        crate::model_types::SmoothingCorrectionAbsence::OuterHessianNotAnalytic {
                            detail: format!("{reason:?}"),
                        },
                    );
                    rho_covariance = None;
                    smoothing_correction = None;
                    smoothing_correction_method = None;
                    smoothing_correction_first_order = None;
                    smoothing_correction_method_first_order = None;
                }
                outcome => {
                    rho_covariance = outcome.rho_covariance().cloned();
                    let (factor, method) = outcome.into_correction_with_method();
                    smoothing_correction_method = method;
                    if beta_covariance_unscaled.is_some() {
                        smoothing_correction = factor
                            .as_ref()
                            .map(crate::estimate::smoothing_correction::smoothing_correction_gram);
                        smoothing_correction_first_order = smoothing_correction.clone();
                        smoothing_correction_method_first_order = smoothing_correction_method;
                    } else {
                        smoothing_correction_factor = factor;
                    }
                }
            }
            // The correction's assembly transients are gone; its `p × r` factor
            // is all that stays live on the factorized branch.
            drop(smoothing_workspace_reservation);
        }

        // Tier-0 marginal-smoothing adequacy diagnostic (#938): while the REML
        // objective is still live, sample the outer criterion around the
        // converged ρ̂ to read the PSIS k̂ that says whether the plug-in +
        // first-order V_ρ correction is adequate. It runs against the SAME
        // objective the fit converged on, so its criterion is the fit's own
        // bit-for-bit.
        //
        // The returned fit does not need it: the covariance above is complete
        // without it, and the diagnostic costs dozens of inner solves plus a
        // fresh ρ-Hessian. So it runs only when the caller requests ρ-posterior
        // inference (`skip_rho_posterior_inference = false`), together with the
        // escalation tiers it grades for (quadrature for K≤4, NUTS over ρ for
        // K≤16, honest Unavailable beyond). Every other fit keeps the typed
        // `NotComputed(InferenceNotRequested)` set above.
        if !opts.skip_rho_posterior_inference {
            (rho_posterior, rho_posterior_escalation) = reml_state.rho_posterior_inference(
                &final_rho,
                // The box is where λ is numerically resolvable, not the
                // posterior's support: a draw past a saturated face is valued by
                // the criterion's exact affine limit from that face, so no
                // posterior mass is dropped when the box edge moves.
                &rho_continuation,
                None,
            );
        }

        // Standard errors: prefer the diagonal of the full inverse when
        // available; otherwise use the factorised Hessian from the EDF pass
        // (in transformed basis) to compute exact diagonal of H_orig⁻¹ =
        // Qs H_t⁻¹ Qs' via chunked solve-on-demand. The chunk width comes
        // from the runtime resource policy's per-chunk byte target: each
        // chunk keeps ~2 dense p×chunk workspaces (the RHS slice and the
        // solved block) live at once.
        let resource_policy = gam_runtime::resource::ResourcePolicy::for_problem(
            gam_runtime::resource::ProblemHints::default(),
        );
        let governor = gam_runtime::resource::MemoryGovernor::global();
        let se_chunk_target_bytes = resource_policy
            .row_chunk_target_bytes
            .min(governor.remaining_bytes());
        let se_chunk_cols = gam_runtime::resource::rows_for_target_bytes(
            se_chunk_target_bytes,
            qs.ncols().saturating_mul(2),
        );
        if let Some(covariance) = beta_covariance.as_ref() {
            // The dense covariance already includes the inequality-truncation
            // correction, and the published standard errors derive from it
            // (#2955). Its diagonal is judged here, where the attribution is.
            //
            // Why an inequality-truncated covariance may show an exactly-zero
            // diagonal, and why that is a measurement rather than a defect
            // (#2705 group A).
            //
            // For an UNCONSTRAINED fit `Σ = φ·H⁻¹` with `H` SPD, so every
            // diagonal entry is strictly positive and a zero would mean the
            // Hessian is singular — which this gate exists to catch, and still
            // does. A TRUNCATED one is a different object: the constraint
            // removes the coordinate's variance along its own normal, and the
            // λ → ∞ limit of that removal is exactly zero. The Gram assembly
            // above computes that limit as a sum of squares, so it reports the
            // clean `0.0` instead of the `±ε·Σ_ii` rounding residue the
            // subtraction used to leave — and a strict `> 0` test would then
            // refuse the fit for producing the right answer.
            let truncation_applied = constrained_removed_variance.is_some();
            for (index, &variance) in covariance.as_array().diag().iter().enumerate() {
                let valid = if zero_covariance_boundary {
                    variance == 0.0
                } else if truncation_applied {
                    variance.is_finite() && variance >= 0.0
                } else {
                    variance.is_finite() && variance > 0.0
                };
                if !valid {
                    let removed = constrained_removed_variance
                        .as_ref()
                        .and_then(|d| d.get(index).copied())
                        .map_or("n/a".to_string(), |v| format!("{v:.6e}"));
                    let allowance = constrained_diagonal_uncertainty
                        .as_ref()
                        .and_then(|d| d.get(index).copied())
                        .map_or("n/a".to_string(), |v| format!("{v:.6e}"));
                    return Err(EstimationError::RemlOptimizationFailed(format!(
                        "posterior covariance diagonal {index} is not positive and representable: \
                         {variance:?} [#2705 attribution: removed_variance_diag={removed} \
                         cubature_allowance={allowance} truncation_applied={truncation_applied}]"
                    )));
                }
            }
        } else if let Some(ref factor_t) = edf_factor {
            // No dense `Σ`: solve the published coordinates' diagonal
            // `diag(M·Σ·Mᵀ)` through the factor, one row of `M·Qs` per solve
            // (#2960), and publish it as the fit's standard errors.
            let correction = constrained_posterior
                .as_ref()
                .map(crate::constrained_posterior::ConstrainedPosteriorGeometry::correction)
                .transpose()
                .map_err(|reason| {
                    EstimationError::RemlOptimizationFailed(format!(
                        "constrained posterior variance correction is unavailable: {reason}"
                    ))
                })?
                .flatten();
            let inverse_diagonal = crate::estimate::penalty::factorized_published_inverse_diagonal(
                &conditioning,
                qs,
                se_chunk_cols,
                |rhs, rows| {
                    factor_t.certified_solve(
                        &pirls_res.stabilizedhessian_transformed,
                        rhs,
                        &format!(
                            "factorized coefficient standard errors at rows {}..{}",
                            rows.start, rows.end
                        ),
                    )
                },
            )?;
            factorized_standard_errors = Some(crate::estimate::penalty::factorized_standard_errors(
                &conditioning,
                &inverse_diagonal,
                cov_scale,
                None,
                correction,
                zero_covariance_boundary,
            )?);
            // #3283: the corrected standard errors of `Vp = Vb + B·Bᵀ` from the
            // same solved diagonal. A constrained fit truncates `Vp` at its own
            // lift, as the dense branch does, from `Vp·Aᵀ = Vb·Aᵀ + B·(Bᵀ·Aᵀ)`:
            // `m` solves against the factor instead of a `p × p` product.
            if let Some(factor) = smoothing_correction_factor.take() {
                let truncation = match constrained_posterior
                    .as_ref()
                    .filter(|geometry| geometry.decline().is_none())
                {
                    Some(geometry) => {
                        let constraints_transpose = geometry.constraints.a.t().to_owned();
                        let solved = factor_t.certified_solve(
                            &pirls_res.stabilizedhessian_transformed,
                            &qs.t().dot(&constraints_transpose),
                            "smoothing-corrected constrained posterior normal geometry",
                        )?;
                        let conditional_times_constraints = qs.dot(&solved) * cov_scale;
                        factorized_marginal_constraint_truncation(
                            geometry,
                            &conditional_times_constraints,
                            &factor,
                        )?
                    }
                    None => Ok(None),
                };
                match truncation {
                    Ok(marginal_correction) => {
                        let standard_errors = crate::estimate::penalty::factorized_standard_errors(
                            &conditioning,
                            &inverse_diagonal,
                            cov_scale,
                            Some(&factor),
                            marginal_correction.as_ref(),
                            false,
                        )?;
                        smoothing_correction_factorized = Some(
                            crate::model_types::FactorizedSmoothingCorrection {
                                factor,
                                standard_errors,
                            },
                        );
                    }
                    Err(reason) => {
                        log::debug!(
                            "[CONSTRAINED-Vp] the factorized smoothing-corrected law could not be \
                             truncated to the feasible set ({reason}); publishing the typed absence"
                        );
                        smoothing_correction_absence = Some(
                            crate::model_types::SmoothingCorrectionAbsence::ConstrainedTruncationRefused {
                                detail: reason,
                            },
                        );
                        smoothing_correction_method = None;
                    }
                }
            }
        } else {
            // `edf_factor` is set on every `compute_inference` fit, so one of
            // the two branches above runs. Reaching here would publish an
            // inference block with neither a covariance nor standard errors and
            // no reason; say which invariant broke instead (gam-2929, gam#2955).
            return Err(EstimationError::RemlOptimizationFailed(
                "coefficient standard errors were requested with neither a dense posterior \
                 covariance nor an inference factor to solve them from"
                    .to_string(),
            ));
        }

        // Vp = Vb + J·V_ρ·Jᵀ, both terms on the SAME dispersion (variance) scale.
        //
        // The smoothing correction is built from the coefficient sensitivities
        // J = dβ̂/dρ = −H⁻¹(λ_k S_k(β̂ − μ_k)), which are linear in β̂, and from
        // V_ρ = (∇²_ρρ V)⁻¹. Under a Gaussian rescaling y → c·y the fit is exactly
        // equivariant: β̂ → c·β̂ (so J → c·J), H is response-scale-invariant, the
        // REML/LAML cost gains only a ρ-independent (n/2)·log(c²) offset (so its
        // ρ-gradient and ρ-Hessian — hence V_ρ — are dispersion-free), and φ̂ → c²·φ̂.
        // Therefore J·V_ρ·Jᵀ ∝ c · c⁰ · c = c², i.e. the correction is already on
        // the c² variance scale, exactly like Vb = φ̂·H⁻¹ ∝ c². It must be added
        // directly to Vb. Multiplying it by cov_scale
        // (≈ c²) again would make the correction scale as c⁴, inflating every
        // predict() interval for large-magnitude responses (#582). cov_scale is
        // applied once, where it belongs: in Vb = scaled_covariance(H⁻¹, cov_scale).
        //
        // #2705 group A — WHICH `Vb` the sum starts from, when the fit carries
        // inequality constraints.
        //
        // `beta_covariance` is the ρ̂-CONDITIONAL posterior covariance and, for a
        // constrained fit, it has already been truncated to the feasible set:
        // `Σ_π = Σ − GΔGᵀ`. Adding `J·V_ρ·Jᵀ` to THAT produced a matrix that is
        // the truncation of neither covariance:
        //
        //     (Σ − GΔGᵀ) + (Vp − Σ)  =  Vp − GΔGᵀ,
        //
        // with `G` and `Δ` derived from `Σ`, not from `Vp`. Along a coordinate
        // the constraint pins, `(GΔGᵀ)_ii` cancels `Σ_ii` to eleven digits, so
        // whatever `(Vp − Σ)_ii` happens to be becomes the WHOLE reported
        // variance — and `Vp − Σ` need not be resolvable against that
        // cancellation (any increment is PSD only as a SUM with `Vb` once the
        // truncation has removed most of `Σ_ii`). On
        // `y ~ s(x, shape=convex)` that left `Σ_ii = 2.30e-2` truncated to
        // `6.23e-13` with a `−3.03e-9` smoothing increment on top, i.e. a
        // materially negative published variance, and `se_from_covariance`
        // refused the fit.
        //
        // The correct composition follows from the estimand. The feasible set
        // constrains β and says nothing about ρ, so the indicator `1_C(β)`
        // factors straight out of the ρ-integral:
        //
        //     ∫ π(β,ρ|y)·1_C(β) dρ  =  1_C(β)·∫ π(β,ρ|y) dρ,
        //
        // i.e. the β-marginal of the TRUNCATED joint posterior is exactly the
        // truncation of the β-marginal of the untruncated one. So the truncation
        // belongs on `Vp`, applied last, with its own `G_p = Vp·Aᵀ·W_p⁻¹` and its
        // own orthant moments at `W_p = A·Vp·Aᵀ` — not inherited from `Σ`.
        //
        // Two properties come with it, both of which the old order lacked: the
        // published matrix is a genuine truncated-Gaussian covariance, so it sits
        // between `P·Vp·Pᵀ ⪰ 0` and `Vp` instead of below both; and the
        // constraint's effect on the reported interval is measured at the width
        // the interval actually has, rather than at the conditional width.
        //
        // The ρ̂-CONDITIONAL `beta_covariance` keeps its own truncation at `Σ` —
        // that one is right, because that estimand really is conditional on ρ̂.
        beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
            (Some(base_cov), Some(corr)) if base_cov.as_array().dim() == corr.dim() => {
                let mut corrected = untruncated_conditional_covariance
                    .as_ref()
                    .unwrap_or_else(|| base_cov.as_array())
                    .clone();
                corrected += corr;
                let truncation = match constrained_posterior.as_ref() {
                    Some(geometry) => {
                        apply_marginal_constraint_truncation(geometry, &mut corrected)?
                    }
                    None => Ok(()),
                };
                match truncation {
                    Ok(()) => {
                        gam_linalg::matrix::symmetrize_in_place(&mut corrected);
                        Some(corrected)
                    }
                    Err(reason) => {
                        log::debug!(
                            "[CONSTRAINED-Vp] the smoothing-corrected covariance could not be \
                             truncated to the feasible set ({reason}); publishing the typed \
                             absence rather than an untruncated marginal, which would over-state \
                             every constrained interval. The rho-hat-conditional covariance is \
                             unaffected."
                        );
                        smoothing_correction_absence = Some(
                            crate::model_types::SmoothingCorrectionAbsence::ConstrainedTruncationRefused {
                                detail: reason.to_string(),
                            },
                        );
                        None
                    }
                }
            }
            (Some(base), Some(corr)) => {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "base covariance shape {:?} does not match smoothing correction {:?}",
                    base.as_array().dim(),
                    corr.dim()
                )));
            }
            _ => None,
        };
        // The published corrected standard errors derive from this matrix
        // (#2955); judge its diagonal here, where the attribution is.
        beta_covariance_corrected
            .as_ref()
            .map(se_from_covariance)
            .transpose()
            .map_err(|error| {
                // #2705 group A. Three shape-constrained fits die here with one
                // byte-identical message that names the CONSUMER's budget and
                // nothing else, so the refusal cannot say which of the three
                // producers summed into this diagonal overran, or by how much.
                // The matrix being read is
                //     Σ = φH⁻¹  −  GΔGᵀ  +  J V_ρ Jᵀ   (then optionally A·Σ·Aᵀ)
                // and only the first term is accurate to floating point. Print
                // the decomposition of the offending entry against each
                // producer's OWN declared resolution.
                let detail = match &error {
                    gam_problem::CovarianceStandardErrorError::NegativeDiagonal {
                        index,
                        value,
                        tolerance,
                    } => {
                        let removed = constrained_removed_variance
                            .as_ref()
                            .and_then(|d| d.get(*index).copied());
                        let allowance = constrained_diagonal_uncertainty
                            .as_ref()
                            .and_then(|d| d.get(*index).copied());
                        let base = beta_covariance
                            .as_ref()
                            .and_then(|c| c.as_array().diag().get(*index).copied());
                        let smoothing = smoothing_correction
                            .as_ref()
                            .and_then(|c| c.diag().get(*index).copied());
                        format!(
                            " [#2705 attribution: index={index} value={value:.17e} arithmetic_tolerance={tolerance:.6e} post_constrained_diag={} smoothing_correction_diag={} removed_variance_diag={} cubature_allowance={} inside_cubature_allowance={}]",
                            base.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            smoothing.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            removed.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            allowance.map_or("n/a".to_string(), |v| format!("{v:.6e}")),
                            allowance.map_or("unknown".to_string(), |a| (-value <= a).to_string()),
                        )
                    }
                    _ => String::new(),
                };
                EstimationError::RemlOptimizationFailed(format!(
                    "corrected coefficient covariance is not a valid standard-error source: {error}{detail}"
                ))
            })?;
    }
    let inference = opts.compute_inference.then(|| FitInference {
        edf_by_block,
        penalty_block_trace,
        edf_rank_bound,
        edf_total,
        smoothing_correction,
        smoothing_correction_method,
        smoothing_correction_first_order,
        smoothing_correction_method_first_order,
        smoothing_correction_absence,
        penalized_hessian: penalized_hessian.clone().into(),
        reparam_qs: Some(pirls_res.reparam_result.qs.clone()),
        dispersion,
        factorized_standard_errors,
        smoothing_correction_factorized,
        beta_covariance_frequentist,
        coefficient_influence,
        weighted_gram,
        identified_subspace,
    });

    let pirls_status = pirls_res.status;
    let likelihood_scale_field = pirls_res.likelihood.scale;

    // Report the fitted dispersion parameter on the family variant for the two
    // families whose *reporting log-likelihood kernel* reads it from the family
    // enum rather than from `likelihood_scale`: Negative-Binomial `theta` (issue
    // #802) and Beta `phi` (issue #1608). For both, `ResponseFamily` carries the
    // parameter directly (`NegativeBinomial { theta }`, `Beta { phi }`), the
    // PIRLS deviance/log-likelihood arms read it off that variant, and the inner
    // solve updated the family variant in lock-step with the scale metadata via
    // `with_negbin_theta` / `with_beta_phi`. But `opts.family` is the *seed* spec
    // (θ/φ at their construction defaults), so cloning it and stopping there would
    // ship the seed dispersion in the saved model while `likelihood_scale` carries
    // the fitted value — the two views diverge and the kernel reads the seed.
    // Threading the fitted dispersion back onto the reported family restores the
    // `with_negbin_theta` / `with_beta_phi` invariant (family variant ⇔ scale
    // metadata are two synchronized views of one estimated parameter) in the
    // terminal output, so every consumer — the diagnose AIC/PSIS-LOO kernel
    // included — sees the data's dispersion instead of the seed.
    //
    // Gamma shape and Tweedie φ are deliberately NOT threaded here: their family
    // variants carry no dispersion (`Gamma` is parameterless, `Tweedie { p }`
    // carries only the power), so their kernels read the fitted scale from
    // `likelihood_scale` directly and there is nothing on the family to sync.
    let mut reported_family = opts.family.clone();
    match likelihood_scale_field {
        LikelihoodScaleMetadata::EstimatedNegBinTheta {
            theta: fitted_theta,
        } => {
            if let ResponseFamily::NegativeBinomial { theta, .. } = &mut reported_family.response {
                *theta = fitted_theta;
            }
        }
        LikelihoodScaleMetadata::EstimatedBetaPhi { phi: fitted_phi } => {
            if let ResponseFamily::Beta { phi } = &mut reported_family.response {
                *phi = fitted_phi;
            }
        }

        // Every other scale metadata is either fixed (nothing was estimated to
        // thread back), or belongs to a family whose variant carries no
        // dispersion at all — Gamma shape and Tweedie φ live only on
        // `likelihood_scale`, as the comment above records. Enumerated so a new
        // estimated-dispersion metadata has to declare here whether its family
        // variant needs syncing.
        LikelihoodScaleMetadata::ProfiledGaussian
        | LikelihoodScaleMetadata::FixedDispersion { .. }
        | LikelihoodScaleMetadata::FixedGammaShape { .. }
        | LikelihoodScaleMetadata::EstimatedGammaShape { .. }
        | LikelihoodScaleMetadata::FixedBetaPhi { .. }
        | LikelihoodScaleMetadata::EstimatedTweediePhi { .. }
        | LikelihoodScaleMetadata::EstimatedDispersion { .. }
        | LikelihoodScaleMetadata::FixedNegBinTheta { .. }
        | LikelihoodScaleMetadata::Unspecified => {}
    }
    // Student-t `(σ̂, ν̂)` are outer hyperparameters carried on the family
    // variant rather than scale metadata; the final fit read them from there.
    if let (
        ResponseFamily::StudentT { sigma, nu },
        ResponseFamily::StudentT {
            sigma: fitted_sigma,
            nu: fitted_nu,
        },
    ) = (&mut reported_family.response, &pirls_res.likelihood.spec.response)
    {
        *sigma = *fitted_sigma;
        *nu = *fitted_nu;
    }
    // The fully-normalized reporting kernel (#2096) reads a CONCRETE dispersion
    // `φ = σ̂²` for Gaussian off `likelihood.scale`. A profiled Gaussian carries
    // only the `ProfiledGaussian` marker (`fixed_phi() == None`), which the
    // kernel maps to NaN by contract (the #1583 no-silent-`φ=1` rule) — so the
    // reported `log_likelihood` (and the AIC built from it) came out NaN for
    // every non-degenerate Gaussian fit. Resolve a positive profiled residual
    // scale `σ̂²` into the reporting spec exactly. The validated boundary
    // estimate `σ̂² = 0` deliberately stays `ProfiledGaussian`: an ordinary
    // normalized Lebesgue density does not exist there, and relabeling it as a
    // positive fixed dispersion would falsify both provenance and density. This
    // is a REPORTING-only substitution: the persisted `likelihood_scale` field
    // below stays `ProfiledGaussian` so downstream consumers still see that the
    // scale was profiled, not user-fixed.
    let reporting_scale = match (&reported_family.response, likelihood_scale_field) {
        (ResponseFamily::Gaussian, LikelihoodScaleMetadata::ProfiledGaussian)
            if !zero_covariance_boundary =>
        {
            LikelihoodScaleMetadata::FixedDispersion {
                phi: standard_deviation * standard_deviation,
            }
        }
        _ => likelihood_scale_field,
    };
    let reported_likelihood = GlmLikelihoodSpec {
        spec: reported_family.clone(),
        scale: reporting_scale,
    };
    // At the validated boundary `σ̂² = 0` the fit reproduces the adjusted
    // response exactly, and no ordinary normalized Lebesgue density exists
    // there — so there is no finite FULL log-likelihood to evaluate, and the
    // kernel below must not be asked for one.
    //
    // Report it the way the deterministic-Gaussian route already reports this
    // same boundary: the value `0` under `UserProvided`, which DECLINES to
    // claim a normalized density rather than fabricating one. That is the
    // established convention for this state, not a new one.
    //
    // This replaces a hard refusal that told the caller to "use the dedicated
    // deterministic-Gaussian shortcut". That shortcut is dispatched by a
    // predicate living at ONE entry point (the formula path), so every other
    // entry — the term-collection entries reach this solver directly — had no
    // way to take the advice and died here instead. The condition is DETECTED
    // here, where the dispersion has actually been estimated, so no entry can
    // miss it; an entry-level predicate can only ever PREDICT this state, and
    // the widening history of `exact_unpenalized_gaussian_beta` (which had to
    // grow from the intercept subspace to any exact affine fit) shows how that
    // prediction keeps coming up short.
    //
    // The SAME reasoning applies to the smoothing criterion, which until #2595
    // had no way to say it: `V_r` profiles the scale, so at `φ̂ = 0` its data
    // term is `½ν·log(D_p/ν) → −∞`. The number the outer optimizer happened to
    // stop at there is a function of the last ulp of β, not a criterion — so it
    // is declined here rather than reported, and `UnifiedFitResult::reml_score`
    // carries the absence all the way to `Summary.raw_reml_score`,
    // `compare_models` and the Bayes-factor path, which now refuse it by name
    // instead of ranking a fabricated value.
    let (log_likelihood, log_likelihood_normalization) = if zero_covariance_boundary {
        (0.0, LogLikelihoodNormalization::UserProvided)
    } else {
        (
            crate::pirls::evaluate_full_log_likelihood_from_eta(
                y_o.view(),
                pirls_res.final_eta.view(),
                &reported_likelihood,
                w_o.view(),
            )?
            .total(),
            LogLikelihoodNormalization::Full,
        )
    };

    let result = ExternalOptimResult {
        beta: reported_beta_orig_internal,
        log_lambdas,
        lambdas: lambdas.to_owned(),
        likelihood_family: reported_family,
        likelihood_scale: likelihood_scale_field,
        log_likelihood_normalization,
        log_likelihood,
        standard_deviation,
        iterations: iters,
        finalgrad_norm,
        outer_converged,
        pirls_status,
        // The Gaussian identity deviance IS this weighted RSS, so it follows the
        // same snap: reporting `1.1e-29` next to `σ̂ = 0` would leave
        // `deviance/(n − edf) ≠ σ̂²` in the same record, and the formula path's
        // exact-fit route already reports an exact zero here.
        deviance: if identity_fit_is_exact {
            0.0
        } else {
            pirls_res.deviance
        },
        stable_penalty_term: pirls_res.stable_penalty_term,
        used_device: pirls_res.used_device,
        max_abs_eta: pirls_res.max_abs_eta,
        constraint_kkt: pirls_res.constraint_kkt.clone(),
        geometry: (opts.compute_inference || needs_constrained_posterior).then(|| FitGeometry {
            coefficient_gauge: gam_problem::Gauge::identity(&[beta_orig_internal.len()]),
            penalized_hessian: penalized_hessian.into(),
            constrained_posterior,
            working: Some(WorkingGeometry {
                weights: pirls_res.solveweights.to_owned(),
                response: pirls_res.solveworking_response.to_owned(),
            }),
        }),
        artifacts: FitArtifacts {
            pirls: Some(pirls_res),
            criterion_certificate: outer_result.criterion_certificate.clone(),
            rho_posterior,
            rho_posterior_escalation,
            rho_covariance,
            // Persist the optimized target's Firth state so saved-model
            // sampling reconstructs the same posterior (#2245 finding 16).
            firth_bias_reduction: cfg.firth_bias_reduction,
            jeffreys_arming_evidence,
            ..Default::default()
        },
        inference,
        covariance_conditional: beta_covariance.map(Array2::from),
        covariance_corrected: beta_covariance_corrected,
        reml_score: (!zero_covariance_boundary).then_some(outer_result.final_value),
        outer_cost_evals: usize::try_from(
            // A panic elsewhere can poison this lock, but the count it guards is
            // a diagnostic that is still perfectly readable; recover it rather
            // than turn a reporting field into a second panic.
            *reml_state
                .arena
                .cost_eval_count
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
        .unwrap_or(usize::MAX),
        inner_pirls_solves: usize::try_from(
            reml_state
                .arena
                .inner_pirls_solve_count
                .load(std::sync::atomic::Ordering::Relaxed),
        )
        .unwrap_or(usize::MAX),
        fitted_link: if let Some(state) = final_mixture_state {
            FittedLinkState::Mixture {
                state,
                covariance: final_mixture_param_covariance,
            }
        } else if let Some(state) = opts.latent_cloglog {
            FittedLinkState::LatentCLogLog { state }
        } else if let Some(state) = final_sas_state {
            if opts.family.is_binomial_sas() {
                FittedLinkState::Sas {
                    state,
                    covariance: final_sas_param_covariance,
                }
            } else if opts.family.is_binomial_beta_logistic() {
                FittedLinkState::BetaLogistic {
                    state,
                    covariance: final_sas_param_covariance,
                }
            } else {
                FittedLinkState::Standard(None)
            }
        } else {
            FittedLinkState::Standard(None)
        },
    };
    // Every inference allocation the governor charges is behind us; release
    // both holds explicitly before handing the assembled result back.
    drop(dense_covariance_reservation);
    drop(factorized_inference_reservation);
    conditioning.backtransform_external_result(result)
}

#[cfg(test)]
mod shipped_joint_point_identity_2727_tests {
    //! #2727 — the post-fit certificate identity check must compare the shipped
    //! point against the certified one over EVERY optimized coordinate.
    //!
    //! The outer optimizer for a flexible-link fit searches
    //! `theta = [rho, link-shape coords]`, so the certificate is minted at a
    //! `K + link_dim` vector while the shipped rho block is `K`. The old check
    //! compared those two directly and so could never agree: six SAS/mixture
    //! fixtures were refused with `converged=true`, `certifies=true` and `|Pg|`
    //! down to `1.446e-6`, on a `Vec::len()` mismatch alone.
    //!
    //! Two arms, and the second is the one that matters. Arm 1 is what the
    //! defect broke — a faithfully shipped joint point must be ACCEPTED. Arm 2
    //! is what the obvious wrong repair breaks: comparing only the rho prefix
    //! turns this over-strict gate into an under-strict one, silently ceasing
    //! to check the link coordinates, which are exactly the coordinates this
    //! lane exists to optimize. Arm 2 fails under that repair and passes here,
    //! so the two arms cannot both be satisfied by a prefix comparison.

    use super::shipped_joint_point_is_certified;
    use ndarray::Array1;

    /// The SAS shape of the reproducer in #2727: one rho, two link coordinates
    /// `(epsilon, log delta)`, i.e. the `1 vs 3` that the old check refused.
    fn sas_reproducer() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let rho = Array1::from_vec(vec![-4.753038138161757]);
        let link = Array1::from_vec(vec![0.6483514447757568, -1.2814780614332404]);
        let certified = Array1::from_vec(vec![
            -4.753038138161757,
            0.6483514447757568,
            -1.2814780614332404,
        ]);
        (rho, link, certified)
    }

    /// ARM 1 — a faithfully shipped joint point is accepted.
    ///
    /// This is the arm the defect broke. On the pre-fix code the operands are a
    /// 1-vector and a 3-vector, so this returns false and the fit is refused.
    #[test]
    fn a_faithfully_shipped_joint_point_is_certified() {
        let (rho, link, certified) = sas_reproducer();
        assert!(
            shipped_joint_point_is_certified(&rho, &link, &certified),
            "the shipped point IS the certified point in every coordinate              (rho={rho:?}, link={link:?}, certified={certified:?}); refusing it              is #2727"
        );
    }

    /// ARM 2 — a link coordinate that does not match must still be refused,
    /// even though the rho prefix matches bitwise.
    ///
    /// This is the discriminating arm. A repair that compares only
    /// `certified[..rho.len()]` passes arm 1 and FAILS this one, which is what
    /// stops the over-strict gate from being repaired into an under-strict one.
    /// Note the rho block here is bitwise identical to the certificate, so a
    /// prefix comparison has nothing to catch.
    #[test]
    fn a_mismatched_link_coordinate_is_refused_though_the_rho_prefix_matches() {
        let (rho, link, certified) = sas_reproducer();
        let mut perturbed = link.clone();
        // One ULP. The check is bitwise by design, so the smallest
        // representable disagreement must already refuse — a tolerance here
        // would be a second, unstated stationarity comparison.
        perturbed[1] = f64::from_bits(perturbed[1].to_bits() + 1);
        assert_ne!(perturbed[1].to_bits(), link[1].to_bits());

        assert!(
            rho.iter()
                .zip(certified.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "precondition: the rho prefix must match bitwise, or this arm would              be refused for arm 1's reason instead of its own"
        );
        assert!(
            !shipped_joint_point_is_certified(&rho, &perturbed, &certified),
            "a shipped link coordinate differing from the certified one must be              refused; accepting it is the under-strict repair of #2727"
        );
    }

    /// A missing link coordinate is a different point, not a shorter one — the
    /// length conjunct the original check got right, kept.
    #[test]
    fn a_dropped_link_coordinate_is_refused() {
        let (rho, _link, certified) = sas_reproducer();
        let truncated = Array1::from_vec(vec![0.6483514447757568]);
        assert!(
            !shipped_joint_point_is_certified(&rho, &truncated, &certified),
            "shipping 2 coordinates against a 3-coordinate certificate must be              refused"
        );
    }

    /// The rho-only arm is unchanged: no link coordinates, and the shipped rho
    /// vector is the whole certified point.
    #[test]
    fn the_rho_only_arm_still_compares_rho_against_the_whole_certificate() {
        let rho = Array1::from_vec(vec![0.25, -1.5]);
        let none = Array1::<f64>::zeros(0);
        assert!(shipped_joint_point_is_certified(&rho, &none, &rho.clone()));

        let mut moved = rho.clone();
        moved[0] = f64::from_bits(moved[0].to_bits() + 1);
        assert!(
            !shipped_joint_point_is_certified(&moved, &none, &rho),
            "a moved rho must still be refused on the rho-only arm"
        );
    }
}

#[cfg(test)]
mod negative_binomial_joint_certificate_tests {
    use super::negbin_theta_stationarity_residual;
    use crate::pirls::NegbinThetaScore;

    fn profile(score: f64, info: f64, band: f64) -> NegbinThetaScore {
        NegbinThetaScore { score, info, band }
    }

    #[test]
    fn theta_residual_is_the_log_scale_newton_displacement() {
        let theta: f64 = 2.0;
        let score: f64 = 3.0;
        let info: f64 = 5.0;
        let expected = (theta * score).abs() / (theta * theta * info - theta * score);
        assert_eq!(
            negbin_theta_stationarity_residual(theta, &profile(score, info, 0.0)),
            expected
        );
        let weight_scale = 1.0e-9;
        let scaled = negbin_theta_stationarity_residual(
            theta,
            &profile(weight_scale * score, weight_scale * info, 0.0),
        );
        assert!(
            (scaled - expected).abs() <= 8.0 * f64::EPSILON * expected.max(1.0),
            "the theta certificate must be invariant to uniform case-weight scaling: {scaled} vs {expected}"
        );
    }

    /// There is no profiling box and so no bound multiplier: the residual is zero
    /// exactly when the score is inside its own rounding band, on either side
    /// (#2469).
    #[test]
    fn theta_residual_is_zero_only_inside_the_score_band() {
        assert_eq!(
            negbin_theta_stationarity_residual(2.0, &profile(1.0e-12, 5.0, 2.0e-12)),
            0.0
        );
        assert_eq!(
            negbin_theta_stationarity_residual(2.0, &profile(-1.0e-12, 5.0, 2.0e-12)),
            0.0
        );
        assert!(negbin_theta_stationarity_residual(2.0, &profile(3.0e-12, 5.0, 2.0e-12)) > 0.0);
        assert!(negbin_theta_stationarity_residual(2.0, &profile(-3.0e-12, 5.0, 2.0e-12)) > 0.0);
    }

    #[test]
    fn theta_residual_rejects_invalid_curvature_or_coordinates() {
        assert!(
            negbin_theta_stationarity_residual(f64::NAN, &profile(0.0, 1.0, 0.0)).is_infinite()
        );
        assert!(negbin_theta_stationarity_residual(1.0, &profile(1.0, 0.0, 0.0)).is_infinite());
        assert!(negbin_theta_stationarity_residual(1.0, &profile(2.0, 1.0, 0.0)).is_infinite());
        assert!(
            negbin_theta_stationarity_residual(1.0, &profile(1.0e-3, 1.0, f64::NAN)).is_infinite()
        );
    }
}
