//! The blockwise inner-fit driver (`inner_blockwise_fit`), the joint
//! Newton polish step, and inner-result assembly, split out of
//! `outer_objective.rs` by concern (#1145). Re-exported via
//! `custom_family` so existing paths stay stable.

use super::blockwise_solve::BlockWorkingSetUpdaterExt;
use super::*;
use gam_solve::row_measure::RowSubsampleMaskExt;

pub(crate) fn beta_cache_keys_match_bitwise(lhs: &Array1<f64>, rhs: &Array1<f64>) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(crate) struct ExactJointModeCurvatureCertificate {
    pub(crate) workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    pub(crate) minimum_whitened_eigenvalue: f64,
    pub(crate) numerical_floor: f64,
    /// Coefficient-space direction of the minimum-curvature mode, expressed in
    /// the FULL joint layout (mapped through the active-face tangent when the
    /// mode was certified on a reduced face). Populated only when the mode has
    /// resolvable negative curvature — the direction a saddle-escape steps
    /// along. `None` otherwise (PSD mode, fully pinned face, or no free
    /// direction). Its exact curvature is `minimum_whitened_eigenvalue`: for the
    /// whitened unit eigenvector `v` with eigenvalue `γ_min`, the raw
    /// coefficient direction `δ = D^{-1/2} v` satisfies `δᵀ H_pen δ = γ_min`, so
    /// a step `s·δ` lowers the quadratic model by `½ s² |γ_min|`.
    pub(crate) negative_curvature_direction: Option<Array1<f64>>,
}

impl ExactJointModeCurvatureCertificate {
    pub(crate) fn has_resolvable_negative_curvature(&self) -> bool {
        self.minimum_whitened_eigenvalue < -self.numerical_floor
    }
}

/// Whether the constrained candidate, rather than an ambient unconstrained
/// spectrum step, owns trust-region globalization.
///
/// A reduced-face candidate already resolved negative curvature inside the
/// only accessible space, `null(A_active)`. Ambient negative curvature cannot
/// invalidate that direction: substituting an unconstrained trust step moves
/// off the certified face while incorrectly retaining its endpoint row ids.
fn constrained_search_delta_owns_trust_step(
    exact_face_kind: Option<bool>,
    has_active_set: bool,
    ambient_spectrum_has_negative_curvature: Option<bool>,
) -> bool {
    exact_face_kind.is_some()
        || (has_active_set && ambient_spectrum_has_negative_curvature == Some(false))
}

/// Reduced-space Newton candidate on a certified current inequality face.
///
/// The global constrained step uses a convex model to discover the active
/// face. Once that face is known, convexifying the Hessian in ambient
/// coefficient space is mathematically wrong: curvature normal to the face is
/// inaccessible, and reflecting it can perturb the tangent Newton equation.
/// This routine instead works in the reduced system
///
///     (Z' H Z) delta_z = Z' r,   delta = Z delta_z,
///
/// where `Z` spans `null(A_active)`. Negative tangent curvature is reflected
/// only as a globalization direction and the step is truncated at the first
/// inactive blocker. When reduced curvature is positive and no blocker is
/// crossed, the returned step is the exact fixed-face Newton step and also
/// requires a nonnegative multiplier certificate.
fn certified_reduced_face_newton_candidate(
    exact_hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    beta: &Array1<f64>,
    constraints: &ConstraintSet,
    active_rows: &[usize],
) -> Result<Option<(Array1<f64>, Vec<usize>, bool)>, String> {
    let p = beta.len();
    if active_rows.is_empty()
        || rhs.len() != p
        || exact_hessian.nrows() != p
        || exact_hessian.ncols() != p
        || constraints.ncols() != p
    {
        return Ok(None);
    }
    let gathered = constraints
        .gather_rows(active_rows)
        .map_err(|error| format!("exact active-face row gather failed: {error}"))?;
    let tangent = match active_constraint_tangent_geometry(&gathered.a)? {
        ActiveConstraintTangentGeometry::FullyPinned => return Ok(None),
        ActiveConstraintTangentGeometry::Tangent(tangent) => tangent,
    };
    let mut reduced_hessian = tangent.t().dot(exact_hessian).dot(&tangent);
    symmetrize_dense_in_place(&mut reduced_hessian);
    let (eigenvalues, eigenvectors) =
        FaerEigh::eigh(&reduced_hessian, faer::Side::Lower).map_err(|error| {
            format!("exact active-face Hessian eigendecomposition failed: {error:?}")
        })?;
    let curvature_scale = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !(curvature_scale.is_finite() && curvature_scale > 0.0) {
        return Ok(None);
    }
    let positive_floor = KKT_REFUSAL_RANK_TOL * curvature_scale;
    if eigenvalues
        .iter()
        .any(|value| !value.is_finite() || value.abs() <= positive_floor)
    {
        return Ok(None);
    }
    let exact_positive_curvature = eigenvalues.iter().all(|value| *value > positive_floor);
    let reduced_rhs = tangent.t().dot(rhs);
    let mut spectral_step = eigenvectors.t().dot(&reduced_rhs);
    for (coefficient, eigenvalue) in spectral_step.iter_mut().zip(eigenvalues.iter()) {
        // Away from a local mode, reflect negative curvature only in the
        // accessible tangent. This is a strict-descent modified-Newton
        // globalization, not an estimator change; as soon as the reduced
        // Hessian is positive it becomes the exact Newton equation above.
        *coefficient /= eigenvalue.abs();
    }
    let mut delta = tangent.dot(&eigenvectors.dot(&spectral_step));
    if delta.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    let directional_descent = rhs.dot(&delta);
    if !(directional_descent.is_finite() && directional_descent > 0.0) {
        return Ok(None);
    }

    // The tangent step can meet a previously inactive inequality. Truncate at
    // the first blocker and carry that row into the returned face so the next
    // cycle solves the exchanged face directly — standard gradient-projection
    // globalization, and the reduced tangent curvature stays exact where the
    // ambient-convex QP would reflect it into a slow crawl (measured on the
    // #2301 n=80 CTN fit: routing every blocker cycle to the full reflected
    // QP marched at ~0.989×/cycle and exhausted the 184-cycle budget with the
    // residual at 0.82 vs tol 1.3e-4).
    //
    // CRITICAL: the truncation lands STRICTLY INSIDE the boundary (fraction
    // from the raw slack, then a 1e-12 inward shave), never past it. The
    // previous behavior added ACTIVE_SET_PRIMAL_FEASIBILITY_TOL to the
    // numerator — deliberately overstepping the boundary by up to 1e-8 so the
    // blocker would register tight — which collided with the trial
    // feasibility gate at the SAME 1e-8 tolerance: the trial classified
    // infeasible, the strictly-interior repair projection failed at the
    // active face, a silent shrink-retry accepted only the HALF step, and the
    // blocker (now half a chord away) was dropped by the accepted-face
    // tightness filter — a bit-frozen fixed point (cycles 29-45: warm_rows=21
    // vs candidate face 22, β pinned at 4.4377, the same 8.194e-7 proposal
    // every cycle, residual pinned at 1.556, diagnosis
    // active_set_incomplete). Landing exactly on the face keeps the trial
    // feasible (violation ≤ 0 passes the gate), the full chord is accepted,
    // and the blocker's remaining slack (~1e-12 of the original) classifies
    // tight under the working-face tolerance, so the exchange completes in
    // one cycle.
    let values_beta = constraints
        .values(beta.view())
        .map_err(|error| format!("exact active-face value evaluation failed: {error}"))?;
    let values_delta = constraints
        .values(delta.view())
        .map_err(|error| format!("exact active-face direction evaluation failed: {error}"))?;
    let mut alpha = 1.0_f64;
    let mut blocking_row = None;
    for row in 0..constraints.nrows() {
        if active_rows.contains(&row) {
            continue;
        }
        let norm = constraints
            .row_norm(row)
            .map_err(|error| format!("exact active-face row norm failed: {error}"))?;
        if !(norm.is_finite() && norm > 0.0) {
            continue;
        }
        let bound = constraints
            .bound(row)
            .map_err(|error| format!("exact active-face row bound failed: {error}"))?;
        if bound == f64::NEG_INFINITY {
            continue;
        }
        let scaled_slack = (values_beta[row] - bound) / norm;
        let scaled_rate = values_delta[row] / norm;
        if scaled_rate < 0.0 {
            let fraction = scaled_slack / -scaled_rate;
            if fraction.is_finite() && fraction >= 0.0 && fraction < alpha {
                alpha = fraction;
                blocking_row = Some(row);
            }
        }
    }
    if !(alpha.is_finite() && alpha > 0.0) {
        // The iterate already presses against an unlisted blocker (zero
        // feasible length along the tangent direction): the working face is a
        // lie and the full QP owns the resolution.
        return Ok(None);
    }
    if alpha < 1.0 {
        alpha *= 1.0 - 1e-12;
        delta *= alpha;
    }
    let candidate = beta + &delta;
    let candidate_values = constraints
        .values(candidate.view())
        .map_err(|error| format!("reduced active-face candidate evaluation failed: {error}"))?;
    for row in 0..constraints.nrows() {
        let norm = constraints
            .row_norm(row)
            .map_err(|error| format!("reduced active-face candidate row norm failed: {error}"))?;
        if !(norm.is_finite() && norm > 0.0) {
            continue;
        }
        let bound = constraints
            .bound(row)
            .map_err(|error| format!("reduced active-face candidate bound failed: {error}"))?;
        if bound != f64::NEG_INFINITY
            && (candidate_values[row] - bound) / norm
                < -gam_solve::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL
        {
            return Ok(None);
        }
    }

    let exact_newton = exact_positive_curvature && alpha >= 1.0;
    if exact_newton {
        // The full reduced Newton solve certifies tangent stationarity. Certify
        // the other KKT half as well: its remaining quadratic gradient must be
        // representable by nonnegative multipliers on this face.
        let quadratic_gradient = exact_hessian.dot(&delta) - rhs;
        let Some((projected, _multipliers)) =
            project_stationarity_residual_on_constraint_cone(&quadratic_gradient, &gathered.a)
        else {
            return Ok(None);
        };
        let residual_inf = projected
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let gradient_scale = quadratic_gradient
            .iter()
            .chain(rhs.iter())
            .map(|value| value.abs())
            .fold(1.0_f64, f64::max);
        if residual_inf > 1e-8 * gradient_scale {
            return Ok(None);
        }
    }
    let mut next_active = active_rows.to_vec();
    if let Some(row) = blocking_row
        && !next_active.contains(&row)
    {
        next_active.push(row);
    }
    Ok(Some((candidate, next_active, exact_newton)))
}

#[cfg(test)]
mod exact_face_newton_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn exact_face_newton_uses_tangent_curvature_not_ambient_reflection() {
        // H is indefinite in ambient space (det(H) = -3), but the active
        // half-space x>=0 pins its inaccessible negative-curvature direction.
        // On the tangent x=0 the exact curvature is +2, so the certified Newton
        // solution is y=1. Ambient eigenvalue reflection changes that tangent
        // equation and therefore cannot own the fixed-face endgame.
        let hessian = array![[-1.0_f64, 1.0], [1.0, 2.0]];
        let rhs = array![0.0_f64, 2.0];
        let beta = array![0.0_f64, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(array![[1.0_f64, 0.0]], array![0.0])
                .expect("x>=0 half-space"),
        );
        let (candidate, active, exact) =
            certified_reduced_face_newton_candidate(&hessian, &rhs, &beta, &constraints, &[0])
                .expect("exact face classification")
                .expect("positive reduced curvature and nonnegative multiplier");
        assert_eq!(active, vec![0]);
        assert!(exact);
        assert!(candidate[0].abs() <= 1e-12);
        assert!((candidate[1] - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn reduced_face_truncates_strictly_inside_the_blocker_and_carries_it() {
        // x>=0 is the current face. The reduced tangent direction meets the
        // inactive y<=0.5 row at half its length. The candidate must stop
        // STRICTLY INSIDE that boundary (never past it — the old
        // +PRIMAL_FEASIBILITY_TOL overshoot collided with the trial
        // feasibility gate at the same tolerance and produced the #2301
        // silent-reject/half-step fixed point), land close enough that the
        // row classifies tight, and carry the blocker in the returned face
        // so the next cycle solves the exchanged face directly.
        let hessian = array![[1.0_f64, 0.0], [0.0, -2.0]];
        let rhs = array![0.0_f64, 2.0];
        let beta = array![0.0_f64, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0_f64, 0.0], [0.0, -1.0]],
                array![0.0, -0.5],
            )
            .expect("x>=0 and y<=0.5"),
        );
        let (candidate, active, exact) =
            certified_reduced_face_newton_candidate(&hessian, &rhs, &beta, &constraints, &[0])
                .expect("reduced face classification")
                .expect("strict tangent descent truncated at the first blocker");
        assert!(candidate[0].abs() <= 1e-12);
        assert!(
            candidate[1] <= 0.5,
            "candidate must never overstep the blocker boundary (y={})",
            candidate[1]
        );
        assert!(
            (candidate[1] - 0.5).abs() <= 1e-9,
            "candidate must land on the blocker face to within the inward \
             shave (y={})",
            candidate[1]
        );
        assert_eq!(active, vec![0, 1]);
        assert!(!exact);
    }

    #[test]
    fn reduced_face_full_step_with_negative_tangent_curvature_stays_available() {
        // Same negative accessible curvature, but the blocker sits beyond the
        // unit step (y<=5), so the full reflected tangent step is feasible:
        // the fast path keeps owning the globalization direction and reports
        // it as non-exact (no fixed-face Newton certificate).
        let hessian = array![[1.0_f64, 0.0], [0.0, -2.0]];
        let rhs = array![0.0_f64, 2.0];
        let beta = array![0.0_f64, 0.0];
        let constraints = ConstraintSet::Dense(
            LinearInequalityConstraints::new(
                array![[1.0_f64, 0.0], [0.0, -1.0]],
                array![0.0, -5.0],
            )
            .expect("x>=0 and y<=5"),
        );
        let (candidate, active, exact) =
            certified_reduced_face_newton_candidate(&hessian, &rhs, &beta, &constraints, &[0])
                .expect("reduced face classification")
                .expect("full-length reflected tangent step is feasible");
        assert!(candidate[0].abs() <= 1e-12);
        assert!((candidate[1] - 1.0).abs() <= 1e-12);
        assert_eq!(active, vec![0]);
        assert!(!exact);
    }

    #[test]
    fn ambient_negative_curvature_cannot_replace_a_reduced_face_direction() {
        assert!(constrained_search_delta_owns_trust_step(
            Some(false),
            true,
            Some(true),
        ));
        assert!(constrained_search_delta_owns_trust_step(
            Some(true),
            true,
            Some(true),
        ));
        assert!(constrained_search_delta_owns_trust_step(
            None,
            true,
            Some(false),
        ));
        assert!(!constrained_search_delta_owns_trust_step(
            None,
            true,
            Some(true),
        ));
    }
}

pub(crate) fn fused_first_attempt_log_likelihood<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    options: &BlockwiseFitOptions,
    specs: &[ParameterBlockSpec],
    states: &[ParameterBlockState],
    trust_attempt: usize,
    joint_workspace_requested: bool,
) -> Result<Option<(f64, Arc<dyn ExactNewtonJointHessianWorkspace>)>, String> {
    if trust_attempt == 0 && joint_workspace_requested {
        joint_line_search_log_likelihood_with_workspace(family, options, specs, states)
    } else {
        Ok(None)
    }
}

/// Rebuild the exact penalized coefficient Hessian at the coefficient vector
/// that an inner solve is about to return.
///
/// A first-order/stall exit inside the Newton cycle is only tentative for a
/// nonconvex family: the cycle's spectrum belongs to its head β, while an
/// accepted step changes β before several later exits can fire. This fresh
/// certificate uses the same structural dense materialization required by the
/// Laplace log-determinant, adds only penalties that belong to the objective,
/// and tests inertia in the scale-aware trust metric. Solver stabilization,
/// reflected curvature, and trace-only ridge are deliberately excluded.
pub(crate) fn exact_joint_mode_curvature_certificate<
    F: CustomFamily + Clone + Send + Sync + 'static,
>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    joint_mode_diagonal_ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    total_p: usize,
    active_constraints: Option<&ActiveLinearConstraintBlock>,
) -> Result<ExactJointModeCurvatureCertificate, String> {
    let workspace =
        family.exact_newton_joint_hessian_workspace_with_options(states, specs, options)?;
    let source = match workspace.as_ref() {
        Some(workspace) => exact_newton_joint_hessian_source_from_workspace(
            workspace,
            total_p,
            MaterializationIntent::LogdetFactorization,
            "fresh exact joint-mode curvature certificate",
        )?,
        None => None,
    };
    let mut hessian = match source {
        Some(source) => materialize_joint_hessian_source(
            &source,
            total_p,
            "fresh exact joint-mode curvature certificate",
        )?,
        None => exact_newton_joint_hessian_symmetrized(
            family,
            states,
            specs,
            total_p,
            "fresh exact joint-mode curvature certificate",
        )?
        .ok_or_else(|| {
            "fresh exact joint-mode curvature certificate requires a joint Hessian".to_string()
        })?,
    };
    let mut metric = joint_penalty_preconditioner_diag(
        &hessian.diag().to_owned(),
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
    );
    if let Some(floor) = family.joint_trust_metric_block_floor(states, specs)?
        && floor.len() == metric.len()
    {
        for (value, floor_value) in metric.iter_mut().zip(floor.iter()) {
            if floor_value.is_finite() && *floor_value > *value {
                *value = *floor_value;
            }
        }
    }
    add_joint_penalty_to_matrix(
        &mut hessian,
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
    );
    if hessian.iter().any(|value| !value.is_finite()) {
        return Err(
            "fresh exact joint-mode curvature certificate found a non-finite penalized Hessian"
                .to_string(),
        );
    }
    // Constrained modes are certified on the active-face TANGENT null(A_act) —
    // the same geometry the terminal determinant integrates over
    // (`active_face_logdet_with_ridge_policy`). Curvature normal to the face
    // is neither integrated by the Laplace approximation nor differentiated by
    // the constrained outer kernel, so full-space indefiniteness there is not
    // evidence against the mode; conversely a saddle WITHIN the face is
    // exactly the point a first-order KKT certificate cannot see (the #979
    // CTN cycle-97 witness: KKT-certified with tangent min_eig = -7.9, which
    // then killed the downstream SPD determinant). A fully pinned mode has no
    // free directions and is trivially certified.
    // Retain the active-face tangent `Z` so a resolved negative-curvature mode
    // (certified in the reduced tangent space) can be mapped back into the full
    // joint coefficient layout as a saddle-escape direction.
    let (certificate_matrix, certificate_metric, tangent) = match active_constraints {
        Some(active) => match active_constraint_tangent_geometry(&active.a)? {
            ActiveConstraintTangentGeometry::FullyPinned => {
                return Ok(ExactJointModeCurvatureCertificate {
                    workspace,
                    minimum_whitened_eigenvalue: f64::INFINITY,
                    numerical_floor: 0.0,
                    negative_curvature_direction: None,
                });
            }
            ActiveConstraintTangentGeometry::Tangent(z) => {
                let reduced = z.t().dot(&hessian).dot(&z);
                // Diagonal of Zᵀ·diag(metric)·Z: the exact positive scaling of
                // the whitening metric expressed on the tangent basis.
                let mut reduced_metric = Array1::<f64>::zeros(z.ncols());
                for j in 0..z.ncols() {
                    let mut projected = 0.0;
                    for k in 0..z.nrows() {
                        projected += metric[k] * z[[k, j]] * z[[k, j]];
                    }
                    reduced_metric[j] = projected;
                }
                (reduced, reduced_metric, Some(z))
            }
        },
        None => (hessian, metric, None),
    };
    let zero_rhs = Array1::<f64>::zeros(certificate_matrix.nrows());
    let spectrum = whitened_spectrum::WhitenedHessianSpectrum::decompose(
        &certificate_matrix,
        &zero_rhs,
        &certificate_metric,
        KKT_REFUSAL_RANK_TOL,
    )?;
    let minimum_whitened_eigenvalue = spectrum.gamma.iter().copied().fold(f64::INFINITY, f64::min);
    // A strict saddle exposes a resolvable negative-curvature eigenvector. Map
    // that whitened mode back to the raw coefficient direction `δ = D^{-1/2} v`
    // (curvature `δᵀ H_pen δ = γ_min` for the unit eigenvector `v`), then lift it
    // through the tangent `Z` when the mode was certified on a reduced face. The
    // resulting full-space direction is the one an inner saddle-escape steps
    // along; a PSD mode carries no such direction.
    let negative_curvature_direction = if minimum_whitened_eigenvalue < -spectrum.numerical_floor {
        let mut argmin = 0usize;
        let mut best = f64::INFINITY;
        for (index, &value) in spectrum.gamma.iter().enumerate() {
            if value < best {
                best = value;
                argmin = index;
            }
        }
        let eigenvector = spectrum.evecs.column(argmin);
        let reduced_direction = Array1::from_iter(
            spectrum
                .d_inv_sqrt
                .iter()
                .zip(eigenvector.iter())
                .map(|(scale, component)| scale * component),
        );
        let full_direction = match tangent.as_ref() {
            Some(z) => z.dot(&reduced_direction),
            None => reduced_direction,
        };
        if full_direction.iter().all(|value| value.is_finite()) {
            Some(full_direction)
        } else {
            None
        }
    } else {
        None
    };
    Ok(ExactJointModeCurvatureCertificate {
        workspace,
        minimum_whitened_eigenvalue,
        numerical_floor: spectrum.numerical_floor,
        negative_curvature_direction,
    })
}

/// Maximum number of times the constrained joint-Newton inner solve steps off a
/// first-order KKT point that certifies as a strict saddle on the active-face
/// tangent before it refuses the fit.
///
/// A first-order-stationary point with a resolvable negative face-tangent
/// eigenvalue is NOT a Laplace mode: the penalized objective strictly decreases
/// along that eigenvector, so the standard second-order response is to step
/// along it and continue the solve, not to refuse. Two escapes clear any
/// isolated saddle the fixed-penalty coefficient objective exposes on the way to
/// a mode; a point that still certifies as a saddle after two feasible escapes
/// is evidence of a genuinely non-modal ρ the outer optimizer must reject, so
/// the honest typed refusal is kept on the final attempt.
const MAX_SADDLE_ESCAPES: usize = 2;

/// Verdict of second-order certification at a constrained first-order KKT point.
enum ConstrainedModeResolution {
    /// The active-face-tangent curvature is PSD: a genuine Laplace mode.
    Certified {
        workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>>,
    },
    /// The point is a strict face-tangent saddle. Step `alpha · direction` (in
    /// the full joint layout, `alpha` carrying the feasible sign) strictly
    /// lowers the objective and stays feasible; the caller applies it and
    /// resumes the inner solve.
    Escape {
        direction: Array1<f64>,
        alpha: f64,
        lambda_min: f64,
    },
}

/// Second-order certification of a constrained first-order KKT point, with a
/// bounded feasible saddle-escape when the active-face tangent is indefinite.
///
/// `saddle_escapes_used` is the number of escapes already spent this solve; once
/// it reaches [`MAX_SADDLE_ESCAPES`] a still-indefinite point yields the honest
/// typed refusal instead of another escape.
fn resolve_constrained_converged_mode<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    states: &[ParameterBlockState],
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    ranges: &[(usize, usize)],
    s_lambdas: &[Array2<f64>],
    joint_mode_diagonal_ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    total_p: usize,
    block_constraints: &[Option<ConstraintSet>],
    cached_active_sets: &[Option<Vec<usize>>],
    saddle_escapes_used: usize,
    objective_tol: f64,
) -> Result<ConstrainedModeResolution, String> {
    // Certify on the tangent of every NUMERICALLY-TIGHT constraint, not only the
    // QP's recorded active set. At a degenerate binding vertex the QP can leave a
    // row with slack below the primal-feasibility tolerance OUT of
    // `cached_active_sets` (a phantom-dual / zero-multiplier omission). Such a row
    // is on the active face all the same: the Laplace tangent must null it, or
    // the certificate over-counts free directions and manufactures a PHANTOM
    // saddle whose negative curvature lives almost entirely normal to that
    // near-tight row — the escape direction then points straight into it and the
    // feasible step collapses to ~1e-11 (the measured CTN witness: lambda_min=
    // -7.1e-1 with alpha=-1e-11, a no-op). Building the face from all tight rows
    // resolves that phantom to a genuine constrained mode, and any REAL saddle
    // keeps a feasible escape (its direction lies in the tight-face tangent, so
    // it has zero rate on every tight row and a meaningful feasible length).
    let feasibility_tol = gam_solve::active_set::ACTIVE_SET_PRIMAL_FEASIBILITY_TOL;
    let mut tight_active_sets: Vec<Option<Vec<usize>>> =
        Vec::with_capacity(block_constraints.len());
    for (block_idx, constraints_opt) in block_constraints.iter().enumerate() {
        let Some(constraints) = constraints_opt else {
            tight_active_sets.push(None);
            continue;
        };
        let block_values = constraints.values(states[block_idx].beta.view())?;
        let mut rows: Vec<usize> = cached_active_sets
            .get(block_idx)
            .and_then(|active| active.clone())
            .unwrap_or_default();
        for row in 0..constraints.nrows() {
            if rows.contains(&row) {
                continue;
            }
            let norm = constraints.row_norm(row)?;
            if !(norm.is_finite() && norm > 0.0) {
                continue;
            }
            let bound = constraints.bound(row)?;
            if bound == f64::NEG_INFINITY {
                continue;
            }
            if (block_values[row] - bound) / norm < feasibility_tol {
                rows.push(row);
            }
        }
        rows.sort_unstable();
        rows.dedup();
        tight_active_sets.push((!rows.is_empty()).then_some(rows));
    }
    let mode_active_block =
        assemble_active_constraint_block(block_constraints, &tight_active_sets, ranges, total_p);
    let certificate = exact_joint_mode_curvature_certificate(
        family,
        states,
        specs,
        options,
        ranges,
        s_lambdas,
        joint_mode_diagonal_ridge,
        joint_bundle,
        total_p,
        mode_active_block.as_ref(),
    )?;
    let lambda_min = certificate.minimum_whitened_eigenvalue;
    let numerical_floor = certificate.numerical_floor;
    if !certificate.has_resolvable_negative_curvature() {
        log::info!(
            "[PIRLS/joint-Newton mode certificate] constrained returned beta certified from fresh exact curvature: lambda_min={lambda_min:.6e}, floor={numerical_floor:.6e}",
        );
        return Ok(ConstrainedModeResolution::Certified {
            workspace: certificate.workspace,
        });
    }
    if saddle_escapes_used >= MAX_SADDLE_ESCAPES {
        return Err(format!(
            "joint Newton tentative convergence rejected by fresh exact returned-mode curvature: lambda_min={lambda_min:.6e} < -floor={numerical_floor:.6e}; an indefinite coefficient point cannot define a Laplace mode (after {saddle_escapes_used} negative-curvature escapes)",
        ));
    }
    let Some(direction) = certificate.negative_curvature_direction else {
        return Err(format!(
            "joint Newton returned-mode curvature is a strict saddle (lambda_min={lambda_min:.6e} < -floor={numerical_floor:.6e}) but the certificate exposed no finite escape direction",
        ));
    };
    // Along the raw coefficient direction `δ` the exact curvature is `γ_min`
    // (`δᵀ H_pen δ = γ_min`), so a step `s·δ` lowers the quadratic model by
    // `½ s² |γ_min|`. Pick the `s` at which that guaranteed second-order decrease
    // clears solver noise, then cap the DISPLACEMENT at a fraction of the current
    // coefficient scale so the escape stays local.
    let gamma_min = lambda_min.abs();
    if !(gamma_min.is_finite() && gamma_min > 0.0) {
        return Err(format!(
            "saddle-escape could not size a step: non-finite curvature magnitude lambda_min={lambda_min:.6e}",
        ));
    }
    let direction_norm = direction.dot(&direction).sqrt();
    if !(direction_norm.is_finite() && direction_norm > 0.0) {
        return Err("saddle-escape direction is degenerate (zero or non-finite norm)".to_string());
    }
    let beta = flatten_state_betas(states, specs);
    let beta_norm = beta.dot(&beta).sqrt();
    let decrease_scaled = (2.0 * objective_tol.max(1e-8) / gamma_min).sqrt();
    let displacement_cap = 0.1 * (1.0 + beta_norm) / direction_norm;
    let base_magnitude = decrease_scaled.min(displacement_cap);
    // Feasibility. The tangent-projected direction satisfies the ACTIVE rows to
    // first order; truncate strictly inside the first INACTIVE blocker, in the
    // scaled-slack terms the active-set solvers use (row norm cancels in the
    // ratio). Both signs of `δ` give the same second-order decrease, so choose
    // the sign that admits the longer feasible step.
    let joint_active =
        flatten_joint_active_set(&tight_active_sets, block_constraints).unwrap_or_default();
    let mut feasible_positive = f64::INFINITY;
    let mut feasible_negative = f64::INFINITY;
    if let Some(joint_constraints) =
        assemble_joint_linear_constraints(block_constraints, ranges, total_p)?
    {
        let values_beta = joint_constraints.values(beta.view())?;
        let values_direction = joint_constraints.values(direction.view())?;
        for row in 0..joint_constraints.nrows() {
            if joint_active.contains(&row) {
                continue;
            }
            let norm = joint_constraints.row_norm(row)?;
            if !(norm.is_finite() && norm > 0.0) {
                continue;
            }
            let bound = joint_constraints.bound(row)?;
            if bound == f64::NEG_INFINITY {
                continue;
            }
            let scaled_slack = (values_beta[row] - bound) / norm;
            if scaled_slack < 0.0 {
                continue;
            }
            let scaled_rate = values_direction[row] / norm;
            if scaled_rate < 0.0 {
                feasible_positive = feasible_positive.min(scaled_slack / -scaled_rate);
            } else if scaled_rate > 0.0 {
                feasible_negative = feasible_negative.min(scaled_slack / scaled_rate);
            }
        }
    }
    let (sign, feasible_cap) = if feasible_positive >= feasible_negative {
        (1.0_f64, feasible_positive)
    } else {
        (-1.0_f64, feasible_negative)
    };
    let feasible_cap = if feasible_cap.is_finite() {
        // Land strictly inside the blocker (matching the reduced-face solver's
        // 1e-12 inward shave), never on or past it.
        feasible_cap * (1.0 - 1e-12)
    } else {
        feasible_cap
    };
    let magnitude = base_magnitude.min(feasible_cap);
    if !(magnitude.is_finite() && magnitude > 0.0) {
        return Err(format!(
            "joint Newton returned-mode curvature is a strict saddle (lambda_min={lambda_min:.6e} < -floor={numerical_floor:.6e}) with no feasible escape length along the negative-curvature tangent",
        ));
    }
    Ok(ConstrainedModeResolution::Escape {
        direction,
        alpha: sign * magnitude,
        lambda_min,
    })
}

pub(crate) fn inner_blockwise_fit<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<BlockwiseInnerResult, String> {
    // Inner-blockwise prelude waypoints. At large-scale n the cold-start
    // path between function entry and the first PIRLS/JN cycle-summary
    // log can run for many minutes (sometimes hours) silently while
    // row-kernel workspace builds run. Emit a `[STAGE] PIRLS/inner`
    // line at each transition so the next failed run pinpoints which
    // named step holds time. Gated on large-scale n so small-fit
    // tests stay quiet.
    let inner_started = std::time::Instant::now();
    let mut states = buildblock_states(family, specs)?;
    refresh_all_block_etas(family, specs, &mut states)?;
    let total_joint_p = specs.iter().map(|spec| spec.design.ncols()).sum::<usize>();
    let total_joint_n = joint_observation_count(&states);
    const INNER_PRELUDE_LOG_MIN_N: usize = 100_000;
    let prelude_log = total_joint_n >= INNER_PRELUDE_LOG_MIN_N;
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=buildblock_states+refresh_etas elapsed={:.3}s n={} p={} blocks={}",
            inner_started.elapsed().as_secs_f64(),
            total_joint_n,
            total_joint_p,
            specs.len(),
        );
    }
    let matrix_free_joint_requested = use_joint_matrix_free_path(total_joint_p, total_joint_n)
        || family.prefers_matrix_free_inner_joint(specs, &states);
    let has_workspace_source = family.inner_coefficient_hessian_hvp_available(specs);
    // Probe the *spec-aware* joint Hessian: it is the canonical source of the
    // coupled joint curvature. A family may override only
    // `exact_newton_joint_hessian_with_specs` (the variant that has access to
    // the realized block designs needed to assemble the cross-block
    // `X_aᵀ diag(w_ab) X_b` blocks — e.g. the Dirichlet common-parameterization
    // family, whose `evaluate` emits diagonal working sets so the spec-less
    // default block assembler returns `None`). Routing the inner joint-Newton
    // availability gate through the spec-less `exact_newton_joint_hessian`
    // would then mis-classify such a family as "no joint Hessian" and drop it
    // onto pure block-diagonal backfitting, which fails to reach KKT on small,
    // concentrated coupled likelihoods. The `_with_specs` path subsumes the
    // spec-less one for every family (single-block / uncoupled delegate
    // identically), so it is the correct probe here.
    let has_joint_exacthessian = if has_workspace_source {
        true
    } else {
        family
            .exact_newton_joint_hessian_with_specs(&states, specs)?
            .is_some()
    };
    // When the family declares its likelihood blocks UNCOUPLED
    // (`∂²L/∂β_a∂β_b = 0` for every a ≠ b) the joint penalized objective is
    // fully separable across blocks: the joint Hessian is exactly
    // block-diagonal and each block carries only its own penalty. On a
    // separable objective block-coordinate descent solves each block's
    // (possibly inequality-constrained) subproblem to its own exact optimum —
    // it IS the joint solve, and each block gets its OWN trust radius, its OWN
    // active-set QP, and its OWN KKT certificate.
    //
    // Forcing the coupled joint-Newton onto such a problem instead couples two
    // independent blocks under ONE shared trust radius and ONE concatenated
    // KKT residual. That is actively harmful when the blocks differ sharply in
    // conditioning — the competing-risks twin time-basis fit (#1025) is the
    // canonical case: two cause-specific baselines share the same I-spline
    // evaluated at the same event times, but one cause sits near its
    // monotonicity-constraint boundary with an O(1e5) hazard-derivative
    // gradient while the other is interior. The shared globalization cannot
    // satisfy both blocks' KKT conditions at once; the joint residual stalls
    // far above tolerance, the inner solve burns its whole cycle budget on
    // every outer ρ-eval, and the fit only survives by falling through to the
    // block-coordinate path anyway (which then converges in a handful of
    // cycles). Route uncoupled multi-block specs straight to that exact
    // separable path. Uncoupled families are routed to blockwise before a joint
    // solve starts, so this stops the engine from attempting — and grinding on
    // — a joint solve it was never required to run.
    //
    // Single-block families and genuinely coupled multi-block families are
    // unaffected: the former never had cross-block coupling to begin with, the
    // latter still take the joint path (their objective is NOT separable, so
    // block-coordinate descent would drop the cross-block ∂²L/∂β_a∂β_b
    // curvature).
    let blocks_separable = specs.len() >= 2 && family.likelihood_blocks_uncoupled();
    let use_joint_newton =
        has_joint_exacthessian && (specs.len() >= 2 || has_workspace_source) && !blocks_separable;
    let joint_workspace_requested = use_joint_newton && has_workspace_source;
    // Row-measure consistency for the outer-score subsample (gam#1135 HT path).
    //
    // `BlockwiseFitOptions::outer_score_subsample` carries a per-row
    // Horvitz–Thompson reweighting. The inner β-solve has several likelihood
    // evaluators that must all agree on ONE row measure for the trust-region
    // ratio `ρ = [F(β) − F(β+δ)] / [−g·δ − ½δᵀHδ]` to be valid:
    //
    //   * the coefficient line search, which ALWAYS evaluates
    //     `log_likelihood_only_with_options` and so applies the subsample
    //     whenever it is present in `options`;
    //   * the joint Hessian, built via
    //     `exact_newton_joint_hessian_workspace_with_options` (HT) when the
    //     workspace path is engaged;
    //   * the entry/reload base objective + gradient from
    //     `load_joint_gradient_evaluation`, which only honours the subsample
    //     through its workspace branch — guarded by
    //     `inner_joint_workspace_gradient_available`. A family that does NOT
    //     advertise that capability (e.g. GaussianLocationScale) falls through
    //     to `family.evaluate` / `exact_newton_joint_gradient_evaluation`, which
    //     ignore `options` and score the FULL data.
    //
    // When the base objective is full-data but the line search is HT, the
    // trust-region numerator compares `F_full(β)` against `F_HT(β+δ)`. The two
    // differ by a β-independent constant (the HT-vs-full log-likelihood gap), so
    // `actual_reduction` stays pinned at that constant even as the step shrinks
    // to machine ε — every attempt rejects, the radius collapses, and the inner
    // solve exits non-converged. That cascades to "no candidate seeds passed
    // outer startup validation" and the whole fit fails — the manifestation is a
    // GaussianLocationScale fit with an outer-score subsample installed (manual
    // or auto-installed at scale) that cannot complete its final inner refit.
    //
    // The subsample is an OUTER-score variance-reduction device, consumed by the
    // outer ψ/ρ derivative path (`psi_hyper`); β̂(ρ) itself must stay the
    // unbiased full-data optimum unless the family can run a FULLY HT-consistent
    // inner solve. It can do so exactly when its entry ll+gradient also honour
    // the subsample, i.e. `inner_joint_workspace_gradient_available` (BMS,
    // survival marginal-slope) — there the line search, Hessian, and base
    // objective are all HT and the contract is preserved. Otherwise (the
    // GaussianLocationScale contract: "inner PIRLS never installs the option, so
    // the inner solve continues to consume the exact full-data log-likelihood")
    // the inner solve must run on full data; strip the subsample so the entry
    // objective, gradient/Hessian, line search, and the trust-region
    // row-measure bookkeeping all agree on the full-data measure.
    let inner_consumes_subsample =
        joint_workspace_requested && family.inner_joint_workspace_gradient_available(specs);
    let stripped_subsample_options;
    let options = if !inner_consumes_subsample && options.outer_score_subsample.is_some() {
        let mut cleaned = options.clone();
        cleaned.outer_score_subsample = None;
        stripped_subsample_options = cleaned;
        &stripped_subsample_options
    } else {
        options
    };
    let inner_tol = options.inner_tol;
    let inner_max_cycles_base = options.inner_max_cycles;
    // Per-outer-call inner-cycle cap. The earlier "adaptive inner cycle
    // cap" doubled this mid-loop on plateaus, but that turned out to be
    // the wrong response to stalled descent (descent ratios pinned at
    // ~0.999 paired with a sub-tolerance objective change is the
    // no-descent signal, not a "give Newton more cycles" signal). The
    // plateau-flat-objective convergence certificate in the inner-cycle
    // body now handles that case directly, so the cap stays fixed at the
    // baseline for the lifetime of this outer call.
    let inner_max_cycles = capped_inner_max_cycles(options, inner_max_cycles_base);
    // Each block's assembled penalty matrix depends only on that block's
    // penalties and smoothing parameters. Build these setup matrices in
    // parallel, but keep the coordinate-descent and line-search loops below
    // strictly serial because each accepted block update changes the state seen
    // by later blocks.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let s_lambdas_launch_started = std::time::Instant::now();
    let s_lambdas_par_iter = (0..specs.len()).into_par_iter().map(|b| {
        let spec = &specs[b];
        let Some(block_log_lambda) = block_log_lambdas.get(b) else {
            return Err(CustomFamilyError::UnsupportedConfiguration {
                reason: format!("missing log-smoothing parameter vector for block {b}"),
            }
            .into());
        };
        if block_log_lambda.len() != spec.penalties.len() {
            return Err(CustomFamilyError::DimensionMismatch {
                reason: format!(
                    "block {b} log-smoothing parameter length {} does not match penalties {}",
                    block_log_lambda.len(),
                    spec.penalties.len()
                ),
            }
            .into());
        }

        let p = spec.design.ncols();
        let lambdas = exact_lambdas_from_log_strengths(
            block_log_lambda,
            &format!("inner block {b} log strength"),
        )?;
        let mut s_lambda = Array2::<f64>::zeros((p, p));
        for (k, s) in spec.penalties.iter().enumerate() {
            s.add_scaled_to(lambdas[k], &mut s_lambda);
        }
        Ok(s_lambda)
    });
    let s_lambdas_collect_started = std::time::Instant::now();
    let s_lambdas_launch_elapsed = s_lambdas_launch_started.elapsed();
    let s_lambdas = s_lambdas_par_iter.collect::<Result<Vec<_>, String>>()?;
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=s_lambdas par_iter launch={:.3}s collect={:.3}s blocks={} (since inner-start={:.3}s)",
            s_lambdas_launch_elapsed.as_secs_f64(),
            s_lambdas_collect_started.elapsed().as_secs_f64(),
            specs.len(),
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let ridge = effective_solverridge(options.ridge_floor);
    let joint_bundle: Option<&gam_problem::JointPenaltyBundle> = options.joint_penalties.as_deref();
    if let Some(bundle) = joint_bundle {
        for (i, spec) in bundle.specs().iter().enumerate() {
            if spec.dim() != total_joint_p {
                return Err(format!(
                    "joint penalty {i}: dim {} != total compiled p {}",
                    spec.dim(),
                    total_joint_p,
                ));
            }
        }
        assert_eq!(bundle.specs().len(), bundle.log_lambdas().len());
    }
    let mut cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; specs.len()];
    if let Some(seed) = warm_start
        && seed.block_beta.len() == states.len()
        && seed.active_sets.len() == states.len()
    {
        if warm_start_matches_block_log_lambdas(seed, block_log_lambdas)
            && let Some(cached) = seed.cached_inner.as_ref()
            && cached.converged
            && cached.block_logdet_h.is_some_and(f64::is_finite)
            && cached.block_logdet_s.is_some_and(f64::is_finite)
            && seed
                .block_beta
                .iter()
                .zip(&states)
                .all(|(beta_seed, state)| beta_seed.len() == state.beta.len())
        {
            for (state, beta_seed) in states.iter_mut().zip(&seed.block_beta) {
                state.beta.assign(beta_seed);
            }
            cached_active_sets = seed.active_sets.clone();
            refresh_all_block_etas(family, specs, &mut states)?;
            let local_ranges = block_param_ranges(specs);
            let local_joint_mode_diagonal_ridge =
                if ridge > 0.0 && options.ridge_policy.accounts_for_objective() {
                    ridge
                } else {
                    0.0
                };
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints = assemble_joint_linear_constraints(
                &block_constraints,
                &local_ranges,
                total_joint_p,
            )?;
            let mut cached_mode_acceptable = true;
            let mut certified_workspace = cached.joint_workspace.clone();
            if has_joint_exacthessian
                && joint_constraints.is_none()
                && !family.joint_jeffreys_term_required()
            {
                match exact_joint_mode_curvature_certificate(
                    family,
                    &states,
                    specs,
                    options,
                    &local_ranges,
                    &s_lambdas,
                    local_joint_mode_diagonal_ridge,
                    joint_bundle,
                    total_joint_p,
                    None,
                ) {
                    Ok(certificate) => {
                        cached_mode_acceptable = !certificate.has_resolvable_negative_curvature();
                        let minimum_whitened_eigenvalue = certificate.minimum_whitened_eigenvalue;
                        let numerical_floor = certificate.numerical_floor;
                        certified_workspace = certificate.workspace;
                        if !cached_mode_acceptable {
                            log::warn!(
                                "[PIRLS/joint-Newton warm-start] refused cached same-rho inner mode: fresh returned-mode curvature lambda_min={:.6e} < -floor={:.6e}; retaining beta only as an uncertified solver seed",
                                minimum_whitened_eigenvalue,
                                numerical_floor,
                            );
                        }
                    }
                    Err(error) => {
                        cached_mode_acceptable = false;
                        certified_workspace = None;
                        log::warn!(
                            "[PIRLS/joint-Newton warm-start] refused cached same-rho inner mode because fresh returned-mode curvature could not be certified ({error}); retaining beta only as an uncertified solver seed"
                        );
                    }
                }
            }
            if cached_mode_acceptable {
                let block_logdet_h = cached.block_logdet_h.ok_or_else(|| {
                    "certified cached inner mode is missing its Hessian logdet".to_string()
                })?;
                let block_logdet_s = cached.block_logdet_s.ok_or_else(|| {
                    "certified cached inner mode is missing its penalty logdet".to_string()
                })?;
                log::info!(
                    "[PIRLS/joint-Newton warm-start] reused cached same-rho inner mode | cycles={} logdet_h={:.6e} logdet_s={:.6e}",
                    cached.cycles,
                    block_logdet_h,
                    block_logdet_s,
                );
                return Ok(BlockwiseInnerResult {
                    block_states: states,
                    terminal_working_sets: cached.terminal_working_sets.clone(),
                    active_sets: normalize_active_sets(cached_active_sets),
                    log_likelihood: cached.log_likelihood,
                    penalty_value: cached.penalty_value,
                    cycles: cached.cycles,
                    converged: cached.converged,
                    block_logdet_h: cached.block_logdet_h,
                    block_logdet_s: cached.block_logdet_s,
                    s_lambdas,
                    joint_workspace: certified_workspace,
                    kkt_residual: cached.kkt_residual.clone(),
                    active_constraints: cached.active_constraints.clone(),
                });
            }
        }
        // Cold-start path: copy prior β where dimensions match
        // (best-effort; mismatched blocks keep the freshly-built
        // initial state).
        for (b, beta_seed) in seed.block_beta.iter().enumerate() {
            if beta_seed.len() == states[b].beta.len() {
                let beta_projected =
                    family.post_update_block_beta(&states, b, &specs[b], beta_seed.clone())?;
                states[b].beta.assign(&beta_projected);
            }
        }
        cached_active_sets = seed.active_sets.clone();
        refresh_all_block_etas(family, specs, &mut states)?;
    }
    let load_joint_started = std::time::Instant::now();
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=load_joint_gradient_evaluation begin use_joint_newton={} joint_workspace_requested={} (since inner-start={:.3}s)",
            use_joint_newton,
            joint_workspace_requested,
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let (
        mut current_log_likelihood,
        mut cached_eval,
        mut cached_joint_gradient,
        mut cached_joint_workspace,
    ) = if use_joint_newton {
        let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
            family,
            specs,
            options,
            &states,
            joint_workspace_requested,
            None,
        )?;
        (log_likelihood, eval, gradient, workspace)
    } else {
        let eval = family.evaluate(&states)?;
        let log_likelihood = eval.log_likelihood;
        (log_likelihood, Some(eval), None, None)
    };
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=load_joint_gradient_evaluation end elapsed={:.3}s log_likelihood={:.6e} has_gradient={} has_workspace={}",
            load_joint_started.elapsed().as_secs_f64(),
            current_log_likelihood,
            cached_joint_gradient.is_some(),
            cached_joint_workspace.is_some(),
        );
    }
    // Validate the one authoritative curvature source at the inner-solve
    // boundary. Workspace families must use that exact source here and in
    // cycle 0; asking `family.evaluate` for block Hessians would assemble the
    // same CTN rowwise-Kronecker Gram a second time at the same beta. Families
    // without a workspace retain the generic block-Hessian guard.
    let validate_started = std::time::Instant::now();
    let mut cached_joint_hessian_source = if joint_workspace_requested {
        let workspace = cached_joint_workspace.as_ref().ok_or_else(|| {
            "joint Newton requested an exact Hessian workspace, but gradient loading retained none"
                .to_string()
        })?;
        Some(
            exact_newton_joint_hessian_source_from_workspace(
                workspace,
                total_joint_p,
                MaterializationIntent::InnerSolve,
                "joint Newton inner prevalidation Hessian source",
            )?
            .ok_or_else(|| {
                "joint Newton exact Hessian workspace supplied no inner-solve curvature source"
                    .to_string()
            })?,
        )
    } else {
        // Gradient-override families (e.g. Gaussian/Binomial location-scale,
        // whose `exact_newton_joint_gradient_evaluation` serves the exact joint
        // score) return no cached evaluation. Materialize it once so the
        // non-workspace block-Hessian guard cannot be skipped (#2108 / #1820).
        if cached_eval.is_none() {
            cached_eval = Some(family.evaluate(&states)?);
        }
        if let Some(eval) = cached_eval.as_ref() {
            validate_block_hessians_finite(eval)?;
        }
        None
    };
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=validate_block_hessians_finite elapsed={:.3}s checked={}",
            validate_started.elapsed().as_secs_f64(),
            cached_eval.is_some() || cached_joint_hessian_source.is_some(),
        );
    }
    let penalty_started = std::time::Instant::now();
    let mut current_penalty = total_quadratic_penalty(
        &states,
        &s_lambdas,
        ridge,
        options.ridge_policy,
        joint_bundle,
        Some(specs),
    );
    if prelude_log {
        log::info!(
            "[STAGE] PIRLS/inner step=total_quadratic_penalty elapsed={:.3}s penalty={:.6e} (prelude_total={:.3}s)",
            penalty_started.elapsed().as_secs_f64(),
            current_penalty,
            inner_started.elapsed().as_secs_f64(),
        );
    }
    let mut lastobjective = -current_log_likelihood + current_penalty;
    let mut converged = false;
    let mut cycles_done = 0usize;
    // Pre-allocate per-block eta backup buffers to avoid O(n) allocation
    // per block per cycle in the backtracking line search.
    let mut eta_backups: Vec<Array1<f64>> =
        states.iter().map(|s| Array1::zeros(s.eta.len())).collect();

    // ── Joint Newton fast path ──
    //
    // When the family provides an exact joint Hessian (GAMLSS location-scale),
    // solve the full (p_mu + p_ls) × (p_mu + p_ls) system in one Newton step
    // per cycle instead of iterating between blocks. This converges quadratically
    // (5-10 steps) instead of linearly (20-100+ blockwise cycles).
    //
    // Generic block-diagonal surrogate families may still fall back to
    // blockwise iteration if the joint surrogate is unavailable. Families that
    // advertise a real coupled joint Hessian must not: the blockwise loop only
    // sees principal blocks, so it drops the cross-block curvature that makes
    // the joint problem well conditioned near saturated optima.

    // `last_residual_tol` mirrors the per-cycle KKT tolerance computed inside
    // the joint-Newton loop (`inner_tol · (1 + max(‖∇L‖∞, ‖Sβ‖∞))`). It must
    // live at function scope so both the post-converged exit block inside
    // `if use_joint_newton` AND the post-block-fit IFT residual builder
    // outside that branch can thread the same tolerance into the
    // `ProjectedKktResidual::with_metadata(...)` builder. Seed at `inner_tol`
    // so a path that skips the loop entirely (no joint-Newton, or zero
    // cycles) still records a finite, non-NaN tolerance on the residual
    // carrier rather than NaN.
    let mut last_residual_tol: f64 = inner_tol;

    if use_joint_newton {
        // Build block ranges for the joint system.
        let ranges: Vec<(usize, usize)> = {
            let mut offset = 0;
            specs
                .iter()
                .map(|s| {
                    let start = offset;
                    offset += s.design.ncols();
                    (start, offset)
                })
                .collect()
        };
        let total_p: usize = ranges.last().map_or(0, |r| r.1);

        // Universal full-span Jeffreys/Firth robustness. Build `Z_J` once and
        // use the same term in the coupled Newton step, objective value, and
        // stationarity checks so a near-separating coefficient is bounded by
        // the likelihood's own Fisher geometry instead of an ad-hoc ridge.
        // `None` (empty coefficient system) leaves every step and objective at
        // the un-augmented inner Newton.
        //
        // Continuous-response families (the canonical example: transformation-
        // normal h(Y|x) ~ N(0,1)) opt out via
        // `joint_jeffreys_term_required() = false`. They have no separation
        // regime, the Fisher information is `O(n)` on every identified
        // direction by construction, and each Jeffreys evaluation costs
        // `p` directional-derivative calls into the family's exact joint
        // Hessian — at large scale (CTN duchon16d, p=144, n=20000) that
        // is the dominant per-cycle cost (~200 s/cycle on three calls per
        // cycle), exhausting the inner budget before the algorithm converges
        // while contributing essentially zero to the gradient/curvature.
        let joint_jeffreys_subspace = if family.joint_jeffreys_term_required() {
            build_joint_jeffreys_subspace(specs, &ranges)?
        } else {
            None
        };
        // FIRTH MERIT BOOKKEEPING (gam#826/#872 — per-cycle Φ fold, not a carried
        // value). `current_penalty` / `lastobjective` hold ONLY the quadratic
        // penalty `½βᵀSβ` (NO Φ). The Firth value `−Φ` is folded into the
        // accept/reject comparison FRESH at each β under the same
        // `jeffreys_skippable_this_cycle` gate the step and KKT residual use, so
        // `old_objective` (old β) and `trialobjective` (trial β) are always on the
        // same objective `−ℓ + ½βᵀSβ − Φ` regardless of whether a cycle skips the
        // term. Carrying Φ in `current_penalty` (the previous design) desynced
        // old-vs-trial by ±Φ whenever the per-cycle skippable decision flipped —
        // and the cycle-0 baseline folded Φ UNCONDITIONALLY while the trial folded
        // it gated, so a skippable cycle 0 saw a spurious `Δobj = ±Φ`, rejected
        // every backtrack, and refused as a `phantom_multiplier` at a zero step
        // (the binomial location-scale coupled non-convergence). SIGN: Firth ADDS
        // ½log|I| to the log-likelihood ⇒ the NLL objective SUBTRACTS Φ, matching
        // the Newton step rhs / KKT residual which ADD `∇Φ` to `∇L − Sβ`.

        let joint_mode_diagonal_ridge =
            if ridge > 0.0 && options.ridge_policy.accounts_for_objective() {
                ridge
            } else {
                0.0
            };

        // Exact joint Newton steps are guarded by two independent mechanisms:
        // family-owned feasibility (`max_feasible_step_size`) and the adaptive
        // trust region below. There is intentionally no family hook for a
        // hard per-attempt coefficient-space clamp; keeping the policy local
        // avoids stale no-op configuration and makes the trust-region behavior
        // explicit at the only place it is used.

        // Cross-cycle convergence carry-over: set at the end of every
        // accepted cycle so the next cycle can distinguish a true KKT
        // optimum on a rank-deficient null mode (objective stuck
        // because every direction is along the null space) from
        // genuine non-convergence. The residual signal does not need
        // a carry-over — `residual <= residual_tol` is the canonical
        // KKT certificate and the end-of-cycle test consumes it
        // directly when it fires.

        // Predicted-reduction tracker for the principled trust-region
        // stopping criterion (Conn-Gould-Toint, *Trust-Region Methods*,
        // Theorem 6.4.6). The Newton model at the accepted step has a
        // predicted decrease `m(0) − m(δ) = −g·δ − 0.5·δ·H·δ`. For an
        // unclipped Newton step (H·δ = −g) this is `0.5·g·H⁻¹·g`, the
        // Newton decrement squared / 2. When the model itself predicts
        // a decrease smaller than the objective tolerance, no descent
        // direction the Hessian can resolve will lower the objective
        // by more than `objective_tol`, and continuing is wall-clock
        // waste regardless of whether the raw gradient residual or
        // step-norm gates have closed.
        //
        // Cross-cycle convergence carry-over: set at the end of every
        // accepted cycle so the next cycle's line-search-failure path
        // can distinguish a true KKT optimum on a rank-deficient
        // Hessian (no meaningful trial step, even though step_inf is
        // O(1) along the null mode) from genuine non-convergence.
        let mut last_cycle_residual_below_tol = false;
        let mut last_cycle_obj_change_below_tol = false;

        let mut joint_trust_radius = 1.0_f64;
        let mut joint_block_trust_radii = vec![1.0_f64; ranges.len()];
        let mut last_accepted_hit_joint_trust_boundary = false;
        // Hard upper bound for the for-loop's range. The cap is fixed at
        // `inner_max_cycles` for the lifetime of this outer call (the
        // earlier mid-loop cap extension was removed in favor of the
        // plateau-flat-objective convergence certificate), but the
        // sentinel pattern is retained — the `.max(200)` floor is a
        // harmless safety pad and the explicit `cycle >= inner_max_cycles`
        // break keeps the existing `continue` statements in the body
        // working
        // (they advance `cycle` via the iterator), unlike a `while` +
        // manual-counter rewrite.
        let inner_loop_hard_ceiling = inner_max_cycles.max(200);
        // Verbose cadence for the inner joint-Newton log block. Boring cycles
        // (first-attempt accepts with no convergence event) emit ONE compact
        // one-liner instead of the 4-line pre-cycle/TR/cycle-summary/convergence
        // block. Verbose cycles (first, last, every 20th, all rejections,
        // convergence events) keep the full detail. JOINT_LOG_VERBOSE_PERIOD is
        // tuned so a 200-cycle inner solve emits ~10 detailed waypoints plus
        // 1 compact line per remaining cycle (~210 lines), down from ~800.
        const JOINT_LOG_VERBOSE_PERIOD: usize = 50;
        // Residual-stall detector for joint Newton. Distinct from the
        // blockwise loglik-frozen divergence detector lower in the file:
        // that one requires the log-likelihood to be unchanged for K
        // cycles AND the per-block Newton step pinned at the cap.
        //
        // Large-scale survival marginal-slope hits a different pattern —
        // the joint objective decreases monotonically by O(1) per cycle
        // (so loglik is NOT frozen), the TR repeatedly clamps proposals
        // with |prop|∞ >> trust_radius, and the post-step KKT residual
        // oscillates in a band orders of magnitude above residual_tol
        // without trending down. Burning the rest of the cycle budget on
        // this pattern reaches inner_max_cycles "non-converged", which
        // then drops the outer optimizer into the first-order bridge
        // fallback with a stale-mode gradient that ‖g‖ ≈ 10⁷ kills BFGS
        // line search at iter 0.
        //
        // Track the best residual seen and the number of cycles since
        // any meaningful improvement (≥10% drop). Once we've burned at
        // least RESIDUAL_STALL_MIN_CYCLES with no improvement AND the
        // TR has been clamping aggressively, exit `converged=false` so
        // the outer optimizer sees a non-converged signal while we still
        // have a finite, in-range β to return (instead of running to the
        // hard ceiling and then handing BFGS a junk gradient).
        const RESIDUAL_STALL_NO_IMPROVE_CYCLES: usize = 30;
        const RESIDUAL_STALL_MIN_CYCLES: usize = 40;
        const RESIDUAL_STALL_IMPROVEMENT_FACTOR: f64 = 0.9;
        // Upper bound on how long a still-descending Φ-merit may VETO the
        // flat-residual stall exit (gam#979 survival marginal-slope hang). The
        // merit-descent veto below (`merit_still_descending_over_window`) was
        // added to protect the gam#1607 transient wiggle — a few cycles where
        // the line search drives the objective down while the KKT residual
        // re-anchors through a gauge null before catching up. That is a
        // TRANSIENT: a healthy or wiggling solve reaches KKT tolerance in a
        // handful of cycles. On the survival marginal-slope monotone-cone DGP,
        // by contrast, the joint block carries a free warp/gauge direction
        // (the #892 flexible-regime family) along which the penalized objective
        // drifts DOWN by O(1) every cycle indefinitely while the KKT residual
        // sits orders of magnitude above tol and never trends toward it. The
        // veto then reads that unbounded drift as "still making progress" and
        // suppresses the flat-residual exit for the ENTIRE cycle budget — the
        // loop grinds to `inner_loop_hard_ceiling` on every one of the ~60
        // outer ρ-evaluations (the ~900s #979 hang), then hands the outer
        // optimizer a non-converged result anyway. Once the residual has been
        // flat (no ≥10% drop) for this many cycles — a large multiple of the
        // stall window, far beyond any legitimate wiggle transient or healthy
        // convergence — the drifting merit is no longer credible evidence of
        // reachable convergence and the veto yields to the honest non-converged
        // exit. This changes nothing for solves that actually converge or
        // wiggle briefly (they exit long before the counter climbs this high);
        // it only rejects a provably-non-stationary ρ sooner.
        const RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES: usize = 4 * RESIDUAL_STALL_NO_IMPROVE_CYCLES;
        let mut best_residual_seen: f64 = f64::INFINITY;
        // Smallest *certified* stationarity residual the solve actually computed,
        // tracked independently of `best_residual_seen` (whose updates are bound
        // to the residual-stall counters at the post-step site below and so are
        // skipped by every head-of-cycle / pre-line-search certificate exit). The
        // terminal verdict reports THIS so a legitimate early-certificate exit
        // (e.g. the cycle-0 pre-line-search KKT exit on intercept-only / already-
        // stationary data) reports the finite residual it certified on instead of
        // the sentinel `inf` — converged=true must never be paired with a non-
        // finite residual in the log (#1040 inner-report truthfulness).
        let mut min_certified_residual: f64 = f64::INFINITY;
        let mut cycles_since_residual_improved: usize = 0;
        // Number of consecutive non-improving cycles after which the
        // conditioning-based self-vanishing Levenberg–Marquardt damping is
        // ARMED inside the spectral-range Newton solve, for EVERY family
        // (#826/#808). The undamped range-restricted Newton step oscillates on a
        // full-rank-but-ill-conditioned penalized Hessian at the oversmoothed-ρ
        // operating point: the tiny-but-above-cutoff curvature of the lightly
        // identified mean/threshold/wiggle block takes an enormous `component/λ`
        // proposal that the trust region clips every cycle, so the residual on
        // that block freezes while its β stays ≈0 (the exact #826 signature).
        // The conditioning-gated `μ = c·‖∇L − Sβ‖∞` caps that component into a
        // bounded descent step. It is SELF-VANISHING (μ → 0 as the residual → 0)
        // so the converged β and the KKT certificate are byte-identical to the
        // undamped solve — zero REML/LAML bias. Arming it on OBSERVED non-
        // progress rather than a static per-family flag keeps the AFT /
        // constant-scale endgame (which converges quadratically and never
        // stalls) byte-identical: a quadratically-converging solve reaches
        // tolerance in a handful of cycles and never trips this threshold, so μ
        // is never engaged there. Only a genuinely oscillating ill-conditioned
        // solve crosses it, which is exactly when the damping is sound. Set a
        // few cycles below the stall-exit window so the damping gets a chance to
        // rescue the solve well before the early-exit / budget tripwire fires.
        // (The conditioning-gated self-vanishing μ this armed now lives ONLY in the
        // test-retained `solve_joint_newton_step_on_spectral_range`; the production
        // joint step takes the exact trust-region multiplier λ instead — gam#979.)
        // Recent KKT-residual values (oldest→newest) used to detect STEADY
        // geometric descent at the certificate-refusal gate. A still-converging
        // Newton direction (residual dropping by a steady factor < 1 each cycle)
        // must not be misclassified as a multiplier/null plateau and exited
        // early (gam#787 duchon centers≥20: the logslope block converges
        // geometrically — residual ~0.33×/cycle — but `linearized_rel ≥ 0.5`
        // routed it into the plateau-refusal break a few cycles short of tol).
        const RESIDUAL_DESCENT_WINDOW: usize = 3;
        let mut residual_descent_history: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(RESIDUAL_DESCENT_WINDOW);
        let mut tr_clamped_during_stall: bool = false;
        // Deterministic slow-geometric-rate stall guard (gam#979 survival
        // marginal-slope). The flat-residual guard below resets its no-improve
        // counter whenever the residual drops ≥10% versus the running best, and
        // the Newton-decrement certificate refuses while the decrement sits a
        // hair above `objective_tol`. A residual crawling down by a small fixed
        // fraction each cycle — the survival marginal-slope oversmoothed-ρ
        // endgame: a stiff penalized Hessian (penalty dominates, eigenvalues
        // ~1e6) yields Newton steps ~1e-5 far INSIDE a large trust radius, so
        // the KKT residual descends geometrically but very slowly (~0.99×/cycle,
        // halving only every ~80 cycles) — clears that 10% bar every ~12 cycles,
        // so NEITHER guard ever fires and the solve grinds ~10³ cycles at ~p³
        // each: minutes-to-hours per outer ρ-evaluation, the measured #979
        // survival "hang" (n≈2500, centers=12 runs past a 900 s wall with no
        // result). This is NOT divergence and NOT a flat stall — the residual is
        // genuinely (geometrically) descending, just far too slowly to reach tol
        // in a practical cycle count. Track a trailing window of residuals so
        // the post-step site can PROJECT, from the window's geometric rate
        // (cycle indices and residual ratios only — fully deterministic, NO
        // wall-clock; cf. the explicit no-wall-clock note at the bottom of the
        // cycle loop), how many more cycles reaching `residual_tol` would take.
        const LINEAR_RATE_WINDOW: usize = 16;
        // If, at the current geometric rate, reaching tol would take more than
        // this many additional cycles, the ρ-evaluation cannot finish in a
        // practical budget: exit `converged=false` with the finite β so the
        // outer optimizer rejects this ρ and moves on (a well-conditioned ρ
        // converges quadratically in a handful of cycles and never reaches the
        // window, so this never touches a healthy solve).
        const LINEAR_RATE_PROJECTION_CAP: usize = 100;
        let mut residual_rate_history: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(LINEAR_RATE_WINDOW + 1);
        // Trailing window of the Φ-augmented merit objective, parallel to
        // `residual_rate_history`, for the merit-descent veto on the two
        // residual-trend stall guards (gam#1607 binomial location-scale-WIGGLE).
        //
        // Both residual-trend guards below (flat-residual no-improve, and the
        // slow-geometric-rate projection) use the trend of the KKT *residual*
        // as a proxy for "can this solve still reach tol in a practical
        // budget". On the wiggle family that proxy is unsound: the model
        // carries an exact additive gauge null (the threshold βₜ and the
        // wiggle-intercept `βwᵀB(q₀)` both shift q = q₀ + Bᵀβw), and as the
        // dynamic basis `B(q₀)` re-anchors during PIRLS the KKT residual is
        // genuinely NON-monotone — it humps up (0.2→0.4) for ~150 cycles
        // before descending to tol — even though the merit objective the line
        // search actually minimizes descends monotonically the whole way and
        // the solve DOES converge (measured: cycle 638, β bounded ≈1.9). A
        // residual-trend guard then reads the transient rise as "diverging /
        // can't reach tol" and bails ~cycle 40, handing the outer optimizer a
        // false non-convergence (the #1607 wiggle fullhessian failure).
        //
        // The merit is the real Lyapunov function: a descending merit IS
        // progress, regardless of the residual's transient shape. So veto a
        // residual-trend stall exit while the merit is still descending
        // robustly over the SAME trailing window. This preserves termination —
        // the merit is bounded below and monotone-nonincreasing under the
        // line search, so it cannot keep clearing a fixed relative-descent bar
        // forever; once it genuinely flattens (the true #979 survival stall:
        // ~1e-5 steps ⇒ merit flat to f64) the veto lifts and the guard fires
        // exactly as before.
        let mut merit_window: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(LINEAR_RATE_WINDOW + 1);
        // Fully-rejected stall guard. The residual-stall guard below
        // (post-grad-reload) only fires on cycles that produced an accepted
        // step, because every termination check it gates lives after the
        // `if !accepted { continue; }` exit at the bottom of the trust-region
        // attempt loop. When every cycle in a row is fully rejected — all
        // JOINT_TRUST_MAX_ATTEMPTS trial steps fail the line-search check —
        // none of those guards ever see the iterate, the cycle loop spins
        // up to `inner_loop_hard_ceiling` cycles, and the inner solver burns
        // ~120 s of wall-clock per outer ρ-evaluation that the outer
        // optimizer will reject anyway. The signature is exact and local:
        // (i) every trust attempt this cycle was rejected by SOME path —
        // model, likelihood, objective, OR feasibility (the four counters
        // partition the JOINT_TRUST_MAX_ATTEMPTS attempts), so `model_rejects +
        // likelihood_rejects + objective_rejects + feasibility_rejects ==
        // JOINT_TRUST_MAX_ATTEMPTS`,
        // AND (ii) the joint trust radius has NOT shrunk relative to the
        // previous fully-rejected cycle. Condition (i) was originally
        // objective-only (`objective_rejects == MAX`, others 0), which never
        // fired on the biobank gauge-flat marginal/logslope fit: there the
        // objective is flat to f64 precision along the residual direction and
        // the BMS line search rejects every trial on the LIKELIHOOD early-exit
        // path, so the guard's increment was unreachable and the loop spun to
        // the cap. A full likelihood-path rejection at a collapsed radius is
        // the same no-descent stall, so any-path full rejection counts.
        // Condition (ii) is what proves no progress is possible: β is
        // reverted to its pre-cycle value on every fully-rejected cycle, so
        // with an identical Newton system AND an identical trust radius the
        // next cycle's trust-region search is byte-deterministically the
        // same as this one's. The radius can stall above the 1e-12 floor
        // when `shrink_active_joint_block_trust_radii` only shrinks blocks
        // that hit their per-block boundary — an interior block keeps its
        // radius forever, so `max(block_radii)` is held by that block while
        // the boundary block's radius collapses to 1e-12 without changing
        // the max. After `FULLY_REJECTED_STALL_MAX_CYCLES` consecutive cycles
        // with both conditions, judge convergence on the identified (range)
        // subspace: a stall at a collapsed radius proves the descent direction
        // is gauge-flat, so if the range-projected KKT residual is at tolerance
        // the fit is at a numerically-stationary penalized optimum and is
        // returned converged; only when the identified-subspace residual is
        // ALSO above tol is this a genuine non-convergence the outer optimizer
        // should reject — exit non-converged so it rejects this ρ cleanly
        // instead of waiting for the cycle cap.
        const FULLY_REJECTED_STALL_MAX_CYCLES: usize = 8;
        let mut prev_rejected_trust_radius: Option<f64> = None;
        let mut consecutive_held_rejected_cycles: usize = 0;
        // Byte-identical fixed-point detector for the fully-rejected stall.
        // Tracks the first-attempt trial objective of the previous fully-
        // rejected cycle; when the current fully-rejected cycle reproduces it
        // bit-for-bit the iterate is provably stationary at the f64 floor (the
        // n≈3e5 marginal/logslope coupling case where the line search rejects
        // every step on a 1-ULP cross-path round-off gap and β reverts
        // identically each cycle). One repeat is conclusive, so the guard fires
        // after two such cycles regardless of the `radius_held` heuristic.
        let mut prev_rejected_first_attempt_objective: Option<f64> = None;
        let mut consecutive_identical_rejected_cycles: usize = 0;
        const IDENTICAL_REJECTED_STALL_MAX_CYCLES: usize = 2;
        // Collapsed-trust-region all-reject-at-floor guard (gam#979 survival
        // hang / binary high-`centers` `IntegrationError`). DISTINCT from the
        // two detectors above:
        //   * `consecutive_held_rejected_cycles` requires the radius to be HELD
        //     relative to the *previous reject* — which it is at any pinned
        //     value, floor or not — and only fires after 8 cycles.
        //   * `consecutive_identical_rejected_cycles` requires the trial
        //     objective to repeat BIT-FOR-BIT, which a near-singular coupled
        //     marginal↔logslope system need not do: tiny non-deterministic
        //     round-off in the per-row tower contraction perturbs the trial
        //     objective in its last ULPs even while the step is otherwise
        //     stuck, so the byte-identical detector never latches.
        // The unambiguous deterministic signal of "stuck and cannot recover" is
        // the trust radius sitting at its absolute `1e-12` floor WHILE every
        // line-search attempt is rejected: no smaller step is representable, so
        // the radius cannot shrink further, and the all-reject means the step
        // makes no progress. After `JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES`
        // consecutive such cycles the loop is provably grinding to its budget on
        // a near-singular system (`phantom_multiplier_with_well_conditioned_H`),
        // so exit cleanly through the SAME identified-subspace / fixed-point
        // certificate path the other two detectors use — converged if the
        // range-space residual is stationary, give-best non-converged otherwise
        // — instead of spinning out the full `inner_max_cycles`. The absolute-
        // floor requirement is why this CANNOT fire on a genuinely progressing
        // fit: a fit that is descending keeps the radius well above `1e-12`
        // (it grows on `rho>0.75`/boundary and only collapses to the floor after
        // a sustained reject streak), so the counter resets on every accepted
        // cycle and never reaches the threshold.
        // Threshold + floor ceiling live in `joint_newton.rs` so the loop and
        // the `joint_newton_collapsed_trust_region_all_reject_exits_before_grinding_budget`
        // unit test assert against one source of truth.
        let mut consecutive_all_reject_at_floor_cycles: usize = 0;
        let mut last_joint_math: Option<JointNewtonMathDiagnostic> = None;
        // Cross-cycle cache of the joint Jeffreys/Firth triple `(β_key, ∇Φ, H_Φ)`
        // (gam#729/#826/#808). Computing `(∇Φ, H_Φ)` costs `p` family
        // directional-derivative calls plus the `½ S Sᵀ` GEMM; for a K-block
        // coupled family that is the dominant per-inner-cycle cost. The post-step
        // KKT residual recomputes the triple at the just-accepted β; the NEXT
        // cycle's head needs the SAME triple at that SAME β. Carry it forward
        // keyed on the flattened β so the head reuses the post-step result instead
        // of recomputing — collapsing two O(p)-directional-derivative evaluations
        // per accepted cycle to one. The key is an exact-equality check on the
        // flattened β (β is byte-identical between an accepted post-step residual
        // and the next head), so the reused term is the exact term at the current
        // iterate — no staleness, no tolerance fudge.
        let mut jeffreys_triple_cache: Option<(Array1<f64>, Array1<f64>, Array2<f64>)> = None;
        // Stash for the structured cert-REFUSED report computed inside the
        // cycle loop, so the post-loop bubbled error (`coupled exact-joint
        // inner solve exited the joint Newton path …`) can emit the same
        // per-block + spectrum breakdown without re-materializing H_pen.
        let mut last_kkt_refusal_report: Option<KktRefusalReport> = None;
        let mut prev_kkt_norm: Option<f64> = None;
        // Convergence-endgame flag for the Jeffreys second-order completion
        // (gam#979): set once the post-step KKT residual enters
        // `JEFFREYS_COMPLETION_RESIDUAL_BAND × residual_tol`, consumed by the
        // next cycle's dense-spectral step assembly.
        let mut jeffreys_completion_endgame = false;
        // Total descent budget across the joint-Newton loop, used by
        // the end-of-loop summary to report `descent_total`.
        let initial_joint_objective: f64 = lastobjective;
        // Per-cycle |Δobjective| history for the geometric-tail trigger of
        // the constrained-stationary certificate below. When the cycles
        // settle into a linear-rate plateau (|Δobj_next| / |Δobj_prev|
        // approaching 1 monotonically over the window), the total
        // *remaining* objective descent is rigorously bounded above by the
        // geometric series sum |Δobj_now| / (1 − max_ratio). When that
        // bound is below `objective_tol` the cert can fire many cycles
        // earlier than waiting for any single |Δobj| to individually
        // cross obj_tol — the bound is mathematically the same precision
        // contract, applied to the asymptotic tail rather than one step.
        const GEOMETRIC_TAIL_WINDOW: usize = 5;
        let mut geometric_tail_history: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(GEOMETRIC_TAIL_WINDOW);
        // A first-order convergence event after an accepted step is tentative
        // until exact curvature at that returned beta proves second-order
        // stationarity. The next ordinary cycle owns that proof and, when it
        // exposes a strict saddle, immediately runs the existing finite-radius
        // More-Sorensen hard case from the same beta.
        let mut returned_mode_curvature_pending = false;
        let mut returned_mode_curvature_certified = false;
        // Constrained analogue of `returned_mode_curvature_pending`: a first-order
        // KKT point on an active face is tentative until the next cycle head
        // certifies its active-face-tangent curvature. On a strict face-tangent
        // saddle that head escapes along the negative-curvature direction and
        // resumes; `saddle_escapes_used` bounds those escapes at
        // `MAX_SADDLE_ESCAPES` before the honest typed refusal.
        let mut returned_constrained_mode_pending = false;
        let mut saddle_escapes_used = 0usize;

        // Fit-level wall-clock budget guard at inner-solve ENTRY. The
        // per-cycle guard below only fires from `cycle > 0`, so it returns a
        // best-effort iterate once a solve has taken at least one step. But on
        // the non-certifying constrained baseline every inner solve early-exits
        // at cycle 0 via the divergence/stall guard, so `cycle > 0` is never
        // reached and the per-cycle guard never fires. The outer startup then
        // drives a whole cascade of fresh solves — one per multistart seed,
        // The exact joint-Hessian route solves the penalized Newton system
        // directly. Extra damping must be wired through an accepted/rejected
        // step policy before it belongs here; keep the matvec faithful to the
        // objective until then.
        'joint_newton_cycles: for cycle in 0..inner_loop_hard_ceiling {
            if cycle >= inner_max_cycles {
                break;
            }
            // Constrained returned-mode second-order certification (gam#979).
            //
            // A constrained first-order KKT point reached last cycle is tentative
            // until its active-face-tangent curvature is certified. Do that here,
            // at the cycle head, before rebuilding the step: a strict face-tangent
            // saddle (curvature the first-order certificate cannot see, the #979
            // CTN witness) is ESCAPED along its negative-curvature direction and
            // the solve resumes at the escaped, feasible point — the standard
            // second-order response to an indefinite stationary point, not a
            // refusal. Only when `MAX_SADDLE_ESCAPES` feasible escapes still land
            // on a saddle does the honest typed refusal fire (inside the helper).
            if returned_constrained_mode_pending {
                returned_constrained_mode_pending = false;
                let escape_block_constraints =
                    collect_block_linear_constraints(family, &states, specs)?;
                let escape_objective_tol = inner_tol * (1.0 + lastobjective.abs());
                match resolve_constrained_converged_mode(
                    family,
                    &states,
                    specs,
                    options,
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    joint_bundle,
                    total_p,
                    &escape_block_constraints,
                    &cached_active_sets,
                    saddle_escapes_used,
                    escape_objective_tol,
                )? {
                    ConstrainedModeResolution::Certified { workspace } => {
                        cached_joint_workspace = workspace;
                        returned_mode_curvature_certified = true;
                        converged = true;
                        cycles_done = cycle;
                        break;
                    }
                    ConstrainedModeResolution::Escape {
                        direction,
                        alpha,
                        lambda_min,
                    } => {
                        for (block_idx, (start, _)) in ranges.iter().copied().enumerate() {
                            for (coefficient_idx, coefficient) in
                                states[block_idx].beta.iter_mut().enumerate()
                            {
                                *coefficient += alpha * direction[start + coefficient_idx];
                            }
                        }
                        refresh_all_block_etas(family, specs, &mut states)?;
                        saddle_escapes_used += 1;
                        log::info!(
                            "[PIRLS/joint-Newton saddle-escape] attempt={} lambda_min={:.6e} alpha={:.6e}",
                            saddle_escapes_used,
                            lambda_min,
                            alpha,
                        );
                        // The escaped point is a fresh iterate; every cross-cycle
                        // progress statistic collected at the saddle is stale.
                        converged = false;
                        returned_mode_curvature_certified = false;
                        last_cycle_residual_below_tol = false;
                        last_cycle_obj_change_below_tol = false;
                        min_certified_residual = f64::INFINITY;
                        best_residual_seen = f64::INFINITY;
                        cycles_since_residual_improved = 0;
                        residual_descent_history.clear();
                        tr_clamped_during_stall = false;
                        residual_rate_history.clear();
                        merit_window.clear();
                        prev_rejected_trust_radius = None;
                        consecutive_held_rejected_cycles = 0;
                        prev_rejected_first_attempt_objective = None;
                        consecutive_identical_rejected_cycles = 0;
                        consecutive_all_reject_at_floor_cycles = 0;
                        last_joint_math = None;
                        last_kkt_refusal_report = None;
                        prev_kkt_norm = None;
                        geometric_tail_history.clear();
                    }
                }
            }
            let verbose_cycle = cycle == 0
                || cycle + 1 == inner_max_cycles
                || (cycle + 1) % JOINT_LOG_VERBOSE_PERIOD == 0;
            // Pre-cycle header line removed: the post-cycle one-liner below
            // carries cycle/objective/Δobj/step/residual/time and on verbose
            // cadence the expanded convergence line additionally carries
            // -loglik and penalty. Suppressing this avoids emitting a second
            // info-level line per cycle just to repeat numbers we already
            // log at end of cycle.
            // Per-cycle phase-timing accumulators. Surface where the inner
            // joint-Newton spends time so a 18-min silent cycle 0 (the
            // bernoulli marginal-slope FLEX large-scale failure mode) becomes a
            // logged timeline at the end of the cycle. Phases:
            //   * hessian: joint Hessian source build (matrix-free workspace
            //     OR dense fallback assembly)
            //   * pcg:     matrix-free QP solve via solve_spd_pcg_with_info_into
            //              (already logs its own diagnostics; we accumulate
            //              here for the end-of-cycle summary)
            //   * line_search: backtracking step-size search (up to 8 attempts)
            //   * grad_reload: post-accept joint gradient + workspace refresh
            let cycle_started = std::time::Instant::now();
            // Top-of-cycle row-measure capture. The trust-region ratio
            // ρ = [F(β) − F(β + δ)] / [−g·δ − ½·δᵀHδ] is only meaningful when
            // every input (Hessian, gradient, objective at β, trial objective
            // at β + δ) is evaluated against the same row measure. We freeze
            // the measure here and re-read it at each of the four sites later
            // in the cycle, then hard-fail (Err) just before ρ if any of them
            // diverged. Cf. `src/solver/row_measure.rs`.
            let tr_row_measure_top =
                gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            let hessian_started = std::time::Instant::now();
            let hessian_scope_guard = gam_runtime::process_monitor::track_scope(format!(
                "joint Newton hessian_qp cycle={cycle} n={total_joint_n} p={total_p}"
            ));
            log::info!(
                "[joint-newton-tr] phase=hessian_qp cycle={} r={:.3e}",
                cycle,
                joint_trust_radius,
            );
            let cycle_log = prelude_log;
            let constraints_started = std::time::Instant::now();
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints =
                assemble_joint_linear_constraints(&block_constraints, &ranges, total_p)?;
            // gam#979: joint simple lower bounds, when the joint constraints are
            // all axis-aligned lower bounds (the survival monotone-baseline-hazard
            // / monotone-smooth case). Threaded into the stationarity certificate
            // so ACTIVE simple-lower-bound multipliers are projected out (the
            // box-bound analog of the linear-constraint projection), instead of
            // their multiplier mass being mis-read as a stationarity defect and
            // mis-refusing a genuinely-optimal constrained iterate.
            let joint_lower_bounds: Option<Array1<f64>> = joint_constraints
                .as_ref()
                .and_then(|c| extract_simple_lower_bounds(c, total_p).ok().flatten())
                .map(|b| b.lower_bounds);
            if cycle_log && cycle == 0 {
                log::info!(
                    "[STAGE] PIRLS/inner step=cycle0 block+joint constraints elapsed={:.3}s n={} p={}",
                    constraints_started.elapsed().as_secs_f64(),
                    total_joint_n,
                    total_p,
                );
            }
            let workspace_build_started = std::time::Instant::now();
            // Get joint Hessian and block gradients from the current evaluation.
            // Hold the cycle's exact-Newton workspace (cache of per-row kernel
            // evaluations at the current β) so a REJECTED cycle can hand it back
            // to `cached_joint_workspace` for the next cycle. After a reject the
            // line search restores β to `old_beta` — exactly the β this workspace
            // was built at — so reusing the cache is bit-identical and skips the
            // O(n) row-kernel re-evaluation (`build_row_kernel_cache`) that
            // otherwise reruns the full data through the per-row CDF/derivative
            // math on every rejected cycle. The converged-exit paths below null
            // this (no carry-forward needed once the inner solve returns).
            let mut hessian_workspace_for_cycle: Option<Arc<dyn ExactNewtonJointHessianWorkspace>> =
                None;
            let joint_hessian_source = if joint_workspace_requested {
                let cached_hit = cached_joint_workspace.is_some();
                let workspace = match cached_joint_workspace.take() {
                    Some(workspace) => workspace,
                    None => family
                        .exact_newton_joint_hessian_workspace_with_options(
                            &states, specs, options,
                        )?
                        .ok_or_else(|| {
                            "joint Newton requested an exact Hessian workspace, but the family returned none"
                                .to_string()
                        })?,
                };
                if cycle_log && cycle == 0 {
                    log::info!(
                        "[STAGE] PIRLS/inner step=cycle0 hessian-workspace cached_hit={} elapsed={:.3}s n={} p={}",
                        cached_hit,
                        workspace_build_started.elapsed().as_secs_f64(),
                        total_joint_n,
                        total_p,
                    );
                }
                hessian_workspace_for_cycle = Some(Arc::clone(&workspace));
                Some(match cached_joint_hessian_source.take() {
                    Some(source) => source,
                    None => {
                        exact_newton_joint_hessian_source_from_workspace(
                            &workspace,
                            total_p,
                            MaterializationIntent::InnerSolve,
                            "joint Newton inner exact-newton operator mismatch",
                        )?
                        .ok_or_else(|| {
                            "joint Newton exact Hessian workspace supplied no inner-solve curvature source"
                                .to_string()
                        })?
                    }
                })
            } else {
                None
            };
            // Row measure observed by the Hessian build above.
            let tr_row_measure_hessian =
                gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            let joint_hessian_source = match joint_hessian_source {
                Some(source) => source,
                None => {
                    // Spec-aware joint Hessian: canonical coupled-curvature
                    // source (see the availability gate above). Families that
                    // only override `_with_specs` (Dirichlet common-parameter)
                    // would otherwise hand back `None` from the spec-less
                    // default and silently drop off the joint-Newton path.
                    let h_joint_opt =
                        family.exact_newton_joint_hessian_with_specs(&states, specs)?;
                    let Some(h_joint) = h_joint_opt else {
                        break; // Fall back to blockwise if joint Hessian unavailable
                    };
                    match symmetrized_square_matrix(
                        h_joint,
                        total_p,
                        "joint Newton inner exact-newton Hessian shape mismatch",
                    ) {
                        Ok(matrix) => JointHessianSource::Dense(matrix),
                        Err(_) => break,
                    }
                }
            };
            let hessian_source_elapsed = workspace_build_started.elapsed();
            if hessian_source_elapsed.as_secs_f64() >= 1.0 || (cycle_log && cycle == 0) {
                let source_kind = if matches!(&joint_hessian_source, JointHessianSource::Dense(_)) {
                    "dense"
                } else {
                    "operator"
                };
                log::info!(
                    "[STAGE] PIRLS/inner step=cycle{} hessian-source joint_workspace_requested={} source={} elapsed={:.3}s n={} p={}",
                    cycle,
                    joint_workspace_requested,
                    source_kind,
                    hessian_source_elapsed.as_secs_f64(),
                    total_joint_n,
                    total_p,
                );
            }

            // Concatenate block gradients and betas.
            let Some(grad_joint) = cached_joint_gradient.clone() else {
                break;
            };
            // Row measure observed by the gradient at β. `cached_joint_gradient`
            // was loaded earlier under `options`; if the auto-subsample
            // installer or any sibling path swapped the mask between then and
            // now, the id captured here will diverge from the rest and the
            // pre-ρ check below will Err. Cf. `src/solver/row_measure.rs`.
            let tr_row_measure_gradient =
                gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            if grad_joint.len() != total_p {
                break;
            }
            let mut beta_joint = Array1::<f64>::zeros(total_p);
            for b in 0..specs.len() {
                let (start, end) = ranges[b];
                beta_joint
                    .slice_mut(ndarray::s![start..end])
                    .assign(&states[b].beta);
            }

            // Non-finite-curvature guard (gam#1088). A `NaN`/`Inf` in the
            // family curvature `H` makes the penalized Hessian `H_pen = H +
            // S(λ)` — and therefore its spectrum — degenerate, so the KKT
            // certificate is structurally unreachable: the spectral step
            // solve produces garbage, the projected residual neither converges
            // nor trends down, and the residual-based divergence/stall guards
            // below (gated on a *finite* residual that a corrupted-but-not-yet-
            // propagated curvature can still leave finite) do not catch it.
            // Left unguarded the loop then burns the full `inner_loop_hard_
            // ceiling` (1200 cycles) on every outer ρ-eval / seed — the
            // multi-hour link-wiggle & location-scale benchmark timeouts. The
            // penalty is finite by construction, so this is a curvature defect:
            // the trial is degenerate. Exit immediately as non-converged with
            // the current finite β so the outer optimizer rejects this ρ-eval
            // cleanly (mirrors the residual divergence guard below), rather
            // than grinding to the ceiling and reporting a `NaN` H_pen
            // spectrum at the refusal point.
            if !joint_hessian_source_curvature_is_finite(&joint_hessian_source) {
                // A non-finite entry at the STARTING iterate (cycle 0) is a
                // contract violation against the family's analytic joint second
                // derivative — the coupled solve cannot even begin — so it is a
                // typed hard failure at the same smooth-regularized logdet
                // boundary that `validate_block_hessians_finite` enforces for a
                // per-block exact-Newton Hessian (gam#1088 fail-loudly contract).
                // A non-finite entry that only emerges at a LATER cycle, after
                // the coupled Newton loop has driven β to an overflowing
                // operating point during outer optimization, is a genuine
                // ρ-degeneracy: exit non-converged with the current finite β so
                // the outer optimizer rejects this ρ cleanly instead of grinding
                // to inner_max_cycles (the multi-hour link-wiggle & location-
                // scale timeouts). Both exit immediately; only the initial-iterate
                // case aborts, because there is no finite progress to hand back.
                if cycle == 0 {
                    joint_hessian_source_finite_check(&joint_hessian_source)?;
                }
                cycles_done = cycle + 1;
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | non-finite-curvature guard (gam#1088): the joint Hessian source carries a non-finite entry, so the penalized Hessian H_pen = H + S(λ) and its spectrum (λ_max/λ_min/cond) are degenerate and the KKT certificate can never be issued; returning unconverged with finite β so the outer optimizer rejects this ρ evaluation instead of grinding to inner_max_cycles={}.",
                    cycle,
                    inner_max_cycles,
                );
                converged = false;
                break;
            }

            let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;
            let joint_hessian_is_dense =
                matches!(&joint_hessian_source, JointHessianSource::Dense(_));
            let joint_solver_diagonal_ridge = stabilized_joint_solver_diagonal_ridge(
                family,
                &joint_hessian_source,
                &ranges,
                &s_lambdas,
                trace_diagonal_ridge,
                options.ridge_floor,
                joint_bundle,
            );
            // CHEAP CONDITIONING PRE-CHECK (always-on robustness, zero-cost on
            // easy/large fits). Before paying for the dense joint-Hessian
            // materialization + `O(p³)` reduced eigendecomposition inside the
            // Jeffreys term, ask whether the term is PROVABLY skippable from a few
            // matrix-free Hessian-vector products against the source we just built.
            // When `true`, the exact conditioning gate is certain to return the
            // zero term, so every Jeffreys call this cycle short-circuits to the
            // exact-zero contribution WITHOUT forming anything dense — byte-
            // identical to the gated-off path, and preserving the matrix-free path
            // on wide well-conditioned fits. Only runs the estimate when a Jeffreys
            // subspace exists and `total_p` is wide enough that the dense eigh is
            // the cost we want to avoid (the helper itself gates on the size
            // threshold and conservatively returns `false` if unsure). Computed
            // once per inner cycle and reused across the cycle's head-KKT, step,
            // and trial-value calls; the conditioning changes slowly across cycles
            // so re-estimating per cycle (one `O(p·k)` burst) is already cheap
            // against the work it guards.
            let jeffreys_skippable_this_cycle: bool = if options.seed_screening {
                // Seed screening only ranks seeds: skip the O(p · per-axis-Hdot)
                // full Jeffreys gradient/curvature loop. The value-only Jeffreys
                // term (folded into the objective baseline / trial penalties via
                // `custom_family_joint_jeffreys_value`, gated independently on
                // `joint_jeffreys_subspace.is_some()`) still bounds the screening
                // score on separating directions; only the per-axis step curvature
                // — the wrong cost class for ranking on a K-block coupled family —
                // is dropped here (gam#729/#808).
                true
            } else if joint_jeffreys_subspace.is_some() {
                // EXPECTED-INFORMATION GUARD (gam#1020): the skippable
                // certificate probes the OBSERVED Hessian source; it only
                // transfers to the Jeffreys gate when the family's Jeffreys
                // information IS the observed Hessian. Expected-information
                // families (probit-class) bypass the pre-check — observed
                // information grows on saturated rows exactly where the
                // expected information collapses and the gate must arm.
                family.joint_jeffreys_information_matches_observed_hessian()
                    && jeffreys_term_skippable_for_source(&joint_hessian_source, total_p)
                        .unwrap_or(false)
            } else {
                false
            };
            let joint_trust_metric_diag = match &joint_hessian_source {
                JointHessianSource::Dense(h_joint) => joint_penalty_preconditioner_diag(
                    &h_joint.diag().to_owned(),
                    &ranges,
                    &s_lambdas,
                    joint_solver_diagonal_ridge,
                    joint_bundle,
                ),
                JointHessianSource::Operator { diagonal, .. } => joint_penalty_preconditioner_diag(
                    diagonal,
                    &ranges,
                    &s_lambdas,
                    joint_solver_diagonal_ridge,
                    joint_bundle,
                ),
            };
            // Scale-aware trust-metric floor for a free-scale-coupled block
            // (#1569). A coupled location-scale survival fit drives some rows to
            // small σ (large `exp(−η_σ)`), which inflates the scale-coupled
            // (location / log-σ) block's likelihood-Hessian diagonal on the rows
            // it loads but UNDERSTATES the per-coordinate curvature scale for
            // coefficients loading mostly on large-σ rows. The affine-covariant
            // Moré–Sorensen step then over-reaches on those coordinates (a tiny
            // metric entry blows up the whitened component `c_k/(γ_k+λ)`), the
            // gain ratio never justifies growing the radius, and the inner solve
            // grinds. The family supplies a per-coordinate floor auto-derived from
            // the scale-predictor magnitude (no knob); we take `max(D_i, floor_i)`,
            // so the floor can only tighten the metric and is a no-op for every
            // family that returns `None`. It shapes the trajectory only — the
            // converged β, the KKT certificate, and the REML/LAML the residual
            // feeds are unchanged.
            let mut joint_trust_metric_diag = joint_trust_metric_diag;
            if let Some(floor) = family.joint_trust_metric_block_floor(&states, specs)?
                && floor.len() == joint_trust_metric_diag.len()
            {
                for (d, f) in joint_trust_metric_diag.iter_mut().zip(floor.iter()) {
                    if f.is_finite() && *f > *d {
                        *d = *f;
                    }
                }
            }
            // HEAD-β JEFFREYS CACHE (gam#729/#808). The full Jeffreys/Firth triple
            // `(Φ, ∇Φ, H_Φ)` costs `p` family directional-derivative calls (the
            // `for k in 0..p` loop in `joint_jeffreys_term`); for a K-block coupled
            // family (Dirichlet/multinomial) that is the dominant per-cycle cost.
            // The head-of-cycle KKT residual, the constrained-QP step, and the
            // spectral/dense Newton step are ALL built at the SAME cycle-start β
            // (`&states`, before any step is accepted), so they need the SAME
            // triple. Compute it ONCE here and reuse, instead of three independent
            // O(p)-directional-derivative evaluations per cycle. The post-step
            // residual below is at the accepted β, so it correctly recomputes.
            // `None` when the term is condition-gated/skippable (∇Φ=0, H_Φ=0).
            let head_beta_key: Array1<f64> = flatten_state_betas(&states, specs);
            let head_jeffreys_term: Option<(Array1<f64>, Array2<f64>)> =
                if jeffreys_skippable_this_cycle {
                    None
                } else if let Some((_, grad_phi, hphi)) = jeffreys_triple_cache
                    .as_ref()
                    .filter(|(key, _, _)| beta_cache_keys_match_bitwise(key, &head_beta_key))
                {
                    // Cross-cycle cache hit: the previous cycle's post-step KKT
                    // residual already computed the exact triple at this β. Reuse.
                    Some((grad_phi.clone(), hphi.clone()))
                } else if let Some(z_joint) = joint_jeffreys_subspace.as_ref() {
                    let term = match custom_family_joint_jeffreys_term(
                        family, &states, specs, &ranges, z_joint,
                    )? {
                        Some((_phi, grad_phi, hphi))
                            if grad_phi.len() == grad_joint.len()
                                && hphi.nrows() == total_p
                                && hphi.ncols() == total_p =>
                        {
                            Some((grad_phi, hphi))
                        }
                        _ => None,
                    };
                    if let Some((grad_phi, hphi)) = term.as_ref() {
                        jeffreys_triple_cache =
                            Some((head_beta_key.clone(), grad_phi.clone(), hphi.clone()));
                    }
                    term
                } else {
                    None
                };
            // Fold the Firth/Jeffreys score `∇Φ` into the head-of-cycle KKT
            // residual when the term is armed, for the same reason as the
            // post-step residual below: the inner objective is `−ℓ + ½βᵀSβ − Φ`,
            // so the certifiable stationarity is `∇L − Sβ + ∇Φ = 0`. Without
            // this the head-of-cycle KKT exit (`current_stationarity_residual ≤
            // residual_tol`) can never fire on the near-separating span, even
            // when the iterate is the Firth optimum. No-op when the Jeffreys
            // term is unavailable or condition-gated to zero.
            let head_kkt_gradient: Option<Array1<f64>> = head_jeffreys_term
                .as_ref()
                .map(|(grad_phi, _hphi)| &grad_joint + grad_phi);
            let current_kkt_norm = exact_newton_joint_stationarity_inf_norm_from_gradient(
                head_kkt_gradient.as_ref().unwrap_or(&grad_joint),
                &states,
                specs,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                &block_constraints,
                Some(cached_active_sets.as_slice()),
                joint_lower_bounds.as_ref(),
            )?;
            if current_kkt_norm.is_finite() {
                min_certified_residual = min_certified_residual.min(current_kkt_norm);
            }
            let pcg_rel_tol = joint_pcg_eisenstat_walker_forcing(prev_kkt_norm, current_kkt_norm);

            {
                let grad_phi_inf = head_jeffreys_term
                    .as_ref()
                    .map(|(g, _)| g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
                    .unwrap_or(0.0);
                let beta_inf_probe = states
                    .iter()
                    .flat_map(|s| s.beta.iter())
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max);
                log::info!(
                    "[979-PROBE] cyc={:>3} firth_armed={} skippable={} |gradPhi|inf={:.3e} kkt={:.3e} |beta|inf={:.3e} endgame={}",
                    cycle,
                    head_jeffreys_term.is_some(),
                    jeffreys_skippable_this_cycle,
                    grad_phi_inf,
                    current_kkt_norm,
                    beta_inf_probe,
                    jeffreys_completion_endgame,
                );
            }

            let solve_joint_constraints_dense = joint_constraints.is_some()
                || !matrix_free_joint_requested
                || joint_hessian_is_dense;
            if cycle == 0 {
                log::info!(
                    "[JN-BRANCH-DIAG #1040] cycle=0 joint_constraints_is_some={} matrix_free_joint_requested={} joint_hessian_is_dense={} solve_joint_constraints_dense={} -> branch={} total_p={} levenberg_on_ill_cond={}",
                    joint_constraints.is_some(),
                    matrix_free_joint_requested,
                    joint_hessian_is_dense,
                    solve_joint_constraints_dense,
                    if solve_joint_constraints_dense && joint_constraints.is_some() {
                        "CONSTRAINED_QP"
                    } else if matrix_free_joint_requested && !joint_hessian_is_dense {
                        "MATRIX_FREE_PCG"
                    } else {
                        "DENSE_SPECTRAL"
                    },
                    total_p,
                    family.levenberg_on_ill_conditioning(),
                );
            }
            // Exact trust-region subproblem factorization (gam#979). Populated on
            // the unconstrained dense-spectral path with the metric-whitened
            // eigendecomposition of the penalized Hessian, so the trust loop below
            // re-solves the *exact* Moré–Sorensen subproblem at each trust radius
            // from one factorization — replacing the dogleg/Cauchy/box-truncation
            // globalization with the single object they all approximate. `None` on
            // the constrained-QP and matrix-free PCG paths, which keep their
            // existing globalization untouched.
            let mut joint_spectrum: Option<whitened_spectrum::WhitenedHessianSpectrum> = None;
            // DENSE-FALLBACK OPERATOR MATERIALIZATION REUSE (gam#1040). On the
            // DENSE_SPECTRAL path the inner Hessian `source` can be a matrix-free
            // `Operator` (BMS flex, large n, p below the matrix-free joint-dim
            // threshold so PCG is not requested): the dense-fallback below then
            // calls `materialize_joint_hessian_source` to form the unpenalized
            // dense `H` ONCE for the spectral `decompose`. Without capturing it,
            // the per-cycle Cauchy leg and the up-to-`JOINT_TRUST_MAX_ATTEMPTS`
            // predicted-reduction matvecs each re-apply the operator's `apply_into`
            // — an `O(n·p)` row sweep over n≈196k rows, ~25× per cycle — when the
            // identical action is already available as an `O(p²)` dense matvec.
            // Capturing the unpenalized dense here and routing those matvecs
            // through a `Dense` source is byte-identical (the dense build IS the
            // operator's action by construction of `materialize_joint_hessian_source`)
            // and removes the dominant residual per-cycle row work on this path.
            let mut materialized_dense_unpenalized: Option<Array2<f64>> = None;
            let (
                candidate_beta,
                joint_active_set,
                joint_step_spectral_nullity,
                joint_reduced_face_kind,
            ) = if solve_joint_constraints_dense
                && let Some(constraints) = joint_constraints.as_ref()
            {
                let mut lhs = match materialize_joint_hessian_source(
                    &joint_hessian_source,
                    total_p,
                    "joint Newton inner constrained Hessian materialization",
                ) {
                    Ok(matrix) => matrix,
                    Err(_) => break,
                };
                let mut exact_lhs = lhs.clone();
                add_joint_penalty_to_matrix(
                    &mut exact_lhs,
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    joint_bundle,
                );
                add_joint_penalty_to_matrix(
                    &mut lhs,
                    &ranges,
                    &s_lambdas,
                    trace_diagonal_ridge,
                    joint_bundle,
                );
                if joint_solver_diagonal_ridge != trace_diagonal_ridge {
                    for d in 0..lhs.nrows() {
                        lhs[[d, d]] += joint_solver_diagonal_ridge - trace_diagonal_ridge;
                    }
                }
                check_linear_feasibility(&beta_joint, constraints, 1e-8)
                    .map_err(|e| format!("joint Newton constrained solve [cycle={cycle}]: {e}"))?;
                let warm_joint_active =
                    flatten_joint_active_set(&cached_active_sets, &block_constraints);
                let lower_bounds = match extract_simple_lower_bounds(constraints, total_p) {
                    Ok(bounds) => bounds,
                    Err(_) => break,
                };
                // Newton IRLS step in absolute-β space:
                //
                //   β_new = H_pen⁻¹ (H_L β + ∇ℓ)
                //
                // where H_pen = H_L + S, derived from Newton's update
                //   β_new = β + H_pen⁻¹(∇ℓ − Sβ)
                //         = H_pen⁻¹(H_pen β + ∇ℓ − Sβ)
                //         = H_pen⁻¹(H_L β + ∇ℓ).
                //
                // The QP `min 0.5 β' H_pen β − rhs_beta' β` has unconstrained
                // optimum β = H_pen⁻¹ rhs_beta, so rhs_beta = H_pen β + (∇ℓ − Sβ)
                // gives the correct Newton update. Passing raw grad_joint (=∇ℓ)
                // would collapse to β = H_pen⁻¹ ∇ℓ, which at the true optimum
                // (∇ℓ = Sβ̂) gives H_pen⁻¹ Sβ̂ ≠ β̂ — wrong fixed point.
                let penalty_beta_joint = apply_joint_block_penalty(
                    &ranges,
                    &s_lambdas,
                    &beta_joint,
                    joint_mode_diagonal_ridge,
                    joint_bundle,
                );
                let mut rhs_step = &grad_joint - &penalty_beta_joint;
                // Reuse the head-β Jeffreys triple (consistently attenuated in
                // `head_jeffreys_term` — both ∇Φ and H_Φ scaled by one scalar,
                // gam#826/#872/#715). Skipped when the cheap pre-check certifies
                // well-conditioning: ∇Φ = 0 and H_Φ = 0 there, so neither
                // rhs_step nor lhs change.
                // PSD PROJECTION (gam#979). The exact divided-difference H_Φ is
                // indefinite exactly where Φ is (mixed-sign reduced spectrum at
                // off-mode trial points). The unconstrained dense-spectral path
                // consumes it exactly — the Moré–Sorensen subproblem handles
                // indefiniteness rigorously — but THIS active-set QP requires a
                // convex model (an indefinite QP cycles its active set and the
                // inner grinds the budget). Use the PSD part of H_Φ here: honest
                // magnitudes (unlike the old `K²` vec-Gram phantom), guaranteed
                // solvable QP, and the exact ∇Φ in the rhs keeps the fixed point
                // unchanged — only the convergence rate on indefinite stretches
                // degrades to the damped-Newton rate the constrained path always
                // had.
                if let Some((grad_phi, hphi)) = head_jeffreys_term.as_ref()
                    && grad_phi.len() == rhs_step.len()
                {
                    rhs_step += grad_phi;
                    exact_lhs += hphi;
                    lhs += &symmetric_psd_projection(hphi);
                }
                // The constrained QP cannot drop ker(H_pen) the way the
                // spectral range solve does. A numerical gauge therefore
                // needs positive curvature so the minimizer is unique, but
                // adding the residual-scaled μ to every diagonal also damps
                // weak IDENTIFIED modes. The 4,800-row CTN measured the result:
                // a stable 79-row face, flat objective, and residual contraction
                // of 0.9998 per cycle. `symmetric_constrained_hessian_geometry`
                // installs μ only on the certified null projector, leaving
                // range(H_pen) on the exact Newton equation. A family that
                // explicitly owns the separate full-rank ill-conditioning case
                // retains its ambient policy.
                //
                // Scale gauge curvature by the PROJECTED stationarity residual,
                // not the raw RHS: at a constrained optimum the raw RHS includes
                // the non-vanishing multiplier mass Aᵀλ, while the projected
                // residual is the actual distance from KKT and tends to zero.
                let rhs_inf = rhs_step.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                let floor_scale = if current_kkt_norm.is_finite() {
                    current_kkt_norm.min(rhs_inf)
                } else {
                    rhs_inf
                };
                let constrained_levenberg_mu = JOINT_SPECTRAL_LEVENBERG_FACTOR * floor_scale;
                // MODIFIED-NEWTON CONVEXIFICATION (gam#1040 / gam#979). The
                // exact survival marginal-slope joint NLL Hessian is INDEFINITE
                // on the flat baseline-hazard λ valley (the linear baseline +
                // the z·exp(logslope) cross-coupling carry genuine negative
                // curvature away from the optimum). The active-set QP below
                // minimizes `½βᵀHβ − rhs_betaᵀβ`; with an indefinite `H` that
                // model has a direction that LOWERS the local quadratic
                // objective while moving AWAY from the KKT point. The
                // trust-region wrapper gates acceptance on the objective-
                // reduction ratio ρ — NOT on the stationarity residual — so it
                // accepts every such step at ρ≈1 and GROWS its radius while the
                // stationarity residual DIVERGES (the measured 3.5e4 → 9.5e6
                // blow-up on the time block). The unconstrained dense-spectral
                // path never exhibits this because `WhitenedHessianSpectrum`
                // already reflects negative-curvature modes to `|γ|`; the
                // constrained branch must do the same to its dense `lhs`.
                // Reflecting (not clamping-to-zero) keeps the curvature
                // magnitude so the QP stays bounded and the step length matches
                // the dense path; at a genuine constrained optimum the reduced
                // Hessian is PSD so this is a no-op and the converged β is
                // unchanged.
                //
                // NEWTON-DECREMENT CERTIFICATE ON THE CONSTRAINED PATH
                // (gam#1040 / gam#1088). The dense-spectral branch populates
                // `joint_spectrum` (line ~1493) so the convergence loop's
                // Newton-decrement exit can terminate the geometric/linear tail
                // when the achievable model descent `½ Σ c_k²/|γ_k|` drops below
                // `objective_tol`. The constrained branch never set it, so a
                // weakly-identified survival-MS fit (the n≈2e5 logslope block,
                // step clamped by the trust region, residual creeping ~7%/cycle)
                // had no early-exit and ground the whole budget. Build the same
                // D-whitened spectrum from the penalized `lhs` (decrement reflects
                // negative modes via `.abs()` internally, so the pre-reflection
                // `lhs` is the right input) and the augmented stationarity RHS, so
                // the decrement read is consistent with the dense path. Diagnostic
                // only for the convergence test — it does NOT change the QP step.
                if let Ok(spectrum) = whitened_spectrum::WhitenedHessianSpectrum::decompose(
                    &lhs,
                    &rhs_step,
                    &joint_trust_metric_diag,
                    KKT_REFUSAL_RANK_TOL,
                ) {
                    joint_spectrum = Some(spectrum);
                }
                let constrained_geometry = symmetric_constrained_hessian_geometry(
                    &lhs,
                    constrained_levenberg_mu,
                    family.levenberg_on_ill_conditioning(),
                )?;
                if cycle <= 2 {
                    let min_eval_raw = constrained_geometry.raw_min_eigenvalue;
                    let min_eval_refl = constrained_geometry.stabilized_min_eigenvalue;
                    log::info!(
                        "[JN-REFLECT-DIAG #1040] cycle={cycle} CONSTRAINED_QP lambda_min_signed_raw={min_eval_raw:.3e} lambda_min_signed_reflected={min_eval_refl:.3e} nullity={} condition={:.3e} (reflection {})",
                        constrained_geometry.nullity,
                        constrained_geometry.condition,
                        if min_eval_refl > min_eval_raw + min_eval_raw.abs() * 1e-9 {
                            "CHANGED the spectrum"
                        } else {
                            "NO-OP (already PSD)"
                        },
                    );
                }
                // The free solve and bound-multiplier KKT test must use this
                // same convexified Hessian. Mixing the reflected step model
                // with the original indefinite curvature or the bare gradient
                // makes release and entry contradict each other (gam#979).
                let lhs = constrained_geometry.matrix;
                let rhs_beta = &lhs.dot(&beta_joint) + &rhs_step;
                let exact_face_candidate = if lower_bounds.is_none() {
                    if let Some(active_rows) = warm_joint_active.as_deref() {
                        certified_reduced_face_newton_candidate(
                            &exact_lhs,
                            &rhs_step,
                            &beta_joint,
                            constraints,
                            active_rows,
                        )?
                    } else {
                        None
                    }
                } else {
                    None
                };
                let exact_face_kind = exact_face_candidate.as_ref().map(|(_, _, exact)| *exact);
                let solve_result = if let Some((candidate, active, _)) = exact_face_candidate {
                    Ok((candidate, active))
                } else if let Some(bounds) = lower_bounds.as_ref() {
                    solve_quadratic_with_simple_lower_bounds(
                        &lhs,
                        &rhs_beta,
                        &beta_joint,
                        bounds,
                        warm_joint_active.as_deref(),
                    )
                } else {
                    gam_solve::active_set::solve_quadratic_with_constraint_set(
                        &lhs,
                        &rhs_beta,
                        &beta_joint,
                        constraints,
                        warm_joint_active.as_deref(),
                    )
                    .map_err(|e| e.to_string())
                };
                match solve_result {
                    Ok((beta_new, active_set)) => {
                        // gam#979 constrained-QP probe (temporary): per-cycle
                        // active-set size + ‖β‖∞ so the failing survival pytest's
                        // captured WARN output distinguishes (a) a THRASHING active
                        // set (rows change every cycle → the QP never settles) from
                        // (c) a STABLE set with a blowing-up free direction
                        // (near-separation: ‖β‖∞ grows unbounded). WARN reaches the
                        // failing-test capture; the cond/nullity/diagnosis come from
                        // the existing `format_structured_log` at the refused exit.
                        log::warn!(
                            "[gam#979 constrained-QP] cycle={} path={} warm_rows={} active_set_rows={} beta_inf={:.4e}",
                            cycle,
                            if exact_face_kind == Some(true) {
                                "exact-face"
                            } else if exact_face_kind == Some(false) {
                                "tangent-face"
                            } else if lower_bounds.is_some() {
                                "simple"
                            } else {
                                "linear"
                            },
                            warm_joint_active.as_ref().map_or(0, |v| v.len()),
                            active_set.len(),
                            beta_new.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
                        );
                        (beta_new, Some(active_set), 0usize, exact_face_kind)
                    }
                    Err(error) => {
                        return Err(format!(
                            "joint constrained Newton QP failed at cycle {cycle} \
                                 (constraint_rows={}, warm_active_rows={}, beta_inf={:.6e}, \
                                 rhs_inf={:.6e}): {error}",
                            constraints.nrows(),
                            warm_joint_active.as_ref().map_or(0, Vec::len),
                            beta_joint
                                .iter()
                                .map(|value| value.abs())
                                .fold(0.0_f64, f64::max),
                            rhs_step
                                .iter()
                                .map(|value| value.abs())
                                .fold(0.0_f64, f64::max),
                        ));
                    }
                }
            } else {
                // Stationarity residual: r = S*beta - gradient (for penalized NLL)
                let penalty_beta = apply_joint_block_penalty(
                    &ranges,
                    &s_lambdas,
                    &beta_joint,
                    joint_mode_diagonal_ridge,
                    joint_bundle,
                );
                let mut rhs = &grad_joint - &penalty_beta;
                // Universal robustness: fold the family-general
                // Jeffreys/Firth curvature `H_Φ` and score `∇Φ` into BOTH the
                // matrix-free PCG step AND the dense spectral fallback below,
                // scoped to the full-span basis `Z_J`. Computed ONCE here
                // so the matvec closure and the RHS share the SAME term and the
                // fallback does not recompute it. The inner objective is
                // `−ℓ + ½βᵀSβ − Φ`, so the Newton system the step must solve is
                //   (H + S_λ + H_Φ) δ = (∇ℓ − S_λβ) + ∇Φ.
                // Previously the PCG matvec applied only `H + S_λ` and its RHS
                // omitted `∇Φ`, so on the matrix-free path (large p / large n)
                // Firth was a SILENT NO-OP: the proper-prior never reached the
                // step that actually moves β, leaving separation/under-
                // identification uncured exactly where the dense route is not
                // taken. The dense route (small p, e.g. BMS p≈51) was already
                // correct. `H_Φ` is the full-span Gauss-Newton surrogate
                // `½ J H_id⁻¹ Jᵀ` (Z_J = identity ⇒ p×p, not low-rank), but the
                // conditioning gate in `joint_jeffreys_term` returns the zero
                // term on every well-conditioned fit, so this only arms on the
                // near-separating span
                // — and `hphi` is materialized once per cycle regardless, so the
                // matvec adds only one O(p²) HVP, preserving the matrix-free
                // path's asymptotics where Firth is negligible (term = `None`).
                // Cheap pre-check certified well-conditioned ⇒ the exact term
                // is the zero contribution (∇Φ = 0, H_Φ = 0). Short-circuit to
                // `None` WITHOUT materializing the dense joint Hessian or running
                // the O(p³) reduced eigendecomposition — this is the matrix-free
                // PCG hot path, where forming a dense p×p H_Φ every cycle was the
                // regression. Byte-identical to the gated-off dense path: `rhs`
                // is left as `∇ℓ − S_λβ` and no H_Φ is folded into the matvec.
                // Reuse the head-β Jeffreys triple (computed once this cycle);
                // this Newton step is built at the same cycle-start β.
                let inner_jeffreys_term: Option<(Array1<f64>, Array2<f64>)> =
                    match head_jeffreys_term.as_ref() {
                        Some((grad_phi, hphi)) if grad_phi.len() == rhs.len() => {
                            rhs += grad_phi;
                            Some((grad_phi.clone(), hphi.clone()))
                        }
                        _ => None,
                    };
                // PSD PROJECTION for the SPD-PCG matvec (gam#979): the exact
                // divided-difference H_Φ can be indefinite at off-mode trial
                // points, which breaks the SPD-CG contract. The matvec uses its
                // PSD part; the dense spectral fallback below keeps the EXACT
                // (possibly indefinite) H_Φ — the Moré–Sorensen subproblem
                // handles it rigorously.
                let inner_jeffreys_hphi: Option<Arc<Array2<f64>>> = inner_jeffreys_term
                    .as_ref()
                    .map(|(_grad_phi, hphi)| Arc::new(symmetric_psd_projection(hphi)));
                let pcg_started = std::time::Instant::now();
                let pcg_requested = matrix_free_joint_requested
                    && !joint_hessian_is_dense
                    && !returned_mode_curvature_pending;
                let mut spectral_nullity_for_step = 0usize;
                let mut delta = if pcg_requested {
                    let preconditioner_diag = match &joint_hessian_source {
                        JointHessianSource::Dense(h_joint) => joint_penalty_preconditioner_diag(
                            &h_joint.diag().to_owned(),
                            &ranges,
                            &s_lambdas,
                            joint_solver_diagonal_ridge,
                            joint_bundle,
                        ),
                        JointHessianSource::Operator { diagonal, .. } => {
                            joint_penalty_preconditioner_diag(
                                diagonal,
                                &ranges,
                                &s_lambdas,
                                joint_solver_diagonal_ridge,
                                joint_bundle,
                            )
                        }
                    };
                    // Pre-allocate the penalty workspace ONCE outside the
                    // PCG closure so each CG iter (called hundreds-to-
                    // thousands of times per outer iter at large scale)
                    // reuses the buffer instead of allocating per call.
                    // RefCell because solve_spd_pcg* expects `Fn` (immutable
                    // borrow of captures) and we need interior mutability
                    // to write into the workspace.
                    let penalty_workspace = RefCell::new(Array1::<f64>::zeros(total_p));
                    // Capture the Jeffreys/Firth curvature for the matvec. When
                    // armed (and nonzero past the conditioning gate) the PCG
                    // operator becomes `H + S_λ + H_Φ`, matching the augmented
                    // RHS `(∇ℓ − S_λβ) + ∇Φ` set above and the dense spectral
                    // fallback. `None` keeps the unaugmented matvec.
                    let pcg_hphi_dense = inner_jeffreys_hphi.clone();
                    let pcg_hphi_op = inner_jeffreys_hphi.clone();
                    match &joint_hessian_source {
                        JointHessianSource::Dense(h_joint) => {
                            gam_linalg::utils::solve_spd_pcg_with_info_into(
                                |v, out| {
                                    // h_joint * v -> out (faer-backed, no alloc)
                                    gam_linalg::faer_ndarray::fast_av_view_into(
                                        h_joint,
                                        v,
                                        out.view_mut(),
                                    );
                                    let mut pen = penalty_workspace.borrow_mut();
                                    apply_joint_block_penalty_into(
                                        &ranges,
                                        &s_lambdas,
                                        v,
                                        joint_solver_diagonal_ridge,
                                        &mut pen,
                                        joint_bundle,
                                    );
                                    *out += &*pen;
                                    if let Some(hphi) = pcg_hphi_dense.as_ref() {
                                        *out += &hphi.dot(v);
                                    }
                                },
                                &rhs,
                                &preconditioner_diag,
                                pcg_rel_tol,
                                JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                            )
                            .map(|(solution, info)| {
                                log_joint_pcg_diagnostics(
                                    cycle,
                                    total_p,
                                    total_joint_n,
                                    &preconditioner_diag,
                                    &info,
                                );
                                solution
                            })
                        }
                        JointHessianSource::Operator { apply_into, .. } => {
                            let apply_h_into = Arc::clone(apply_into);
                            gam_linalg::utils::solve_spd_pcg_with_info_into(
                                |v, out| {
                                    if let Err(error) = apply_h_into(v, out) {
                                        log::warn!(
                                            "joint Newton inner operator matvec failed: {error}"
                                        );
                                        out.fill(0.0);
                                    }
                                    let mut pen = penalty_workspace.borrow_mut();
                                    apply_joint_block_penalty_into(
                                        &ranges,
                                        &s_lambdas,
                                        v,
                                        joint_solver_diagonal_ridge,
                                        &mut pen,
                                        joint_bundle,
                                    );
                                    *out += &*pen;
                                    if let Some(hphi) = pcg_hphi_op.as_ref() {
                                        *out += &hphi.dot(v);
                                    }
                                },
                                &rhs,
                                &preconditioner_diag,
                                pcg_rel_tol,
                                JOINT_PCG_MAX_ITER_MULTIPLIER * total_p.max(1),
                            )
                            .map(|(solution, info)| {
                                log_joint_pcg_diagnostics(
                                    cycle,
                                    total_p,
                                    total_joint_n,
                                    &preconditioner_diag,
                                    &info,
                                );
                                solution
                            })
                        }
                    }
                } else {
                    None
                };
                if pcg_requested {
                    log::info!(
                        "[PIRLS/joint-PCG] cycle {:>3} | n={} p={} solved={} elapsed={:.3}s",
                        cycle,
                        total_joint_n,
                        total_p,
                        delta.is_some(),
                        pcg_started.elapsed().as_secs_f64()
                    );
                }
                if delta.is_none() {
                    if pcg_requested {
                        break;
                    }
                    let mut lhs_true = match materialize_joint_hessian_source(
                        &joint_hessian_source,
                        total_p,
                        "joint Newton inner dense fallback Hessian materialization",
                    ) {
                        Ok(matrix) => matrix,
                        Err(_) => break,
                    };
                    // Capture the unpenalized dense `H` for the rest of this
                    // cycle (gam#1040): the Cauchy leg and trust-region
                    // predicted-reduction matvecs below can then reuse it as a
                    // cheap `O(p²)` dense matvec instead of re-applying a
                    // matrix-free operator `O(n·p)` per attempt. Only when the
                    // source is an `Operator` — a `Dense` source already gives
                    // those matvecs the fast path, so cloning would be waste.
                    if matches!(&joint_hessian_source, JointHessianSource::Operator { .. }) {
                        materialized_dense_unpenalized = Some(lhs_true.clone());
                    }
                    // Snapshot the Jeffreys information matrix only when a
                    // family supplies the contracted completion. The generic
                    // pairwise fallback costs p(p+1)/2 full second-directional
                    // Hessian passes; at biobank scale (BMS p=35, n≈196k) it
                    // turns a near-converged polishing cycle into ~50s of row
                    // work. Without a contracted hook the divided-difference
                    // H_phi model remains first-order correct and the KKT
                    // certificate owns convergence.
                    let jeffreys_completion_requested =
                        family.joint_jeffreys_information_contracted_trace_hessian_available();
                    let h_info_for_completion = (jeffreys_completion_endgame
                        && inner_jeffreys_term.is_some()
                        && jeffreys_completion_requested)
                        .then(|| family.joint_jeffreys_information_with_specs(&states, specs))
                        .transpose()?
                        .flatten();
                    add_joint_penalty_to_matrix(
                        &mut lhs_true,
                        &ranges,
                        &s_lambdas,
                        joint_mode_diagonal_ridge,
                        joint_bundle,
                    );
                    // Universal robustness: add the
                    // family-general Jeffreys curvature `H_Phi` to the
                    // penalized Hessian. This is the Tier-B coupled-Newton form
                    // of Firth: the reduced Fisher information `Z_J^T H Z_J`
                    // supplies the missing O(n) curvature that bounds a
                    // near-separating coefficient to O(1). When the Jeffreys
                    // term is unavailable, the step stays unaugmented.
                    //
                    // `∇Φ` is NOT re-added here: `rhs` (and thus `spectral_rhs`)
                    // already carries `+∇Φ` from the single shared computation
                    // above, and we REUSE that same `H_Φ` here rather than
                    // recomputing the (O(p) directional-derivative) term — the
                    // dense fallback and the matrix-free PCG step now solve the
                    // SAME Jeffreys-augmented Newton system.
                    let spectral_rhs = rhs.clone();
                    if let Some((_grad_phi, hphi)) = inner_jeffreys_term.as_ref() {
                        lhs_true += hphi;
                        // ENDGAME EXACTNESS (gam#979). The divided-difference
                        // H_Φ omits the second-directional-Hessian remainder
                        // `½ tr(K · D_ab)`; near a Firth-active mode that
                        // remainder is comparable to the kept curvature, so
                        // Newton converges only linearly (a residual sawtooth
                        // plateauing just above the certificate tolerance —
                        // enough mode noise to swamp outer finite differences
                        // and feed the IFT near-flat-kernel amplification).
                        // Once the residual enters the convergence band, add
                        // the exact completion so the model is the true
                        // Hessian of the Φ-augmented objective and the endgame
                        // is quadratic. A family contracted trace hook can
                        // supply it at any width; the pairwise `p(p+1)/2`
                        // fallback remains limited to moderate p. `None`
                        // degrades safely to the divided-difference model.
                        if let (Some(h_info), Some(z_joint)) = (
                            h_info_for_completion.as_ref(),
                            joint_jeffreys_subspace.as_ref(),
                        ) && let Some(completion) =
                            custom_family_joint_jeffreys_second_order_completion(
                                family, &states, specs, h_info, z_joint, false,
                            )?
                        {
                            // TRUST-REGION GATE (gam#1607): fold the completion
                            // only when `H_Φ + completion` stays PSD. In the
                            // near-separable regime the `−½ tr(K·D_ab)` remainder
                            // explodes negative and cancels `H_Φ`, leaving the
                            // augmented inner Hessian strongly indefinite. The
                            // trust-region spectral solve below would reflect those
                            // negative modes, but that turns the quadratic endgame
                            // back into a reflected-descent crawl that plateaus
                            // above the certificate tolerance ("inner solve exited
                            // before convergence"). Keeping the bounded PSD `H_Φ`
                            // model preserves the linear-but-monotone endgame the
                            // divided-difference solve already certifies.
                            if custom_family_jeffreys_completion_preserves_psd(hphi, &completion) {
                                lhs_true += &completion;
                            }
                        }
                    }
                    // Single metric-whitened eigendecomposition drives BOTH the
                    // seed step and every trust-region re-solve this cycle
                    // (gam#979). The prior code ran a SECOND O(p³)
                    // eigendecomposition of the raw Hessian here purely to form
                    // the seed step — doubling the dominant per-cycle cost on the
                    // ~5 s/cycle ill-conditioned survival marginal-slope inner.
                    // The exact trust-region multiplier λ (chosen so ‖δ‖_D = r)
                    // subsumes the old self-vanishing Levenberg-μ seed: `decompose`
                    // whitens by the trust metric so the penalty (λ~e²⁴) and the
                    // likelihood scales are throttled uniformly — the scale
                    // invariance the multiplicative μ approximated. `lhs_true`
                    // already carries the penalty and the Firth/Jeffreys curvature
                    // H_Φ and `spectral_rhs` the augmented stationarity RHS, so the
                    // subproblem model matches the predicted-reduction model and the
                    // accept/reject gain ratio exactly.
                    let spectrum = whitened_spectrum::WhitenedHessianSpectrum::decompose(
                        &lhs_true,
                        &spectral_rhs,
                        &joint_trust_metric_diag,
                        KKT_REFUSAL_RANK_TOL,
                    )?;
                    // Seed = the unconstrained (Moore–Penrose, range-restricted)
                    // exact step, so cycle 0 can take the full Newton step on a
                    // well-conditioned model (the cycle-0 radius bump below relies
                    // on this); the trust loop re-solves at finite radius for every
                    // subsequent attempt. An indefinite model reflects negative
                    // curvature to |λ|, exactly as the prior spectral solve did.
                    let spectral_step = spectrum.trust_region_step(f64::INFINITY);
                    spectral_nullity_for_step = spectral_step.nullity;
                    // gam#979: Levenberg shift-to-PD of the SEED search direction on
                    // the rigid ill-conditioned path. When the whitened inner Hessian
                    // is indefinite the Moré–Sorensen step reflects the negative
                    // modes to |λ|; on the near-separable coupled marginal-slope
                    // surface those reflected modes then ride a poor gain ratio that
                    // shrinks the single scalar trust radius, which clamps the
                    // *well-conditioned* modes that DO carry real descent — the
                    // measured "reflected-descent crawl" where the residual plateaus
                    // (‖g‖~1e-1) while the objective creeps down ~1e-3/cycle and the
                    // solve exhausts its budget without ever reaching residual_tol
                    // (the binary twin of the survival-MS oversmoothed-ρ endgame).
                    // Once the residual has stalled for a few cycles, seed the trust
                    // loop from a genuinely convex modified-Newton step: add
                    // μ·D_trust (μ just above |λ_min| so the reflected modes become
                    // gently positive while the well-conditioned modes, |λ|≫μ, keep
                    // their full Newton step) and re-solve once. This is a modified-
                    // Newton SEARCH DIRECTION only — the trust-region accept/reject
                    // still judges it against the true `lhs_true` model, `joint_spectrum`
                    // stays the exact (unshifted) spectrum for the trust re-solves and
                    // the Newton-decrement certificate, and μ is applied ONLY while the
                    // Hessian is indefinite (it vanishes once the endgame becomes PD),
                    // so a healthy convex fit is byte-unchanged and no non-minimum can
                    // be certified. Family-gated on `levenberg_on_ill_conditioning()`.
                    const JOINT_REFLECTED_CONVEXIFY_STALL_WINDOW: usize = 3;
                    const JOINT_REFLECTED_CONVEXIFY_MARGIN: f64 = 1.5;
                    let mut seed_delta = spectral_step.delta;
                    if family.levenberg_on_ill_conditioning()
                        && spectral_step.reflected_negative_modes > 0
                        && cycles_since_residual_improved >= JOINT_REFLECTED_CONVEXIFY_STALL_WINDOW
                        && spectral_step.most_negative_eigenvalue.is_finite()
                        && spectral_step.most_negative_eigenvalue < 0.0
                    {
                        let mu = spectral_step.most_negative_eigenvalue.abs()
                            * JOINT_REFLECTED_CONVEXIFY_MARGIN;
                        if mu.is_finite() && mu > 0.0 {
                            let mut lhs_convex = lhs_true.clone();
                            for d in 0..lhs_convex.nrows() {
                                lhs_convex[[d, d]] += mu * joint_trust_metric_diag[d];
                            }
                            if let Ok(convex_spectrum) =
                                whitened_spectrum::WhitenedHessianSpectrum::decompose(
                                    &lhs_convex,
                                    &spectral_rhs,
                                    &joint_trust_metric_diag,
                                    KKT_REFUSAL_RANK_TOL,
                                )
                            {
                                let convex_step = convex_spectrum.trust_region_step(f64::INFINITY);
                                if convex_step.reflected_negative_modes == 0
                                    && convex_step.delta.iter().all(|v| v.is_finite())
                                {
                                    log::info!(
                                        "[PIRLS/joint-Newton] cycle {cycle:>3} | gam#979 \
                                             Levenberg shift-to-PD seed: μ={mu:.3e}·D convexified \
                                             {} reflected mode(s) (λ_min={:.3e}) after {} stalled \
                                             cycle(s); seeding the trust loop from the convex \
                                             modified-Newton step to break the reflected-descent \
                                             crawl",
                                        spectral_step.reflected_negative_modes,
                                        spectral_step.most_negative_eigenvalue,
                                        cycles_since_residual_improved,
                                    );
                                    seed_delta = convex_step.delta;
                                }
                            }
                        }
                    }
                    if spectral_step.reflected_negative_modes > 0 {
                        log::info!(
                            "[PIRLS/joint-Newton] cycle {cycle:>3} | indefinite inner \
                                 Hessian: reflected {}/{} negative-curvature modes to |λ| \
                                 (λ_min={:.3e}); proceeding with modified-Newton descent step \
                                 under trust-region globalization",
                            spectral_step.reflected_negative_modes,
                            total_p,
                            spectral_step.most_negative_eigenvalue,
                        );
                    }
                    {
                        log::info!(
                            "[979-DIAG] cycle {cycle:>3} spectral solve: nullity@{:.0e}={}/{} \
                             |P0 rhs|∞={:.3e} |P+ rhs|∞={:.3e} λ_min+={:.3e} λ_max={:.3e} reflected={}",
                            spectral_step.rank_tol,
                            spectral_step.nullity,
                            total_p,
                            spectral_step.null_rhs_inf,
                            spectral_step.range_rhs_inf,
                            spectral_step.lambda_min_positive,
                            spectral_step.lambda_max_abs,
                            spectral_step.reflected_negative_modes,
                        );
                    }
                    delta = Some(seed_delta);
                    // The same factorization powers every trust-radius re-solve
                    // in the loop below (gam#979) — no second eigendecomposition.
                    // `spectrum` is the EXACT (unshifted) Hessian: the gam#979
                    // Levenberg shift-to-PD above only reshapes the SEED direction,
                    // so the trust re-solves and the Newton-decrement certificate
                    // keep judging against the true model.
                    joint_spectrum = Some(spectrum);
                }

                let Some(delta) = delta else {
                    break; // Fall back to blockwise
                };
                if !delta.iter().all(|v| v.is_finite()) {
                    break; // Fall back to blockwise
                }
                (
                    beta_joint.clone() + &delta,
                    None,
                    spectral_nullity_for_step,
                    None,
                )
            };
            // Hessian-source build (and any QP solve immediately above) are
            // done by the time we reach `delta`. Capture the wall-clock
            // before the line-search phase so the end-of-cycle summary can
            // attribute time correctly between the Hessian/QP and the
            // backtracking step search.
            let hessian_and_qp_elapsed = hessian_started.elapsed();
            drop(hessian_scope_guard);
            let line_search_started = std::time::Instant::now();
            log::info!(
                "[joint-newton-tr] phase=line_search cycle={} r={:.3e} hessian_qp_elapsed={:.3}s",
                cycle,
                joint_trust_radius,
                hessian_and_qp_elapsed.as_secs_f64(),
            );
            let delta = &candidate_beta - &beta_joint;
            // Effective Hessian source for the remaining per-cycle matvecs
            // (Cauchy leg + trust-region predicted reduction). When the dense
            // fallback above materialized a matrix-free `Operator` to dense, route
            // those matvecs through that `Dense` snapshot so each is an `O(p²)`
            // GEMV rather than an `O(n·p)` operator row-sweep repeated up to
            // `JOINT_TRUST_MAX_ATTEMPTS` times (gam#1040). Byte-identical action
            // (the dense build IS the operator's action by construction); falls
            // back to the original source when no dense snapshot was taken (the
            // already-`Dense` and PCG paths).
            let dense_snapshot_source =
                materialized_dense_unpenalized.map(JointHessianSource::Dense);
            let effective_hessian_source: &JointHessianSource = dense_snapshot_source
                .as_ref()
                .unwrap_or(&joint_hessian_source);

            // Trust-region globalization for the joint Newton proposal.  The
            // previous implementation used up to eight backtracking likelihood
            // evaluations (each can build the exact joint workspace at large-scale
            // scale).  Here the step is truncated before evaluation and the
            // single trial objective is accepted only when the actual decrease
            // is positive relative to the local quadratic model.
            let step_inf = delta.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);

            let old_beta: Vec<Array1<f64>> = states.iter().map(|s| s.beta.clone()).collect();
            // Firth value Φ at the OLD (start-of-cycle) β, folded under the SAME
            // skippable gate the trial uses below — so `actual_reduction =
            // old_objective − trialobjective` compares two points on one objective
            // `−ℓ + ½βᵀSβ − Φ` (gam#826/#872). `lastobjective` is the pure
            // quadratic-penalized objective; subtract the gated old-β Φ here.
            let old_phi = if !jeffreys_skippable_this_cycle {
                joint_jeffreys_subspace
                    .as_ref()
                    .map(|z_joint| {
                        custom_family_joint_jeffreys_value(family, &states, specs, &ranges, z_joint)
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let old_objective = lastobjective - old_phi;
            // Row measure observed by the objective at β. `lastobjective` was
            // set on the previous cycle (or at function entry) under `options`;
            // see top-of-cycle capture for rationale.
            let tr_row_measure_old_objective =
                gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
            let mut accepted = false;
            let mut accepted_joint_workspace: Option<Arc<dyn ExactNewtonJointHessianWorkspace>> =
                None;
            let mut line_search_attempts = 0usize;

            // Pure Newton must take a full step on the first cycle of an
            // exact quadratic problem (i.e. converge in one cycle when the
            // model is exact). The trust-region globalization above must not
            // truncate the very first proposal merely because the hard-coded
            // initial radius (1.0) is smaller than the natural Newton-step norm.
            //
            // There are two norms in play:
            //   * the constrained-QP / dogleg paths truncate per block against
            //     `joint_block_trust_radii`;
            //   * the exact spectral trust-region path solves one global
            //     Moré–Sorensen problem against `joint_trust_radius`.
            //
            // The old cycle-0 bump only raised the per-block radii and then set the
            // global radius to `max(block_norms)`. For a multiblock exact quadratic
            // with a diagonal metric that leaves a full Newton step like
            // `[0.8, 0.8]` inside every per-block ball (`max = 0.8`) but outside
            // the global spectral ball (`sqrt(0.8² + 0.8²) = 1.13`). Once the
            // constrained branch started populating `joint_spectrum` for the
            // Newton-decrement certificate, the line search correctly used the
            // spectral path and incorrectly clipped that exact feasible Newton
            // step to radius 1.0, preventing one-cycle KKT convergence. Bump the
            // global radius to the full metric norm while still bumping each block
            // radius to its own block norm; this keeps the first exact Newton step
            // untruncated in both globalization modes and leaves the standard
            // adaptive shrink/expand for subsequent cycles.
            if cycle == 0 && joint_step_spectral_nullity == 0 {
                let initial_global_norm =
                    joint_trust_region_metric_step_norm(&delta, &joint_trust_metric_diag);
                let initial_block_norms = joint_trust_region_block_metric_norms(
                    &delta,
                    &ranges,
                    &joint_trust_metric_diag,
                );
                for (radius, norm) in joint_block_trust_radii.iter_mut().zip(initial_block_norms) {
                    if norm.is_finite() && norm > *radius {
                        *radius = norm;
                    }
                }
                let block_radius = joint_block_trust_radii
                    .iter()
                    .copied()
                    .fold(0.0_f64, f64::max);
                joint_trust_radius = if initial_global_norm.is_finite() {
                    block_radius.max(initial_global_norm)
                } else {
                    block_radius
                };
                if !joint_trust_radius.is_finite() || joint_trust_radius <= 0.0 {
                    joint_trust_radius = 1.0;
                }
            }

            let penalty_beta = apply_joint_block_penalty(
                &ranges,
                &s_lambdas,
                &beta_joint,
                joint_mode_diagonal_ridge,
                joint_bundle,
            );
            // Stationarity RHS for the trust-region quadratic model. When the
            // Jeffreys/Firth term is armed the inner objective is `−ℓ+½βᵀSβ+Φ`, so
            // the model RHS is `∇L − Sβ + ∇Φ` — the SAME augmented RHS the Newton
            // step solves and the H_Φ-augmented `hpen_delta` below pairs with. Using
            // the bare `∇L − Sβ` here desyncs `predicted_reduction` from the
            // augmented step + the Φ-augmented `actual_reduction`, which is what
            // froze the coupled K-block line search (gam#729/#715). No-op when the
            // term is condition-gated/unavailable (∇Φ=0).
            let mut rhs = &grad_joint - &penalty_beta;
            if let Some((grad_phi, _hphi)) = head_jeffreys_term.as_ref()
                && grad_phi.len() == rhs.len()
            {
                rhs += grad_phi;
            }
            let beta_inf = states
                .iter()
                .flat_map(|s| s.beta.iter().copied())
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let step_tol = inner_tol * (1.0 + beta_inf);
            let objective_tol = inner_tol * (1.0 + old_objective.abs());
            // Scale the KKT residual tolerance against the natural magnitude
            // of ‖Sβ − ∇L‖∞ (i.e. max(‖∇L‖∞, ‖Sβ‖∞)), not the objective. The
            // gradient and Sβ scale independently of the likelihood — at
            // large scale with |β|∞ ~ 10²–10³ and non-trivial smoothing,
            // ‖Sβ‖∞ can sit orders of magnitude above |obj| and FP noise
            // alone keeps the residual above any obj-scaled tol, so KKT is
            // never certified even when the iterate is the true optimum.
            let grad_inf = grad_joint
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max);
            let penalty_inf = penalty_beta
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max);
            let residual_tol = inner_tol * (1.0 + grad_inf.max(penalty_inf));
            last_residual_tol = residual_tol;
            let current_stationarity_residual = current_kkt_norm;
            // Local-mode certificate: first-order KKT and a small Newton
            // proposal are sufficient only when the exact penalized Hessian
            // has no resolvable negative curvature. CTN's squared SCOP shape
            // chart contains finite strict saddles where both the score and the
            // unconstrained reflected-Newton proposal are exactly zero. Calling
            // those points converged bypasses the finite-radius Moré–Sorensen
            // hard case below and hands an indefinite matrix to the outer LAML
            // determinant. The spectrum uses the same numerical-rank threshold
            // for this certificate and for the hard-case step, so a direction
            // that blocks convergence is guaranteed to be actionable.
            // Conditioning the valid local-minimum exit on additional evidence
            // of objective progress in the previous cycle would refuse to
            // recognize convergence at a starting point that already sits
            // at the optimum (e.g. balanced data with an intercept-only
            // fit, where ∇ℓ vanishes by symmetry from cycle 0 and the
            // Newton step is identically zero so the trust-region search
            // can never produce a strictly negative actual reduction).
            let has_resolvable_negative_curvature = joint_spectrum
                .as_ref()
                .is_some_and(|spectrum| spectrum.has_resolvable_negative_curvature());
            if returned_mode_curvature_pending {
                returned_mode_curvature_pending = false;
                if joint_spectrum.is_none() {
                    return Err(
                        "returned-mode curvature cycle did not produce the required exact joint spectrum"
                            .to_string(),
                    );
                }
                if has_resolvable_negative_curvature {
                    log::info!(
                        "[PIRLS/joint-Newton mode certificate] tentative first-order convergence revoked at returned beta; continuing this exact spectral cycle through the finite-radius negative-curvature hard case"
                    );
                    converged = false;
                    returned_mode_curvature_certified = false;
                    last_cycle_residual_below_tol = false;
                    last_cycle_obj_change_below_tol = false;
                    min_certified_residual = f64::INFINITY;
                    best_residual_seen = f64::INFINITY;
                    cycles_since_residual_improved = 0;
                    residual_descent_history.clear();
                    tr_clamped_during_stall = false;
                    residual_rate_history.clear();
                    merit_window.clear();
                    prev_rejected_trust_radius = None;
                    consecutive_held_rejected_cycles = 0;
                    prev_rejected_first_attempt_objective = None;
                    consecutive_identical_rejected_cycles = 0;
                    consecutive_all_reject_at_floor_cycles = 0;
                    last_joint_math = None;
                    last_kkt_refusal_report = None;
                    prev_kkt_norm = None;
                    geometric_tail_history.clear();
                } else {
                    log::info!(
                        "[PIRLS/joint-Newton mode certificate] returned beta certified by the next exact spectral cycle"
                    );
                    returned_mode_curvature_certified = true;
                    cached_joint_workspace = hessian_workspace_for_cycle.take();
                    cycles_done = cycle;
                    break;
                }
            }
            if current_stationarity_residual <= residual_tol
                && step_inf <= step_tol
                && !has_resolvable_negative_curvature
            {
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | pre-line-search converged: proposal_inf={:.3e} (tol={:.3e}) | residual={:.3e} (tol={:.3e})",
                    cycle,
                    step_inf,
                    step_tol,
                    current_stationarity_residual,
                    residual_tol,
                );
                // Pre-line-search convergence: β did not move this cycle (the
                // proposal was at the step-tolerance floor), so the cycle
                // workspace is still at the converged β and the post-loop
                // covariance/IFT assembly can reuse it instead of rebuilding the
                // full per-row kernel cache at the same β.
                cached_joint_workspace = hessian_workspace_for_cycle.take();
                cycles_done = cycle;
                converged = true;
                returned_mode_curvature_certified = joint_constraints.is_none()
                    && joint_jeffreys_subspace.is_none()
                    && joint_spectrum.is_some();
                break;
            }
            if current_stationarity_residual <= residual_tol
                && step_inf <= step_tol
                && has_resolvable_negative_curvature
            {
                log::info!(
                    "[PIRLS/joint-Newton] cycle {cycle:>3} | first-order stationary strict saddle; refusing convergence and invoking the finite-radius negative-curvature hard case"
                );
            }

            // Trust-region retries preserve the objective-decrease guarantee
            // when the initial radius is too optimistic. If the Newton proposal
            // is not a descent direction for the penalized quadratic model,
            // switch once to a diagonally preconditioned gradient step and keep
            // the same exact full-objective accept/reject test.
            const JOINT_TRUST_MAX_ATTEMPTS: usize = 24;
            let mut search_delta = delta.clone();
            let search_joint_active_set: Option<Vec<usize>> = joint_active_set.clone();
            // A constrained Newton step can discover a different critical cone.
            // Residual/objective-rate samples collected on the previous active
            // face are not evidence about stationarity on the new face: the KKT
            // projection itself changes when the active rows change. Remember the
            // accepted transition so the post-step convergence machinery can
            // start its plateau/descent evidence from this face only (gam#979).
            let active_face_before_step =
                flatten_joint_active_set(&cached_active_sets, &block_constraints);
            let mut accepted_active_face_changed = false;
            let mut tried_preconditioned_descent = false;
            // Dogleg Cauchy leg (gam#826/#808). Compute the unconstrained Cauchy
            // point of the penalized (Firth-augmented) quadratic model ONCE per
            // cycle: the M-metric steepest-descent direction `p_sd = M⁻¹·rhs`
            // and its curvature `p_sd·H·p_sd` (a coupled Hessian-vector product,
            // so it must be hoisted out of the radius-shrink loop). When the
            // Newton step exceeds a block's trust radius the dogleg blends
            // toward this Cauchy leg, guaranteeing at least the Cauchy decrease
            // even when the spectral Newton step is numerically frozen at the
            // oversmoothed seed (the high-curvature log_sigma block's Newton
            // component is `O(g/λ) ≈ 5e-21`). `joint_active_set` is the
            // unconstrained joint Newton path; the constrained-QP path keeps its
            // own globalization, so the dogleg is only built (and used) when no
            // active set is in force.
            // Only the dogleg/box-truncation globalization (no spectrum) ever
            // consumes the Cauchy leg; when the exact Moré–Sorensen spectrum is
            // present the trust loop re-solves from it and `dogleg_cauchy` is dead.
            // Skipping its construction there removes one coupled Hessian-vector
            // product per cycle — an `O(n·p)` operator row-sweep on the matrix-free
            // DENSE_SPECTRAL path that produced no value (gam#1040).
            let dogleg_cauchy: Option<Array1<f64>> =
                if search_joint_active_set.is_none() && joint_spectrum.is_none() {
                    let mut p_sd = Array1::<f64>::zeros(total_p);
                    for (i, (r, w)) in rhs.iter().zip(joint_trust_metric_diag.iter()).enumerate() {
                        p_sd[i] = r / positive_joint_diagonal_entry(*w);
                    }
                    let mut h_psd = Array1::<f64>::zeros(total_p);
                    let mut cauchy_penalty_scratch = Array1::<f64>::zeros(total_p);
                    match apply_joint_penalized_hessian_into_with_workspace(
                        effective_hessian_source,
                        &ranges,
                        &s_lambdas,
                        joint_mode_diagonal_ridge,
                        &p_sd,
                        &mut h_psd,
                        &mut cauchy_penalty_scratch,
                        joint_bundle,
                    ) {
                        Ok(()) => {
                            if let Some((_grad_phi, hphi)) = head_jeffreys_term.as_ref() {
                                h_psd += &hphi.dot(&p_sd);
                            }
                            let cauchy = joint_cauchy_step(&rhs, &p_sd, &h_psd);
                            if cauchy.iter().all(|v| v.is_finite()) {
                                Some(cauchy)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                } else {
                    None
                };
            let mut model_rejects = 0usize;
            let mut likelihood_rejects = 0usize;
            let mut objective_rejects = 0usize;
            // Feasibility-path rejections (gam#979 survival monotone cone). The two
            // constrained-path `continue`s — the `apply_joint_feasibility_limit`
            // α-crush `Err` (current iterate infeasible / no positive step) and the
            // `project_point_strictly_into_feasible_cone` `None` (degenerate /
            // empty-interior cone at this trial) — consume a trust attempt but were
            // NOT counted by any of model/likelihood/objective. On the survival
            // marginal-slope monotone-cone pathology this is the DOMINANT reject
            // path (the trial step keeps crossing the binding time-derivative cone
            // at slack≈0), so `model + likelihood + objective < MAX_ATTEMPTS`
            // ALWAYS, `all_attempts_rejected` was permanently false, and the
            // fully-rejected stall guard below NEVER armed — the inner joint-Newton
            // spun to `inner_loop_hard_ceiling` every outer ρ-evaluation (the 1322 s
            // hang; #1040). A feasibility rejection IS a "no descent the local model
            // can reconcile at this β" signal exactly like an objective rejection,
            // so counting it restores the partition invariant
            // `model + likelihood + objective + feasibility == MAX_ATTEMPTS` the
            // stall guard relies on. Off the constrained pathology this counter
            // stays 0 (those `continue`s are never taken on a feasible/unconstrained
            // arm), so every converging fit is byte-identical.
            let mut feasibility_rejects = 0usize;
            let mut first_likelihood_reject: Option<String> = None;
            // Fixed-point signature for the fully-rejected stall guard. The
            // FIRST trust attempt of a cycle evaluates the proposal at the
            // pre-cycle β (the attempt that has not yet shrunk the radius), so
            // its trial objective is a deterministic function of (β, S, λ)
            // alone. On a fully-rejected cycle β is reverted to that same
            // pre-cycle value, so if two consecutive fully-rejected cycles
            // report a *byte-identical* first-attempt trial objective the next
            // cycle's entire Newton system, trust-region search, and reject
            // outcome are provably byte-identical — the iterate is at an exact
            // fixed point and every further cycle is a pure no-op. This is
            // strictly stronger evidence than the `radius_held` heuristic (which
            // can keep resetting while the boundary block's radius oscillates),
            // so it lets the guard fire in two cycles instead of grinding to the
            // 8-cycle held-radius count or the hard ceiling. Captured only on
            // the first attempt; a successful first-attempt likelihood-reject
            // leaves it `None` (and the early-exit reject path captures it
            // explicitly below) so a non-finite trial cannot masquerade as a
            // fixed point.
            let mut first_attempt_trial_objective: Option<f64> = None;
            // Frozen-step line-search short-circuit (n≈3e5 marginal-slope floor
            // stall). Once the joint trust radius is pinned (the shrink rule
            // clamps every block radius at the `1e-12` floor, so a reject can no
            // longer reduce it), the Moré–Sorensen / dogleg step the next attempt
            // builds is a deterministic function of the unchanged radii and the
            // reverted β — byte-identical to this attempt, hence the same trial
            // objective. The floor-stalled cycle therefore logged
            // `JOINT_TRUST_MAX_ATTEMPTS` identical `reject_floor` lines, each a
            // redundant full-data (320k-row) line-search sweep (~0.5 s apiece),
            // every cycle until the cross-cycle stall guard fired — pure waste on
            // the dominant cost of the inner solve. Track the previous rejected
            // attempt's trial objective; when the radius is held AND the current
            // rejected trial reproduces it bit-for-bit the step is provably frozen
            // and the remaining attempts are no-ops, so stop. `frozen_floor_full_reject`
            // records that the cycle was nonetheless fully rejected, preserving the
            // `all_attempts_rejected` partition the cross-cycle stall guard relies
            // on; `first_attempt_trial_objective` is still captured on attempt 0,
            // so the byte-identical cross-cycle detector is unaffected and the
            // converged/non-converged decision is byte-identical to exhausting the
            // loop.
            let mut prev_rejected_attempt_objective: Option<f64> = None;
            let mut frozen_floor_full_reject = false;
            // Coalesce consecutive trust-region attempts whose accept/reject
            // outcome and numeric signature round to the same values, so a long
            // run of identical retries collapses into a single "attempts a..b
            // (×N)" line at flush time instead of spamming one line per try.
            let mut tr_log_sig: Option<String> = None;
            let mut tr_log_first: usize = 0;
            let mut tr_log_last: usize = 0;
            // Hoist the two full-size scratch buffers used in the predicted-
            // reduction computation outside the trust-region attempt loop.
            // The loop runs up to JOINT_TRUST_MAX_ATTEMPTS times per outer
            // Newton step, so allocating these per-attempt would add O(total_p)
            // heap traffic on every radius shrink/expand iteration.
            let mut hpen_delta = Array1::<f64>::zeros(total_p);
            let mut tr_penalty_scratch = Array1::<f64>::zeros(total_p);
            for trust_attempt in 0..JOINT_TRUST_MAX_ATTEMPTS {
                line_search_attempts = trust_attempt + 1;
                accepted_joint_workspace = None;
                // Dogleg globalization (gam#826/#808): when the unconstrained
                // Newton path is in force and a finite Cauchy leg was built,
                // construct the dogleg blend of the Cauchy and Newton points at
                // the current per-block radii. Otherwise (constrained-QP path,
                // or after the preconditioned-descent fallback replaced
                // `search_delta`) fall back to box-truncating the search step.
                let mut trial_delta;
                // gam#979: set when the constrained-QP candidate is taken
                // untruncated because the global α-crush would otherwise collapse
                // a feasible, within-trust QP step (see the gated bypass below).
                let mut qp_feasible_bypass = false;
                let mut block_step_norms = if let Some(spectrum) = joint_spectrum.as_ref() {
                    // POSITIVE-DEFINITE CONSTRAINED PATH (gam#979 CTN).
                    //
                    // `search_delta` is the authoritative active-set QP Newton
                    // step. When the true penalized Hessian has no negative
                    // curvature, the reflected QP matrix is identical to that
                    // Hessian, so replacing this step with an unconstrained
                    // More-Sorensen step changes both the critical cone and the
                    // local quadratic problem. That replacement used to happen
                    // whenever the QP step exceeded the trust radius. On the
                    // 4,800-row Duchon CTN it converted a feasible O(1e-2) QP
                    // proposal into an O(1e-4) projected step, repeatedly changed
                    // the active face, and left the KKT residual near 20--30.
                    //
                    // Global scaling by one alpha is the correct globalization
                    // for this case. Both beta and beta + search_delta are cone-
                    // feasible, so every point on their convex segment remains
                    // feasible. The QP objective is convex and no larger at its
                    // minimizer than at zero, so the same segment is descending.
                    // Use one alpha across every block (per-block clipping would
                    // change direction and can leave the cone). The ordinary
                    // family feasibility limiter still runs below for any
                    // additional nonlinear constraint not represented by the QP.
                    let constrained_search_delta_is_authoritative =
                        constrained_search_delta_owns_trust_step(
                            joint_reduced_face_kind,
                            search_joint_active_set.is_some(),
                            Some(spectrum.has_resolvable_negative_curvature()),
                        );
                    // CONSTRAINED-PATH REFLECTED-QP RESCUE (gam#979 n3000 grind).
                    //
                    // On the constrained path `search_delta` is the *reflected*
                    // active-set QP step: the QP convexifies the indefinite
                    // survival-marginal-slope penalized Hessian by reflecting its
                    // negative-curvature modes to `|γ|` (`symmetric_negative_
                    // curvature_reflected`, line ~1490). The reflection changes
                    // WHICH monotone-derivative-guard rows bind, so the QP settles
                    // to a step whose active set is INCOMPLETE relative to the true
                    // KKT (`active_set_incomplete`): the huge time-block stationarity
                    // residual is never absorbed by the reported multipliers, and
                    // taking that step and then globally α-crushing it (the
                    // `apply_joint_feasibility_limit` fraction-to-boundary below)
                    // collapses β to ~1e-4/cycle, grinding the whole 30-cycle budget
                    // (the measured 478s Weibull-n3000 hang / 600s CI TIMEOUTs).
                    //
                    // The gam#979 `qp_feasible_bypass` rescue that skips the α-crush
                    // lived only in the box-truncation branch below, which became
                    // UNREACHABLE once the decrement diagnostic started populating
                    // `joint_spectrum` on the constrained path (line ~1488) — routing
                    // it into this branch. Resurrect the rescue here, but with the
                    // EXACT Moré–Sorensen step from the UNREFLECTED penalized Hessian
                    // (`spectrum`, decomposed from the true `lhs` at line ~1482): it
                    // handles the indefiniteness rigorously (no reflection, no
                    // incomplete active set), and the convex monotone cone projection
                    // just below (gam#1108) restores feasibility. Because the cone is
                    // convex the projected iterate is feasible, so the α-crush is
                    // skipped (`qp_feasible_bypass`). GATED on the α-crush pathology
                    // (α below the crush threshold) exactly as the original bypass
                    // was, so every healthy constrained arm (α≈1, the binary BMS
                    // score-warp monotonicity fit) takes the byte-identical
                    // reflected-QP-step path unchanged.
                    let constrained_alpha_would_crush = search_joint_active_set.is_some()
                        && match compute_joint_feasibility_alpha(
                            family,
                            &states,
                            &ranges,
                            &search_delta,
                        ) {
                            Ok((alpha, _)) => alpha < JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
                            // Err = current iterate infeasible / no positive step;
                            // fall through to the reflected-QP step, which surfaces the
                            // same Err downstream and shrinks the radius.
                            Err(_) => false,
                        };
                    if constrained_search_delta_is_authoritative {
                        // A full-space convex QP direction and a reduced-face
                        // direction both own their feasible chord. In the latter
                        // case, ambient negative curvature is inaccessible and
                        // has already been handled inside the face tangent.
                        // Scaling the whole chord to the trust radius preserves
                        // A_active·delta=0 and cone feasibility; replacing it
                        // with an ambient spectrum step destroys both invariants.
                        trial_delta = search_delta.clone();
                        let qp_norms = joint_trust_region_block_metric_norms(
                            &trial_delta,
                            &ranges,
                            &joint_trust_metric_diag,
                        );
                        let alpha_trust = qp_norms
                            .iter()
                            .zip(joint_block_trust_radii.iter())
                            .filter(|(norm, _)| norm.is_finite() && **norm > 0.0)
                            .map(|(norm, radius)| (radius / norm).min(1.0))
                            .fold(1.0_f64, f64::min);
                        if alpha_trust.is_finite() && alpha_trust < 1.0 {
                            trial_delta.mapv_inplace(|value| value * alpha_trust);
                            qp_norms.iter().map(|norm| norm * alpha_trust).collect()
                        } else {
                            qp_norms
                        }
                    } else if constrained_alpha_would_crush {
                        qp_feasible_bypass = true;
                        trial_delta = spectrum.trust_region_step(joint_trust_radius).delta;
                        joint_trust_region_block_metric_norms(
                            &trial_delta,
                            &ranges,
                            &joint_trust_metric_diag,
                        )
                    } else {
                        // Exact Moré–Sorensen trust-region step at the current radius
                        // (gam#979). The step already lies in the `D`-metric ball, so
                        // no dogleg blend or box-truncation is applied: on a shrink the
                        // direction is RE-SOLVED (bending toward the gradient), the
                        // property the dogleg/truncation lacked. Re-solving reuses the
                        // cached factorization at O(p) cost. On the constrained path the
                        // resulting (unconstrained) step is projected back onto the cone
                        // just below (gam#1108), preserving this step's fast convergence
                        // while keeping every accepted iterate feasible.
                        //
                        // If the already-computed Newton/QP step lies inside the
                        // current global trust ball, take it directly instead of asking
                        // the trust-region solver to recover the boundary solution at
                        // `r == ‖δ_N‖`. The boundary multiplier is mathematically zero
                        // in that case, but finite precision can produce a tiny positive
                        // multiplier and perturb an exact quadratic one-step solve by
                        // O(1e-6), which is large relative to the inner KKT floor. The
                        // direct step is the exact unconstrained minimizer of the local
                        // model and is still trust-region feasible.
                        let search_norm = joint_trust_region_metric_step_norm(
                            &search_delta,
                            &joint_trust_metric_diag,
                        );
                        if !spectrum.has_resolvable_negative_curvature()
                            && search_norm.is_finite()
                            && joint_trust_radius.is_finite()
                            && search_norm <= joint_trust_radius * (1.0 + 1e-12)
                        {
                            trial_delta = search_delta.clone();
                        } else {
                            trial_delta = spectrum.trust_region_step(joint_trust_radius).delta;
                        }
                        joint_trust_region_block_metric_norms(
                            &trial_delta,
                            &ranges,
                            &joint_trust_metric_diag,
                        )
                    }
                } else if let Some(cauchy) = dogleg_cauchy.as_ref()
                    && !tried_preconditioned_descent
                {
                    trial_delta = Array1::<f64>::zeros(total_p);
                    joint_dogleg_step_to_block_metric_radii(
                        &search_delta,
                        cauchy,
                        &ranges,
                        &joint_trust_metric_diag,
                        &joint_block_trust_radii,
                        &mut trial_delta,
                    )
                } else {
                    // Box-truncation branch — taken on the CONSTRAINED-QP path
                    // (search_joint_active_set is Some, so no spectrum / dogleg).
                    // `search_delta` is the active-set QP's Newton step, FEASIBLE
                    // by construction.
                    //
                    // gam#979 GATED QP-FEASIBILITY BYPASS. The default behaviour
                    // box-truncates this feasible QP step per-block (which can push
                    // it off the monotone cone face) and then the global
                    // fraction-to-boundary α-crush scales the whole joint step by a
                    // single α; on a binding monotonicity row at slack≈0 that α
                    // collapses to ~1e-4, freezing β and hanging the inner solve.
                    //
                    // Bypass that ONLY when the observable pathology is present —
                    // never on a healthy step — so every currently-converging arm
                    // (binary BMS score-warp monotonicity especially) is byte-
                    // identical. The three gate conditions:
                    //   (i)   the candidate came from the constrained QP
                    //         (search_joint_active_set.is_some()),
                    //   (ii)  the QP step is already within the joint trust region
                    //         (step_norm ≤ joint_trust_radius), so truncation is
                    //         not needed for globalization, AND
                    //   (iii) the α the legacy path WOULD apply is below the crush
                    //         threshold (it would gut the step).
                    // When all hold, take the untruncated QP step and let the
                    // magnitude-preserving cone projection below enforce
                    // feasibility. Otherwise truncate exactly as before.
                    let qp_norms = joint_trust_region_block_metric_norms(
                        &search_delta,
                        &ranges,
                        &joint_trust_metric_diag,
                    );
                    let qp_step_norm = qp_norms.iter().copied().fold(0.0_f64, f64::max);
                    let within_trust = qp_step_norm.is_finite()
                        && joint_trust_radius.is_finite()
                        && qp_step_norm <= joint_trust_radius;
                    let alpha_would_crush = if search_joint_active_set.is_some() {
                        match compute_joint_feasibility_alpha(
                            family,
                            &states,
                            &ranges,
                            &search_delta,
                        ) {
                            Ok((alpha, _)) => alpha < JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
                            // Err = current iterate infeasible / no positive step;
                            // fall back to the legacy truncate + α path, which will
                            // surface the same Err and shrink the radius.
                            Err(_) => false,
                        }
                    } else {
                        false
                    };
                    if search_joint_active_set.is_some() && alpha_would_crush {
                        // gam#979 survival n3000 grind. The constrained active-set QP
                        // returns a FEASIBLE point and the monotone cone is convex, so a
                        // step that would trip the fraction-to-boundary α-crush (α below
                        // threshold) must NOT be per-block-truncated: per-block truncation
                        // pushes the joint iterate OFF the cone face, and the α-crush then
                        // collapses it to ~1e-2, freezing β while a huge time-block
                        // gradient persists → the 30-cycle budget grind. Instead skip the
                        // α-crush (feasible by construction) and:
                        //   * within trust      → take the untruncated feasible QP step;
                        //   * exceeds the radius → scale the WHOLE joint step by a SINGLE
                        //     global scalar to the block radii, which stays feasible by
                        //     cone convexity (β and β+δ both feasible ⇒ β+αδ feasible),
                        //     unlike per-block truncation which breaks it.
                        // A feasible boundary step with ρ≈1 then lets the TR GROW the
                        // radius back, curing the collapse-and-grind after one bad
                        // far-field-nonlinear step shrank it. (Reached only on the
                        // observable pathology — constrained candidate whose α WOULD
                        // crush — so every healthy/converging constrained arm is
                        // untouched.)
                        qp_feasible_bypass = true;
                        trial_delta = search_delta.clone();
                        if within_trust {
                            qp_norms
                        } else {
                            let alpha_trust = qp_norms
                                .iter()
                                .zip(joint_block_trust_radii.iter())
                                .filter(|(norm, _)| norm.is_finite() && **norm > 0.0)
                                .map(|(norm, radius)| (radius / norm).min(1.0))
                                .fold(1.0_f64, f64::min);
                            if alpha_trust.is_finite() && alpha_trust < 1.0 {
                                trial_delta.mapv_inplace(|v| v * alpha_trust);
                                qp_norms.iter().map(|n| n * alpha_trust).collect()
                            } else {
                                qp_norms
                            }
                        }
                    } else {
                        trial_delta = search_delta.clone();
                        truncate_joint_step_to_block_metric_radii(
                            &mut trial_delta,
                            &ranges,
                            &joint_trust_metric_diag,
                            &joint_block_trust_radii,
                        )
                    }
                };
                // FEASIBILITY ENFORCEMENT (gam#979 survival flex non-convergence).
                //
                // The global `apply_joint_feasibility_limit` enforces feasibility
                // by a single fraction-to-boundary scalar `α` applied to the WHOLE
                // joint step. On a binding monotonicity row at slack≈0 with
                // negative drift, `α` collapses to ~1e-4, globally crushing the
                // step so β moves ~1e-4/cycle while a huge time-block gradient
                // (|g|≈720) persists: the objective drifts down ~50/cycle but the
                // KKT residual never clears, the inner joint-Newton grinds the full
                // cycle budget, and the seed is rejected — the survival
                // marginal-slope hang.
                //
                // When the gated bypass above fired (`qp_feasible_bypass`), the
                // pathology is present (constrained-QP candidate, within trust, α
                // below the crush threshold): SKIP the α-crush and let the
                // magnitude-preserving cone projection below enforce feasibility,
                // which keeps the unconstrained step components and only corrects
                // the binding directions. Off the pathology the α-crush runs
                // exactly as before — a no-op when `α = 1` (healthy), the legacy
                // scaling when `α ∈ [threshold, 1)` — so every converging arm is
                // byte-identical. The α-crush `Err` (current iterate infeasible /
                // no positive step) still triggers a radius shrink + retry.
                if !qp_feasible_bypass
                    && apply_joint_feasibility_limit(family, &states, &ranges, &mut trial_delta)
                        .is_err()
                {
                    feasibility_rejects += 1;
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                    continue;
                }
                // CONSTRAINED-PATH FEASIBILITY PROJECTION (gam#1108 / gam#979). The
                // trust-region trial step (Moré–Sorensen / dogleg / box-trunc) is
                // taken in the UNCONSTRAINED D-metric ball, so the step can cross
                // the monotone time-derivative cone `Aβ ≥ b`. The next cycle's
                // `check_linear_feasibility` gate would then reject the accepted
                // iterate — the interval-censored survival warm-start abort.
                // Project the trial iterate back onto the cone with the exact
                // identity-Hessian active-set projection, preserving the trust
                // step's fast convergence while guaranteeing every accepted iterate
                // is feasible. This is the feasibility mechanism for the gated
                // QP-bypass case (gam#979, where the α-crush is skipped) and the
                // safety net for any truncation-induced infeasibility on the
                // constrained path. No-op when the joint design is unconstrained or
                // the trial is already feasible (the common case — including a
                // bypassed QP step, which is feasible by construction);
                // `block_step_norms` is recomputed from the projected step just
                // below so the trust-radius bookkeeping stays consistent.
                if let Some(constraints) = joint_constraints.as_ref() {
                    let trial_beta = &beta_joint + &trial_delta;
                    if check_linear_feasibility(&trial_beta, constraints, 1e-8).is_err() {
                        match gam_solve::active_set::project_point_strictly_into_feasible_constraint_set(
                            &trial_beta,
                            constraints,
                        ) {
                            Some(projected) => {
                                trial_delta = &projected - &beta_joint;
                            }
                            None => {
                                // Projection failed to find a strictly-interior
                                // point (degenerate / empty-interior cone at this
                                // trial). Since the global α-crush is gated off on
                                // the constrained path, preserve the old safety
                                // net here: shrink the active block trust radii and
                                // retry, exactly as the α-crush `Err` branch did
                                // (gam#979). Without this an infeasible trial would
                                // reach the next cycle's `check_linear_feasibility`
                                // QP gate and hard-error.
                                feasibility_rejects += 1;
                                joint_trust_radius = shrink_active_joint_block_trust_radii(
                                    &mut joint_block_trust_radii,
                                    &block_step_norms,
                                    0.25,
                                );
                                continue;
                            }
                        }
                    }
                }
                block_step_norms = joint_trust_region_block_metric_norms(
                    &trial_delta,
                    &ranges,
                    &joint_trust_metric_diag,
                );
                let step_norm = block_step_norms.iter().copied().fold(0.0_f64, f64::max);
                let trial_step_inf = trial_delta
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0_f64, f64::max);
                let step_hit_trust_boundary = block_step_norms
                    .iter()
                    .zip(&joint_block_trust_radii)
                    .any(|(step_norm, radius)| {
                        joint_block_step_hit_trust_boundary(*step_norm, *radius)
                    });
                // Predicted reduction must use the TRUE penalized Hessian
                // (the one that appears in `f(β) = -ℓ + ½βᵀSβ + ½·joint_mode_diagonal_ridge·‖β‖²`),
                // NOT the SPD-stabilized version. The stabilizing shift
                // in `joint_solver_diagonal_ridge` is purely a solver-side
                // tool to make the Newton system invertible when H_NLL
                // has negative eigenvalues; it is not part of the true
                // objective the trial-likelihood evaluator computes.
                //
                // If we use `joint_solver_diagonal_ridge` here, then for
                // any Newton step lying in null(H_true) (e.g. the
                // marginal-block cancellation direction in the saturated
                // probit regime — see
                // `marginal_block_hessian_cancels_in_saturated_regime`),
                // predicted = ½·rhs·δ while actual = rhs·δ, giving ρ = 2
                // exactly. The trust-region loop then accepts the step
                // (ρ > 0.75 expands the radius), and the same regime
                // repeats every cycle — exactly the large-scale-saturated
                // failure trace. Pinned by
                // `ridge_stabilization_gap_produces_exact_rho_two_in_null_direction`.
                //
                // `hpen_delta` and `tr_penalty_scratch` are hoisted outside
                // this loop; the workspace variant reuses them without
                // allocating per attempt.
                hpen_delta.fill(0.0);
                if apply_joint_penalized_hessian_into_with_workspace(
                    effective_hessian_source,
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    &trial_delta,
                    &mut hpen_delta,
                    &mut tr_penalty_scratch,
                    joint_bundle,
                )
                .is_err()
                {
                    break;
                }
                // JEFFREYS/FIRTH CURVATURE IN THE TRUST-REGION MODEL (gam#729/#715).
                // When the Jeffreys term is armed, the inner objective the merit
                // (`trialobjective = −ℓ + ½βᵀSβ + Φ`) measures and the Newton step
                // (`(H+Sλ+H_Φ)δ = ∇L−Sβ+∇Φ`) target both include the Firth term, so
                // the trust-region quadratic model's curvature MUST include `H_Φδ`
                // too. Omitting it (bare `(H+Sλ)δ`) makes `predicted_reduction`
                // inconsistent with the H_Φ-augmented `rhs` and the Φ-augmented
                // `actual_reduction`: for a coupled K-block family near the Firth
                // optimum (residual floored at ‖∇Φ‖) the resulting trust_ratio is
                // wrong, the line search rejects the genuine descent step (accepts
                // ~0), and β freezes with the residual stalled at a constant ≫ tol
                // — the unbounded-cycle non-convergence the inner solve exhibits on
                // the Dirichlet/multinomial fits. Adding `H_Φδ` makes the model
                // curvature match the augmented system the step solves and the
                // merit the accept test uses, so the step is accepted and the
                // residual descends. No-op when the term is condition-gated (∇Φ=0,
                // H_Φ=0) or unavailable.
                if let Some((_grad_phi, hphi)) = head_jeffreys_term.as_ref() {
                    let hphi_delta = hphi.dot(&trial_delta);
                    hpen_delta += &hphi_delta;
                }
                let predicted_reduction =
                    joint_quadratic_predicted_reduction(&rhs, &hpen_delta, &trial_delta);
                let linearized_next_kkt_inf = hpen_delta
                    .iter()
                    .zip(rhs.iter())
                    .map(|(hpen, rhs)| (hpen - rhs).abs())
                    .fold(0.0_f64, f64::max);
                // Reject only non-descent directions on the quadratic model.
                // A small-but-positive predicted reduction is what Newton
                // *should* produce near the optimum of a large-magnitude
                // objective: ½δᵀHδ scales with curvature×step², so it can be
                // far below the (relative) objective_tol = inner_tol·(1+|obj|)
                // while still being a correct Newton step. Trust-region ρ
                // shrink/expand handles small-but-valid Newton steps; the
                // preconditioned branch below is only for model-invalid
                // directions, and preserves linear constraints when present.
                //
                // NEAR-FLOOR CARVE-OUT (gam#787 binary matern centers=12). When
                // the Newton proposal is already at the step-tolerance floor —
                // `step_inf ≤ 4·step_tol`, the same round-off band the cert path
                // uses — the iterate is doing KKT polishing on a flat objective,
                // not global descent: there `predicted_reduction = rhs·δ − ½δᵀHδ`
                // is two near-equal O(step²) quantities and its SIGN is round-off
                // noise (a true Newton step gives +½δᵀHδ but the damped/range-
                // restricted spectral solve leaves rhs·δ a hair below ½δᵀHδ). The
                // `predicted_reduction ≤ 0` branch then mistook this for a model-
                // invalid direction and substituted `joint_preconditioned_descent_delta`,
                // a step sized for OBJECTIVE descent (diagonal-preconditioned
                // gradient, O(900×) larger than the polishing proposal). That step
                // bought a round-off-level objective gain but catapulted the KKT
                // residual off a near-converged iterate (‖∇L−Sβ‖ 1.7e-4 → 4.7e-1),
                // which then never recovered — every later cycle re-triggered the
                // same substitution (proposal stays pred≤0), pinning the residual
                // far above tol until the cycle budget exhausted → seed rejected →
                // hard raise. At the step floor we instead take the tiny proposal
                // as-is and let the trust-region noise-floor guard accept it at
                // ρ=1 (it neither helps nor hurts the objective beyond round-off),
                // so the inner keeps polishing the KKT residual to tol.
                let proposal_at_step_floor = joint_proposal_at_step_floor(step_inf, step_tol);
                if (!predicted_reduction.is_finite() || predicted_reduction <= 0.0)
                    && !proposal_at_step_floor
                {
                    model_rejects += 1;
                    // CONSTRAINED-PATH GUARD (#1108). The preconditioned-descent
                    // substitution replaces `search_delta` with an UNCONSTRAINED
                    // diagonally-preconditioned gradient step (`δ = M⁻¹·rhs`). That
                    // direction respects neither the active set nor the linear
                    // inequality cone `Aβ ≥ b`, and nothing downstream re-projects
                    // it: a constrained family that maintains feasibility purely
                    // through the QP (e.g. `LatentSurvivalFamily`, whose
                    // `max_feasible_step_size` is `None` and whose
                    // `post_update_block_beta` is the identity) has no barrier clip
                    // in `apply_joint_feasibility_limit` to pull the gradient step
                    // back onto the monotone time-derivative cone. The trial β then
                    // leaves the cone, the objective-descent test ACCEPTS it (the
                    // gradient step does lower the unconstrained merit), and the
                    // NEXT cycle's `check_linear_feasibility` rejects the accepted
                    // iterate as an "infeasible iterate" (raw `Aβ−b` violation
                    // ~5.5e-3) — aborting the whole interval-censored warm start.
                    // The QP's `search_delta` is a feasible-to-feasible chord
                    // (`candidate_beta − beta_joint`, both endpoints in the convex
                    // cone), so box-truncating it to a SMALLER trust radius keeps
                    // every sub-step feasible. On the constrained path we therefore
                    // never swap in the unconstrained descent direction; we only
                    // shrink the radius and re-truncate the constrained chord. The
                    // comment on the preconditioned branch already promised it
                    // "preserves linear constraints when present" — this makes the
                    // implementation honor that contract.
                    let constrained_path_active = search_joint_active_set.is_some();
                    if !tried_preconditioned_descent && !constrained_path_active {
                        match joint_preconditioned_descent_delta(
                            effective_hessian_source,
                            &ranges,
                            &s_lambdas,
                            joint_solver_diagonal_ridge,
                            &rhs,
                            joint_bundle,
                        ) {
                            Ok(descent_delta) => {
                                search_delta = descent_delta;
                            }
                            Err(_) => {
                                joint_trust_radius = shrink_active_joint_block_trust_radii(
                                    &mut joint_block_trust_radii,
                                    &block_step_norms,
                                    0.25,
                                );
                            }
                        }
                        tried_preconditioned_descent = true;
                    } else {
                        joint_trust_radius = shrink_active_joint_block_trust_radii(
                            &mut joint_block_trust_radii,
                            &block_step_norms,
                            0.25,
                        );
                    }
                    continue;
                }

                for b in 0..specs.len() {
                    let (start, end) = ranges[b];
                    let mut trial_beta = old_beta[b].clone();
                    trial_beta += &trial_delta.slice(ndarray::s![start..end]);
                    let projected =
                        family.post_update_block_beta(&states, b, &specs[b], trial_beta.clone())?;
                    reject_constrained_post_update_repair(
                        b,
                        &specs[b],
                        &trial_beta,
                        &projected,
                        block_constraints[b].as_ref(),
                    )?;
                    states[b].beta.assign(&projected);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                let mut trial_penalty = total_quadratic_penalty(
                    &states,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    joint_bundle,
                    Some(specs),
                );
                // Jeffreys objective contribution at the trial point keeps the
                // accept/reject objective consistent with the Jeffreys-modified
                // Newton step. `states` already holds the trial coefficients
                // (assigned + eta-refreshed above). No-op when the Jeffreys term
                // is unavailable or condition-gated to zero. When the cheap pre-
                // check certified this cycle well-conditioned, the step used H_Φ=0
                // / ∇Φ=0, so the consistent accept/reject objective also uses Φ=0:
                // skipping here keeps value and step on the SAME objective (the
                // value/step consistency the term exists to enforce) and avoids the
                // dense H/eigh at the trial point. The 8× conditioning margin makes
                // a single damped Newton step incapable of crossing the gate.
                // SUBTRACT Φ: the inner NLL objective is `−ℓ + ½βᵀSβ − Φ` (Firth
                // adds ½log|I| to the log-likelihood). Must match the cycle-0
                // baseline, the Newton step, and the KKT residual — INCLUDING the
                // `jeffreys_skippable_this_cycle` gate, so that on a well-conditioned
                // cycle the trial, the step (H_Φ=0/∇Φ=0), and the residual all sit
                // on the SAME Φ=0 objective (gam#729/#715 sign fix; the baseline and
                // post-accept folds carry the matching skippable gate).
                if !jeffreys_skippable_this_cycle
                    && let Some(z_joint) = joint_jeffreys_subspace.as_ref()
                {
                    trial_penalty -= custom_family_joint_jeffreys_value(
                        family, &states, specs, &ranges, z_joint,
                    );
                }
                // Cheap-LL line-search path: rejected backtracking attempts
                // discard the exact-Newton workspace they build, so we evaluate
                // just the scalar full-data log-likelihood for the accept/reject
                // decision and only build the full state once the step is
                // accepted (via the gradient reload below).
                //
                // EARLY-EXIT THRESHOLD MUST BOUND THE NLL, NOT THE FULL OBJECTIVE
                // (was a stall — gam#787/#785, duchon centers≥20). The family's
                // `bernoulli_margslope_line_search_ll_with_early_exit` short-
                // circuits the row sweep when the accumulated `-Σ wᵢ log CDF` (the
                // NLL ALONE — no penalty, no Jeffreys Φ) exceeds the threshold; its
                // monotone-lower-bound proof is valid only for the NLL term. But the
                // accept test is on the FULL augmented objective
                // `F = -ℓ + ½βᵀSβ + Φ_trial`, accepted iff `F ≤ old_objective + slack`,
                // i.e. iff `-ℓ_trial ≤ old_objective + slack − penalty_trial`. Passing
                // the full `old_objective` as the NLL threshold therefore over-rejects
                // by exactly `penalty_trial`: where the trial penalty is NEGATIVE
                // (the Jeffreys term subtracts Φ, and `½βᵀSβ` can be net-negative
                // under the reparam) the NLL threshold sits BELOW the true accept
                // bound, so the early exit kills net-descent steps the trust region
                // would accept — every backtracking attempt false-rejects, the radius
                // collapses, and the inner exits non-converged at cycle ~2 (seed
                // rejected pre-solver → hard raise, β pinned). Subtract the trial
                // penalty so the threshold is the NLL the trial must beat.
                let line_search_options =
                    coefficient_line_search_options(options, old_objective + 1e-10 - trial_penalty);
                // Accept-on-first-attempt fast path (gam#979 `gradient_reload`
                // cost). On the FIRST trust-region attempt of a cycle the step
                // is the undamped (radius-bumped) Newton proposal, which on the
                // common ρ≈1 `hold_inside` large-scale pattern accepts outright.
                // The cheap scalar sweep below would then run a full row stream
                // and immediately discard it, leaving `gradient_reload` to
                // re-stream every row at the SAME β to build the gradient
                // workspace — the ~5s redundant second pass per accepted cycle.
                //
                // Instead, when a workspace gradient source is available, build
                // the joint-Newton workspace ONCE at the trial β and read its
                // `joint_log_likelihood_evaluation()` (the same `Σ wᵢ log Φ` the
                // cheap sweep computes, on the same row measure — both derive
                // from `options`). The materialised per-row cache is threaded
                // forward as `accepted_joint_workspace`, so on accept the reload
                // short-circuits through `joint_gradient_evaluation()` with NO
                // second stream — collapsing the accepted cycle to one row pass.
                //
                // Only the first attempt takes this path: it is the only one
                // expected to accept, so a rejected first attempt pays a single
                // full (non-early-exited) sweep — paid back many-fold on the
                // dominant accept-on-first-attempt cycle. Later backtracking
                // attempts keep the cheap early-exiting sweep (they are expected
                // to reject and the workspace they would build is discarded).
                // Capability absence is the only reason to use the scalar path.
                // Once the family advertises fused likelihood evidence, an error
                // or a missing advertised value is a broken workspace contract,
                // not an infeasible trial: silently replaying the scalar family
                // path could change the row measure and let structurally invalid
                // Hessian/gradient evidence participate in the trust ratio.
                let fused_first_attempt = fused_first_attempt_log_likelihood(
                    family,
                    options,
                    specs,
                    &states,
                    trust_attempt,
                    joint_workspace_requested,
                )?;
                let trial_ll = if let Some((value, workspace)) = fused_first_attempt {
                    accepted_joint_workspace = Some(workspace);
                    value
                } else {
                    match joint_line_search_log_likelihood(family, &line_search_options, &states) {
                        Ok((value, workspace)) => {
                            accepted_joint_workspace = workspace;
                            value
                        }
                        Err(e) => {
                            likelihood_rejects += 1;
                            if first_likelihood_reject.is_none() {
                                first_likelihood_reject = Some(e);
                            }
                            for (b, old) in old_beta.iter().enumerate() {
                                states[b].beta.assign(old);
                            }
                            refresh_all_block_etas(family, specs, &mut states)?;
                            joint_trust_radius = shrink_active_joint_block_trust_radii(
                                &mut joint_block_trust_radii,
                                &block_step_norms,
                                0.25,
                            );
                            continue;
                        }
                    }
                };
                let trialobjective = -trial_ll + trial_penalty;
                if trust_attempt == 0 && trialobjective.is_finite() {
                    // Deterministic fixed-point signature (see declaration). The
                    // first attempt evaluates at the unshrunk pre-cycle β, so this
                    // value identifies the iterate exactly.
                    first_attempt_trial_objective = Some(trialobjective);
                }
                // Row measure observed by the trial objective at β + δ. The
                // line-search helper above runs under `coefficient_line_search_options`,
                // which now preserves `outer_score_subsample` and disables
                // any further auto-install; if either contract is broken the
                // id will diverge from `tr_row_measure_top` and we Err below.
                let tr_row_measure_trial =
                    gam_solve::row_measure::RowSubsampleMask::from_options(options, total_joint_n);
                // Hard invariant: the trust-region ratio numerator (objective
                // at β minus trial at β+δ) and denominator (rhs·δ − ½δᵀH δ)
                // MUST share a row measure with the Hessian/gradient build.
                // Bubble out via `Err` rather than panic; this function
                // already returns `Result<_, String>`.
                let top_id = tr_row_measure_top.id;
                if tr_row_measure_hessian.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         Hessian id 0x{:016x} differs from top-of-cycle id 0x{:016x} \
                         (cycle {}); the joint Hessian was built against a different \
                         row mask than the trust-region globalization captured at the \
                         top of the cycle. ρ would compare ½δᵀHδ on one measure to \
                         F(β)−F(β+δ) on another.",
                        tr_row_measure_hessian.id, top_id, cycle
                    ));
                }
                if tr_row_measure_gradient.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         gradient id 0x{:016x} differs from top-of-cycle id 0x{:016x} \
                         (cycle {}); `cached_joint_gradient` was loaded against a \
                         different row mask than the trust-region globalization \
                         captured at the top of the cycle. rhs·δ in the predicted \
                         reduction would not match the rest of the ρ inputs.",
                        tr_row_measure_gradient.id, top_id, cycle
                    ));
                }
                if tr_row_measure_old_objective.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         objective-at-β id 0x{:016x} differs from top-of-cycle id \
                         0x{:016x} (cycle {}); `lastobjective` was computed against \
                         a different row mask than the trust-region globalization \
                         captured at the top of the cycle.",
                        tr_row_measure_old_objective.id, top_id, cycle
                    ));
                }
                if tr_row_measure_trial.id != top_id {
                    return Err(format!(
                        "trust-region row-measure invariant violated: \
                         trial-objective id 0x{:016x} differs from top-of-cycle id \
                         0x{:016x} (cycle {}, attempt {}); the line-search trial \
                         likelihood evaluated against a different row mask than the \
                         Hessian/gradient/old-objective build. Cf. \
                         `coefficient_line_search_options` and \
                         `install_auto_outer_subsample_options`.",
                        tr_row_measure_trial.id, top_id, cycle, trust_attempt
                    ));
                }
                let actual_reduction = old_objective - trialobjective;
                let trust_update = update_joint_trust_region_radius(
                    joint_trust_radius,
                    step_norm,
                    actual_reduction,
                    predicted_reduction,
                    old_objective,
                );
                let old_radius = joint_trust_radius;
                // Classify the outcome of this attempt so the diagnostic line
                // says *why* the step was taken or rejected rather than just
                // dumping numbers. The four phases partition the post-log
                // branches below; computing them up front lets the log line
                // and the dispatch agree.
                let floor_reached = trust_update.accepted
                    && current_stationarity_residual <= residual_tol
                    && !has_resolvable_negative_curvature
                    && joint_objective_floor_reached(
                        old_objective,
                        trialobjective,
                        actual_reduction,
                        predicted_reduction,
                        objective_tol,
                    );
                let roundoff_slack = joint_objective_roundoff_slack(old_objective, trialobjective);
                let secondary_ok = !floor_reached
                    && trialobjective.is_finite()
                    && trust_update.accepted
                    && trialobjective <= old_objective + roundoff_slack;
                let phase: &'static str = if floor_reached {
                    "converged"
                } else if secondary_ok {
                    "accepted"
                } else if trust_update.accepted {
                    "stall"
                } else {
                    "reject"
                };
                if floor_reached || secondary_ok {
                    for (block_radius, block_step_norm) in joint_block_trust_radii
                        .iter_mut()
                        .zip(block_step_norms.iter())
                    {
                        let block_update = update_joint_trust_region_radius(
                            *block_radius,
                            *block_step_norm,
                            actual_reduction,
                            predicted_reduction,
                            old_objective,
                        );
                        if block_update.radius >= *block_radius
                            || joint_block_step_hit_trust_boundary(*block_step_norm, *block_radius)
                        {
                            *block_radius = block_update.radius;
                        }
                    }
                    joint_trust_radius = joint_block_trust_radii
                        .iter()
                        .copied()
                        .fold(0.0_f64, f64::max);
                } else {
                    joint_trust_radius = shrink_active_joint_block_trust_radii(
                        &mut joint_block_trust_radii,
                        &block_step_norms,
                        0.25,
                    );
                }
                let radius_held =
                    (joint_trust_radius - old_radius).abs() <= 1e-12 * old_radius.abs().max(1.0);
                let joint_math = JointNewtonMathDiagnostic {
                    old_kkt_inf: current_kkt_norm,
                    linearized_next_kkt_inf,
                    predicted_reduction,
                    actual_reduction,
                    trust_ratio: trust_update.rho,
                    step_inf: trial_step_inf,
                    proposal_inf: step_inf,
                };
                let radius_field = if radius_held {
                    format!("r={:.3e} (held)", old_radius)
                } else {
                    format!("r={:.3e}->{:.3e}", old_radius, joint_trust_radius)
                };
                // Surface the TR-policy decision so future failures
                // distinguish "TR is throttling Newton" from "TR is not
                // the bottleneck — Newton itself finds short steps".
                // For the large-scale linear-convergence pattern the policy
                // is consistently `hold_inside` (ρ≈1, |δ| ≪ radius),
                // which proves the TR is not what is keeping the step
                // small — that came up before via "(held)" alone but
                // the explicit decision label makes the inference
                // immediate instead of requiring step/radius arithmetic
                // in the reader's head.
                let tr_attempt_sig = format!(
                    "{:<9}  ρ={:+.3e}  Δobj={:+.3e}  pred={:+.3e}  {}  decision={:<22}  |δ|={:.3e}  |δ|∞={:.3e}  |prop|∞={:.3e}",
                    phase,
                    trust_update.rho,
                    actual_reduction,
                    predicted_reduction,
                    radius_field,
                    trust_update.decision.label(),
                    step_norm,
                    trial_step_inf,
                    step_inf,
                );
                match tr_log_sig.as_deref() {
                    Some(prev) if prev == tr_attempt_sig.as_str() => {
                        tr_log_last = line_search_attempts;
                    }
                    Some(prev) => {
                        if tr_log_first == tr_log_last {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                                cycle,
                                tr_log_first,
                                prev,
                            );
                        } else {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                                cycle,
                                tr_log_first,
                                tr_log_last,
                                tr_log_last - tr_log_first + 1,
                                prev,
                            );
                        }
                        tr_log_sig = Some(tr_attempt_sig);
                        tr_log_first = line_search_attempts;
                        tr_log_last = line_search_attempts;
                    }
                    None => {
                        tr_log_sig = Some(tr_attempt_sig);
                        tr_log_first = line_search_attempts;
                        tr_log_last = line_search_attempts;
                    }
                }
                if floor_reached {
                    if let Some(sig) = tr_log_sig.take() {
                        if tr_log_first == tr_log_last {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                                cycle,
                                tr_log_first,
                                sig,
                            );
                        } else {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                                cycle,
                                tr_log_first,
                                tr_log_last,
                                tr_log_last - tr_log_first + 1,
                                sig,
                            );
                        }
                    }
                    for (b, old) in old_beta.iter().enumerate() {
                        states[b].beta.assign(old);
                    }
                    refresh_all_block_etas(family, specs, &mut states)?;
                    last_joint_math = Some(joint_math);
                    accepted = true;
                    converged = true;
                    returned_mode_curvature_certified = joint_constraints.is_none()
                        && joint_jeffreys_subspace.is_none()
                        && joint_spectrum.is_some();
                    // Constrained (non-Jeffreys) acceptance is tentative until the
                    // next cycle head certifies the active-face-tangent curvature
                    // and escapes a strict saddle (gam#979). A trust-floor accept
                    // is a common saddle signature (negative curvature keeps every
                    // step rejected), so it must route through the same check.
                    if joint_constraints.is_some() && joint_jeffreys_subspace.is_none() {
                        returned_constrained_mode_pending = true;
                        continue 'joint_newton_cycles;
                    }
                    break;
                }
                if secondary_ok {
                    if let Some(sig) = tr_log_sig.take() {
                        if tr_log_first == tr_log_last {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                                cycle,
                                tr_log_first,
                                sig,
                            );
                        } else {
                            log::info!(
                                "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                                cycle,
                                tr_log_first,
                                tr_log_last,
                                tr_log_last - tr_log_first + 1,
                                sig,
                            );
                        }
                    }
                    if let Some(joint_active_set) = search_joint_active_set.as_ref() {
                        // The QP reports the face at its full candidate endpoint,
                        // but globalization may accept only a strict subsegment of
                        // that feasible chord. Endpoint-only rows are then still
                        // slack at `states` and cannot own either the next warm
                        // KKT system or terminal tangent-space evidence. Filter the
                        // sparse candidate face against the actual accepted beta;
                        // the active-set ratio test rediscovers a discarded row
                        // exactly when a later iterate reaches it.
                        let accepted_beta = flatten_state_betas(&states, specs);
                        let accepted_joint_active = if let Some(constraints) =
                            joint_constraints.as_ref()
                        {
                            gam_solve::active_set::constraint_set_rows_tight_at_point(
                                constraints,
                                &accepted_beta,
                                joint_active_set,
                            )
                            .map_err(|error| {
                                format!(
                                    "accepted constrained-Newton face classification failed: {error}"
                                )
                            })?
                        } else {
                            joint_active_set.clone()
                        };
                        let accepted_face = if accepted_joint_active.is_empty() {
                            None
                        } else {
                            Some(accepted_joint_active.clone())
                        };
                        accepted_active_face_changed = accepted_face != active_face_before_step;
                        cached_active_sets =
                            scatter_joint_active_set(&accepted_joint_active, &block_constraints);
                    }
                    last_joint_math = Some(joint_math);
                    last_accepted_hit_joint_trust_boundary = step_hit_trust_boundary;
                    accepted = true;
                    break;
                }
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                objective_rejects += 1;
                // Frozen-step short-circuit (see declaration). `radius_held` here
                // means the post-reject shrink did not change the joint trust
                // radius — i.e. the radii are pinned at the `1e-12` floor. If the
                // trial objective also reproduces the previous rejected attempt's
                // bit-for-bit, the dogleg/Moré–Sorensen step is frozen and every
                // remaining attempt would re-reject the identical step: skip the
                // redundant full-data sweeps and let the cross-cycle stall guard
                // certify the fixed point.
                if radius_held && trialobjective.is_finite() {
                    if let Some(prev) = prev_rejected_attempt_objective {
                        if prev.to_bits() == trialobjective.to_bits() {
                            frozen_floor_full_reject = true;
                            break;
                        }
                    }
                    prev_rejected_attempt_objective = Some(trialobjective);
                } else {
                    prev_rejected_attempt_objective = None;
                }
            }
            if let Some(sig) = tr_log_sig.take() {
                if tr_log_first == tr_log_last {
                    log::info!(
                        "[PIRLS/joint-Newton/TR cycle={} attempt={}] {}",
                        cycle,
                        tr_log_first,
                        sig,
                    );
                } else {
                    log::info!(
                        "[PIRLS/joint-Newton/TR cycle={} attempts={}..{} ×{}] {}",
                        cycle,
                        tr_log_first,
                        tr_log_last,
                        tr_log_last - tr_log_first + 1,
                        sig,
                    );
                }
            }
            let line_search_elapsed = line_search_started.elapsed();
            if accepted && converged {
                log::info!(
                    "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=true hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} reject_model={} reject_likelihood={} reject_objective={} reject_feasibility={} first_likelihood_reject={} grad_reload=0.000s total={:.3}s",
                    cycle,
                    hessian_and_qp_elapsed.as_secs_f64(),
                    line_search_elapsed.as_secs_f64(),
                    line_search_attempts,
                    model_rejects,
                    likelihood_rejects,
                    objective_rejects,
                    feasibility_rejects,
                    first_likelihood_reject.as_deref().unwrap_or("none"),
                    cycle_started.elapsed().as_secs_f64(),
                );
                // Accepted step moved β; the cycle workspace is at the OLD
                // (pre-step) β, so it must NOT be carried into the post-loop
                // covariance/IFT assembly (which needs the converged β). Drop it.
                cached_joint_workspace = None;
                cycles_done = cycle + 1;
                break;
            }
            if !accepted {
                // Retry the joint Newton loop from the same state after a
                // failed trust-region search. Falling through into blockwise
                // would switch a coupled exact-Hessian problem onto a
                // principal-block surrogate, which is the ridge-drift failure
                // mode this path is meant to avoid. The trust-region radius
                // already collapsed via the attempt loop's shrink rules, so
                // the next cycle's Newton proposal will be evaluated under
                // a tighter L2 bound without any parallel adaptation here.
                log::info!(
                    "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=false hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} reject_model={} reject_likelihood={} reject_objective={} reject_feasibility={} first_likelihood_reject={} grad_reload=0.000s total={:.3}s",
                    cycle,
                    hessian_and_qp_elapsed.as_secs_f64(),
                    line_search_elapsed.as_secs_f64(),
                    line_search_attempts,
                    model_rejects,
                    likelihood_rejects,
                    objective_rejects,
                    feasibility_rejects,
                    first_likelihood_reject.as_deref().unwrap_or("none"),
                    cycle_started.elapsed().as_secs_f64(),
                );
                // Restore original betas
                for (b, old) in old_beta.iter().enumerate() {
                    states[b].beta.assign(old);
                }
                refresh_all_block_etas(family, specs, &mut states)?;
                // β is now back at `old_beta`, the exact β this cycle's
                // exact-Newton workspace was built at. A rejected cycle does NOT
                // run the post-accept gradient reload (which is what otherwise
                // re-stashes a workspace), so without this the next cycle's
                // `cached_joint_workspace.take()` is `None` and re-streams all n
                // rows through the per-row kernel cache build — pure redundancy
                // at the identical β. Hand the still-valid workspace back so the
                // next cycle hits the cache. Bit-identical: same family, same β,
                // so the rebuilt cache would be byte-for-byte this one; the inner
                // solve still re-derives the Newton step and runs to its KKT
                // certificate unchanged. On the loop-exit `break`s below this is
                // a harmless assignment to a value that is then dropped.
                cached_joint_workspace = hessian_workspace_for_cycle.take();
                // If the previous cycle's bookkeeping certified KKT
                // stationarity (residual ≤ tol and objective change ≤
                // tol), the line-search failure here is round-off on a
                // rank-deficient null mode rather than non-convergence:
                // the proposed `H⁻¹ g` step stays O(1) along the null
                // direction at the optimum, every trial moves β along
                // it without changing the objective, and round-off
                // flips the sign of `actual − predicted` so the
                // sufficient-decrease check rejects every trial. The
                // iterate ALREADY satisfies the first-order optimality
                // conditions; we accept that as convergence rather
                // than fail the outer "inner solve did not converge"
                // panic on a fully resolved fit.
                if last_cycle_residual_below_tol && last_cycle_obj_change_below_tol {
                    converged = true;
                    // Constrained (non-Jeffreys) acceptance is tentative until the
                    // next cycle head certifies the active-face-tangent curvature
                    // and escapes a strict saddle (gam#979).
                    if joint_constraints.is_some() && joint_jeffreys_subspace.is_none() {
                        returned_constrained_mode_pending = true;
                        returned_mode_curvature_certified = false;
                        continue 'joint_newton_cycles;
                    }
                    break;
                }
                // Fully-rejected stall guard. See the constant declaration
                // at the top of this function for the full rationale. The
                // condition is: every trust attempt this cycle was rejected by
                // SOME path (model OR likelihood OR objective OR feasibility; the
                // four reject counters partition the JOINT_TRUST_MAX_ATTEMPTS
                // attempts) AND
                // the joint trust radius did not shrink relative to the previous
                // fully-rejected cycle. Both together prove the next cycle's
                // Newton system, trust radius, and trust-region search are
                // bytewise identical to this cycle's — there is no descent
                // direction the local quadratic model can reconcile at this β.
                //
                // The earlier form required objective_rejects ==
                // JOINT_TRUST_MAX_ATTEMPTS && likelihood_rejects == 0, so it
                // NEVER fired on the biobank gauge-flat marginal/logslope fit:
                // there the objective is flat to f64 precision along the
                // residual direction and the BMS line search rejects every
                // trial on the *likelihood* early-exit path
                // (likelihood_rejects == 24), so the stall guard's increment
                // condition was unreachable and the loop spun to its cap. A
                // full rejection by the likelihood path at a collapsed trust
                // radius is the same numerically-flat-no-descent stall as a
                // full objective rejection; counting either lets the guard fire.
                let all_attempts_rejected = frozen_floor_full_reject
                    || model_rejects + likelihood_rejects + objective_rejects + feasibility_rejects
                        == JOINT_TRUST_MAX_ATTEMPTS;
                let radius_held_since_last_reject = match prev_rejected_trust_radius {
                    Some(prev) => {
                        joint_trust_radius.is_finite()
                            && prev.is_finite()
                            && joint_trust_radius >= prev * (1.0 - 1e-12)
                    }
                    None => false,
                };
                if all_attempts_rejected && radius_held_since_last_reject {
                    consecutive_held_rejected_cycles =
                        consecutive_held_rejected_cycles.saturating_add(1);
                } else {
                    consecutive_held_rejected_cycles = 0;
                }
                prev_rejected_trust_radius = Some(joint_trust_radius);
                // Byte-identical fixed-point detector. A fully-rejected cycle
                // whose first-attempt trial objective reproduces the previous
                // fully-rejected cycle's value bit-for-bit proves β reverted to
                // the same iterate and the Newton system is identical, so every
                // further cycle is a provable no-op. This is stronger than the
                // held-radius count and fires in two cycles, fixing the n≈3e5
                // marginal/logslope grind where the held-radius path's off-by-one
                // let the inner solve spin past the wall-clock budget (gam#979).
                match (
                    all_attempts_rejected,
                    first_attempt_trial_objective,
                    prev_rejected_first_attempt_objective,
                ) {
                    (true, Some(current), Some(prev)) if current.to_bits() == prev.to_bits() => {
                        consecutive_identical_rejected_cycles =
                            consecutive_identical_rejected_cycles.saturating_add(1);
                    }
                    _ => {
                        consecutive_identical_rejected_cycles = 0;
                    }
                }
                if all_attempts_rejected {
                    prev_rejected_first_attempt_objective = first_attempt_trial_objective;
                } else {
                    prev_rejected_first_attempt_objective = None;
                }
                // Collapsed-trust-region all-reject-at-floor detector (gam#979).
                // Increment only when EVERY attempt this cycle was rejected AND
                // the joint trust radius has reached its absolute `1e-12` floor:
                // the radius cannot shrink further and the step makes no
                // progress, so the next cycle is forced to repeat this one. Any
                // accepted cycle (handled below via the post-grad-reload reset)
                // or any cycle whose radius is still above the floor breaks the
                // streak, so a progressing fit never accumulates it.
                let all_attempts_rejected_at_floor_this_cycle = all_attempts_rejected
                    && joint_trust_radius_at_absolute_floor(joint_trust_radius);
                if all_attempts_rejected_at_floor_this_cycle {
                    consecutive_all_reject_at_floor_cycles =
                        consecutive_all_reject_at_floor_cycles.saturating_add(1);
                } else {
                    consecutive_all_reject_at_floor_cycles = 0;
                }
                let collapsed_floor_exit = joint_collapsed_floor_all_reject_exit(
                    consecutive_all_reject_at_floor_cycles,
                    all_attempts_rejected_at_floor_this_cycle,
                );
                if consecutive_held_rejected_cycles >= FULLY_REJECTED_STALL_MAX_CYCLES
                    || consecutive_identical_rejected_cycles >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                    || collapsed_floor_exit
                {
                    let last_math_summary = last_joint_math
                        .as_ref()
                        .map(|math| {
                            format!(
                                "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                                math.old_kkt_inf,
                                math.linearized_next_kkt_inf,
                                math.actual_reduction,
                                math.predicted_reduction,
                                math.trust_ratio,
                                math.scalar_model_relative_error(),
                                math.step_inf,
                                math.proposal_inf,
                            )
                        })
                        .unwrap_or_else(|| "last_newton_math=<none>".to_string());
                    let stall_trigger = if consecutive_identical_rejected_cycles
                        >= IDENTICAL_REJECTED_STALL_MAX_CYCLES
                    {
                        format!(
                            "byte-identical first-attempt trial objective for {} consecutive \
                             fully-rejected cycles (exact fixed point)",
                            consecutive_identical_rejected_cycles
                        )
                    } else if consecutive_all_reject_at_floor_cycles
                        >= JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES
                    {
                        format!(
                            "{} consecutive fully-rejected cycles with the joint trust radius \
                             collapsed to its absolute 1e-12 floor (no smaller step \
                             representable, step makes no progress)",
                            consecutive_all_reject_at_floor_cycles
                        )
                    } else {
                        format!(
                            "{} consecutive fully-rejected cycles with joint trust radius held",
                            consecutive_held_rejected_cycles
                        )
                    };
                    log::warn!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | fully-rejected stall \
                         early-exit: every trust-region attempt rejected (by any of the model / \
                         likelihood / objective paths) — {} at joint trust radius {:.3e}. Reverted β \
                         + identical Newton system mean the next cycle's step is byte-identical to \
                         this one's; no accepted descent step is reachable from this iterate under the \
                         current local model. {}. The strict KKT residual has not converged, so \
                         returning non-converged.",
                        cycle,
                        stall_trigger,
                        joint_trust_radius,
                        last_math_summary,
                    );
                    converged = false;
                    break;
                }
                // CONTINUE rather than break (gam#826/#872/#715). The comment
                // above documents the intent — "retry the joint Newton loop from
                // the same state after a failed trust-region search" — but the old
                // code BROKE instead, giving up after a SINGLE cycle of failed line
                // search. On a severely near-separating coupled fit (matern
                // binomial location-scale, quasi-separating multinomial, flexible
                // linkwiggle) the cycle-0 Newton proposal is huge (the separation
                // gradient ÷ the Firth-bounded curvature), the trust region clamps
                // it, and the clamped step does not yet reduce the merit — so the
                // FIRST cycle's backtracking exhausts without acceptance. The
                // attempt loop already shrank `joint_trust_radius` /
                // `joint_block_trust_radii` (carried across cycles), so the NEXT
                // cycle re-proposes under the tighter radius and eventually accepts
                // a productive step — standard trust-region globalization. Breaking
                // at cycle 0 aborted the coupled solve ("exited the joint Newton
                // path before convergence — no math snapshot") before the trust
                // region could adapt. The inner cycle cap and the residual-stall /
                // trust-region-floor guards above still bound the loop, so a
                // genuinely stuck fit exits with a diagnosed non-convergence rather
                // than spinning. Falling through to blockwise (the old `break`)
                // would switch the coupled exact-Hessian problem onto a
                // principal-block surrogate (the ridge-drift mode this path avoids).
                if joint_workspace_requested {
                    cached_joint_hessian_source = Some(joint_hessian_source);
                }
                continue;
            }

            let grad_reload_started = std::time::Instant::now();
            log::info!(
                "[joint-newton-tr] phase=gradient_reload cycle={} attempts={} r={:.3e}",
                cycle,
                line_search_attempts,
                joint_trust_radius,
            );
            let (log_likelihood, gradient, eval, workspace) = load_joint_gradient_evaluation(
                family,
                specs,
                options,
                &states,
                joint_workspace_requested,
                accepted_joint_workspace.take(),
            )?;
            let grad_reload_elapsed = grad_reload_started.elapsed();
            // Reset the fully-rejected stall guard's bookkeeping: an accepted
            // cycle moved β and may have grown the trust radius, so the next
            // rejected-cycle comparison must start fresh rather than carry
            // forward a stale radius snapshot from the previous reject streak.
            prev_rejected_trust_radius = None;
            consecutive_held_rejected_cycles = 0;
            // An accepted step moved β, so the fixed-point signature is stale;
            // reset it so a later reject streak compares only consecutive
            // fully-rejected cycles at the SAME iterate.
            prev_rejected_first_attempt_objective = None;
            consecutive_identical_rejected_cycles = 0;
            // An accepted step moved β and (via the trust-region grow rules)
            // lifts the radius off its floor, so the collapsed-floor all-reject
            // streak no longer holds; reset it (gam#979).
            consecutive_all_reject_at_floor_cycles = 0;
            // Accepted-cycle timing breakdown is debug-only. The per-cycle
            // info line below already includes total cycle time; emitting a
            // four-phase split on every verbose cycle adds a redundant info
            // line. Rejected cycles still keep the detailed phase log since
            // the reject reason and per-phase split is the diagnostic.
            log::debug!(
                "[PIRLS/joint-Newton/cycle-summary] cycle={} accepted=true hessian_qp={:.3}s line_search={:.3}s line_search_attempts={} grad_reload={:.3}s total={:.3}s",
                cycle,
                hessian_and_qp_elapsed.as_secs_f64(),
                line_search_elapsed.as_secs_f64(),
                line_search_attempts,
                grad_reload_elapsed.as_secs_f64(),
                cycle_started.elapsed().as_secs_f64(),
            );
            current_log_likelihood = log_likelihood;
            cached_joint_gradient = gradient;
            cached_eval = eval;
            cached_joint_workspace = workspace;
            current_penalty = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            // `current_penalty` / `lastobjective` stay the pure quadratic-penalized
            // objective (NO Φ folded in) — the Firth value is applied per cycle at
            // each β (see `old_objective` above and `trialobjective` below). The
            // gated Φ at the accepted β is captured separately so the convergence
            // `objective_change` compares the augmented objective at the new vs old
            // β consistently (gam#826/#872).
            lastobjective = -current_log_likelihood + current_penalty;
            let new_phi = if !jeffreys_skippable_this_cycle {
                joint_jeffreys_subspace
                    .as_ref()
                    .map(|z_joint| {
                        custom_family_joint_jeffreys_value(family, &states, specs, &ranges, z_joint)
                    })
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let accepted_step_inf = states
                .iter()
                .zip(old_beta.iter())
                .flat_map(|(state, old)| {
                    state
                        .beta
                        .iter()
                        .zip(old.iter())
                        .map(|(new, old)| (new - old).abs())
                })
                .fold(0.0_f64, f64::max);
            cycles_done = cycle + 1;

            macro_rules! finish_post_step_convergence {
                () => {{
                    converged = true;
                    if joint_constraints.is_none() && joint_jeffreys_subspace.is_none() {
                        returned_mode_curvature_pending = true;
                        returned_mode_curvature_certified = false;
                        continue 'joint_newton_cycles;
                    }
                    // Constrained (non-Jeffreys) first-order convergence is
                    // tentative until the next cycle head certifies the
                    // active-face-tangent curvature and, on a strict saddle,
                    // escapes along it (gam#979). Jeffreys-augmented families keep
                    // their first-order contract and their own augmented-objective
                    // certificate, so they still break straight to the post-loop.
                    if joint_jeffreys_subspace.is_none() {
                        returned_constrained_mode_pending = true;
                        returned_mode_curvature_certified = false;
                        continue 'joint_newton_cycles;
                    }
                    break;
                }};
            }

            // Check convergence via joint stationarity. When the family-general
            // Firth/Jeffreys term is armed, the penalized objective the inner
            // Newton actually optimizes is `−ℓ + ½βᵀSβ − Φ`, so its KKT
            // stationarity is `∇L − Sβ + ∇Φ = 0`. The Newton STEP already folds
            // `∇Φ` into its RHS (`spectral_rhs += grad_phi`), but the bare
            // `exact_newton_joint_stationarity_*` residual omits it — at the
            // Firth fixed point `∇L − Sβ = −∇Φ`, so the certificate floors at
            // `‖∇Φ‖∞` and never certifies, stalling the inner solve on exactly
            // the near-separating span Firth is meant to bound (the residual the
            // outer REML then rejects). Fold `∇Φ` into the gradient used for the
            // KKT residual so the convergence criterion matches the augmented
            // objective the step descends. No-op when the Jeffreys term is
            // unavailable or condition-gated to zero.
            let Some(gradient) = cached_joint_gradient.as_ref() else {
                break;
            };
            let jeffreys_augmented_gradient: Option<Array1<f64>> = if jeffreys_skippable_this_cycle
            {
                // Well-conditioned ⇒ ∇Φ = 0, so the KKT residual is the bare
                // stationarity (and floors at 0, not ‖∇Φ‖) — matching the step,
                // which folded H_Φ=0/∇Φ=0 this cycle. Avoids the dense H/eigh.
                None
            } else if let Some(z_joint) = joint_jeffreys_subspace.as_ref() {
                match custom_family_joint_jeffreys_term(family, &states, specs, &ranges, z_joint)? {
                    Some((_phi, grad_phi, hphi))
                        if grad_phi.len() == gradient.len()
                            && hphi.nrows() == total_p
                            && hphi.ncols() == total_p =>
                    {
                        let augmented = gradient + &grad_phi;
                        // Cache the exact triple at the just-accepted β so the next
                        // cycle's head reuses it instead of recomputing the
                        // O(p)-directional-derivative + GEMM term (gam#729).
                        let post_beta_key = flatten_state_betas(&states, specs);
                        jeffreys_triple_cache = Some((post_beta_key, grad_phi, hphi));
                        Some(augmented)
                    }
                    _ => None,
                }
            } else {
                None
            };
            let residual_gradient = jeffreys_augmented_gradient.as_ref().unwrap_or(gradient);
            if accepted_active_face_changed {
                // The accepted QP step changed the critical cone. In particular,
                // the Duchon CTN separator can enlarge the face substantially
                // (the issue-979 4,800-row replay changed 94 -> 274 rows). The
                // projected residual then jumps because it is a different KKT
                // system, while the preceding objective/residual samples describe
                // the old face. Feeding those samples to the constrained-
                // stationary or slow-rate guards creates a false plateau and
                // refuses the very first iterate on the newly discovered face.
                //
                // Reset every cross-cycle progress statistic whose inference
                // assumes a fixed stationarity system. This does not accept an
                // iterate or loosen a tolerance: the current residual is computed
                // below on the new face and must build fresh evidence and satisfy
                // the unchanged KKT certificate.
                min_certified_residual = f64::INFINITY;
                best_residual_seen = f64::INFINITY;
                cycles_since_residual_improved = 0;
                residual_descent_history.clear();
                tr_clamped_during_stall = false;
                residual_rate_history.clear();
                merit_window.clear();
                geometric_tail_history.clear();
                last_kkt_refusal_report = None;
                log::info!(
                    "[PIRLS/joint-Newton active-face] cycle {} | accepted critical-cone transition; reset fixed-face convergence histories and require a fresh KKT certificate",
                    cycle,
                );
            }
            let residual = exact_newton_joint_stationarity_inf_norm_from_gradient(
                residual_gradient,
                &states,
                specs,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                &block_constraints,
                Some(cached_active_sets.as_slice()),
                joint_lower_bounds.as_ref(),
            )?;
            prev_kkt_norm = Some(residual);
            // Record this cycle's KKT residual for the steady-geometric-descent
            // test at the certificate-refusal gate below (gam#787 centers≥20).
            if residual.is_finite() {
                min_certified_residual = min_certified_residual.min(residual);
                residual_descent_history.push_back(residual);
                while residual_descent_history.len() > RESIDUAL_DESCENT_WINDOW {
                    residual_descent_history.pop_front();
                }
            }

            // Scale-aware tolerances. The objective check was already
            // relative (`inner_tol * (1 + |obj|)`), but the step and
            // residual checks were absolute against the bare `inner_tol`
            // — at large scale (n ≈ 320k), β iterates can keep moving
            // by ~1e-5 per cycle along the monotonicity-feasible
            // manifold even after the likelihood has gone flat, and the
            // joint gradient ‖·‖_∞ is O(|obj|), not O(1). Running
            // 50-100 cycles past objective convergence is the
            // dominant inner-PIRLS cost at large scale. Switching to
            // relative scaling (`inner_tol * (1 + ‖β‖_∞)` for steps,
            // `inner_tol * (1 + |obj|)` for the gradient residual)
            // exits PIRLS as soon as the optimum is statistically
            // resolved, without loosening behavior at small n where
            // ‖β‖_∞ ≈ 1 and |obj| ≈ 1 give tolerances within 2× of
            // the historical absolute 1e-6.
            let beta_inf = states
                .iter()
                .flat_map(|s| s.beta.iter().copied())
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            let step_tol = inner_tol * (1.0 + beta_inf);
            let objective_tol = inner_tol * (1.0 + lastobjective.abs());
            // KKT residual tolerance must scale with the natural magnitude of
            // ‖Sβ − ∇L‖∞ (i.e. max(‖∇L‖∞, ‖Sβ‖∞)), not the objective. At
            // large scale with |β|∞ in the 10²–10³ range the gradient and
            // penalty norms can sit orders of magnitude above |obj| and FP
            // noise alone keeps the residual above any obj-scaled tol. The
            // pre-line-search check at the head of the cycle already uses
            // `inner_tol * (1 + max(grad_inf, pen_inf))`; using only grad_inf
            // here created an asymmetry where the same convergence criterion
            // would accept at one site and reject at the other, and on
            // marginal-slope models where Sβ is the larger term it shrank
            // the post-accept tolerance below the achievable FP floor.
            let mut block_gradient_norms = Vec::with_capacity(states.len());
            let mut block_penalty_norms = Vec::with_capacity(states.len());
            for (block_idx, (start, end)) in ranges.iter().copied().enumerate() {
                block_gradient_norms.push(
                    gradient
                        .slice(s![start..end])
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max),
                );
                let mut penalty_block = s_lambdas[block_idx].dot(&states[block_idx].beta);
                if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                    penalty_block += &states[block_idx].beta.mapv(|v| ridge * v);
                }
                block_penalty_norms.push(
                    penalty_block
                        .iter()
                        .map(|x: &f64| x.abs())
                        .fold(0.0_f64, f64::max),
                );
            }
            let grad_inf = block_gradient_norms.iter().copied().fold(0.0_f64, f64::max);
            let pen_inf = block_penalty_norms.iter().copied().fold(0.0_f64, f64::max);
            // Firth/Jeffreys score magnitude. The convergence residual is the
            // AUGMENTED stationarity `∇L − Sβ + ∇Φ`, so `∇Φ` is a first-class term
            // whose own numerical scale sets the achievable KKT floor: `∇Φ` is a
            // trace `½ tr(H_id⁻¹ Z_Jᵀ Ḣ Z_J)` formed from a FLOORED reduced-info
            // pseudo-inverse, so its components carry O(‖∇Φ‖·ε_floor) round-off
            // that the augmented residual cannot polish below. Scaling the KKT
            // tolerance by `max(grad, pen, ‖∇Φ‖)` (not just grad/pen) makes the
            // certificate reachable for coupled K-block Firth fits whose data
            // gradient is small but whose Firth score is O(1): otherwise the
            // augmented residual plateaus a few × above an unattainably tight
            // `inner_tol·(1+grad)` tol and the solve refuses just short of
            // convergence (gam#729/#715 — the residual stalled at ~8.8e-6 against a
            // ~1e-6 tol). No-op when the term is condition-gated (∇Φ=0).
            let firth_score_inf = head_jeffreys_term
                .as_ref()
                .map(|(grad_phi, _hphi)| grad_phi.iter().map(|v| v.abs()).fold(0.0_f64, f64::max))
                .unwrap_or(0.0);
            let residual_tol = inner_tol * (1.0 + grad_inf.max(pen_inf).max(firth_score_inf));
            // Arm the Jeffreys second-order endgame completion (gam#979) once
            // the residual enters the convergence band; latched (never
            // un-armed) so the endgame model cannot oscillate between the
            // divided-difference and exact Hessians across cycles.
            if residual.is_finite() && residual <= JEFFREYS_COMPLETION_RESIDUAL_BAND * residual_tol
            {
                jeffreys_completion_endgame = true;
            }
            // Active-set-projected stationarity residual vector (multiplier
            // mass of every pinned bound row already subtracted). Keep the full
            // vector so the constrained-stationary certificate can distinguish
            // represented active-set multipliers from unresolved KKT mass.
            let projected_residual_vec =
                exact_newton_joint_projected_stationarity_vector_from_gradient(
                    gradient,
                    &states,
                    specs,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    &block_constraints,
                    Some(cached_active_sets.as_slice()),
                    joint_penalty_stationarity_score(options, specs, &states).as_ref(),
                )?;
            let block_stationarity_norms = {
                let mut offset = 0usize;
                states
                    .iter()
                    .map(|state| {
                        let start = offset;
                        let end = start + state.beta.len();
                        offset = end;
                        projected_residual_vec
                            .slice(ndarray::s![start..end])
                            .iter()
                            .map(|x: &f64| x.abs())
                            .fold(0.0_f64, f64::max)
                    })
                    .collect::<Vec<_>>()
            };
            // gam#1082 perf: a per-cycle #979 divergence-trace logging block
            // lived here and computed — EVERY inner cycle for the first 40
            // cycles, purely to feed two `log::info!` lines — a FULL O((P·M)³)
            // eigendecomposition of the penalized-Hessian range, a
            // penalty-matrix min-eigenvalue, and per-penalty quadratic forms.
            // On any penalized family with a penalty null space (every
            // `select=TRUE` double-penalty tp-smooth model, including the
            // multinomial smooth-by-factor fit) the eigh's `nullity > 0` branch
            // actually ran, so each outer REML evaluation paid up to 40
            // redundant O(p³) eigendecompositions inside its inner joint-Newton.
            // That diagnostic instrumentation — not the outer iteration count —
            // was the dominant wall-clock cost (the #1082 overrun the outer
            // rel-cost decouple could not touch, because the cost is
            // per-inner-cycle, not per-outer-iteration). The trace has served
            // its #979 purpose and is removed from the production hot path; the
            // strict residual and per-block diagnostics remain available without
            // introducing a second numerical rank decision.
            let near_convergence = residual <= 10.0 * residual_tol;
            // Augmented-objective change: `(quad(new) − Φ_gated(new)) −
            // (quad(old) − Φ_gated(old))`. `lastobjective` is quadratic-only and
            // `old_objective` already carries `−old_phi`, so subtract the accepted
            // β's `new_phi` here to keep both endpoints on the Φ-augmented merit
            // (gam#826/#872). On a skippable cycle both phis are 0 ⇒ identical to
            // the bare quadratic change.
            let signed_obj_change = (lastobjective - new_phi) - old_objective;
            let objective_change = signed_obj_change.abs();

            // Per-cycle observability for the convergence test. Surfaces
            // WHICH criterion is binding (proposed step, accepted step,
            // residual, objective change) at every iteration so CI logs
            // distinguish "Newton hasn't proposed a small step yet"
            // (algorithm still working) from "step is small but residual
            // won't drop below tol" (tolerance scaling problem). Without
            // this, the only visible signal is the objective itself,
            // which is insufficient to choose the right algorithmic
            // remedy.
            //
            // gam#979 discriminator: the PER-BLOCK projected stationarity
            // breakdown. The aggregate `residual` alone cannot distinguish a
            // genuinely-coupled stall from one block dragging the others — for
            // the survival marginal↔logslope grind the question "is the total
            // residual dominated by a single block (the multiplicative
            // z·exp(logslope) coupling channel), or spread evenly (global
            // conditioning)?" is answerable only from the split. `block_resid`
            // is already computed above for the convergence test, so surfacing
            // it per cycle is free; reading it across a 75 s repro under
            // RUST_LOG=info tells whether the slowdown is a single stuck block
            // (curvature/coupling channel) or an evenly slow descent
            // (conditioning) — without it the four #979 candidates are not
            // separable from the timeline.
            let block_resid_sig = block_stationarity_norms
                .iter()
                .map(|n| format!("{n:.3e}"))
                .collect::<Vec<_>>()
                .join(",");
            log::info!(
                "[PIRLS/joint-Newton convergence] cycle {:>3} | step_inf={:.3e} (tol={:.3e}) | accepted_step_inf={:.3e} | residual={:.3e} (tol={:.3e}) | per_block_resid=[{}] | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e}",
                cycle,
                step_inf,
                step_tol,
                accepted_step_inf,
                residual,
                residual_tol,
                block_resid_sig,
                objective_change,
                objective_tol,
                beta_inf,
            );

            // gam#1082 perf: a tightly-gated `#1040 inner-conditioning probe`
            // lived here. Once the inner joint-Newton stalled (residual stuck
            // above tol for `RESIDUAL_STALL_NO_IMPROVE_CYCLES` cycles), it
            // eigendecomposed the FULL P·M penalized Hessian (O((P·M)³)) plus an
            // O(p²) Rayleigh-quotient loop EVERY cycle thereafter, purely to feed
            // one `log::info!`. The gate's whole point is "the solve is
            // grinding" — exactly the regime where it then fires on EVERY one of
            // the remaining (up to `inner_max_cycles`) cycles, turning a stall
            // into an O(p³)-per-cycle crawl (a dominant face of the #1082
            // multinomial wall-clock overrun: the cost is per-stalled-cycle, not
            // per-outer-iteration). The diagnostic is removed from the hot path;
            // the inner solve's own stall handling (trust-region clamp and
            // Newton-decrement certificate) governs
            // termination, and the cheap per-cycle convergence line above already
            // surfaces residual/step/per-block-residual for observability.

            if verbose_cycle || near_convergence {
                log::info!(
                    "[PIRLS/JN] cyc={:>3}/{} obj={:.6e} -loglik={:.6e} pen={:.3e} Δobj={:+.3e} |δ|∞={:.3e} accepted_|δ|∞={:.3e} resid={:.3e} (tol={:.3e}) obj_tol={:.3e} step_tol={:.3e} |β|∞={:.3e} attempts={} t={:.3}s",
                    cycle,
                    inner_max_cycles,
                    lastobjective,
                    -current_log_likelihood,
                    current_penalty,
                    signed_obj_change,
                    step_inf,
                    accepted_step_inf,
                    residual,
                    residual_tol,
                    objective_tol,
                    step_tol,
                    beta_inf,
                    line_search_attempts,
                    cycle_started.elapsed().as_secs_f64(),
                );
            } else {
                log::info!(
                    "[PIRLS/JN] cyc={:>3}/{} obj={:.6e} Δobj={:+.3e} |δ|∞={:.3e} resid={:.3e} attempts={} t={:.3}s",
                    cycle,
                    inner_max_cycles,
                    lastobjective,
                    signed_obj_change,
                    accepted_step_inf,
                    residual,
                    line_search_attempts,
                    cycle_started.elapsed().as_secs_f64(),
                );
            }

            // Divergence guard: a non-finite KKT residual, objective, or
            // log-likelihood means the inner joint Newton has diverged (NaN
            // mass propagating from a near-unidentified penalized block — the
            // binomial location-scale shared-basis log-σ deviation channel is
            // the canonical trigger, gam#554). Every convergence and
            // residual-stall exit below is gated on finite `<=` comparisons,
            // which a NaN residual silently defeats; left unguarded the loop
            // then grinds the full `inner_loop_hard_ceiling` on every outer
            // ρ-eval and every startup seed, which is the multi-hour "hang".
            // Treat it as immediate non-convergence so the outer optimizer
            // rejects this point cleanly instead of burning the budget.
            if !residual.is_finite()
                || !lastobjective.is_finite()
                || !current_log_likelihood.is_finite()
            {
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | divergence guard: non-finite inner state (residual={:.3e}, objective={:.3e}, -loglik={:.3e}); returning unconverged so the outer optimizer rejects this ρ evaluation instead of running to inner_max_cycles.",
                    cycle,
                    residual,
                    lastobjective,
                    -current_log_likelihood,
                );
                converged = false;
                break;
            }

            // KKT convergence: a small post-step residual is the
            // canonical optimality certificate for the penalized
            // objective. ‖∇L(β) − Sβ‖∞ ≤ residual_tol means the
            // iterate is at a KKT point to numerical precision and
            // further iteration cannot reduce it; the step magnitude
            // is irrelevant once the residual signal has fired.
            //
            // Tying convergence to a small step instead would refuse
            // to recognise quadratic-rate single-shot convergence:
            // exact Newton on an exact quadratic produces one full
            // step that lands at the optimum, so ‖delta‖∞ equals the
            // initial distance ‖β* − β₀‖∞ no matter how exact the
            // model is. Pairing a residual check with a step-size
            // requirement structurally rejects this entirely-correct
            // cycle-0 termination, leaving inner_max_cycles=1 callers
            // unable to certify convergence on a problem that was
            // solved exactly in one Newton step.
            if joint_inner_kkt_converged(residual, residual_tol) {
                finish_post_step_convergence!();
            }
            // Newton-decrement convergence certificate (gam#1040 / gam#1088).
            //
            // The strict / identified-subspace / constrained certificates all
            // gate on the penalized stationarity residual ‖∇L − Sβ‖∞ reaching
            // `residual_tol`. On a weakly-identified (near-flat) carrying block
            // — the survival marginal↔logslope alias, the binomial link-wiggle
            // block, the gaussian/binomial location-scale μ block — that residual
            // can stall ORDERS above tol (`g` is O(1e2) along a direction whose
            // penalized curvature `γ` is tiny) while every step the trust region
            // admits is clamped, so neither the residual nor the step-norm gate
            // ever closes and the loop grinds to the cycle ceiling, the outer
            // REML rejects ρ after ρ, and the fit times out (the #1040/#1088
            // benchmark hangs). Yet the ACHIEVABLE objective improvement is
            // `g²/(2γ)` — the Newton decrement — and on such a direction it is
            // far below `objective_tol`: no step the local quadratic model can
            // resolve lowers the penalized objective by more than `objective_tol`.
            // By the Conn–Gould–Toint stopping criterion (*Trust-Region Methods*,
            // Thm 6.4.6) the iterate is then the penalized optimum to within
            // tolerance, on the entire identifiable subspace — the residual's
            // un-resolvable mass lives on near-null directions the outer IFT
            // pseudo-inverse projects out (gam#553). The decrement is read off
            // the SAME D-whitened seed spectrum the step is built from (range
            // modes only; the null space contributes none), so it is exactly the
            // model decrease of the unconstrained modified-Newton step. A genuine
            // defect (real curvature AND large gradient) yields a LARGE decrement,
            // so this never certifies a non-converged iterate.
            //
            // Precondition (gam#1082): the original gate required the LAST cycle's
            // `objective_change ≤ objective_tol` to "confirm we are AT the plateau,
            // not one big step away." That precondition is the multinomial
            // smooth-by-factor blocker: the coupled-softmax select=TRUE gauge mode
            // is a NEAR-null (weak-but-above-`KKT_REFUSAL_RANK_TOL` curvature), so
            // the iterate keeps DRIFTING along it with a small but nonzero
            // `objective_change` every cycle (exactly the gam#979 survival
            // signature) — `objective_change ≤ objective_tol` never holds, the
            // decrement certificate never fires, and the solve crawls to
            // `inner_max_cycles` paying one ~p³ Newton-step eigh per cycle (the
            // eu-stack-profiled #1082 blow-up). But the decrement bound is itself
            // the correct, curvature-aware stopping test: by Conn–Gould–Toint Thm
            // 6.4.6 `decrement ≤ objective_tol` ALONE certifies the iterate is the
            // penalized optimum to tolerance — no model-resolvable step (gauge
            // drift included) lowers the objective by more than tol. So the
            // objective-flat precondition is replaced by the RESIDUAL-STALL window
            // (`cycles_since_residual_improved ≥ DECREMENT_STALL_WINDOW`): the
            // certificate fires once the raw residual has stopped descending and
            // the decrement confirms no resolvable improvement remains. This reuses
            // the EXACT degeneracy classification the Newton step uses (the
            // decrement skips every `|γ_k| ≤ null_cutoff` mode), so it catches the
            // near-null gauge direction the raw-`H_pen` range projection's absolute
            // `1e-10·λ_max` cutoff misses — without ever accepting a genuinely
            // curved (large-decrement) unconverged iterate. A still-progressing
            // solve never reaches the stall window (its residual keeps improving,
            // resetting the counter).
            //
            // Plateau disjunct (gam#1607 gaussian/binomial homoscedastic location-
            // scale). The residual-stall window alone has a complementary blind
            // spot to the multinomial drift it was built for: a near-flat scale
            // ridge (homoscedastic data → the log_σ block is weakly identified, the
            // μ block's penalized residual floors a few ×10⁻² above `residual_tol`
            // with a tiny `decrement`) keeps the raw residual JITTERING by >10% per
            // cycle around its plateau (0.031 → 0.024 → 0.028), so the 10%-drop test
            // resets `cycles_since_residual_improved` to 0 every cycle and the stall
            // window NEVER reaches DECREMENT_STALL_WINDOW within the (outer-capped,
            // ~12-cycle) refit budget. The OBJECTIVE, however, is genuinely flat
            // there (`objective_change` ~10⁻⁵ ≪ `objective_tol`) — that is the very
            // signal the original gam#1082 precondition used before it was narrowed
            // to the stall window for the multinomial gauge-drift case (where the
            // objective keeps changing). Restoring it as a DISJUNCTIVE alternative
            // recovers the homoscedastic case without touching multinomial (which
            // still fires via the stall window): both disjuncts gate the SAME
            // rigorous `decrement ≤ objective_tol` Conn–Gould–Toint stopping test
            // below, so neither can certify a genuinely reducible (large-decrement)
            // iterate — a fit one resolvable step from the optimum has a large
            // decrement (fails the bound) regardless of which precondition admits
            // it, and a fit still making real objective progress has
            // `objective_change > objective_tol` (fails this disjunct) AND a
            // descending residual that resets the stall window (fails the other).
            const DECREMENT_STALL_WINDOW: usize = 3;
            let decrement_precondition = cycles_since_residual_improved >= DECREMENT_STALL_WINDOW
                || objective_change <= objective_tol;
            if decrement_precondition
                && let Some(decrement) = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.newton_decrement())
                && decrement.is_finite()
                && decrement <= objective_tol
                // Conditioning-robust safety (gam#1449): the raw decrement above
                // excludes every `|γ_k| ≤ null_cutoff = max(rank_tol·λ_max,
                // numerical_floor)` mode. On a badly-scaled penalized Hessian
                // `rank_tol·λ_max` can swallow a mode with small-but-REAL curvature
                // AND real signal — a weakly-identified direction, not gauge —
                // whose achievable improvement `c²/(2|γ|)` the raw decrement then
                // silently ignores. Require that improvement, measured over the
                // genuinely-curved band (above the machine-rank `numerical_floor`,
                // a conditioning-robust cutoff that does NOT scale with λ_max), to
                // ALSO be within tolerance before certifying. The genuine numerical
                // null space (below `numerical_floor`) still contributes nothing,
                // and the step is unchanged; this only HARDENS the stopping test so
                // a weakly-identified real mode blocks premature certification.
                && let Some(weak_decrement) = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.weakly_identified_decrement())
                && weak_decrement.is_finite()
                && weak_decrement <= objective_tol
            {
                // Audit witness (#1082): the residual mass this certificate
                // EXCLUDES as gauge-null. The decrement bound is sound only when
                // that excluded mass truly lies on penalty-null directions; if it
                // is large the certificate may have discarded a weakly-identified
                // real mode (the `null_cutoff = rank_tol·λ_max` ill-conditioning
                // edge), so emit it at WARN to keep the decision auditable.
                let excluded_null_residual = joint_spectrum
                    .as_ref()
                    .map(|spectrum| spectrum.null_residual_inf())
                    .unwrap_or(0.0);
                if excluded_null_residual > residual_tol.max(1e-6) {
                    log::warn!(
                        "[PIRLS/joint-Newton convergence] cycle {cycle:>3} | Newton-decrement \
                         certificate fired with LARGE excluded near-null residual \
                         ={excluded_null_residual:.3e} (> tol={residual_tol:.3e}); the stopping \
                         rule treated this mass as free gauge. Sound iff it lies on genuine \
                         penalty-null directions — flagged for joint-stationarity audit (#1082)."
                    );
                }
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | Newton-decrement certificate (gam#1040/#1088/#1082): \
                     residual={:.3e} (tol={:.3e}) stalled above tol for {} cycles on a weakly-identified block (last \
                     |Δobjective|={:.3e}, drifting along a near-null gauge mode), but the unconstrained modified-Newton \
                     step's predicted objective decrease (Newton decrement ½gᵀH⁻¹g over identified modes, the SAME \
                     |γ_k|≤null_cutoff degeneracy classification the Newton step uses)={:.3e} ≤ objective_tol={:.3e} \
                     — no model-resolvable step lowers the penalized objective by more than tolerance, so the \
                     iterate is the REML optimum on the identifiable subspace (Conn–Gould–Toint Thm 6.4.6); \
                     the un-resolvable residual mass lies on near-null directions the outer IFT projects out.",
                    cycle,
                    residual,
                    residual_tol,
                    cycles_since_residual_improved,
                    objective_change,
                    decrement,
                    objective_tol,
                );
                // Record the residual this exit certified on so the terminal
                // line reports a finite certified residual (#1040 truthfulness):
                // the converged status is earned by the decrement bound, and the
                // finite stationarity residual at this iterate is the honest
                // certificate witness.
                if residual.is_finite() {
                    min_certified_residual = min_certified_residual.min(residual);
                }
                finish_post_step_convergence!();
            }

            // Noise-floor KKT certificate.
            //
            // Reading the joint stationarity residual ‖∇L(β) − Sβ‖_∞ at finite
            // precision picks up rounding mass from the X'WX assembly and the
            // per-block penalty contraction. For well-conditioned problems
            // that floor sits well below `residual_tol`, so the strict path
            // fires and this branch is dormant. For tightly converged inner
            // states where the Newton iterate is already at the analytic
            // optimum but every additional step changes the objective by less
            // than `objective_tol` and the recomputed residual lands just
            // above `residual_tol` due to arithmetic noise, the strict path
            // alone refuses to certify convergence — even though no further
            // useful descent direction exists. Burning hundreds of identical
            // descent cycles past that point neither tightens the inner
            // optimum (the noise floor sets a hard lower bound on ‖rhs‖) nor
            // gives the outer optimizer more hyperparameter information; it
            // just causes the outer wrapper to reject every seed as
            // "inner did not converge" and downstream callers to mark the
            // analytic outer Hessian as unavailable.
            //
            // Combining two independent post-step signals — objective change
            // within scale-aware tolerance AND residual within the same KKT
            // tolerance — supplies the missing certificate without weakening
            // the envelope-theorem requirement. A residual above tolerance
            // can be a free Hessian-null gradient component, not an active
            // multiplier, so it must not be accepted by an objective-flatness
            // rule.
            //
            // Distinct from the strict path because the strict path is silent
            // on objective change;
            // distinct from the trust-region floor certificate at the head
            // of the cycle because that one fires only when the trust radius
            // has collapsed to its 1e-12 floor with all attempts rejected,
            // whereas this branch fires when the trust region is still open
            // but each accepted step is no longer producing detectable
            // objective progress.
            let objective_change = signed_obj_change.abs();
            if objective_change.is_finite() {
                geometric_tail_history.push_back(objective_change);
                while geometric_tail_history.len() > GEOMETRIC_TAIL_WINDOW {
                    geometric_tail_history.pop_front();
                }
            }
            if objective_change <= objective_tol && residual <= residual_tol {
                log::info!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | noise-floor KKT certificate: residual={:.3e} <= tol={:.3e}, |Δobjective|={:.3e} <= obj_tol={:.3e}",
                    cycle,
                    residual,
                    residual_tol,
                    objective_change,
                    objective_tol,
                );
                finish_post_step_convergence!();
            }

            // Constrained-stationary certificate.
            //
            // The inner Newton system is `Hδ = -g`, solved over the
            // active-constraint-aware subspace (the QP step path).  When
            // the *unprojected* gradient `g` carries a large Lagrange-
            // multiplier component pointing into the constraint —
            // i.e. some β coordinates are pinned at the bound or against
            // the family's structural constraint surface — the linear
            // solve correctly DOES NOT try to eliminate that component,
            // because doing so would push β infeasibly.  The signature of
            // this state is precise and entirely local to the most recent
            // accepted step:
            //
            //   • `‖g + Hδ‖∞ / ‖g‖∞ ≥ 0.5` — the linear solve neutralised
            //     ≤ 50 % of g; the remainder is structurally outside the
            //     solver's range, i.e. it's a Lagrange multiplier of the
            //     active constraints, not a defect of the linear solve.
            //   • `|actual − pred| / max(|pred|, …) ≤ 1e-3` — the local
            //     quadratic Newton model agrees with the actual objective
            //     change to roundoff, so the Hessian and gradient are
            //     correct AT this β.  The "stuck" residual is not noise
            //     in the linearisation; it's a real multiplier.
            //   • `|Δobjective| ≤ objective_tol` — the objective has
            //     ceased moving meaningfully.
            //   • `|δ|∞ ≤ step_tol` — the accepted feasible Newton step is
            //     exhausted. Objective flatness alone is not a terminal
            //     signal on large survival fits: a step of O(1e-2..1e-1)
            //     can still continue reducing the KKT residual after the
            //     objective first crosses tolerance.
            //
            // Together these four are the rigorous certificate that
            // Newton has reached a constrained-stationary point: further
            // cycles would reproduce the same plateau (the diagnostic in
            // PIRLS/JN/math shows `‖g+Hδ‖/‖g‖` constant near 1 cycle
            // after cycle, the very signature this certificate names).
            //
            // The 0.5 threshold on `linearized_rel` is conservative —
            // an unconstrained Newton step has `linearized_rel ≈ 1e-12`;
            // a step deliberately constrained to a (k-1)-dim subspace
            // leaves the orthogonal Lagrange direction in the residual
            // and `linearized_rel ≈ |λ|/|g| > 0`, typically 0.9+ in
            // practice when the multiplier dominates.  Anything ≥ 0.5
            // is unambiguously in the constrained-stationary regime;
            // unconstrained Newton with `linearized_rel ≥ 0.5` would
            // have already failed the trust-region's scalar model test
            // and been rejected upstream.
            if let Some(math) = last_joint_math.as_ref() {
                let linearized_rel = math.linearized_rel();
                let scalar_model_relerr = math.scalar_model_relative_error();
                let geometric_tail_bound = if geometric_tail_history.len() == GEOMETRIC_TAIL_WINDOW
                {
                    let values = geometric_tail_history.iter().copied().collect::<Vec<_>>();
                    let mut max_ratio = 0.0_f64;
                    let mut valid = true;
                    for pair in values.windows(2) {
                        let prev = pair[0];
                        let next = pair[1];
                        if prev <= 0.0 || next < 0.0 || !prev.is_finite() || !next.is_finite() {
                            valid = false;
                            break;
                        }
                        let ratio = next / prev;
                        if !ratio.is_finite() || ratio >= 1.0 {
                            valid = false;
                            break;
                        }
                        max_ratio = max_ratio.max(ratio);
                    }
                    if valid {
                        Some(objective_change / (1.0 - max_ratio).max(1.0e-12))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let certificate_decision = constrained_stationary_certificate_decision(
                    math,
                    objective_change,
                    objective_tol,
                    step_tol,
                    geometric_tail_bound,
                    residual,
                    residual_tol,
                );
                if !matches!(
                    certificate_decision,
                    ConstrainedStationaryCertificate::NotCandidate
                ) {
                    // The `linearized_rel >= 0.5` signal is necessary but not
                    // sufficient. It proves either (a) g carries a Lagrange
                    // multiplier of an active constraint that the QP's active
                    // set already represents — in which case the *projected*
                    // residual is at tolerance — or (b) H is rank-deficient
                    // in the direction of g, so Hδ ≈ 0 along the null
                    // direction regardless of whether g is a multiplier or a
                    // real defect. Case (b) is the survival marginal-slope
                    // pathology at large scale: H σ_min ≈ 1e-12 and Newton
                    // genuinely cannot move g, but the residual is NOT a
                    // captured multiplier — it's an unresolved KKT defect in
                    // the H-null subspace.
                    //
                    // The projected residual computed at the top of this
                    // block (line ~12055) already subtracts the multiplier
                    // mass of every row in `cached_active_sets`. If that
                    // residual is at tolerance, case (a) holds and the
                    // certificate is honest. If it's still orders of
                    // magnitude above tolerance, case (b) holds: certifying
                    // here would hand the unified evaluator a
                    // `kkt_residual` with norm ≈ ‖g‖ which then gets
                    // amplified by H⁻¹_proj in the cost/gradient IFT
                    // corrections, contaminating the envelope formula and
                    // triggering the "envelope-gradient consistency"
                    // tripwire downstream. Bail with `converged = false` so
                    // the outer optimizer rejects this ρ cleanly, exactly
                    // as it would on any other non-converged inner exit.
                    let cert_residual_factor = 1.0;
                    if matches!(
                        certificate_decision,
                        ConstrainedStationaryCertificate::Accept
                    ) {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained-stationary certificate: \
                             linear-solve neutralised {:.1}% of g (the remaining {:.1}% is a Lagrange multiplier \
                             of the active constraint set, not an unresolved gradient); \
                             scalar Newton model agrees with reality to relerr={:.3e} (Hessian+gradient are correct \
                             at this β); projected residual={:.3e} ≤ {:.1}×tol={:.3e} (multipliers captured by active set); \
                             |Δobjective|={:.3e}, geometric_tail_bound={:.3e}, obj_tol={:.3e}; further cycles cannot reduce the \
                             multiplier mass and would reproduce this plateau indefinitely; \
                             active-set multiplier mass will be projected out of the KKT residual \
                             before the outer IFT correction is assembled",
                            cycle,
                            (1.0 - linearized_rel) * 100.0,
                            linearized_rel * 100.0,
                            scalar_model_relerr,
                            residual,
                            cert_residual_factor,
                            cert_residual_factor * residual_tol,
                            objective_change,
                            geometric_tail_bound.unwrap_or(objective_change),
                            objective_tol,
                        );
                        finish_post_step_convergence!();
                    }
                    // Constrained exact-fixed-point acceptance (gam#797).
                    //
                    // We reach here only with the iterate ALREADY proven stationary
                    // (objective + step exhausted, `linearized_rel >= 0.5` so the
                    // residual is multiplier/null mass, `scalar_relerr <= 1e-3` so
                    // the quadratic model is exact), the strict/range-space/noise
                    // certificates having declined. For a CONSTRAINED block the
                    // remaining residual can be a genuine active-constraint Lagrange
                    // multiplier that the active-set QP under-identified (it reports
                    // only rows it drove tight during a non-degenerate step, so a
                    // monotone derivative-guard row tight at the optimum but never
                    // explicitly stepped is missing), leaving the cone projection
                    // unable to decompose `r = A_activeᵀ λ` and the residual stuck
                    // far above tol on an iterate that is EXACTLY the constrained
                    // optimum (the `active_set_incomplete` refusal; gam#797 survival
                    // marginal/logslope/time blocks).
                    //
                    // When (a) the joint Newton has reached a numerical FIXED POINT
                    // — the accepted step and objective change are both at the
                    // machine-epsilon floor relative to the iterate, so no further
                    // progress is mathematically possible — (b) the local quadratic
                    // model is exact (`scalar_relerr` tiny), and (c) the design
                    // carries linear inequality constraints AND `H_pen` has NO
                    // numerical null space (so the residual is an active-constraint
                    // multiplier, NOT an H-null/rank-deficient defect, which the
                    // range-space certificate above already handles), the iterate is
                    // a bona fide constrained KKT point. The active-constraint
                    // multiplier mass is projected out of the KKT residual by the
                    // unified evaluator's active-constraint-aware IFT correction
                    // before the envelope gradient, exactly as for an explicitly
                    // captured multiplier, so certifying here is correct. Gated
                    // strictly on a fixed point with no H-null, so a genuinely
                    // non-converged or rank-deficient iterate is never accepted.
                    let any_block_constrained = block_constraints.iter().any(|c| c.is_some());
                    let beta_scale = states
                        .iter()
                        .flat_map(|s| s.beta.iter().copied())
                        .map(f64::abs)
                        .fold(0.0_f64, f64::max)
                        .max(1.0);
                    let fixed_point_floor = 64.0 * f64::EPSILON * beta_scale;
                    let objective_floor = 64.0 * f64::EPSILON * (1.0 + lastobjective.abs());
                    let at_numerical_fixed_point = accepted_step_inf.is_finite()
                        && accepted_step_inf <= fixed_point_floor
                        && objective_change <= objective_floor
                        && scalar_model_relerr <= 1e-3;
                    if any_block_constrained && at_numerical_fixed_point {
                        // Materialize H_pen = H + S(λ) (+ model ridge) and count its
                        // numerical null space at the shared rank tolerance: nullity == 0
                        // ⇒ the stuck residual is NOT an H-null/rank-deficient defect
                        // (that case is handled by the range-space certificate above) but
                        // a genuine active-constraint multiplier.
                        let hpen_nullity = materialize_joint_hessian_source(
                            &joint_hessian_source,
                            total_p,
                            "constrained fixed-point nullity check",
                        )
                        .ok()
                        .map(|mut h_pen| {
                            let model_diagonal_ridge =
                                if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                                    ridge
                                } else {
                                    0.0
                                };
                            add_joint_penalty_to_matrix(
                                &mut h_pen,
                                &ranges,
                                &s_lambdas,
                                model_diagonal_ridge,
                                None,
                            );
                            symmetrize_dense_in_place(&mut h_pen);
                            symmetric_penalized_hessian_nullity(&h_pen)
                        })
                        .unwrap_or(None);
                        if hpen_nullity == Some(0) {
                            log::info!(
                                "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained fixed-point certificate:                                  accepted_step_inf={:.3e} ≤ {:.3e} and |Δobjective|={:.3e} ≤ {:.3e} (numerical fixed point),                                  scalar_relerr={:.3e}, linearized_rel={:.3e}; H_pen has no numerical null space so the                                  residual={:.3e} is an active-constraint Lagrange multiplier (the QP under-identified the                                  binding rows), projected out of the KKT residual by the active-constraint-aware IFT                                  correction before the envelope gradient — the iterate is a constrained KKT point",
                                cycle,
                                accepted_step_inf,
                                fixed_point_floor,
                                objective_change,
                                objective_floor,
                                scalar_model_relerr,
                                linearized_rel,
                                residual,
                            );
                            finish_post_step_convergence!();
                        }
                    }
                    // Still-converging guard (gam#787 duchon centers≥20). The
                    // certificates above all declined, so the iterate would be
                    // refused as a multiplier/null plateau. But the
                    // `linearized_rel ≥ 0.5` + flat-objective signature that
                    // routed us here ALSO holds for a logslope block whose
                    // objective is already at its Φ-bounded floor while the KKT
                    // residual is still polishing by a STEADY geometric factor
                    // each cycle. Refusing there rejects the seed a few cycles
                    // short of `residual_tol` (→ outer seed-rejection → raise).
                    // If the residual is in steady geometric descent over the
                    // recent window, the direction is genuinely converging, not
                    // plateaued: keep iterating (bounded by the inner cycle cap)
                    // rather than refuse. The genuine plateau (flat/oscillating
                    // residual above tol) fails this test and refuses as before.
                    if residual_in_steady_geometric_descent(&residual_descent_history) {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | certificate declined but residual in steady geometric descent (history={:?}, residual={:.3e}, tol={:.3e}); continuing to convergence rather than refusing as a plateau",
                            cycle,
                            residual_descent_history,
                            residual,
                            residual_tol,
                        );
                        continue;
                    }
                    // EARLY-CYCLE CARVE-OUT (gam#826/#872). The phantom-multiplier
                    // refusal asserts that the residual is a captured Lagrange
                    // multiplier / H-null mass that Newton genuinely cannot move —
                    // a claim that requires EVIDENCE of a plateau. The candidate
                    // conditions above (objective + step exhausted, linearized_rel ≥
                    // 0.5) are ALSO satisfied transiently when a single Newton step
                    // is small because the augmented (Firth) curvature `H_Φ` is
                    // legitimately large in the `∇Φ` direction at an oversmoothed
                    // cycle-0 seed: the step `(H+Sλ+H_Φ)⁻¹(∇L−Sβ+∇Φ)` is tiny (high
                    // curvature ⇒ short step) and ONE step undershoots the
                    // nonquadratic Firth optimum, so `step_inf` and `|Δobj|` look
                    // exhausted while the residual is still O(‖∇Φ‖) ≫ tol. Refusing
                    // there at cycle 0 (no descent history yet) aborts the coupled
                    // binomial location-scale / flexible-linkwiggle fit before the
                    // inner has taken the handful of cycles it needs to walk the
                    // curved Firth basin to its optimum. When the residual is still
                    // ORDERS above tol and we lack a full descent window to prove a
                    // genuine plateau, keep iterating — the inner cycle cap and the
                    // residual-stall / trust-region-floor guards still bound the
                    // loop and diagnose a true non-convergence. A genuine multiplier
                    // plateau (residual flat across the window) is caught once the
                    // history fills, exactly as before. The threshold is the same
                    // `RESIDUAL_DESCENT_WINDOW` the descent test uses, so this only
                    // defers the refusal until there is enough history to make it,
                    // never weakens it.
                    let residual_far_above_tol = residual.is_finite()
                        && residual_tol.is_finite()
                        && residual > cert_residual_factor * residual_tol;
                    if residual_far_above_tol
                        && residual_descent_history.len() < RESIDUAL_DESCENT_WINDOW
                    {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | constrained-stationary refusal DEFERRED: residual={:.3e} ≫ tol={:.3e} but only {} descent samples (< {} window) — too early to prove a multiplier/null plateau vs a high-curvature Firth-basin transient; continuing",
                            cycle,
                            residual,
                            residual_tol,
                            residual_descent_history.len(),
                            RESIDUAL_DESCENT_WINDOW,
                        );
                        continue;
                    }
                    // UNCONSTRAINED MODEL-STATIONARY ACCEPTANCE (gam#826/#808/#715).
                    //
                    // The phantom-multiplier refusal asserts the residual is a
                    // captured Lagrange multiplier of an active constraint that
                    // the QP could not decompose. That diagnosis is categorically
                    // IMPOSSIBLE when there is no active constraint at all: a
                    // residual cannot be a phantom multiplier of a constraint that
                    // does not exist. For a fully UNCONSTRAINED coupled fit
                    // (multinomial softmax; the location-scale flat blocks) on a
                    // near-flat Fisher surface (`diag(p)−ppᵀ → 0`, or the
                    // high-curvature/low-curvature `log_sigma` block) the
                    // Firth-augmented stationarity residual `‖∇L−Sβ+∇Φ‖` floors
                    // LEGITIMATELY above `4·residual_tol`: the absolute curvature
                    // is tiny so `residual_tol = inner_tol·(1+grad/pen/firth)` is
                    // tiny too, yet the Newton/dogleg step exhausts before the
                    // residual drops below that band — `residual_tol` is scaled by
                    // the gradient magnitude and does not see the flat-Fisher
                    // absolute-curvature floor. The well-conditioned spectrum keeps
                    // the conditioning-keyed Levenberg gate (`COND_NEWTON_SAFETY`)
                    // off, so neither LM nor the cond-armed dogleg engages, and
                    // every seed is refused as `phantom_multiplier_with_well_
                    // conditioned_H`.
                    //
                    // When the model itself certifies stationarity — the standard
                    // trust-region "predicted decrease ≈ 0" criterion, here the
                    // `at_numerical_fixed_point` flag (accepted step at the
                    // machine-eps floor, |Δobj| at the eps floor, scalar model
                    // exact to relerr ≤ 1e-3) — AND no further progress is being
                    // made (the steady-geometric-descent test above declined) AND
                    // we have a full descent window (the early-cycle deferral above
                    // passed, so this is a proven plateau not a Firth-basin
                    // transient), an unconstrained iterate is a bona fide
                    // first-order optimum: the quadratic model says no step can
                    // reduce the residual further, and there is no constraint whose
                    // multiplier the residual could otherwise represent. The
                    // residual that remains lives where the model is flat
                    // (vanishing curvature), so it carries no `gᵀ∂β/∂ρ` envelope
                    // contribution the outer IFT could not already neutralise
                    // through its penalty-projected pseudo-inverse. Accept.
                    //
                    // This does NOT regress #729 (coupled Dirichlet): that fit
                    // converges to a genuine `residual < residual_tol` and exits
                    // via the strict KKT certificate long before this branch, and
                    // even if reached it has a curved (non-flat) Fisher surface so
                    // its model is not at a fixed point with a residual stuck above
                    // tol. It does NOT mask a real non-convergence: a still-moving
                    // iterate fails `at_numerical_fixed_point` (its step / |Δobj|
                    // are above the eps floor), and a rank-deficient H-null defect
                    // is the CONSTRAINED concern the fixed-point certificate above
                    // already handles via its nullity check.
                    // The certificate-candidate conditions that routed us into
                    // this block already PROVE model stationarity for the
                    // unconstrained case: `objective_exhausted` + `step_inf ≤
                    // step_tol` (the model's minimizer is at this β), `scalar_relerr
                    // ≤ 1e-3` (the quadratic model is exact), and `linearized_rel ≥
                    // 0.5` (‖g+Hδ‖ ≈ ‖g‖, so `Hδ ≈ 0` — the residual lives in the
                    // flat/near-null subspace of H, exactly a flat-Fisher direction
                    // for an unconstrained fit). We do NOT additionally require the
                    // far stricter machine-eps `at_numerical_fixed_point` here: on a
                    // flat Fisher surface the dogleg keeps taking a small step at
                    // the `step_tol` floor every cycle, so `accepted_step_inf` floors
                    // a hair above `64·eps·|β|` and the eps-fixed-point flag never
                    // sets even though the model is stationary. The `step_tol` floor
                    // (`inner_tol·(1+|β|∞)`) is the principled stationarity gate; the
                    // eps floor is for the constrained-multiplier certificate, where
                    // a tighter proof is warranted because a wrong accept biases the
                    // constraint-aware IFT kernel.
                    let any_active_set_rows = cached_active_sets
                        .iter()
                        .any(|maybe| maybe.as_ref().is_some_and(|rows| !rows.is_empty()));
                    let unconstrained_fit = !any_block_constrained && !any_active_set_rows;
                    if unconstrained_fit {
                        log::info!(
                            "[PIRLS/joint-Newton convergence] cycle {:>3} | unconstrained model-stationary certificate (gam#826/#808/#715): \
                             no active constraint (active_set_rows_total=0) so the residual={:.3e} cannot be a phantom multiplier; \
                             the iterate is a numerical fixed point (accepted_step_inf={:.3e}, |Δobjective|={:.3e}, scalar_relerr={:.3e}) \
                             on a flat Fisher surface where residual_tol={:.3e} sits below the absolute-curvature floor; \
                             linearized_rel={:.3e}, |Δobjective| exhausted and residual not in steady descent → genuine first-order optimum, accepting",
                            cycle,
                            residual,
                            accepted_step_inf,
                            objective_change,
                            scalar_model_relerr,
                            residual_tol,
                            linearized_rel,
                        );
                        finish_post_step_convergence!();
                    }
                    // Structured per-block + per-spectrum refusal report.
                    // The legacy one-line refusal log printed only aggregate
                    // numbers (linearized_rel, scalar_relerr, residual,
                    // |Δobj|) and was not actionable on models with many
                    // blocks: it could not identify WHICH smooth carried
                    // the unresolved mass, nor whether H_pen was genuinely
                    // rank-deficient (the "polynomial null space slipped
                    // past absorption" pathology). Cost: one dense
                    // materialize + symmetric eigh on H_pen at this β,
                    // sub-millisecond for typical p, executed once per
                    // refusal (the loop breaks immediately after).
                    let report = compute_kkt_refusal_report(
                        cycle,
                        &states,
                        specs,
                        &s_lambdas,
                        &ranges,
                        cached_joint_gradient.as_ref(),
                        &cached_active_sets,
                        &block_constraints,
                        Some(&joint_hessian_source),
                        total_p,
                        ridge,
                        options.ridge_policy,
                        accepted_step_inf,
                        step_inf,
                        joint_trust_radius,
                        residual_tol,
                        objective_tol,
                        step_tol,
                        objective_change,
                        residual,
                        Some(&math),
                    );
                    log::warn!(
                        "{}",
                        report.format_structured_log(cert_residual_factor * residual_tol)
                    );
                    last_kkt_refusal_report = Some(report);
                    converged = false;
                    break;
                }
            }

            // INVESTIGATION NOTE — do NOT soft-accept here.
            //
            // The outer objective is V(ρ) = f(β*(ρ), ρ), where β*(ρ)
            // satisfies g(β*,ρ)=∇_β f=0.  The envelope/IFT gradient used
            // by the outer optimizer is
            //
            //   dV/dρ_j = ∂f/∂ρ_j
            //
            // only at g=0.  At a non-stationary β, the actual chain rule is
            //
            //   d f(β(ρ),ρ)/dρ_j = ∂f/∂ρ_j + gᵀ ∂β/∂ρ_j.
            //
            // A soft certificate based only on small Δf discards the second
            // term without proving it is small.  The projected pseudo-inverse
            // in the outer trace path removes null-space components of g, but
            // any range-space component still contributes gᵀ∂β/∂ρ and gives
            // ARC/BFGS a biased outer gradient.  The `[PIRLS/JN/math]` line
            // above now prints the actual Newton identity:
            //
            //   old_kkt = ‖g‖∞,
            //   linearized_next = ‖g + Hδ‖∞ = ‖Hδ-rhs‖∞,
            //   new_kkt = ‖g(β+δ)‖∞,
            //   scalar_model relerr = |actual-pred|/max(1,|pred|).
            //
            // That is the proof surface. The diagnostic reports the measured
            // linear solve residual, post-step KKT residual, scalar model
            // error, and step sizes directly; downstream analysis should use
            // those numbers rather than this solver attaching labels.

            // Residual-stall early-exit. The strict and noise-floor
            // certificates above require the KKT residual to land within
            // a small multiple of residual_tol. On survival marginal-slope
            // at large scale the residual oscillates in a band that is
            // orders of magnitude above tol without trending down while
            // the unconstrained proposal has |prop|∞ in the 10³–10⁶ range,
            // the TR clamps it, and each clamped step moves β by O(1)
            // without driving ‖∇L − Sβ‖∞ closer to KKT.
            //
            // Spending the remaining cycle budget on this pattern hits
            // inner_max_cycles "non-converged", which then routes the
            // outer optimizer through the first-order bridge with a stale
            // same-ρ inner mode and a gradient of magnitude 10⁷ that kills
            // BFGS line search at iter 0 (the failure mode pinned in the
            // commit messages of 6578e884 and 1c181d1f).
            //
            // Track the best residual seen so far and the number of
            // cycles since any meaningful improvement (≥ 10 % drop). Once
            // the inner has burned at least RESIDUAL_STALL_MIN_CYCLES
            // without progress and the accepted step kept hitting the
            // trust-region clamp, return `converged = false` with the current
            // finite β. A stalled residual above the strict KKT tolerance is
            // not converted into convergence by a pointwise Hessian rank test.
            if residual.is_finite() {
                if residual < RESIDUAL_STALL_IMPROVEMENT_FACTOR * best_residual_seen {
                    best_residual_seen = residual;
                    cycles_since_residual_improved = 0;
                    tr_clamped_during_stall = false;
                } else {
                    cycles_since_residual_improved =
                        cycles_since_residual_improved.saturating_add(1);
                    if last_accepted_hit_joint_trust_boundary {
                        tr_clamped_during_stall = true;
                    }
                }
                // Trailing window of post-step residuals for the deterministic
                // slow-geometric-rate stall projection (gam#979 survival). Kept
                // at length ≤ LINEAR_RATE_WINDOW+1 so the front is the residual
                // exactly LINEAR_RATE_WINDOW cycles back.
                if residual_rate_history.len() > LINEAR_RATE_WINDOW {
                    residual_rate_history.pop_front();
                }
                residual_rate_history.push_back(residual);
            }
            // Trailing window of the Φ-augmented merit, kept in lockstep with
            // `residual_rate_history` so its front is the merit exactly
            // LINEAR_RATE_WINDOW cycles back. Powers the merit-descent veto on
            // the two residual-trend stall guards below (gam#1607 wiggle).
            if lastobjective.is_finite() {
                if merit_window.len() > LINEAR_RATE_WINDOW {
                    merit_window.pop_front();
                }
                merit_window.push_back(lastobjective);
            }
            // Is the merit still descending robustly across the trailing
            // window? A residual-trend stall verdict is premature while it is:
            // the line search is making real progress on the actual objective,
            // and the KKT residual's transient non-monotonicity (the wiggle
            // gauge-null re-anchoring, gam#1607) is not evidence of being
            // stuck. "Robustly" = the merit dropped by more than the
            // accumulated objective tolerance over the window — i.e. by more
            // than the convergence machinery would call flat — so a merit that
            // has genuinely plateaued (the #979 survival stall, ~1e-5 steps ⇒
            // merit flat to f64) does NOT clear the bar and the guard fires as
            // before. The per-cycle `objective_tol` (relative, scale-aware) is
            // the natural unit; require the window drop to exceed
            // LINEAR_RATE_WINDOW × objective_tol so a window of merely
            // tolerance-scale dithering counts as flat.
            let merit_still_descending_over_window = || -> bool {
                if merit_window.len() <= LINEAR_RATE_WINDOW {
                    // Not enough history yet to judge a window-scale trend;
                    // don't veto on partial information (the guards have their
                    // own RESIDUAL_STALL_MIN_CYCLES floor anyway).
                    return false;
                }
                let oldest = *merit_window.front().unwrap();
                let newest = *merit_window.back().unwrap();
                if !oldest.is_finite() || !newest.is_finite() {
                    return false;
                }
                let drop = oldest - newest;
                drop > (LINEAR_RATE_WINDOW as f64) * objective_tol
            };
            // Deterministic tol-reachability exemption for the two counter-based
            // stall guards below (gam#979 CTN endgame). The ≥10%-drop
            // "improvement" test cannot distinguish a genuinely flat residual
            // from a slow monotone endgame descent closing on tol: a residual
            // walking 1.06×tol → tol over ~a dozen cycles never produces a
            // single 10% drop, so `cycles_since_residual_improved` climbs and
            // the guards (or the merit-veto cap expiring) kill a solve that is
            // cycles away from certifying — measured on the #979 CTN smoke,
            // where one run certifies at cycle ~122 (r=9.05e-3 ≤ tol=9.56e-3)
            // and another is killed at cycle 123 by the veto-cap expiry with
            // the residual 6% above tol and still descending. The trailing
            // -window geometric projection already trusted by the slow-rate
            // guard is the honest discriminator: if it projects reaching
            // `residual_tol` within LINEAR_RATE_PROJECTION_CAP cycles the
            // residual IS trending to KKT and a stall verdict is false, so the
            // guards defer to the projection guard (which fires precisely when
            // reachability fails). The genuine stall shapes keep exiting: a
            // flat or rising window projects `unreachable`, and the historic
            // #979 hang (~0.99×/cycle orders above tol) projects far past the
            // cap. Deterministic: cycle indices and residual ratios only; the
            // `inner_max_cycles` ceiling remains the hard backstop.
            let residual_tol_reachable_within_cap = residual_rate_history.len()
                > LINEAR_RATE_WINDOW
                && !gam_solve::loop_guard::slow_geometric_rate_exceeds_projection_cap(
                    residual,
                    *residual_rate_history.front().unwrap(),
                    LINEAR_RATE_WINDOW,
                    residual_tol,
                    LINEAR_RATE_PROJECTION_CAP,
                );
            if cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
                && cycles_since_residual_improved >= RESIDUAL_STALL_NO_IMPROVE_CYCLES
                && tr_clamped_during_stall
                && !residual_tol_reachable_within_cap
            {
                let last_math_summary = last_joint_math
                    .as_ref()
                    .map(|math| {
                        format!(
                            "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                            math.old_kkt_inf,
                            math.linearized_next_kkt_inf,
                            math.actual_reduction,
                            math.predicted_reduction,
                            math.trust_ratio,
                            math.scalar_model_relative_error(),
                            math.step_inf,
                            math.proposal_inf,
                        )
                    })
                    .unwrap_or_else(|| "last_newton_math=<none>".to_string());
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | residual-stall early-exit: residual={:.3e} best_seen={:.3e} no_improve_cycles={} accepted_step_inf={:.3e} trust_radius={:.3e} block_stationarity_inf={:?} {}; returning unconverged with finite β so the outer optimizer rejects this ρ evaluation before inner_max_cycles.",
                    cycle,
                    residual,
                    best_residual_seen,
                    cycles_since_residual_improved,
                    accepted_step_inf,
                    joint_trust_radius,
                    block_stationarity_norms,
                    last_math_summary,
                );
                // Record a structured KKT-refusal report at the stall iterate so
                // the bubbled IntegrationFailed error carries the per-block
                // residual breakdown + H_pen spectrum instead of the opaque
                // "no joint Newton math snapshot" string (gam#979/#1040). This is
                // the dominant non-convergence exit for the survival
                // marginal-slope monotone-cone DGP; without a report the cause of
                // the abort is invisible past serialization.
                cycles_done = cycle + 1;
                let report = compute_kkt_refusal_report(
                    cycle,
                    &states,
                    specs,
                    &s_lambdas,
                    &ranges,
                    cached_joint_gradient.as_ref(),
                    &cached_active_sets,
                    &block_constraints,
                    Some(&joint_hessian_source),
                    total_p,
                    ridge,
                    options.ridge_policy,
                    accepted_step_inf,
                    step_inf,
                    joint_trust_radius,
                    residual_tol,
                    objective_tol,
                    step_tol,
                    objective_change,
                    residual,
                    last_joint_math.as_ref(),
                );
                last_kkt_refusal_report = Some(report);
                converged = false;
                break;
            }

            // KKT convergence: small residual plus EITHER a small
            // Newton step (tight quadratic-rate convergence, lets β
            // polish to machine precision), confirmed stagnation
            // (`accepted_step_inf <= step_tol` AND `objective_change
            // <= objective_tol`, the rank-deficient null-mode case),
            // OR a stricter stationarity certificate where both the
            // residual and objective change are an additional factor of
            // `inner_tol` below their scale-aware tolerances. The last
            // branch is deliberately stricter than the public tolerance:
            // it handles machine-precision null directions where β can
            // still move by about `step_tol` but the KKT residual and
            // objective are already over-polished. Using objective
            // stagnation alone is not sufficient; the residual guard is
            // what preserves first-order correctness.
            let superconverged_residual_tol = inner_tol * residual_tol;
            let superconverged_objective_tol = inner_tol * objective_tol;
            let superconverged_stationarity = residual <= superconverged_residual_tol
                && objective_change <= superconverged_objective_tol;
            if residual <= residual_tol
                && (step_inf <= step_tol
                    || (accepted_step_inf <= step_tol && objective_change <= objective_tol)
                    || superconverged_stationarity)
            {
                log::info!(
                    "[JN-EXIT] cycle={cycle} reason=strict_kkt residual={residual:.3e} residual_tol={residual_tol:.3e} obj_change={objective_change:.3e} objective_tol={objective_tol:.3e} accepted_step_inf={accepted_step_inf:.3e} step_tol={step_tol:.3e}",
                );
                // This branch certifies on `residual ≤ residual_tol`; record it
                // so the terminal line reports the finite certified residual
                // rather than the `inf` stall sentinel (#1040 truthfulness).
                if residual.is_finite() {
                    min_certified_residual = min_certified_residual.min(residual);
                }
                finish_post_step_convergence!();
            }
            // Carry the KKT-stationarity / objective-stagnation signals
            // into the next cycle so the line-search-failure path above
            // can recognise a true KKT optimum on a rank-deficient null
            // mode. See that path for the full rationale.
            last_cycle_residual_below_tol = residual <= residual_tol;
            last_cycle_obj_change_below_tol = objective_change <= objective_tol;

            // Flat-residual stall early-exit (gam#1040/#979/#370/#859).
            //
            // The `tr_clamped_during_stall` residual-stall exit above only fires
            // when the accepted step kept hitting the trust-region boundary. A
            // distinct but equally terminal stall reaches neither it nor any
            // acceptance certificate: the KKT residual stops improving (no ≥10%
            // drop for the full `RESIDUAL_STALL_NO_IMPROVE_CYCLES` window) while
            // the accepted steps stay strictly INSIDE the trust region (so
            // `tr_clamped_during_stall` never latches) and the objective keeps
            // drifting just above `objective_tol` (so the relative-objective
            // plateau exit's flat streak never completes). This is the measured
            // "[joint-newton-tr] cycles 1000+" wall on the binomial location-scale
            // / bms-flex / CTN inner solves: without an exit the loop grinds the
            // remaining budget to `inner_loop_hard_ceiling` on every outer
            // ρ-evaluation, then hands the outer optimizer a non-converged result
            // anyway.
            //
            // Reaching this point means every acceptance certificate above already
            // DECLINED this cycle — the residual is above `residual_tol`, its
            // range-space component is above tolerance (so the iterate is NOT
            // stationary on the identifiable subspace), and there is no
            // constrained-multiplier signature. The honest action mirrors the
            // `tr_clamped` `converged=false` exit: stop and return the current
            // finite β as NON-converged so the outer optimizer rejects this ρ
            // cleanly. This is purely a termination/perf guard — it certifies
            // nothing (`converged=false`) and so cannot bias the envelope
            // gradient; it only rejects the same non-optimum sooner. The `≥10%
            // drop` reset of `cycles_since_residual_improved` keeps a
            // geometrically-descending solve (residual dropping by a steady factor
            // each cycle) from ever reaching the window — only a genuinely flat
            // residual does.
            if residual.is_finite()
                && residual > residual_tol
                && cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
                && cycles_since_residual_improved >= RESIDUAL_STALL_NO_IMPROVE_CYCLES
                && !residual_tol_reachable_within_cap
                && (!merit_still_descending_over_window()
                    || cycles_since_residual_improved >= RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES)
            {
                log::warn!(
                    "[PIRLS/joint-Newton convergence] cycle {:>3} | flat-residual stall early-exit (gam#1040/#979): residual={:.3e} (tol={:.3e}) best_seen={:.3e} stalled {} cycles with steps inside the trust region (tr_clamped={}) and no acceptance certificate satisfied; the residual is neither trending toward KKT nor stationary on the identifiable subspace, so returning unconverged with finite β instead of grinding to inner_max_cycles={}.",
                    cycle,
                    residual,
                    residual_tol,
                    best_residual_seen,
                    cycles_since_residual_improved,
                    tr_clamped_during_stall,
                    inner_max_cycles,
                );
                cycles_done = cycle + 1;
                converged = false;
                break;
            }

            // Slow-geometric-rate stall early-exit (gam#979 survival marginal-slope).
            //
            // Distinct from the flat-residual exit above (residual NOT improving
            // for the no-improve window) and the Newton-decrement certificate
            // (decrement ≤ objective_tol). Here the residual IS descending, just
            // geometrically and far too slowly to reach tol in a practical cycle
            // count — the survival marginal-slope oversmoothed-ρ endgame (stiff
            // penalized Hessian → ~1e-5 Newton steps far inside a large trust
            // radius → residual ~0.99×/cycle). Project, from the trailing
            // window's geometric rate, the additional cycles to reach
            // `residual_tol`; if that exceeds LINEAR_RATE_PROJECTION_CAP the
            // ρ-evaluation cannot finish in a practical budget, so return the
            // finite β as NON-converged and let the outer optimizer reject this
            // ρ cleanly instead of grinding ~10³ cycles to inner_max_cycles (the
            // #979 "hang"). DETERMINISTIC: cycle indices and residual ratios
            // only, no wall-clock (cf. the no-wall-clock note below). Certifies
            // nothing (`converged=false`) so it cannot bias the envelope
            // gradient; it only rejects an impractical-to-finish iterate sooner.
            // A still-progressing (quadratic / fast-geometric) solve reaches tol
            // in a handful of cycles and never fills the window, so this never
            // fires on a healthy fit.
            if residual.is_finite()
                && residual > residual_tol
                && cycle + 1 >= RESIDUAL_STALL_MIN_CYCLES
                && residual_rate_history.len() > LINEAR_RATE_WINDOW
                && (!merit_still_descending_over_window()
                    || cycle + 1 >= RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES)
            {
                let oldest = *residual_rate_history.front().unwrap();
                // Single source of truth for the slow-geometric-rate projection
                // (gam#979): deterministic cycle-count projection, no wall-clock.
                let too_slow = gam_solve::loop_guard::slow_geometric_rate_exceeds_projection_cap(
                    residual,
                    oldest,
                    LINEAR_RATE_WINDOW,
                    residual_tol,
                    LINEAR_RATE_PROJECTION_CAP,
                );
                if too_slow {
                    log::warn!(
                        "[PIRLS/joint-Newton convergence] cycle {:>3} | slow-geometric-rate stall early-exit (gam#979): residual={:.3e} (tol={:.3e}) descending at ~{:.4}×/cycle over the last {} cycles — projected >{} more cycles to reach tol; the residual is converging but far too slowly to finish in a practical budget (the survival marginal-slope oversmoothed-ρ endgame), so returning unconverged with finite β instead of grinding to inner_max_cycles={}.",
                        cycle,
                        residual,
                        residual_tol,
                        (residual / oldest).powf(1.0 / (LINEAR_RATE_WINDOW as f64)),
                        LINEAR_RATE_WINDOW,
                        LINEAR_RATE_PROJECTION_CAP,
                        inner_max_cycles,
                    );
                    cycles_done = cycle + 1;
                    converged = false;
                    break;
                }
            }

            // NOTE: there is deliberately NO wall-clock-driven "adaptive
            // early-exit" here. A convergence verdict that fires when a cycle's
            // wall-clock happens to fall below a fraction of a running EMA is
            // non-deterministic — under CPU contention (a parallel sweep) the
            // same fit accepts at a different iterate than it does run alone,
            // which cascades into a different accepted outer state (gam#979's
            // sequential-versus-parallel instability). It also
            // accepts iterates up to 10× outside the real KKT/objective
            // tolerance, biasing the REML/LAML criterion the inner residual
            // feeds. Convergence is certified ONLY by the mathematical tests
            // above (KKT residual / Newton step / objective change at their
            // scale-aware tolerances); whether convergence is *reachable within
            // the cycle budget* is judged by the deterministic descent-rate
            // guard alongside the residual-stall detector above.
        }

        if converged {
            let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
            let joint_constraints =
                assemble_joint_linear_constraints(&block_constraints, &ranges, total_p)?;
            // A full-space PSD test is exact for the unconstrained CTN mode.
            // Constrained modes certify on the active-face tangent (the
            // critical-cone surrogate under strict complementarity) — the same
            // Z the terminal determinant uses, so a mode this certificate
            // accepts can never fail the downstream SPD logdet on curvature
            // grounds. Jeffreys-augmented families still require their own
            // augmented-objective certificate and retain the first-order
            // contract here.
            if !returned_mode_curvature_certified && joint_jeffreys_subspace.is_none() {
                let mode_active_block = if joint_constraints.is_some() {
                    crate::blockwise_solve::assemble_active_constraint_block(
                        &block_constraints,
                        &cached_active_sets,
                        &ranges,
                        total_p,
                    )
                } else {
                    None
                };
                let certificate = exact_joint_mode_curvature_certificate(
                    family,
                    &states,
                    specs,
                    options,
                    &ranges,
                    &s_lambdas,
                    joint_mode_diagonal_ridge,
                    joint_bundle,
                    total_p,
                    mode_active_block.as_ref(),
                )?;
                let has_negative_curvature = certificate.has_resolvable_negative_curvature();
                let minimum_whitened_eigenvalue = certificate.minimum_whitened_eigenvalue;
                let numerical_floor = certificate.numerical_floor;
                cached_joint_workspace = certificate.workspace;
                if has_negative_curvature {
                    return Err(format!(
                        "joint Newton tentative convergence rejected by fresh exact returned-mode curvature: lambda_min={:.6e} < -floor={:.6e}; an indefinite coefficient point cannot define a Laplace mode",
                        minimum_whitened_eigenvalue, numerical_floor,
                    ));
                } else {
                    log::info!(
                        "[PIRLS/joint-Newton mode certificate] returned beta certified from fresh exact curvature: lambda_min={:.6e}, floor={:.6e}",
                        minimum_whitened_eigenvalue,
                        numerical_floor,
                    );
                }
            }
        }

        // Explicit terminal verdict for the joint-Newton inner solve.
        //
        // The per-cycle `[PIRLS/JN] cyc=N/MAX … resid=… (tol=…)` line prints
        // the KKT/step/objective gaps at every cycle but never states which
        // criterion *terminated* the loop, so the final visible line on a
        // budget-exhausted solve looks identical to an ordinary mid-run cycle
        // (gam#744). A reader scanning a sweep log cannot tell a fit that
        // reached a stationary point from one that simply ran out of cycles
        // with the residual still orders of magnitude above tolerance and only
        // the objective stalled. Emit one authoritative line, on every exit
        // path, naming the terminating condition: `converged` is the honest
        // status the result carries downstream, `budget_exhausted` distinguishes
        // "ran the full cap" from an early certificate/divergence exit, and the
        // residual/step/objective stall flags say *why*. A budget-exhausted,
        // non-converged exit is logged at WARN so it is impossible to miss even
        // when per-cycle INFO is filtered out; a clean convergence is INFO.
        {
            let budget_exhausted = cycles_done >= inner_max_cycles;
            // Hard convergence-truthfulness invariant (#1040): a converged exit
            // is, by construction, certified on a finite stationarity residual
            // ≤ tol (every `converged = true` path above is gated on a finite
            // residual / range-space check and records it into
            // `min_certified_residual`). If — through any path — `converged` is
            // set without a finite certified residual on record, the solve has
            // NOT actually certified convergence; reporting `converged=true …
            // best_residual_inf=inf` is the self-contradicting status #1040
            // flags. The honest status is then non-converged: downgrade it so
            // the outer REML/LAML evaluation rejects this ρ rather than
            // consuming a phantom optimum certified on no finite residual.
            if !gam_solve::loop_guard::inner_convergence_is_truthful(
                converged,
                min_certified_residual,
            ) {
                log::warn!(
                    "[PIRLS/joint-Newton terminal] cycle {cycles_done}/{inner_max_cycles}: a converged \
                     exit fired without any finite certified stationarity residual on record \
                     (min_certified_residual is non-finite) — this would report \
                     converged=true with best_residual_inf=inf, a convergence-truthfulness \
                     violation (#1040). Downgrading to non-converged so the outer optimizer \
                     rejects this evaluation."
                );
                converged = false;
            }
            let terminator = if converged {
                "KKT/certificate-converged"
            } else if budget_exhausted {
                "budget-exhausted (max cycles reached)"
            } else {
                "early-exit non-converged (divergence/stall guard)"
            };
            // `solve_wall` (whole inner-solve elapsed) + `cycles` make the
            // per-solve cost explicit on ONE line: gam#979's "outer
            // multiplication" candidate is read off by counting these terminal
            // lines across a repro and summing their wall-times, and the
            // overhead candidate by comparing `solve_wall / cycles` against the
            // [joint-newton-tr] phase splits. Together with the per-cycle
            // `per_block_resid` (which block stalls) and the existing TR line
            // (ρ gain-ratio + decision: model infidelity vs TR throttling), a
            // single RUST_LOG=info run separates all four #979 candidates.
            //
            // Report `min_certified_residual` (the smallest stationarity residual
            // the solve actually computed) rather than the stall-tracker
            // `best_residual_seen`: the latter is only written at the post-step
            // residual site, so a head-of-cycle / pre-line-search certificate exit
            // (cycle-0 KKT exit on already-stationary data) left it at the sentinel
            // `inf` and the line read `converged=true … best_residual_inf=inf`, a
            // self-contradicting status (#1040 inner-report truthfulness). A
            // converged exit always certified on a finite residual ≤ tol, so the
            // reported residual is finite whenever `converged` (every converged=true
            // path is gated on a `≤ tol` check of a residual recorded above).
            let reported_residual_below_tol = last_cycle_residual_below_tol
                || (converged && min_certified_residual <= last_residual_tol);
            let verdict = format!(
                "[PIRLS/joint-Newton terminal] converged={} terminator={} cycles={}/{} \
                 solve_wall={:.3}s best_residual_inf={:.3e} (tol={:.3e}) last_residual_below_tol={} \
                 last_obj_change_below_tol={} objective={:.6e}; this is the status the inner \
                 solve reports to the outer REML/LAML evaluation — a non-converged exit \
                 (residual ≫ tol with only the objective stalled) is rejected, not accepted",
                converged,
                terminator,
                cycles_done,
                inner_max_cycles,
                inner_started.elapsed().as_secs_f64(),
                min_certified_residual,
                last_residual_tol,
                reported_residual_below_tol,
                last_cycle_obj_change_below_tol,
                lastobjective,
            );
            if converged {
                log::info!("{verdict}");
            } else {
                log::warn!("{verdict}");
            }
        }

        // If joint Newton converged, skip the blockwise loop entirely.
        if converged {
            // The accepted-step cache is keyed by the exact coefficient bits.
            // Nothing between the accepted step and this terminal branch mutates
            // beta, so a hit is the authoritative Jeffreys derivative artifact at
            // the returned mode. A miss deliberately falls through to the normal
            // computation; approximate/stale reuse is never allowed here.
            let final_beta_key = flatten_state_betas(&states, specs);
            let final_jeffreys_cache = jeffreys_triple_cache.as_ref().filter(|(beta_key, _, _)| {
                beta_cache_keys_match_bitwise(beta_key, &final_beta_key)
            });
            let penalty_value = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let active_constraints = {
                let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
                assemble_active_constraint_block(
                    &block_constraints,
                    &cached_active_sets,
                    &ranges,
                    total_p,
                )
                .map(std::sync::Arc::new)
            };
            let (block_logdet_h, block_logdet_s) = blockwise_logdet_terms_with_workspace(
                family,
                specs,
                &mut states,
                block_log_lambdas,
                options,
                cached_joint_workspace.clone(),
                final_jeffreys_cache.map(|(_, _, hphi)| hphi),
                active_constraints.as_deref(),
            )?;
            // The IFT/outer KKT residual must be the AUGMENTED stationarity
            // `∇L − Sβ + ∇Φ` the inner Newton actually drove to zero — NOT the bare
            // `∇L − Sβ`. With the Firth term armed, `∇L − Sβ = −∇Φ` at the
            // converged β, so the bare residual's null-space component equals ∇Φ
            // (O(‖∇Φ‖), e.g. 2.49 for the coupled Dirichlet). The outer evaluator's
            // range-projected IFT validity gate (`projected_into_reduced_range`)
            // then sees that ‖∇Φ‖ of "unresolved mass outside the reduced range"
            // and rejects EVERY seed at outer startup validation ("no candidate
            // seeds passed", gam#729/#715). Folding ∇Φ into the gradient makes the
            // residual the genuinely-near-zero augmented stationarity the inner
            // certified, so the gate passes. No-op when the term is
            // condition-gated/unavailable (∇Φ=0).
            let augmented_joint_gradient: Option<Array1<f64>> = match cached_joint_gradient.as_ref()
            {
                Some(gradient) => match final_jeffreys_cache {
                    Some((_, grad_phi, _)) if grad_phi.len() == gradient.len() => {
                        Some(gradient + grad_phi)
                    }
                    _ => match joint_jeffreys_subspace.as_ref() {
                        Some(z_joint) => match custom_family_joint_jeffreys_term(
                            family, &states, specs, &ranges, z_joint,
                        )? {
                            Some((_phi, grad_phi, _hphi)) if grad_phi.len() == gradient.len() => {
                                Some(gradient + &grad_phi)
                            }
                            _ => None,
                        },
                        None => None,
                    },
                },
                None => None,
            };
            let ift_gradient = augmented_joint_gradient
                .as_ref()
                .or(cached_joint_gradient.as_ref());
            let joint_penalty_score = joint_penalty_stationarity_score(options, specs, &states);
            let kkt_residual = exact_newton_joint_kkt_residual_for_ift_from_cached_gradient(
                family,
                specs,
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                Some(cached_active_sets.as_slice()),
                ift_gradient,
                joint_penalty_score.as_ref(),
            )?;
            let kkt_residual =
                require_projected_kkt_residual(kkt_residual, "joint-Newton converged exit")?;
            // Thread the cert tolerance + free subspace rank through to
            // the unified evaluator's certificate so the outer
            // optimiser's InnerStatus carrier sees honest numbers
            // instead of NaN / None.
            let active_set_rows_total: usize = cached_active_sets
                .iter()
                .map(|maybe| maybe.as_ref().map(|v| v.len()).unwrap_or(0))
                .sum();
            let free_rank_at_cert = total_p.saturating_sub(active_set_rows_total);
            let kkt_residual = kkt_residual.with_metadata(last_residual_tol, free_rank_at_cert);
            // Build the joint active-constraint block for the unified
            // evaluator's constraint-aware kernel
            // `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`. Returns `None` when
            // the family has no declared inequality constraints, or when
            // no rows are currently active at the cert point; in either
            // case the consumer-side `with_active_constraints` helper
            // degrades back to the bare penalty-projected pseudo-inverse.
            return Ok(BlockwiseInnerResult {
                block_states: states,
                terminal_working_sets: cached_eval
                    .as_ref()
                    .map(|eval| eval.blockworking_sets.clone()),
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: current_log_likelihood,
                penalty_value,
                cycles: cycles_done,
                converged,
                block_logdet_h: Some(block_logdet_h),
                block_logdet_s: Some(block_logdet_s),
                s_lambdas,
                joint_workspace: cached_joint_workspace.clone(),
                kkt_residual: Some(kkt_residual),
                active_constraints,
            });
        }
        if cycles_done >= inner_max_cycles {
            if !converged {
                // Engine-level diagnostic. Emit measured quantities only:
                // objective movement, coefficient scale, per-block dimensions,
                // per-block β and gradient scales, the unprojected stationarity
                // norm at exit, the Hessian source shape, and the last accepted
                // Newton identity diagnostics. The outer error path has no
                // access to these internals, so this line is the complete
                // numerical record needed to decide the next fix.
                let block_grad_norms: Vec<f64> = match cached_joint_gradient.as_ref() {
                    Some(joint_grad) => {
                        let mut acc = 0usize;
                        states
                            .iter()
                            .map(|s| {
                                let n = s.beta.len();
                                let end = (acc + n).min(joint_grad.len());
                                let nrm = if acc < end {
                                    joint_grad
                                        .slice(ndarray::s![acc..end])
                                        .iter()
                                        .map(|x: &f64| x.abs())
                                        .fold(0.0_f64, f64::max)
                                } else {
                                    f64::NAN
                                };
                                acc += n;
                                nrm
                            })
                            .collect()
                    }
                    None => vec![f64::NAN; states.len()],
                };
                let block_widths: Vec<usize> = states.iter().map(|s| s.beta.len()).collect();
                let block_beta_inf: Vec<f64> = states
                    .iter()
                    .map(|s| s.beta.iter().map(|x: &f64| x.abs()).fold(0.0_f64, f64::max))
                    .collect();
                let descent_total = initial_joint_objective - lastobjective;
                let beta_inf_final = states
                    .iter()
                    .flat_map(|s| s.beta.iter().copied())
                    .map(f64::abs)
                    .fold(0.0_f64, f64::max);
                let block_diag_default =
                    !family.exact_newton_joint_hessian_beta_dependent() && specs.len() >= 2;
                let exit_unprojected_kkt_inf = cached_joint_gradient
                    .as_ref()
                    .and_then(|joint_grad| {
                        exact_newton_joint_stationarity_vector_from_gradient(
                            joint_grad,
                            &states,
                            specs,
                            &s_lambdas,
                            ridge,
                            options.ridge_policy,
                        )
                        .ok()
                    })
                    .map(|residual| {
                        residual
                            .iter()
                            .map(|x: &f64| x.abs())
                            .fold(0.0_f64, f64::max)
                    })
                    .unwrap_or(f64::NAN);
                let last_math_summary = last_joint_math
                    .as_ref()
                    .map(|math| {
                        format!(
                            "last_newton_math={{old_kkt={:.3e}, linearized_next={:.3e}, actual={:+.3e}, pred={:+.3e}, rho={:+.3e}, scalar_relerr={:.3e}, step_inf={:.3e}, proposal_inf={:.3e}}}",
                            math.old_kkt_inf,
                            math.linearized_next_kkt_inf,
                            math.actual_reduction,
                            math.predicted_reduction,
                            math.trust_ratio,
                            math.scalar_model_relative_error(),
                            math.step_inf,
                            math.proposal_inf,
                        )
                    })
                    .unwrap_or_else(|| "last_newton_math=<none>".to_string());
                log::warn!(
                    "[PIRLS/joint-Newton] cycle={} budget-exhausted without KKT: objective_start={:.6e} objective_end={:.6e} objective_drop={:+.3e} beta_inf={:.3e} exit_unprojected_kkt_inf={:.3e} total_p={} total_n={} block_widths={:?} block_beta_inf={:?} block_grad_inf={:?} block_diag_hessian_default={} {}; rejecting this outer REML/LAML evaluation",
                    cycles_done,
                    initial_joint_objective,
                    lastobjective,
                    descent_total,
                    beta_inf_final,
                    exit_unprojected_kkt_inf,
                    total_p,
                    total_joint_n,
                    block_widths,
                    block_beta_inf,
                    block_grad_norms,
                    block_diag_default,
                    last_math_summary,
                );
                {
                    // Budget exhaustion is a failed *inner mode at this rho*, not
                    // malformed user input.  Propagate it as a finite
                    // `converged=false` inner result so the outer objective can
                    // reject/back off this smoothing point (the same contract used
                    // by non-exact families) instead of bubbling an
                    // `InvalidInput` through the custom-family string boundary.
                    // This matters on the survival/location-scale flat baseline
                    // valley: some startup rho candidates are numerically
                    // non-certifying, but neighbouring rho values are perfectly
                    // fit-able, so aborting the whole fit prevents the optimizer
                    // from ever leaving the valley.
                    let block_diag = if let Some(report) = last_kkt_refusal_report.as_ref() {
                        report.format_bubbled_error()
                    } else {
                        let block_constraints =
                            collect_block_linear_constraints(family, &states, specs)?;
                        let report = compute_kkt_refusal_report(
                            cycles_done,
                            &states,
                            specs,
                            &s_lambdas,
                            &ranges,
                            cached_joint_gradient.as_ref(),
                            &cached_active_sets,
                            &block_constraints,
                            None,
                            total_p,
                            ridge,
                            options.ridge_policy,
                            f64::NAN,
                            f64::NAN,
                            f64::NAN,
                            last_residual_tol,
                            f64::NAN,
                            f64::NAN,
                            f64::NAN,
                            exit_unprojected_kkt_inf,
                            last_joint_math.as_ref(),
                        );
                        report.format_bubbled_error()
                    };
                    log::warn!(
                        "coupled exact-joint inner solve exhausted the joint Newton budget without KKT convergence after {cycles_done} cycle(s) — {block_diag}; returning a non-converged inner mode for outer-rho rejection"
                    );
                }
            }
            let penalty_value = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let active_constraints = {
                let local_ranges = block_param_ranges(specs);
                let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
                let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
                assemble_active_constraint_block(
                    &block_constraints,
                    &cached_active_sets,
                    &local_ranges,
                    local_total_p,
                )
                .map(std::sync::Arc::new)
            };
            let (block_logdet_h, block_logdet_s) = if converged {
                let (h, s) = blockwise_logdet_terms_with_workspace(
                    family,
                    specs,
                    &mut states,
                    block_log_lambdas,
                    options,
                    cached_joint_workspace.clone(),
                    None,
                    active_constraints.as_deref(),
                )?;
                (Some(h), Some(s))
            } else {
                (None, None)
            };
            return Ok(BlockwiseInnerResult {
                block_states: states,
                terminal_working_sets: cached_eval
                    .as_ref()
                    .map(|eval| eval.blockworking_sets.clone()),
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: current_log_likelihood,
                penalty_value,
                cycles: cycles_done,
                converged,
                block_logdet_h,
                block_logdet_s,
                s_lambdas,
                joint_workspace: cached_joint_workspace.clone(),
                kkt_residual: None,
                active_constraints,
            });
        }
        {
            // An early exit from an exact joint path is a non-certifying inner
            // mode at the current rho, not invalid input. The selected joint
            // solver is authoritative even for a single tensor block: falling
            // through to coordinate iteration would silently switch algorithms
            // after a failed certificate, discard its active-set provenance,
            // and grind a second cycle budget before reaching the same outer-rho
            // rejection. Families whose objective is deliberately separable are
            // routed to blockwise before this joint path starts. Return the
            // current finite iterate with `converged=false` so the outer
            // optimizer can reject this rho and continue.
            let block_diag = last_kkt_refusal_report
                .as_ref()
                .map(KktRefusalReport::format_bubbled_error)
                .unwrap_or_else(|| {
                    "structured KKT refusal report unavailable: no joint Newton math snapshot"
                        .to_string()
                });
            log::warn!(
                "coupled exact-joint inner solve exited the joint Newton path before convergence — {block_diag}; returning a non-converged inner mode for outer-rho rejection"
            );
            let penalty_value = total_quadratic_penalty(
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let active_constraints = {
                let local_ranges = block_param_ranges(specs);
                let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
                let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
                assemble_active_constraint_block(
                    &block_constraints,
                    &cached_active_sets,
                    &local_ranges,
                    local_total_p,
                )
                .map(std::sync::Arc::new)
            };
            return Ok(BlockwiseInnerResult {
                block_states: states,
                terminal_working_sets: cached_eval
                    .as_ref()
                    .map(|eval| eval.blockworking_sets.clone()),
                active_sets: normalize_active_sets(cached_active_sets),
                log_likelihood: current_log_likelihood,
                penalty_value,
                cycles: cycles_done,
                converged: false,
                block_logdet_h: None,
                block_logdet_s: None,
                s_lambdas,
                joint_workspace: cached_joint_workspace.clone(),
                kkt_residual: None,
                active_constraints,
            });
        }
    }

    let mut cached_eval = match cached_eval {
        Some(eval) => eval,
        None => family.evaluate(&states)?,
    };
    lastobjective = -cached_eval.log_likelihood + current_penalty;

    // Divergence-detection state for the blockwise loop.
    //
    // Some family parameterizations (e.g. BernoulliMarginalSlopeFamily with
    // linkwiggle + scorewarp) carry a near-null direction in the joint
    // Hessian when the link-deviation basis's empirical anchor — fixed at
    // the rigid-pilot η₀ when the basis is constructed — drifts during
    // PIRLS as the location/spatial blocks update η₀. The Newton step
    // becomes dominated by that null direction and is clamped at
    // MAX_NEWTON_STEP every cycle while β grows linearly along it; the
    // log-likelihood stays frozen, only the penalty changes (slowly).
    // Without an early-exit the loop runs to inner_max_cycles producing
    // the same -loglik over and over, which at large scale (each cycle
    // ~0.5s) burns ~50s per ρ-cost call and stacks up to a 2400s timeout.
    //
    // Detect the pattern and bail with `converged = false` so the cost
    // call returns Err / +∞, BFGS κ-optim backs off the divergent ρ
    // region, and the outer loop progresses instead of grinding.

    // Per-block trust-region radius in the block's penalized-Hessian metric.
    // Updated each cycle by `update_joint_trust_region_radius` (the same
    // function the joint-Newton path uses) on a real model-vs-truth rho
    // computed from each block's penalized quadratic. Using the curvature
    // metric here avoids the same starvation mechanism fixed in the joint
    // path: one near-null coordinate in a block must not raw-rescale every
    // other coordinate in that block. The η-overflow safety half of the
    // previous static `MAX_NEWTON_STEP = 20.0` is owned by the family's
    // `max_feasible_step_size` barrier check, called by the line search below;
    // this variable handles only the algorithmic trust-region half. The
    // initial seed value is the family-declared safe step for a fresh fit; the
    // function then adapts it freely (clamped to [1e-12, 1e6] by the function
    // itself, same as the joint path).
    const BLOCK_NEWTON_STEP_INITIAL: f64 = 20.0;
    let mut block_max_step: Vec<f64> = vec![BLOCK_NEWTON_STEP_INITIAL; specs.len()];

    let mut prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
    // Frozen-loglik streak rides the shared window discipline
    // (loop_guard::FlatStreak, #968); the frozen-loglik predicate and the
    // clamped-step side condition below stay local — they are policy about
    // what counts as flat, which this loop rightly owns.
    let mut frozen_loglik_streak =
        gam_solve::loop_guard::FlatStreak::new(DIVERGENCE_FROZEN_LOGLIK_CYCLES);
    // Coordinate descent visits each block in turn, so `max_proposed_step`
    // (the per-cycle max across blocks) only fires the cap on cycles where
    // the divergent block is the active one. On a near-null direction this
    // produces an alternation pattern (e.g. cap, cap, small, cap, small,
    // cap, …) and a strict "consecutive cycles where step is clamped"
    // requirement resets the counter every time another block's smaller
    // step dominates the per-cycle maximum. The frozen-loglik signal,
    // however, is a property of the joint state — it stays true across
    // every cycle of the alternation. Track frozen-loglik consecutively
    // and require that `step_clamped` was observed AT LEAST ONCE inside
    // the frozen run (rather than EVERY cycle).
    let mut clamped_step_in_frozen_run: bool = false;
    const DIVERGENCE_FROZEN_LOGLIK_CYCLES: usize = 8;

    let is_dynamic = family.block_geometry_is_dynamic();
    for cycle in 0..inner_max_cycles {
        // Fires at the top of each blockwise coordinate cycle so we can count
        // iterations from CI logs when a benchmark hangs inside the first
        // outer-eval. Emitted at info-level: same rationale as the joint-Newton
        // sibling above — silent-grind diagnosis without debug logs.
        log::info!(
            "[PIRLS/blockwise coord] cycle {:>3}/{} | -loglik {:.6e} | penalty {:.6e} | objective {:.6e}",
            cycle,
            inner_max_cycles,
            -cached_eval.log_likelihood,
            current_penalty,
            lastobjective,
        );
        let mut max_proposed_beta_step = 0.0_f64;
        let mut max_accepted_beta_step = 0.0_f64;
        let mut trust_boundary_hit_in_cycle = false;

        let mut objective_cycle_prev = lastobjective;
        // Reuse cached evaluation from end of previous cycle (or initial eval).
        // For dynamic families, the end-of-cycle evaluation is also reused here
        // instead of re-evaluating redundantly — the state hasn't changed since
        // the last cycle's final evaluate.
        let mut cycle_eval = std::mem::replace(
            &mut cached_eval,
            FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: Vec::new(),
            },
        );
        if cycle_eval.blockworking_sets.len() != specs.len() {
            return Err(format!(
                "family returned {} block working sets, expected {}",
                cycle_eval.blockworking_sets.len(),
                specs.len()
            ));
        }
        // Track whether any block was modified this cycle (for dynamic families,
        // we only need to re-evaluate before block b if a previous block changed).
        let mut any_block_modified = false;
        for b in 0..specs.len() {
            if is_dynamic && any_block_modified {
                // Only re-evaluate if a previous block in this cycle actually
                // modified coefficients. Skips the redundant evaluate for the
                // first block (b=0) since cached_eval is still valid.
                refresh_all_block_etas(family, specs, &mut states)?;
                cycle_eval = family.evaluate(&states)?;
                if cycle_eval.blockworking_sets.len() != specs.len() {
                    return Err(format!(
                        "family returned {} block working sets, expected {}",
                        cycle_eval.blockworking_sets.len(),
                        specs.len()
                    ));
                }
            }

            let spec = &specs[b];
            let work = &cycle_eval.blockworking_sets[b];
            let linear_constraints = family.block_linear_constraints(&states, b, spec)?;
            let s_lambda = &s_lambdas[b];
            let updater = work.updater();
            let update = updater.compute_update_step(&BlockUpdateContext {
                family,
                states: &states,
                spec,
                block_idx: b,
                s_lambda,
                options,
                linear_constraints: linear_constraints.as_ref(),
                cached_active_set: cached_active_sets[b].as_deref(),
            })?;
            if let Some(active_set) = update.active_set {
                cached_active_sets[b] = Some(active_set);
            }
            let beta_new_raw = update.beta_new_raw;
            let beta_new = family.post_update_block_beta(&states, b, spec, beta_new_raw.clone())?;
            reject_constrained_post_update_repair(
                b,
                spec,
                &beta_new_raw,
                &beta_new,
                linear_constraints.as_ref(),
            )?;
            let beta_old = states[b].beta.clone();
            let raw_delta = &beta_new - &beta_old;
            // Per-block trust-region radius in the block's local
            // penalized-Hessian metric. The cap is the current value of
            // `block_max_step[b]`, updated below via
            // `update_joint_trust_region_radius` once we know rho.
            let block_cap = block_max_step[b];
            let (delta, step_metric_norm) = truncate_block_step_to_metric_radius(
                spec,
                work,
                s_lambda,
                raw_delta,
                block_cap,
                ridge,
                options.ridge_policy,
            )?;
            let step_hit_trust_boundary =
                joint_block_step_hit_trust_boundary(step_metric_norm, block_cap);
            trust_boundary_hit_in_cycle |= step_hit_trust_boundary;
            // Capture the objective at the start of this block update so
            // we can compute the true `actual_reduction` once the line
            // search has finished. `objective_cycle_prev` is the running
            // total: it advances inside the line search whenever a trial
            // is accepted, so we must snapshot it here.
            let obj_before_block = objective_cycle_prev;
            let old_block_penalty =
                block_quadratic_penalty(&beta_old, s_lambda, ridge, options.ridge_policy);
            let step_beta_inf = delta.iter().copied().map(f64::abs).fold(0.0, f64::max);
            max_proposed_beta_step = max_proposed_beta_step.max(step_beta_inf);
            if step_beta_inf <= inner_tol {
                continue;
            }

            // Damped update: require non-increasing penalized objective under dynamic geometry.
            // Precompute X * delta once so line-search eta updates are O(n) not O(np).
            // Reuse pre-allocated eta backup to avoid O(n) allocation per block per cycle.
            let eta_checkpoint = BlockEtaCheckpoint::capture_reuse(&states[b], &mut eta_backups[b]);
            let x_delta = if !is_dynamic {
                Some(spec.solver_design().matrixvectormultiply(&delta))
            } else {
                None
            };
            let mut accepted = false;
            // Barrier-aware step ceiling: families with natural log-barrier
            // terms (e.g. log(h') in transformation-normal) report the maximum
            // feasible step fraction so the line search never evaluates the
            // likelihood outside its domain.
            let barrier_ceiling = family
                .max_feasible_step_size(&states, b, &delta)?
                .unwrap_or(1.0);
            // Reuse trial_beta_buf to avoid allocation per backtracking trial.
            let mut trial_beta_buf = beta_old.clone();
            let mut accepted_bt: usize = usize::MAX;
            for bt in 0..8 {
                let alpha = (0.5f64.powi(bt)).min(barrier_ceiling);
                trial_beta_buf.assign(&beta_old);
                trial_beta_buf.scaled_add(alpha, &delta);
                let trial_beta =
                    family.post_update_block_beta(&states, b, spec, trial_beta_buf.clone())?;
                reject_constrained_post_update_repair(
                    b,
                    spec,
                    &trial_beta_buf,
                    &trial_beta,
                    linear_constraints.as_ref(),
                )?;
                states[b].beta = trial_beta;
                // Use precomputed X*delta when geometry is static and beta wasn't modified.
                if let Some(ref xd) = x_delta {
                    if states[b].beta == trial_beta_buf {
                        eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                    } else {
                        refresh_single_block_eta(family, specs, &mut states, b)?;
                    }
                } else {
                    refresh_single_block_eta(family, specs, &mut states, b)?;
                }
                let trial_block_penalty =
                    block_quadratic_penalty(&states[b].beta, s_lambda, ridge, options.ridge_policy);
                let trial_penalty = current_penalty - old_block_penalty + trial_block_penalty;
                let line_search_options = coefficient_line_search_options(
                    options,
                    objective_cycle_prev - trial_penalty + 1e-10,
                );
                let trial_ll =
                    match family.log_likelihood_only_with_options(&states, &line_search_options) {
                        Ok(value) => value,
                        Err(_) => {
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                            continue;
                        }
                    };
                let trialobjective = -trial_ll + trial_penalty;
                if trialobjective.is_finite() && trialobjective <= objective_cycle_prev + 1e-10 {
                    objective_cycle_prev = trialobjective;
                    current_penalty = trial_penalty;
                    accepted = true;
                    accepted_bt = bt as usize;
                    break;
                }
            }
            // Trust-region update for this block, using the same
            // `update_joint_trust_region_radius` strategy the
            // joint-Newton path uses. Predicted reduction is computed
            // from the per-block penalized quadratic model:
            //
            //   Q(β + αδ) ≈ Q(β) − α·rhs·δ + 0.5·α²·δ·H_pen·δ
            //   predicted_reduction(α) = α·(rhs·δ) − 0.5·α²·(δ·H_pen·δ)
            //
            // where `rhs = score − S·β (− ridge·β)` is the penalized
            // gradient (in maximize-direction) and `H_pen = H + S
            // (+ ridge·I)` is the penalized observed information.
            // Actual reduction is the true penalized objective change
            // measured by the line search; rho = actual / predicted is
            // the standard model-vs-truth ratio that drives the same
            // 0.25 / 0.75 grow-shrink rules `update_joint_trust_region_radius`
            // already implements for the joint path.
            let alpha_accepted = if accepted {
                0.5_f64.powi(accepted_bt as i32)
            } else {
                0.0
            };
            let (rhs_block, hpen_delta_full): (Array1<f64>, Array1<f64>) = match work {
                BlockWorkingSet::ExactNewton { gradient, .. } => {
                    let mut rhs = gradient - &s_lambda.dot(&beta_old);
                    if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                        rhs.scaled_add(-ridge, &beta_old);
                    }
                    let hpen = block_penalized_hessian_vector(
                        spec,
                        work,
                        s_lambda,
                        &delta,
                        ridge,
                        options.ridge_policy,
                    );
                    (rhs, hpen)
                }
                BlockWorkingSet::Diagonal {
                    working_response,
                    working_weights,
                } => {
                    // IRLS local-quadratic gradient and Hessian:
                    //   rhs = X^T W (z − Xβ) − Sβ
                    //   H_pen δ = X^T W X δ + Sδ
                    let solver_design = spec.solver_design();
                    let xb = solver_design.matrixvectormultiply(&beta_old);
                    let resid = working_response - &xb;
                    let w_resid = &resid * working_weights;
                    let mut rhs = solver_design.transpose_vector_multiply(&w_resid);
                    rhs -= &s_lambda.dot(&beta_old);
                    if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                        rhs.scaled_add(-ridge, &beta_old);
                    }
                    let hpen = block_penalized_hessian_vector(
                        spec,
                        work,
                        s_lambda,
                        &delta,
                        ridge,
                        options.ridge_policy,
                    );
                    (rhs, hpen)
                }
            };
            let rhs_dot_delta = rhs_block.dot(&delta);
            let delta_dot_hpen = delta.dot(&hpen_delta_full);
            let predicted_reduction = alpha_accepted * rhs_dot_delta
                - 0.5 * alpha_accepted * alpha_accepted * delta_dot_hpen;
            let actual_reduction = obj_before_block - objective_cycle_prev;
            let trust_update = update_joint_trust_region_radius(
                block_max_step[b],
                alpha_accepted * step_metric_norm,
                actual_reduction,
                predicted_reduction,
                obj_before_block,
            );
            block_max_step[b] = trust_update.radius;
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
                if let BlockWorkingSet::ExactNewton { gradient, .. } = work {
                    let mut raw_descent = gradient - &s_lambda.dot(&beta_old);
                    if options.ridge_policy.accounts_for_objective() && ridge > 0.0 {
                        raw_descent -= &beta_old.mapv(|v| ridge * v);
                    }
                    let (descent_dir, descent_metric_norm) = truncate_block_step_to_metric_radius(
                        spec,
                        work,
                        s_lambda,
                        raw_descent,
                        block_cap,
                        ridge,
                        options.ridge_policy,
                    )?;
                    trust_boundary_hit_in_cycle |=
                        joint_block_step_hit_trust_boundary(descent_metric_norm, block_cap);
                    let dir_norm = descent_dir.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    if dir_norm > inner_tol {
                        // Precompute X * descent_dir once for incremental eta updates.
                        let x_descent = if !is_dynamic {
                            Some(spec.solver_design().matrixvectormultiply(&descent_dir))
                        } else {
                            None
                        };
                        let descent_barrier_ceiling = family
                            .max_feasible_step_size(&states, b, &descent_dir)?
                            .unwrap_or(1.0);
                        for bt in 0..12 {
                            let alpha = (0.5f64.powi(bt)).min(descent_barrier_ceiling);
                            trial_beta_buf.assign(&beta_old);
                            trial_beta_buf.scaled_add(alpha, &descent_dir);
                            let trial_beta = family.post_update_block_beta(
                                &states,
                                b,
                                spec,
                                trial_beta_buf.clone(),
                            )?;
                            reject_constrained_post_update_repair(
                                b,
                                spec,
                                &trial_beta_buf,
                                &trial_beta,
                                linear_constraints.as_ref(),
                            )?;
                            states[b].beta = trial_beta;
                            if let Some(ref xd) = x_descent {
                                if states[b].beta == trial_beta_buf {
                                    eta_checkpoint.restore_eta_with_step(&mut states[b], alpha, xd);
                                } else {
                                    refresh_single_block_eta(family, specs, &mut states, b)?;
                                }
                            } else {
                                refresh_single_block_eta(family, specs, &mut states, b)?;
                            }
                            let trial_block_penalty = block_quadratic_penalty(
                                &states[b].beta,
                                s_lambda,
                                ridge,
                                options.ridge_policy,
                            );
                            let trial_penalty =
                                current_penalty - old_block_penalty + trial_block_penalty;
                            let line_search_options = coefficient_line_search_options(
                                options,
                                objective_cycle_prev - trial_penalty + 1e-10,
                            );
                            let trial_ll = match family
                                .log_likelihood_only_with_options(&states, &line_search_options)
                            {
                                Ok(value) => value,
                                Err(_) => {
                                    states[b].beta.assign(&beta_old);
                                    eta_checkpoint.restore_eta(&mut states[b]);
                                    continue;
                                }
                            };
                            let trialobjective = -trial_ll + trial_penalty;
                            if trialobjective.is_finite()
                                && trialobjective <= objective_cycle_prev + 1e-10
                            {
                                objective_cycle_prev = trialobjective;
                                current_penalty = trial_penalty;
                                accepted = true;
                                break;
                            }
                            states[b].beta.assign(&beta_old);
                            eta_checkpoint.restore_eta(&mut states[b]);
                        }
                    }
                }
            }
            if !accepted {
                states[b].beta.assign(&beta_old);
                eta_checkpoint.restore_eta(&mut states[b]);
            } else {
                let accepted_step = states[b]
                    .beta
                    .iter()
                    .zip(beta_old.iter())
                    .map(|(new, old)| (new - old).abs())
                    .fold(0.0_f64, f64::max);
                max_accepted_beta_step = max_accepted_beta_step.max(accepted_step);
                any_block_modified = true;
            }
            // Recycle the checkpoint's buffer back into the pre-allocated pool.
            eta_backups[b] = eta_checkpoint.into_buffer();
        }

        // For non-dynamic families, incremental eta updates within the block loop
        // maintain correct etas. Only refresh from scratch for dynamic-geometry families
        // where block interactions may require recomputation.
        if is_dynamic {
            refresh_all_block_etas(family, specs, &mut states)?;
        }
        cached_eval = family.evaluate(&states)?;
        current_penalty = total_quadratic_penalty(
            &states,
            &s_lambdas,
            ridge,
            options.ridge_policy,
            joint_bundle,
            Some(specs),
        );
        let objective = -cached_eval.log_likelihood + current_penalty;
        let objective_change = (objective - lastobjective).abs();
        lastobjective = objective;
        cycles_done = cycle + 1;

        // Divergence guard (mirrors the joint-Newton sibling, gam#554): a
        // non-finite objective / log-likelihood means a near-unidentified
        // penalized block has propagated NaN mass through the coordinate
        // descent. Every convergence and divergence-frozen exit below is a
        // finite `<=` comparison that NaN silently defeats, so without this
        // the loop grinds the full `inner_max_cycles` on every outer ρ-eval
        // and startup seed. Break unconverged so the outer optimizer rejects
        // this point immediately instead of burning the budget.
        if !objective.is_finite() || !cached_eval.log_likelihood.is_finite() {
            log::warn!(
                "[PIRLS/blockwise convergence] cycle {:>3} | divergence guard: non-finite inner state (objective={:.3e}, -loglik={:.3e}); returning unconverged so the outer optimizer rejects this ρ evaluation instead of running to inner_max_cycles.",
                cycle,
                objective,
                -cached_eval.log_likelihood,
            );
            converged = false;
            break;
        }

        // Scale-aware tolerances — see the matching joint-Newton path
        // above for the rationale. At large scale absolute step/residual
        // tolerances against `inner_tol = 1e-6` keep this loop spinning
        // long after the objective has gone flat.
        let beta_inf = states
            .iter()
            .flat_map(|s| s.beta.iter().copied())
            .map(f64::abs)
            .fold(0.0_f64, f64::max);
        let step_tol = inner_tol * (1.0 + beta_inf);
        let objective_tol = inner_tol * (1.0 + objective.abs());
        let residual_tol = objective_tol;
        // For single-block models the blockwise iteration IS the joint
        // iteration, so block-conditional convergence implies joint
        // convergence.  The exact_newton_joint_stationarity check can
        // stall at ~10x the tolerance due to numerical differences
        // between the block-conditional and joint gradient formulations,
        // causing 100s of wasted cycles on an already-converged solution.
        let exact_joint_stationarity_ok = if has_joint_exacthessian && specs.len() >= 2 {
            exact_newton_joint_stationarity_inf_norm(
                family,
                specs,
                &cached_eval,
                &states,
                &s_lambdas,
                ridge,
                options.ridge_policy,
                None,
            )?
            .map(|residual| residual <= residual_tol)
            .unwrap_or(true)
        } else {
            true
        };
        log::info!(
            "[PIRLS/blockwise convergence] cycle {:>3} | max_proposed_step={:.3e} (tol={:.3e}) | max_accepted_step={:.3e} | obj_change={:.3e} (tol={:.3e}) | beta_inf={:.3e} | joint_stationarity_ok={}",
            cycle,
            max_proposed_beta_step,
            step_tol,
            max_accepted_beta_step,
            objective_change,
            objective_tol,
            beta_inf,
            exact_joint_stationarity_ok,
        );

        // Divergence early-exit. See the rationale block at the top of
        // this loop. We treat "log-likelihood unchanged + Newton step
        // pinned at the trust-region cap" as a near-null direction
        // signature and break out unconverged once it persists for
        // DIVERGENCE_FROZEN_LOGLIK_CYCLES consecutive iterations. Tracking
        // log-likelihood (not objective) is essential: when the null mode
        // dominates, only the penalty drifts cycle-to-cycle, so
        // `objective_change` stays above tol while -loglik is genuinely
        // frozen.
        let loglik_change_for_divergence_check =
            (cached_eval.log_likelihood - prev_log_likelihood_for_divergence_check).abs();
        let loglik_frozen_tol_for_divergence_check =
            inner_tol * (1.0 + cached_eval.log_likelihood.abs());
        let step_clamped_for_divergence_check = trust_boundary_hit_in_cycle;
        let loglik_frozen =
            loglik_change_for_divergence_check <= loglik_frozen_tol_for_divergence_check;
        let frozen_verdict = frozen_loglik_streak.note(loglik_frozen);
        if loglik_frozen {
            if step_clamped_for_divergence_check {
                clamped_step_in_frozen_run = true;
            }
        } else {
            clamped_step_in_frozen_run = false;
        }
        prev_log_likelihood_for_divergence_check = cached_eval.log_likelihood;
        if frozen_verdict == gam_solve::loop_guard::LoopVerdict::Plateaued
            && clamped_step_in_frozen_run
        {
            log::warn!(
                "[PIRLS/blockwise convergence] divergence early-exit at cycle {} | -loglik={:.6e} frozen for {} consecutive cycles | max_proposed_step={:.3e} (trust-boundary hit observed in frozen run) | step_tol={:.3e}; near-null Hessian direction detected — returning unconverged so the outer optimizer backs off this region instead of running to inner_max_cycles.",
                cycle,
                -cached_eval.log_likelihood,
                frozen_loglik_streak.streak(),
                max_proposed_beta_step,
                step_tol,
            );
            converged = false;
            break;
        }

        // NOTE: there is deliberately NO wall-clock-driven "adaptive
        // early-exit" here — the same discipline the joint-Newton sibling loop
        // documents above. A verdict that fires when a cycle's wall-clock falls
        // below a fraction of a running EMA is non-deterministic: under CPU
        // contention (a parallel sweep) the same fit accepts at a different
        // iterate than it does run alone, and it accepts iterates up to 10×
        // outside the real KKT/objective tolerance, biasing the REML/LAML
        // criterion the inner residual feeds. Convergence is certified ONLY by
        // the exact stationarity gate below.
        if max_accepted_beta_step <= step_tol && objective_change <= objective_tol {
            if exact_joint_stationarity_ok || max_proposed_beta_step <= step_tol {
                converged = true;
            }
            break;
        }
    }

    // ── Polishing joint Newton step ──
    //
    // For block-coupled multi-block families (e.g. GAMLSS wiggle), Gauss-Seidel
    // blockwise iteration can reach step_inf < inner_tol while the joint KKT
    // residual (||Sβ − grad_ℓ||_∞) remains at ~10× inner_tol. This is because
    // each block is solved conditionally on other blocks' current values —
    // block-conditional stationarity does not imply joint stationarity when
    // the likelihood couples blocks off-diagonally.
    //
    // Once blockwise has placed β near the true joint optimum, a single (or
    // a few) damped joint Newton steps can tighten the joint residual to the
    // floor set by β magnitudes. This polishing phase is essential for the
    // outer REML gradient formula (which assumes exact β̂ stationarity); a
    // non-converged β̂ produces large envelope-theorem violations in the
    // analytic outer gradient.
    if use_joint_newton && !converged {
        polish_joint_newton_step(
            family,
            specs,
            options,
            &s_lambdas,
            ridge,
            joint_bundle,
            inner_tol,
            &cached_active_sets,
            &mut states,
            &mut cached_eval,
            &mut current_penalty,
            &mut converged,
        )?;
    }

    assemble_inner_blockwise_result(
        family,
        specs,
        states,
        block_log_lambdas,
        options,
        s_lambdas,
        ridge,
        joint_bundle,
        cached_active_sets,
        &cached_eval,
        converged,
        cycles_done,
        last_residual_tol,
        has_joint_exacthessian,
    )
}

/// Polishing joint-Newton step for the blockwise fall-through path of
/// [`inner_blockwise_fit`].
///
/// For block-coupled multi-block families (e.g. GAMLSS wiggle), Gauss-Seidel
/// blockwise iteration can reach `step_inf < inner_tol` while the joint KKT
/// residual (`||Sβ − grad_ℓ||_∞`) remains at ~10× `inner_tol`. Once blockwise
/// has placed β near the joint optimum, a few damped joint-Newton steps tighten
/// the joint residual to the floor set by β magnitudes; this is essential for the
/// outer REML gradient formula (which assumes exact β̂ stationarity).
///
/// Behavior is identical to the inline loop it replaced: the `?`-propagation, the
/// per-iteration `break` exits (gradient/Hessian unavailable, non-finite delta,
/// solver failure, residual-tolerance reached, line-search failure) and the
/// inner backtracking-search `continue` are preserved verbatim. Mutates `states`,
/// `cached_eval`, `current_penalty`, and `converged` in place exactly as before.
pub(crate) fn polish_joint_newton_step<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    s_lambdas: &[Array2<f64>],
    ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    inner_tol: f64,
    cached_active_sets: &[Option<Vec<usize>>],
    states: &mut Vec<ParameterBlockState>,
    cached_eval: &mut FamilyEvaluation,
    current_penalty: &mut f64,
    converged: &mut bool,
) -> Result<(), String> {
    let ranges_joint: Vec<(usize, usize)> = {
        let mut offset = 0;
        specs
            .iter()
            .map(|s| {
                let start = offset;
                offset += s.design.ncols();
                (start, offset)
            })
            .collect()
    };
    let total_p_joint: usize = ranges_joint.last().map_or(0, |r| r.1);
    let joint_mode_diagonal_ridge = if ridge > 0.0 && options.ridge_policy.accounts_for_objective()
    {
        ridge
    } else {
        0.0
    };
    let trace_diagonal_ridge = joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE;

    // Allow up to a few polishing steps. The blockwise endpoint is close
    // to optimum, so step sizes should be small and line search should
    // accept full steps quickly.
    const POLISH_MAX_ITER: usize = 16;
    for _polish_iter in 0..POLISH_MAX_ITER {
        // Re-evaluate at current β to get the joint gradient and Hessian.
        refresh_all_block_etas(family, specs, states)?;
        let eval_for_polish = family.evaluate(states)?;
        let grad_full =
            match exact_newton_joint_gradient_from_eval(&eval_for_polish, specs, states)? {
                Some(g) => g,
                None => break,
            };
        // Spec-aware joint Hessian: canonical coupled-curvature source
        // (see the joint-Newton availability gate). Families overriding
        // only `_with_specs` return `None` from the spec-less default.
        let h_joint_opt = family.exact_newton_joint_hessian_with_specs(states, specs)?;
        let Some(h_joint) = h_joint_opt else { break };
        let mut h_dense = match symmetrized_square_matrix(
            h_joint,
            total_p_joint,
            "joint polish Hessian shape mismatch",
        ) {
            Ok(matrix) => matrix,
            Err(_) => break,
        };
        let h_unpenalized_dense = h_dense.clone();
        add_joint_penalty_to_matrix(
            &mut h_dense,
            &ranges_joint,
            s_lambdas,
            trace_diagonal_ridge,
            joint_bundle,
        );
        let joint_polish_diagonal_ridge = stabilized_joint_solver_diagonal_ridge(
            family,
            &JointHessianSource::Dense(h_unpenalized_dense),
            &ranges_joint,
            s_lambdas,
            trace_diagonal_ridge,
            options.ridge_floor,
            joint_bundle,
        );
        if joint_polish_diagonal_ridge != trace_diagonal_ridge {
            for d in 0..h_dense.nrows() {
                h_dense[[d, d]] += joint_polish_diagonal_ridge - trace_diagonal_ridge;
            }
        }

        let mut beta_joint = Array1::<f64>::zeros(total_p_joint);
        for b in 0..specs.len() {
            let (start, end) = ranges_joint[b];
            beta_joint
                .slice_mut(ndarray::s![start..end])
                .assign(&states[b].beta);
        }
        let penalty_beta = apply_joint_block_penalty(
            &ranges_joint,
            s_lambdas,
            &beta_joint,
            joint_mode_diagonal_ridge,
            joint_bundle,
        );
        let rhs = &grad_full - &penalty_beta;

        // Respect constraints that block line search on the boundary.
        // Gauss-Seidel blockwise leaves the joint KKT residual at a floor
        // around |λ_k S_k β̂| for boundary-active components. The residual
        // magnitude on FREE components is a better measure of whether we
        // should keep polishing: if β_i is clipped at the boundary and
        // KKT multiplier μ_i > 0, then rhs[i] is the multiplier, not a
        // free-space gradient violation.
        let block_constraints_now = collect_block_linear_constraints(family, states, specs)?;
        let joint_constraints_now = assemble_joint_linear_constraints(
            &block_constraints_now,
            &ranges_joint,
            total_p_joint,
        )?;
        let mut active_mask: Vec<bool> = vec![false; total_p_joint];
        if let Some(ref constraints) = joint_constraints_now
            && let Ok(Some(bounds)) = extract_simple_lower_bounds(constraints, total_p_joint)
        {
            for (idx, (bound, beta_val)) in bounds
                .lower_bounds
                .iter()
                .zip(beta_joint.iter())
                .enumerate()
            {
                if *bound > f64::NEG_INFINITY && (*beta_val - *bound).abs() < 1e-12 {
                    active_mask[idx] = true;
                }
            }
        }
        let res_inf_free = rhs
            .iter()
            .zip(active_mask.iter())
            .filter(|(_, active)| !**active)
            .map(|(v, _)| v.abs())
            .fold(0.0_f64, f64::max);
        // Scale-aware residual tolerance — the joint stationarity
        // residual ‖∇ℓ − Sβ‖_∞ scales with |obj| (≈ O(n) at large-scale
        // scale), so the historical absolute `inner_tol = 1e-6` is
        // unachievable here even at the true minimum. Same rationale
        // as the joint-Newton convergence test above.
        let polish_obj = -cached_eval.log_likelihood + *current_penalty;
        let polish_residual_tol = inner_tol * (1.0 + polish_obj.abs());
        if res_inf_free <= polish_residual_tol {
            *converged = true;
            break;
        }

        // Solve constrained Newton system if simple bounds are present,
        // else unconstrained.
        let delta = if let Some(ref constraints) = joint_constraints_now {
            let warm = flatten_joint_active_set(cached_active_sets, &block_constraints_now);
            let lower_bounds_opt = extract_simple_lower_bounds(constraints, total_p_joint)
                .ok()
                .flatten();
            if let Some(bounds) = lower_bounds_opt.as_ref() {
                match solve_quadratic_with_simple_lower_bounds(
                    &h_dense,
                    &rhs,
                    &beta_joint,
                    bounds,
                    warm.as_deref(),
                ) {
                    Ok((beta_new, _active)) => &beta_new - &beta_joint,
                    Err(_) => break,
                }
            } else {
                match gam_solve::active_set::solve_quadratic_with_constraint_set(
                    &h_dense,
                    &rhs,
                    &beta_joint,
                    constraints,
                    warm.as_deref(),
                ) {
                    Ok((beta_new, _active)) => &beta_new - &beta_joint,
                    Err(_) => break,
                }
            }
        } else {
            let solver = gam_linalg::utils::StableSolver::new();
            let factor = match solver.factorize(&h_dense) {
                Ok(factor) => factor,
                Err(_) => break,
            };
            let mut direction = rhs.clone();
            let mut direction_matrix =
                gam_linalg::faer_ndarray::array1_to_col_matmut(&mut direction);
            factor.solve_in_place(direction_matrix.as_mut());
            if !direction.iter().all(|value| value.is_finite()) {
                break;
            }
            direction
        };
        if !delta.iter().all(|v| v.is_finite()) {
            break;
        }
        // Keep polishing until the free-space joint residual is small; a
        // tiny delta alone is not a certificate of stationarity.
        // Damped line search with projection.
        let old_states: Vec<ParameterBlockState> = states.clone();
        let old_obj = -eval_for_polish.log_likelihood + *current_penalty;
        let mut accepted_polish = false;
        for bt in 0..10 {
            let alpha = 0.5f64.powi(bt);
            for b in 0..specs.len() {
                let (start, end) = ranges_joint[b];
                let mut trial_beta = old_states[b].beta.clone();
                trial_beta.scaled_add(alpha, &delta.slice(ndarray::s![start..end]));
                let projected =
                    family.post_update_block_beta(&old_states, b, &specs[b], trial_beta.clone())?;
                reject_constrained_post_update_repair(
                    b,
                    &specs[b],
                    &trial_beta,
                    &projected,
                    block_constraints_now[b].as_ref(),
                )?;
                states[b].beta.assign(&projected);
            }
            refresh_all_block_etas(family, specs, states)?;
            let trial_ll = match family.log_likelihood_only(states) {
                Ok(v) => v,
                Err(_) => {
                    for (b, s) in old_states.iter().enumerate() {
                        states[b] = s.clone();
                    }
                    refresh_all_block_etas(family, specs, states)?;
                    continue;
                }
            };
            let trial_penalty = total_quadratic_penalty(
                states,
                s_lambdas,
                ridge,
                options.ridge_policy,
                joint_bundle,
                Some(specs),
            );
            let trial_obj = -trial_ll + trial_penalty;
            if trial_obj.is_finite() && trial_obj <= old_obj + 1e-12 {
                *current_penalty = trial_penalty;
                *cached_eval = family.evaluate(states)?;
                accepted_polish = true;
                break;
            }
        }
        if !accepted_polish {
            // Restore and stop polishing.
            for (b, s) in old_states.iter().enumerate() {
                states[b] = s.clone();
            }
            refresh_all_block_etas(family, specs, states)?;
            break;
        }
    }
    Ok(())
}

/// Final result assembly for the blockwise / polish fall-through path of
/// [`inner_blockwise_fit`]. Computes the penalty value, the block log-dets, the
/// (converged-only) projected KKT residual for the IFT, and the active-constraint
/// block, then moves `states`, `s_lambdas`, and `cached_active_sets` into the
/// returned [`BlockwiseInnerResult`]. Before log-determinant assembly, every
/// unconstrained converged result with exact joint curvature is re-certified at
/// the coefficient vector being returned; this includes modes minted by the
/// blockwise fall-through and joint-polish paths.
pub(crate) fn assemble_inner_blockwise_result<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    mut states: Vec<ParameterBlockState>,
    block_log_lambdas: &[Array1<f64>],
    options: &BlockwiseFitOptions,
    s_lambdas: Vec<Array2<f64>>,
    ridge: f64,
    joint_bundle: Option<&gam_problem::JointPenaltyBundle>,
    cached_active_sets: Vec<Option<Vec<usize>>>,
    cached_eval: &FamilyEvaluation,
    converged: bool,
    cycles_done: usize,
    last_residual_tol: f64,
    exact_joint_curvature_available: bool,
) -> Result<BlockwiseInnerResult, String> {
    let local_ranges = block_param_ranges(specs);
    let local_total_p = local_ranges.last().map(|(_, end)| *end).unwrap_or(0);
    let block_constraints = collect_block_linear_constraints(family, &states, specs)?;
    let joint_constraints =
        assemble_joint_linear_constraints(&block_constraints, &local_ranges, local_total_p)?;
    let joint_mode_diagonal_ridge = if ridge > 0.0 && options.ridge_policy.accounts_for_objective()
    {
        ridge
    } else {
        0.0
    };
    let mut certified_workspace = None;
    if converged
        && exact_joint_curvature_available
        && joint_constraints.is_none()
        && !family.joint_jeffreys_term_required()
    {
        let certificate = exact_joint_mode_curvature_certificate(
            family,
            &states,
            specs,
            options,
            &local_ranges,
            &s_lambdas,
            joint_mode_diagonal_ridge,
            joint_bundle,
            local_total_p,
            None,
        )?;
        let has_negative_curvature = certificate.has_resolvable_negative_curvature();
        let minimum_whitened_eigenvalue = certificate.minimum_whitened_eigenvalue;
        let numerical_floor = certificate.numerical_floor;
        certified_workspace = certificate.workspace;
        if has_negative_curvature {
            return Err(format!(
                "blockwise/joint-polish tentative convergence rejected by fresh exact returned-mode curvature: lambda_min={:.6e} < -floor={:.6e}; an indefinite coefficient point cannot define a Laplace mode",
                minimum_whitened_eigenvalue, numerical_floor,
            ));
        }
        log::info!(
            "[PIRLS/blockwise mode certificate] returned beta certified from fresh exact curvature: lambda_min={:.6e}, floor={:.6e}",
            minimum_whitened_eigenvalue,
            numerical_floor,
        );
    }

    // Reuse cached evaluation from the last cycle's end (or the initial eval if 0 cycles ran).
    let penalty_value = total_quadratic_penalty(
        &states,
        &s_lambdas,
        ridge,
        options.ridge_policy,
        joint_bundle,
        Some(specs),
    );

    let active_constraints = {
        assemble_active_constraint_block(
            &block_constraints,
            &cached_active_sets,
            &local_ranges,
            local_total_p,
        )
        .map(std::sync::Arc::new)
    };
    let (block_logdet_h, block_logdet_s) = if converged {
        let (h, s) = blockwise_logdet_terms_with_workspace(
            family,
            specs,
            &mut states,
            block_log_lambdas,
            options,
            certified_workspace.clone(),
            None,
            active_constraints.as_deref(),
        )?;
        (Some(h), Some(s))
    } else {
        (None, None)
    };
    let kkt_residual = if converged {
        match exact_newton_joint_gradient_from_eval(cached_eval, specs, &states)? {
            Some(gradient) => {
                let active_set_rows_total: usize = cached_active_sets
                    .iter()
                    .map(|maybe| maybe.as_ref().map(|v| v.len()).unwrap_or(0))
                    .sum();
                let free_rank_at_cert = local_total_p.saturating_sub(active_set_rows_total);
                exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
                    &gradient,
                    specs,
                    &states,
                    &s_lambdas,
                    ridge,
                    options.ridge_policy,
                    &block_constraints,
                    Some(cached_active_sets.as_slice()),
                    joint_penalty_stationarity_score(options, specs, &states).as_ref(),
                )?
                .map(|r| r.with_metadata(last_residual_tol, free_rank_at_cert))
            }
            None => None,
        }
    } else {
        // Inner did not converge; no caller should trust an IFT correction
        // at a non-KKT iterate.
        None
    };

    Ok(BlockwiseInnerResult {
        block_states: states,
        terminal_working_sets: Some(cached_eval.blockworking_sets.clone()),
        active_sets: normalize_active_sets(cached_active_sets),
        log_likelihood: cached_eval.log_likelihood,
        penalty_value,
        cycles: cycles_done,
        converged,
        block_logdet_h,
        block_logdet_s,
        s_lambdas,
        joint_workspace: certified_workspace,
        kkt_residual,
        active_constraints,
    })
}

/// Borrowed derivative provider for joint models that wraps closures with
/// non-`'static` lifetimes.
///
/// The closures borrow data from the calling stack frame (family, synced states,
/// specs), so we use borrowed closures with a non-`'static` lifetime.
/// Instead we borrow the closures and implement `HessianDerivativeProvider` directly.
///
/// # Sign convention
///
/// The unified evaluator passes `v_k = H⁻¹(A_k β̂)` to `hessian_derivative_correction`.
/// By the implicit function theorem, `dβ̂/dρ_k = −v_k`. The stored `compute_dh`
/// expects the actual perturbation direction `δβ`, so we negate `v_k` before calling it.
pub(crate) struct BorrowedJointDerivProvider<'a> {
    pub(crate) compute_dh: &'a DriftDerivFn<'a>,
    pub(crate) compute_dh_many: Option<&'a DriftDerivManyFn<'a>>,
    pub(crate) compute_d2h: &'a DriftSecondDerivFn<'a>,
    /// Optional batched second-derivative callback. The unified evaluator's
    /// outer-Hessian ρ-ρ pair loop precomputes all K(K+1)/2 (v_k, v_l, u_kl)
    /// triples and calls this once per outer Hessian assembly when set, so
    /// families that fuse the per-row D²H walk across pairs (e.g. survival
    /// marginal-slope which scans n rows once per outer eval) replace
    /// K(K+1)/2 separate row-walks with one. The default `None` falls back
    /// to the per-pair `compute_d2h` dispatch and preserves the historical
    /// dispatch cost.
    pub(crate) compute_d2h_many: Option<&'a DriftSecondDerivManyFn<'a>>,
    pub(crate) family_outer_hessian_operator: Option<Arc<dyn gam_problem::HessianOperator>>,
}
