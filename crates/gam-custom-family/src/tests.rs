//! Unit tests for the custom-family blockwise carrier. Declared from `mod.rs`
//! as `#[cfg(test)] mod tests;`; reaches the FD helper via `super::test_support`.

use super::*;

#[derive(Clone)]
pub(crate) struct BatchedOuterHessianTestFamily {
    pub(crate) matrix: Array2<f64>,
}

pub(crate) struct TestHessianOperator {
    pub(crate) matrix: Array2<f64>,
}

impl gam_problem::HessianOperator for TestHessianOperator {
    fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    fn apply_into(
        &self,
        v: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), opt::ObjectiveEvalError> {
        out.assign(&self.matrix.dot(v));
        Ok(())
    }

    fn materialization(&self) -> opt::HessianMaterialization {
        opt::HessianMaterialization::Explicit
    }

    fn materialize_dense(&self) -> Result<Array2<f64>, opt::ObjectiveEvalError> {
        Ok(self.matrix.clone())
    }
}

impl CustomFamily for BatchedOuterHessianTestFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![],
        })
    }

    fn outer_hyper_hessian_hvp_available(&self, _: &[ParameterBlockSpec]) -> bool {
        true
    }

    fn outer_hyper_hessian_operator(
        &self,
        _: &[ParameterBlockSpec],
    ) -> Option<Arc<dyn gam_problem::HessianOperator>> {
        Some(Arc::new(TestHessianOperator {
            matrix: self.matrix.clone(),
        }))
    }
}

#[test]
pub(crate) fn blockwise_fit_from_parts_accepts_stacked_solver_eta_with_canonical_geometry_rows() {
    let canonical_design = DesignMatrix::from(Array2::ones((2, 1)));
    let stacked_design = DesignMatrix::from(Array2::ones((6, 1)));
    let spec = ParameterBlockSpec {
        name: "stacked".to_string(),
        design: canonical_design,
        offset: Array1::zeros(2),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: Some(stacked_design),
        stacked_offset: Some(Array1::zeros(6)),
    };
    let state = ParameterBlockState {
        beta: array![0.25],
        eta: Array1::zeros(6),
    };
    let fit = blockwise_fit_from_parts(
        BlockwiseFitResultParts {
            block_states: vec![state],
            log_likelihood: -1.0,
            log_lambdas: Array1::zeros(0),
            lambdas: Array1::zeros(0),
            covariance_conditional: Some(Array2::eye(1)),
            stable_penalty_term: 0.0,
            penalized_objective: 1.0,
            outer_iterations: 0,
            outer_gradient_norm: Some(0.0),
            criterion_certificate: None,
            inner_cycles: 0,
            outer_converged: true,
            geometry: Some(FitGeometry {
                coefficient_gauge: gam_problem::gauge::Gauge::identity(&[1]),
                penalized_hessian: Array2::eye(1).into(),
                working_weights: Array1::ones(2),
                working_response: Array1::zeros(2),
            }),
            precomputed_edf: Some((1.0, Vec::new(), vec![1.0], Vec::new())),
            joint_log_lambdas: None,
        },
        &[spec],
    )
    .expect("stacked solver eta should assemble against canonical geometry rows");

    assert_eq!(fit.block_states[0].eta.len(), 6);
    assert_eq!(fit.geometry.as_ref().unwrap().working_weights.len(), 2);
}

#[test]
pub(crate) fn custom_family_geometry_keeps_active_precision_under_rectangular_raw_lift() {
    let active_hessian = array![[5.0, 1.5], [1.5, 3.0]];
    let outer = Gauge {
        t_full: array![[1.0, 0.0], [0.5, 2.0], [-1.0, 3.0]],
        affine_shift: array![4.0, -2.0, 1.0],
        block_starts_raw: vec![0, 3],
        block_starts_reduced: vec![0, 2],
    };
    let geometry = FitGeometry {
        coefficient_gauge: Gauge::identity(&[2]),
        penalized_hessian: active_hessian.clone().into(),
        working_weights: array![1.0],
        working_response: array![0.0],
    };

    let lifted = lift_fit_geometry_through_gauge(&outer, Some(geometry))
        .expect("rectangular raw lift")
        .expect("geometry remains available");

    assert_eq!(lifted.penalized_hessian.as_array(), &active_hessian);
    assert_eq!(lifted.coefficient_gauge.t_full, outer.t_full);
    assert_eq!(lifted.coefficient_gauge.affine_shift, outer.affine_shift);
    assert_eq!(lifted.coefficient_gauge.raw_widths(), vec![3]);
    assert_eq!(lifted.coefficient_gauge.reduced_widths(), vec![2]);
}

#[test]
pub(crate) fn batched_outer_hessian_terms_materialize_to_exact_small_matrix() {
    let exact = array![[4.0, -1.0], [-1.0, 3.0]];
    let family = BatchedOuterHessianTestFamily {
        matrix: exact.clone(),
    };
    // rho.len() must equal sum(spec.penalties.len()); empty specs ⇒ empty rho.
    let terms = family
        .batched_outer_hessian_terms(&[], &[], &[], &Array1::<f64>::zeros(0), None)
        .expect("batched Hessian hook succeeds")
        .expect("test family exposes batched HVP terms");
    let operator = match terms.outer_hessian {
        gam_problem::HessianValue::Operator(operator) => operator,
        _ => panic!("batched hook should expose an operator"),
    };
    let dense = operator
        .apply_mat(Array2::<f64>::eye(2).view())
        .expect("operator materializes on small exact case");
    assert_eq!(dense, exact);
}

#[test]
pub(crate) fn batched_outer_hessian_operator_selected_only_for_hessian_eval() {
    let family = BatchedOuterHessianTestFamily {
        matrix: array![[2.0, 0.5], [0.5, 5.0]],
    };
    let selected = custom_family_batched_outer_hessian_operator(
        &family,
        &[],
        &[],
        &[],
        &Array1::<f64>::zeros(0),
        None,
        EvalMode::ValueGradientHessian,
    )
    .expect("selection check succeeds");
    assert!(
        selected.is_some(),
        "supported Hessian/HVP families should select the batched operator path"
    );

    let not_selected = custom_family_batched_outer_hessian_operator(
        &family,
        &[],
        &[],
        &[],
        &Array1::<f64>::zeros(0),
        None,
        EvalMode::ValueAndGradient,
    )
    .expect("non-Hessian selection check succeeds");
    assert!(
        not_selected.is_none(),
        "batched Hessian terms must not run for gradient-only evaluations"
    );
}

#[test]
pub(crate) fn batched_outer_gradient_override_rejected_when_jeffreys_curvature_is_active() {
    assert!(
        batched_outer_gradient_contract_allows_override(None),
        "released objective without robust Jeffreys curvature may use a family-owned batched gradient"
    );

    let zero_hphi = Array2::<f64>::zeros((2, 2));
    assert!(
        batched_outer_gradient_contract_allows_override(Some(&zero_hphi)),
        "a gated zero Jeffreys curvature leaves the batched gradient contract unchanged"
    );

    let active_hphi = array![[0.0, 0.0], [0.0, 1.0e-6]];
    assert!(
        !batched_outer_gradient_contract_allows_override(Some(&active_hphi)),
        "nonzero H_phi changes the logdet operator and needs the unified H_phi-aware gradient"
    );
}

use approx::assert_relative_eq;
use faer::sparse::{SparseColMat, Triplet};
use gam_linalg::matrix::DesignMatrix;
use gam_models::gamlss::{BinomialLocationScaleFamily, BinomialLocationScaleWiggleFamily};
use gam_test_support::binomial_location_scale_base_fixture;
use ndarray::{Array1, Array2, array};

#[test]
pub(crate) fn joint_preconditioner_preserves_negative_observed_curvature_scale() {
    let base_diagonal = array![-12.0, 3.0, 0.0];
    let penalty = array![[2.0, -1.0], [-1.0, 2.0]];
    let diagonal =
        joint_penalty_preconditioner_diag(&base_diagonal, &[(0, 2), (2, 3)], &[penalty], 0.5, None);

    // The penalty contributes its absolute row sum (3) to the first block;
    // the ridge contributes 0.5 everywhere.  In particular, -12 is a
    // magnitude-12 trust scale rather than a direction collapsed to the floor.
    assert_eq!(diagonal, array![15.5, 6.5, 0.5]);
}

pub(crate) fn assert_kronecker_factored_matches_dense(
    left: Array2<f64>,
    right: Array2<f64>,
    vectors: Vec<Array1<f64>>,
) {
    let penalty = PenaltyMatrix::KroneckerFactored { left, right };
    let dense = penalty.to_dense();
    for v in vectors {
        let factored_dot = penalty.dot(&v);
        let dense_dot = dense.dot(&v);
        for i in 0..v.len() {
            assert!(
                (factored_dot[i] - dense_dot[i]).abs() <= 1.0e-14,
                "Kronecker dot mismatch at component {i}: factored={}, dense={}",
                factored_dot[i],
                dense_dot[i],
            );
        }

        let factored_quad = penalty.quadratic_form(&v);
        let dense_quad = v.dot(&dense_dot);
        assert!(
            (factored_quad - dense_quad).abs() <= 1.0e-14,
            "Kronecker quadratic form mismatch: factored={factored_quad}, dense={dense_quad}",
        );
    }
}

#[test]
pub(crate) fn kronecker_factored_dot_and_quadratic_form_match_dense_row_major_operator() {
    let left_diag = array![[10.0, 0.0], [0.0, 100.0]];
    let right_diag = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
    let mut diag_vectors = Vec::new();
    for i in 0..6 {
        let mut v = Array1::<f64>::zeros(6);
        v[i] = 1.0;
        diag_vectors.push(v);
    }
    diag_vectors.push(array![0.25, -1.5, 2.0, 0.75, -0.5, 3.25]);
    assert_kronecker_factored_matches_dense(left_diag, right_diag, diag_vectors);

    let left_nondiag = array![[1.0, 2.0], [3.0, 4.0]];
    let right_nondiag = array![[0.0, 1.0], [1.0, 0.0]];
    let mut nondiag_vectors = Vec::new();
    for i in 0..4 {
        let mut v = Array1::<f64>::zeros(4);
        v[i] = 1.0;
        nondiag_vectors.push(v);
    }
    nondiag_vectors.push(array![1.25, -0.75, 2.5, -3.0]);
    assert_kronecker_factored_matches_dense(left_nondiag, right_nondiag, nondiag_vectors);
}

/// The marker-free coupled-joint-Hessian gate (#727, #729) trusts a family
/// that returns a genuinely coupled joint Hessian — nonzero off-diagonal
/// blocks — without a hand-set `has_explicit_joint_hessian()`. Pin the
/// structural probe that drives every `_with_specs` dispatch: block-diagonal
/// (the trait default) is NOT coupling, a single nonzero off-block IS, and a
/// shape disagreement must never be claimed as coupling.
pub(crate) fn solve_blockweighted_system(
    x: &DesignMatrix,
    y_star: &Array1<f64>,
    w: &Array1<f64>,
    s_lambda: &Array2<f64>,
    ridge_floor: f64,
    ridge_policy: RidgePolicy,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    if y_star.len() != n || w.len() != n {
        return Err(CustomFamilyError::DimensionMismatch {
            reason: "weighted-system dimension mismatch".to_string(),
        }
        .into());
    }
    let xtwy = x.compute_xtwy(w, y_star)?;
    x.solve_systemwith_policy(w, &xtwy, Some(s_lambda), ridge_floor, ridge_policy)
        .map_err(|_| "block solve failed after ridge retries".to_string())
}

#[test]
pub(crate) fn default_inner_cycle_budget_covers_large_scale_joint_newton_tail() {
    let options = BlockwiseFitOptions::default();

    assert_eq!(
        options.inner_max_cycles,
        DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES
    );
    assert!(
        options.inner_max_cycles > 300,
        "startup validation must not reject still-descending exact joint solves at the old cap"
    );
}

#[test]
pub(crate) fn joint_penalty_subspace_trace_matches_projected_logdet_derivative() {
    let ranges = vec![(0, 3)];
    let s_lambda = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
    let penalties = vec![s_lambda];
    let h = array![[4.0, 0.2, 7.0], [0.2, 9.0, -3.0], [7.0, -3.0, 30.0]];
    // `∂Sλ/∂ρ` is supported on range(Sλ) (here the leading 2×2 block, the
    // positive-eigenvalue subspace of `S`). Since #901 the kernel is the
    // full spectral `M⁺`, whose trace differentiates `log|H+Sλ|₊` exactly
    // for EVERY drift; a range(Sλ)-supported drift exercises the same
    // contract the production `∂Sλ/∂ρ` does (and is where the old
    // range(Sλ)-block kernel and `M⁺` agree, so this pin is stable
    // across the kernel generalization).
    let drift = array![[0.7, -0.4, 0.0], [-0.4, 1.3, 0.0], [0.0, 0.0, 0.0]];

    let (logdet, kernel) = joint_penalty_subspace_trace_parts(
        &JointHessianSource::Dense(h.clone()),
        &ranges,
        &penalties,
        3,
        0.0,
        None,
        None,
    )
    .expect("projection parts build");
    let kernel = kernel.expect("rank-deficient penalty still has an identified subspace");
    // Kernel basis = kept eigenvectors of M = H + Sλ (full rank 3 here),
    // NOT the rank-2 range(Sλ) basis of the pre-#901 reduced kernel.
    assert_eq!(kernel.u_s.ncols(), 3);
    // logdet is the FULL identifiable-subspace `log|H + Sλ|₊`. Here H + Sλ
    // is full rank (3), so this is the ordinary log-det of
    //   M = [[5, 0.2, 7], [0.2, 11, -3], [7, -3, 30]],  det(M) = 1056.4.
    let m = array![[5.0, 0.2, 7.0], [0.2, 11.0, -3.0], [7.0, -3.0, 30.0]];
    let (m_evals, _) = m.eigh(faer::Side::Lower).expect("M eigendecomposition");
    let expected_logdet: f64 = m_evals.iter().map(|&v| v.ln()).sum();
    assert_relative_eq!(logdet, expected_logdet, epsilon = 1e-10);

    let analytic = kernel.trace_projected_logdet(&drift);
    let eps = 1.0e-6;
    let h_plus = &h + &(drift.mapv(|v| eps * v));
    let h_minus = &h - &(drift.mapv(|v| eps * v));
    let (logdet_plus, _) = joint_penalty_subspace_trace_parts(
        &JointHessianSource::Dense(h_plus),
        &ranges,
        &penalties,
        3,
        0.0,
        None,
        None,
    )
    .expect("plus projection parts build");
    let (logdet_minus, _) = joint_penalty_subspace_trace_parts(
        &JointHessianSource::Dense(h_minus),
        &ranges,
        &penalties,
        3,
        0.0,
        None,
        None,
    )
    .expect("minus projection parts build");
    let finite_difference = (logdet_plus - logdet_minus) / (2.0 * eps);

    assert_relative_eq!(
        analytic,
        finite_difference,
        epsilon = 1e-8,
        max_relative = 1e-8
    );
}

#[test]
pub(crate) fn joint_outer_gradient_uses_projected_trace_for_rank_deficient_penalty() {
    let ranges = vec![(0, 3)];
    let rho = array![0.0];
    let beta = array![1.0, -1.0, 3.0];
    let s_lambda = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
    let h = array![[4.0, 0.2, 7.0], [0.2, 9.0, -3.0], [7.0, -3.0, 30.0]];
    let spec = ParameterBlockSpec {
        name: "surface".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (1, 3),
        ))),
        offset: Array1::zeros(1),
        penalties: vec![PenaltyMatrix::Dense(s_lambda.clone())],
        nullspace_dims: vec![1],
        initial_log_lambdas: rho.clone(),
        initial_beta: Some(beta.clone()),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![spec];
    let inner = BlockwiseInnerResult {
        block_states: vec![ParameterBlockState {
            beta: beta.clone(),
            eta: Array1::zeros(1),
        }],
        active_sets: vec![None],
        log_likelihood: 0.0,
        penalty_value: 0.5 * beta.dot(&fast_av(&s_lambda, &beta)),
        cycles: 1,
        converged: true,
        block_logdet_h: 0.0,
        block_logdet_s: 0.0,
        s_lambdas: vec![s_lambda.clone()],
        joint_workspace: None,
        kkt_residual: None,
        active_constraints: None,
    };
    let per_block = vec![rho.clone()];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: false,
        ..BlockwiseFitOptions::default()
    };
    let no_dh = |_direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> { Ok(None) };
    let no_d2h = |_u: &Array1<f64>, _v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
        Ok(None)
    };

    let projected = joint_outer_evaluate(
        &inner,
        &specs,
        &per_block,
        &rho,
        &beta,
        JointHessianSource::Dense(h.clone()),
        &ranges,
        3,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        true,
        true,
        false,
        true,
        EvalMode::ValueAndGradient,
        &options,
        gam_problem::RhoPrior::Flat,
        PseudoLogdetMode::Smooth,
        &no_dh,
        None,
        &no_d2h,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("projected outer evaluation succeeds");

    let unprojected = joint_outer_evaluate(
        &inner,
        &specs,
        &per_block,
        &rho,
        &beta,
        JointHessianSource::Dense(h.clone()),
        &ranges,
        3,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        true,
        true,
        false,
        false,
        EvalMode::ValueAndGradient,
        &options,
        gam_problem::RhoPrior::Flat,
        PseudoLogdetMode::Smooth,
        &no_dh,
        None,
        &no_d2h,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("unprojected outer evaluation succeeds");

    let (_, kernel) = joint_penalty_subspace_trace_parts(
        &JointHessianSource::Dense(h.clone()),
        &ranges,
        std::slice::from_ref(&s_lambda),
        3,
        0.0,
        None,
        None,
    )
    .expect("projection kernel builds");
    let projected_trace = kernel
        .expect("rank-deficient penalty has positive subspace")
        .trace_projected_logdet(&s_lambda);
    let expected_gradient =
        0.5 * beta.dot(&fast_av(&s_lambda, &beta)) + 0.5 * projected_trace - 0.5 * 2.0;

    assert_relative_eq!(
        projected.gradient[0],
        expected_gradient,
        epsilon = 1e-12,
        max_relative = 1e-12
    );
    // Post gh#752/#901 contract: the trace kernel is the FULL spectral
    // pseudo-inverse `M⁺ = (H+Sλ)⁺` over range(H+Sλ). On a NONSINGULAR `M`
    // (this fixture) that is exactly `M⁻¹`, so the projected route and the
    // full-space operator route compute the same generalized determinant
    // and the same ρ-trace — the projection must be INVARIANT here. (The
    // historical assertion that they differ encoded the pre-#752 range(Sλ)
    // reduction, which dropped the penalty-null likelihood curvature and
    // was itself the bug. The case where the routes genuinely diverge — a
    // singular `M` whose ker(H+Sλ) the pseudo-logdet must drop — is
    // asserted in `joint_outer_gradient_projected_trace_drops_joint_null`.)
    assert_relative_eq!(
        projected.gradient[0],
        unprojected.gradient[0],
        epsilon = 1e-8,
        max_relative = 1e-8
    );
}

/// The discriminating case for `project_hessian_logdet`: a joint Hessian
/// whose ker(H) overlaps ker(Sλ), so `M = H + Sλ` is genuinely singular.
/// The projected route must drop the unidentified direction (pseudo-logdet
/// + `M⁺` trace kernel over range(M)) and produce the exact closed-form
/// gradient; a full-space `M⁻¹` route has no finite answer here. This is
/// the routing guard the nonsingular fixture above cannot provide (there
/// the two routes coincide by design).
#[test]
pub(crate) fn joint_outer_gradient_projected_trace_drops_joint_null() {
    let ranges = vec![(0, 3)];
    let rho = array![0.0];
    let beta = array![1.0, -1.0, 3.0];
    let s_lambda = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
    // ker(h) = span(e3) = ker(s_lambda) ⇒ M = H + Sλ is singular with the
    // unidentified direction e3.
    let h = array![[4.0, 0.2, 0.0], [0.2, 9.0, 0.0], [0.0, 0.0, 0.0]];
    let spec = ParameterBlockSpec {
        name: "surface".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (1, 3),
        ))),
        offset: Array1::zeros(1),
        penalties: vec![PenaltyMatrix::Dense(s_lambda.clone())],
        nullspace_dims: vec![1],
        initial_log_lambdas: rho.clone(),
        initial_beta: Some(beta.clone()),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![spec];
    let inner = BlockwiseInnerResult {
        block_states: vec![ParameterBlockState {
            beta: beta.clone(),
            eta: Array1::zeros(1),
        }],
        active_sets: vec![None],
        log_likelihood: 0.0,
        penalty_value: 0.5 * beta.dot(&fast_av(&s_lambda, &beta)),
        cycles: 1,
        converged: true,
        block_logdet_h: 0.0,
        block_logdet_s: 0.0,
        s_lambdas: vec![s_lambda.clone()],
        joint_workspace: None,
        kkt_residual: None,
        active_constraints: None,
    };
    let per_block = vec![rho.clone()];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: false,
        ..BlockwiseFitOptions::default()
    };
    let no_dh = |_direction: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> { Ok(None) };
    let no_d2h = |_u: &Array1<f64>, _v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
        Ok(None)
    };

    let projected = joint_outer_evaluate(
        &inner,
        &specs,
        &per_block,
        &rho,
        &beta,
        JointHessianSource::Dense(h.clone()),
        &ranges,
        3,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        true,
        true,
        false,
        true,
        EvalMode::ValueAndGradient,
        &options,
        gam_problem::RhoPrior::Flat,
        PseudoLogdetMode::Smooth,
        &no_dh,
        None,
        &no_d2h,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("projected outer evaluation succeeds on a singular joint Hessian");

    // Independent closed form. On the identified (e1,e2) quotient,
    // M = H + S = [[5, 0.2], [0.2, 11]], so
    // tr(M⁻¹S) = (11·1 + 5·2) / (5·11 - 0.2²) = 21/54.96.
    // The penalty quadratic is βᵀSβ = 1² + 2(-1)² = 3 and rank(S)=2.
    // Keep this arithmetic independent of the production trace kernel so a
    // shared kernel defect cannot satisfy both sides of the assertion.
    let hand_projected_trace = 21.0 / 54.96;
    let expected_gradient = 0.5 * 3.0 + 0.5 * hand_projected_trace - 0.5 * 2.0;

    assert!(
        projected.objective.is_finite(),
        "pseudo-logdet objective must stay finite when ker(H+Sλ) is dropped"
    );
    assert_relative_eq!(
        projected.gradient[0],
        expected_gradient,
        epsilon = 1e-10,
        max_relative = 1e-10
    );
}

// Experimental scan documenting that on THIS fixture's geometry the
// joint_outer_evaluate path does not show divergence between
// project_hessian_logdet=true and =false at large-scale ρ: the dominant
// term ½ λ β'Sβ grows linearly in λ regardless of projection, and the trace
// pair cancels in both routes here. The clustered-PC marginal-slope failure
// (#808/#787) is a DIFFERENT geometry — a near-collinear penalty-null trend
// whose likelihood determinant the range(Sλ)-only route drops. That route is
// now disabled for all marginal-slope families: the project_hessian_logdet
// flag at every joint_outer_evaluate/_efs call site reads
// `use_projected_penalty_logdet()` (default true), so value and analytic
// gradient share the range(H+Sλ) generalized determinant.
#[test]
pub(crate) fn large_scale_rho_scan_joint_outer_evaluate_is_projection_invariant() {
    // Same fixture shape as the rank-deficient projected-trace test,
    // but with H_unpen scaled to data-Hessian magnitude (n ~ 2e5).
    let ranges = vec![(0, 3)];
    let beta = array![1.0, -1.0, 3.0];
    let s_unit: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
    let n_scale = 2.0e5_f64;
    let h: Array2<f64> =
        array![[4.0, 0.2, 7.0], [0.2, 9.0, -3.0], [7.0, -3.0, 30.0]].mapv(|v| v * n_scale);

    let no_dh = |_d: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> { Ok(None) };
    let no_d2h = |_u: &Array1<f64>, _v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
        Ok(None)
    };

    let mut g_un_at_10 = 0.0_f64;
    let mut g_pr_at_10 = 0.0_f64;

    for &rho_val in &[0.0_f64, 2.0, 4.0, 6.0, 8.0, 10.0] {
        let lam = rho_val.exp();
        let rho = array![rho_val];
        let s_lambda = s_unit.mapv(|v| v * lam);

        let spec = ParameterBlockSpec {
            name: "surface".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((1, 3)),
            )),
            offset: Array1::zeros(1),
            penalties: vec![PenaltyMatrix::Dense(s_unit.clone())],
            nullspace_dims: vec![1],
            initial_log_lambdas: rho.clone(),
            initial_beta: Some(beta.clone()),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        };
        let specs = vec![spec];
        let inner = BlockwiseInnerResult {
            block_states: vec![ParameterBlockState {
                beta: beta.clone(),
                eta: Array1::zeros(1),
            }],
            active_sets: vec![None],
            log_likelihood: 0.0,
            penalty_value: 0.5 * lam * beta.dot(&fast_av(&s_unit, &beta)),
            cycles: 1,
            converged: true,
            block_logdet_h: 0.0,
            block_logdet_s: 0.0,
            s_lambdas: vec![s_lambda.clone()],
            joint_workspace: None,
            kkt_residual: None,
            active_constraints: None,
        };
        let per_block = vec![rho.clone()];
        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: false,
            ..BlockwiseFitOptions::default()
        };

        // project_hessian_logdet = true (current main behavior)
        let projected = joint_outer_evaluate(
            &inner,
            &specs,
            &per_block,
            &rho,
            &beta,
            JointHessianSource::Dense(h.clone()),
            &ranges,
            3,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            true,
            true,
            false,
            true,
            EvalMode::ValueAndGradient,
            &options,
            gam_problem::RhoPrior::Flat,
            PseudoLogdetMode::Smooth,
            &no_dh,
            None,
            &no_d2h,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("projected eval ok");

        // project_hessian_logdet = false (the 0.1.92 / pre-fix behavior)
        let unprojected = joint_outer_evaluate(
            &inner,
            &specs,
            &per_block,
            &rho,
            &beta,
            JointHessianSource::Dense(h.clone()),
            &ranges,
            3,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            true,
            true,
            false,
            false,
            EvalMode::ValueAndGradient,
            &options,
            gam_problem::RhoPrior::Flat,
            PseudoLogdetMode::Smooth,
            &no_dh,
            None,
            &no_d2h,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("unprojected eval ok");

        let g_un = unprojected.gradient[0];
        let g_pr = projected.gradient[0];
        if rho_val == 10.0 {
            g_un_at_10 = g_un.abs();
            g_pr_at_10 = g_pr.abs();
        }
    }

    // Finding: at this fixture geometry the two routes agree to
    // ~1e-6 relative precision at every ρ in [0, 10].  Both grow
    // linearly in λ (≈ ½ λ β'Sβ + bounded trace contribution).
    // The optimizer-visible blow-up in large-scale therefore cannot be
    // a missing projection in joint_outer_evaluate — it must live
    // in the survival-marginal-slope custom gradient path.
    let rel_diff = (g_un_at_10 - g_pr_at_10).abs() / g_pr_at_10.max(1e-30);
    assert!(
        rel_diff < 1e-4,
        "projection should be near-invariant on this fixture at rho=10; \
             got g_un={:.6e}, g_pr={:.6e}, rel_diff={:.3e}",
        g_un_at_10,
        g_pr_at_10,
        rel_diff
    );
}

// ── Large-scale reproducer for the marginal-slope ρ-saturation
// failure ────────────────────────────────────────────────────────────
//
// Failure being investigated:
//   outer iter=60, |g|=4.18e13, three of four ρ-coords pinned at the
//   box bound ±10 (`with_rho_bound(10.0)`). The dominant explicit term
//   ½λβ'Sβ at large scale (n≈2e5, p≈60, β'Sβ~10⁴, λ=exp(10)≈22k) is
//   only ~10⁸ — observed gradient is ~10¹³, FIVE orders of magnitude
//   beyond what the projected-trace kernel cancellation predicts.
//
// The existing `large_scale_rho_scan_joint_outer_evaluate_is_projection_invariant`
// test uses single-block, p=3, nullspace_dims=1, and supplies
// `compute_dh = Ok(None)` — that path SKIPS the trace pair entirely and
// therefore cannot reproduce the failure. The large-scale fit has:
//   - 3 blocks (time_surface, marginal_surface, logslope_surface)
//   - 4 penalty coords (time:1, marginal:2 [anisotropic], logslope:1)
//   - Duchon-shape penalties: large nullspace_dims (d+1=4 for d=3 PCs)
//     producing rank-deficient S with many zero eigenvalues
//   - n ~ 2e5 → H_unpen scale ~ n × diag-of-design-Gram
//   - Realistic `compute_dh(d)` returning the per-coord penalty drift
//     ∂H/∂ρ_k = λ_k S_k (chained through the direction d)
//
// This test reproduces the SHAPE: builds large-scale-dimensioned blocks
// with rank-deficient Duchon-shape penalties, scales H to large-scale
// magnitude, supplies a realistic penalty-drift `compute_dh`, evaluates
// `joint_outer_evaluate` at the actual failure ρ point
// [time=10, marg=10, marg=10, logslope=4.5], and asserts every gradient
// entry is BOUNDED by a physically reasonable multiple of the dominant
// ½λβ'Sβ term.
//
// If this test passes with reasonable bounds: the bug is NOT in
//   joint_outer_evaluate itself — it must live in the marginal-slope-
//   specific drift derivatives (`evaluate_exact_newton_joint_gradient_*`
//   in survival_marginal_slope.rs) that feed the closure.
// If this test fails: joint_outer_evaluate has a numerical defect that
//   surfaces at large scale + realistic Ḣ. We then bisect inside the
//   evaluator.
//
#[test]
pub(crate) fn large_scale_multiblock_outer_gradient_with_realistic_drift_is_bounded() {
    // LargeScale-realistic dimensions for binary-outcome marginal-slope.
    // Duchon(PC1,PC2,PC3, centers=10, order=1) → p_basis = centers +
    // null_basis(d+1=4) = 14 columns per spatial block, nullspace dim=4.
    // The actual fit has time_surface with a different basis (B-spline
    // along entry/exit age) — we approximate with p_time=10, null=2.
    let p_time = 10usize;
    let p_marg = 14usize;
    let p_logs = 14usize;
    let p_total = p_time + p_marg + p_logs;

    // Block ranges in the joint coefficient vector.
    let ranges = vec![
        (0, p_time),
        (p_time, p_time + p_marg),
        (p_time + p_marg, p_total),
    ];

    // ── Build rank-deficient Duchon-shape penalty matrices.
    // S = U diag(σ) Uᵀ where σ has `nullspace_dims` trailing zeros.
    // We use deterministic orthonormal columns from a simple QR of a
    // structured matrix to mimic the eigenstructure without random.
    fn build_duchon_shape(p: usize, nullspace: usize, signal_scale: f64) -> Array2<f64> {
        // Diagonal eigenvalue spectrum, geometric decay across the
        // signal subspace then zeros on the nullspace.
        let rank = p - nullspace;
        let mut eigvals = vec![0.0_f64; p];
        for i in 0..rank {
            // 1.0, 0.5, 0.25, ... — physical Duchon penalty spectrum
            // has spectrum decaying like 1/k for high-frequency modes;
            // geometric decay is a faithful caricature.
            eigvals[i] = signal_scale * 0.5_f64.powi(i as i32);
        }
        // Use a deterministic orthogonal basis: discrete cosine basis.
        // U[i,j] = sqrt(2/p) cos(π (i+0.5) j / p) for j>0; U[i,0]=1/√p.
        let mut u = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            u[[i, 0]] = 1.0 / (p as f64).sqrt();
            for j in 1..p {
                u[[i, j]] = (2.0 / p as f64).sqrt()
                    * (std::f64::consts::PI * (i as f64 + 0.5) * j as f64 / p as f64).cos();
            }
        }
        // S = U diag(eigvals) Uᵀ.
        let mut s = Array2::<f64>::zeros((p, p));
        for k in 0..p {
            if eigvals[k] == 0.0 {
                continue;
            }
            for i in 0..p {
                for j in 0..p {
                    s[[i, j]] += eigvals[k] * u[[i, k]] * u[[j, k]];
                }
            }
        }
        s
    }

    // time_surface: 1 penalty (nullspace=2: constant + linear in age).
    let s_time = build_duchon_shape(p_time, 2, 1.0);
    // marginal_surface: 2 penalties (nullspace=4 each, anisotropic).
    let s_marg_0 = build_duchon_shape(p_marg, 4, 1.0);
    let s_marg_1 = build_duchon_shape(p_marg, 4, 0.7);
    // logslope_surface: 1 penalty (nullspace=4).
    let s_logs = build_duchon_shape(p_logs, 4, 1.0);

    // ── Failure-point ρ = [10, 10, 10, 4.5]. λ = exp(ρ).
    let rho = array![10.0_f64, 10.0, 10.0, 4.5];
    let lams: Array1<f64> = rho.mapv(f64::exp);

    // λ-scaled S matrices (per-block, in block-local indexing — this
    // is what BlockwiseInnerResult.s_lambdas stores).
    let s_lambdas_local: Vec<Array2<f64>> = vec![
        s_time.mapv(|v| v * lams[0]),
        // marginal block has TWO penalties — they are summed into one
        // local s_lambda (this matches how BlockwiseInnerResult stores
        // a per-block sum of all penalties in that block):
        (&s_marg_0 * lams[1]) + &(&s_marg_1 * lams[2]),
        s_logs.mapv(|v| v * lams[3]),
    ];

    // β at large scale: |β|∞ ~ 1, β'Sβ ~ trace(S) ~ O(p) ~ 10.
    let beta_flat = Array1::<f64>::from_iter((0..p_total).map(|i| ((i as f64) * 0.13).sin()));

    // ── Large-scale joint unpenalized Hessian.
    // Real survival Hessian = Xᵀ W X with W diagonal and n=2e5. We
    // mimic the SCALE by H = n * (I + small dense perturbation).
    let n_scale = 2.0e5_f64;
    let mut h = Array2::<f64>::eye(p_total) * n_scale;
    // Add a small off-diagonal coupling to make it non-trivial but SPD.
    for i in 0..p_total {
        for j in 0..p_total {
            if i != j {
                let v = 0.05_f64
                    * n_scale
                    * ((i as f64 - j as f64).abs() / p_total as f64).exp().recip();
                h[[i, j]] = v;
            }
        }
    }

    // ── Hessian β-chain closure.
    // CONTRACT: `compute_dh(v_k)` takes a β-space direction `v_k`
    // (length p_total = `∂β/∂ρ_k` under the envelope) and returns
    // `D_beta H[v_k]` — the third-order tensor of H contracted with
    // `v_k`. The penalty-drift component `λ_k S_k` is added by
    // `joint_outer_evaluate` automatically from `inner.s_lambdas` —
    // this closure adds ONLY the β-chained piece.
    //
    // For an idealized H_unpen that is independent of β (linear model
    // limit, no nonlinear inner geometry), `D_beta H = 0` and the
    // closure returns `Ok(None)`. This is exactly the regime the
    // existing single-block `large_scale_rho_scan_*` test exercises
    // and finds projection-invariant. The marginal-slope family's
    // Hessian DOES depend on β (through the joint geometry), so the
    // closure is non-trivial in production — and that is the
    // candidate source of the gradient blowup.
    //
    // This test takes the idealized path (`Ok(None)`) so any blowup
    // observed here is attributable to `joint_outer_evaluate`'s
    // multi-block / rank-deficient-S handling alone. If this test
    // PASSES (gradient bounded), the bug must live in the family's
    // `hessian_derivative_correction_result` β-chain — not in the
    // evaluator. If it FAILS, the evaluator itself has the defect at
    // large scale + Duchon-shape S.
    let no_dh = |_v_k: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> { Ok(None) };
    let compute_dh = no_dh;
    let no_d2h = |_u: &Array1<f64>, _v: &Array1<f64>| -> Result<Option<DriftDerivResult>, String> {
        Ok(None)
    };

    // ── ParameterBlockSpec for each block.
    let mk_spec = |name: &str,
                   p: usize,
                   penalties: Vec<Array2<f64>>,
                   null: usize,
                   rho_block: Array1<f64>|
     -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::<f64>::zeros((1, p)),
            )),
            offset: Array1::zeros(1),
            penalties: penalties.into_iter().map(PenaltyMatrix::Dense).collect(),
            nullspace_dims: vec![null],
            initial_log_lambdas: rho_block,
            initial_beta: Some(beta_flat.slice(s![..p]).to_owned()),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    };
    let specs = vec![
        mk_spec(
            "time_surface",
            p_time,
            vec![s_time.clone()],
            2,
            array![rho[0]],
        ),
        mk_spec(
            "marginal_surface",
            p_marg,
            vec![s_marg_0.clone(), s_marg_1.clone()],
            4,
            array![rho[1], rho[2]],
        ),
        mk_spec(
            "logslope_surface",
            p_logs,
            vec![s_logs.clone()],
            4,
            array![rho[3]],
        ),
    ];

    let per_block = vec![array![rho[0]], array![rho[1], rho[2]], array![rho[3]]];

    let inner = BlockwiseInnerResult {
        block_states: vec![
            ParameterBlockState {
                beta: beta_flat.slice(s![0..p_time]).to_owned(),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: beta_flat.slice(s![p_time..p_time + p_marg]).to_owned(),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: beta_flat.slice(s![p_time + p_marg..p_total]).to_owned(),
                eta: Array1::zeros(1),
            },
        ],
        active_sets: vec![None, None, None],
        log_likelihood: 0.0,
        penalty_value: 0.5
            * (lams[0]
                * beta_flat.slice(s![0..p_time]).dot(&fast_av(
                    &s_time,
                    &beta_flat.slice(s![0..p_time]).to_owned(),
                ))
                + lams[1]
                    * beta_flat.slice(s![p_time..p_time + p_marg]).dot(&fast_av(
                        &s_marg_0,
                        &beta_flat.slice(s![p_time..p_time + p_marg]).to_owned(),
                    ))
                + lams[2]
                    * beta_flat.slice(s![p_time..p_time + p_marg]).dot(&fast_av(
                        &s_marg_1,
                        &beta_flat.slice(s![p_time..p_time + p_marg]).to_owned(),
                    ))
                + lams[3]
                    * beta_flat.slice(s![p_time + p_marg..p_total]).dot(&fast_av(
                        &s_logs,
                        &beta_flat.slice(s![p_time + p_marg..p_total]).to_owned(),
                    ))),
        cycles: 1,
        converged: true,
        block_logdet_h: 0.0,
        block_logdet_s: 0.0,
        s_lambdas: s_lambdas_local,
        joint_workspace: None,
        kkt_residual: None,
        active_constraints: None,
    };

    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: false,
        ..BlockwiseFitOptions::default()
    };

    let projected = joint_outer_evaluate(
        &inner,
        &specs,
        &per_block,
        &rho,
        &beta_flat,
        JointHessianSource::Dense(h.clone()),
        &ranges,
        p_total,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        true,
        true,
        false,
        true,
        EvalMode::ValueAndGradient,
        &options,
        gam_problem::RhoPrior::Flat,
        PseudoLogdetMode::Smooth,
        &compute_dh,
        None,
        &no_d2h,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("large-scale projected eval");

    // Physical-bound check: ½λ_k β'_k S_k β_k is the dominant explicit
    // term per coord. For large-scale shape this is ~10⁸ at ρ=10 with
    // β-scale O(1). The full gradient including the projected trace
    // pair should be of THE SAME ORDER (or smaller after cancellation),
    // never 10⁵× larger.
    let dominant_terms = [
        0.5 * lams[0]
            * beta_flat.slice(s![0..p_time]).dot(&fast_av(
                &s_time,
                &beta_flat.slice(s![0..p_time]).to_owned(),
            )),
        0.5 * lams[1]
            * beta_flat.slice(s![p_time..p_time + p_marg]).dot(&fast_av(
                &s_marg_0,
                &beta_flat.slice(s![p_time..p_time + p_marg]).to_owned(),
            )),
        0.5 * lams[2]
            * beta_flat.slice(s![p_time..p_time + p_marg]).dot(&fast_av(
                &s_marg_1,
                &beta_flat.slice(s![p_time..p_time + p_marg]).to_owned(),
            )),
        0.5 * lams[3]
            * beta_flat.slice(s![p_time + p_marg..p_total]).dot(&fast_av(
                &s_logs,
                &beta_flat.slice(s![p_time + p_marg..p_total]).to_owned(),
            )),
    ];
    assert_eq!(
        projected.gradient.len(),
        dominant_terms.len(),
        "projected gradient dimension changed"
    );
    for (k, (&g, &dominant_term)) in projected
        .gradient
        .iter()
        .zip(dominant_terms.iter())
        .enumerate()
    {
        // Bound: trace pair adds ~p contributions, plus H⁻¹ Ḣ trace
        // bounded by Σ |λ_k| / |H_diag| × p ~ λ_k p / n ~ tiny at
        // large scale. Total gradient should be within 10× of the
        // dominant term (allowing for projection-correction sign).
        let bound = dominant_term.abs().max(1.0) * 100.0;
        assert!(g.is_finite(), "gradient[{k}] is non-finite: {g}");
        assert!(
            g.abs() <= bound,
            "gradient[{k}] = {:.6e} exceeds physical bound 100·|½λβ'Sβ| = {:.6e} \
                 (dominant_term={:.6e}); this reproduces the large-scale blowup \
                 inside joint_outer_evaluate.",
            g,
            bound,
            dominant_term
        );
    }
}

#[test]
pub(crate) fn direct_joint_hyper_inner_tolerance_follows_outer_target() {
    let options = BlockwiseFitOptions {
        inner_tol: 1e-6,
        outer_tol: 1e-5,
        inner_max_cycles: 100,
        ..BlockwiseFitOptions::default()
    };
    let (eval_options, strict_warm_start) =
        derivative_quality_options_and_warm_start(&options, None, true);

    assert_eq!(
        eval_options.inner_tol, options.outer_tol,
        "default exact joint-hyper eval should use the outer optimizer scale"
    );
    assert_eq!(eval_options.inner_max_cycles, options.inner_max_cycles);
    assert!(
        strict_warm_start.is_none(),
        "loosening to the outer scale should not discard cached inner state"
    );
    let large_scale_objective = 3.689e5;
    let posted_residual = 6.788e-1;
    let posted_objective_change = 4.209e-2;
    let eval_tol = eval_options.inner_tol * (1.0 + large_scale_objective);
    assert!(
        posted_residual <= 2.0 * eval_tol && posted_objective_change <= eval_tol,
        "the exact outer startup validation should accept numerically flat inner solves at outer scale"
    );
    let (rho_default, _) = derivative_quality_options_and_warm_start(&options, None, false);
    assert_eq!(
        rho_default.inner_tol, options.inner_tol,
        "rho-only exact joint-hyper eval must preserve the rho-only outer surface"
    );

    let tighter_options = BlockwiseFitOptions {
        inner_tol: 1e-3,
        outer_tol: 1e-5,
        inner_max_cycles: 100,
        ..BlockwiseFitOptions::default()
    };
    let (tightened, _) = derivative_quality_options_and_warm_start(&tighter_options, None, true);
    assert_eq!(tightened.inner_tol, tighter_options.outer_tol);
    assert_eq!(tightened.inner_max_cycles, 200);

    let (rho_only, _) = derivative_quality_options_and_warm_start(&tighter_options, None, false);
    assert_eq!(rho_only.inner_tol, tighter_options.inner_tol);
    assert_eq!(rho_only.inner_max_cycles, tighter_options.inner_max_cycles);

    let explicitly_tight_options = BlockwiseFitOptions {
        inner_tol: 1e-12,
        outer_tol: 1e-10,
        inner_max_cycles: 100,
        ..BlockwiseFitOptions::default()
    };
    let (explicitly_tight, _) =
        derivative_quality_options_and_warm_start(&explicitly_tight_options, None, true);
    assert_eq!(
        explicitly_tight.inner_tol, 1e-12,
        "an explicitly sub-default inner tolerance should be honored down to the explicit direct joint-hyper floor instead of being loosened to outer_tol"
    );
    assert_eq!(explicitly_tight.inner_max_cycles, 100);
}

#[test]
pub(crate) fn exact_spatial_joint_hyper_inner_tolerance_follows_spatial_outer_target() {
    let options = BlockwiseFitOptions {
        inner_tol: 1e-6,
        outer_tol: 1e-10,
        inner_max_cycles: 200,
        ..BlockwiseFitOptions::default()
    };
    let spatial_outer_tol = 1e-4;
    let eval_input = joint_hyper_options_for_outer_tolerance(&options, spatial_outer_tol);
    let (eval_options, strict_warm_start) =
        derivative_quality_options_and_warm_start(&eval_input, None, true);

    assert_eq!(eval_options.outer_tol, spatial_outer_tol);
    assert_eq!(
        eval_options.inner_tol, spatial_outer_tol,
        "exact spatial [rho, psi] evaluations should certify beta only to the tolerance of the outer optimizer consuming the derivative"
    );
    assert!(
        strict_warm_start.is_none(),
        "loosening an over-tight caller tolerance should preserve the cached inner state"
    );

    let large_scale_objective = 3.689e5;
    let posted_residual_plateau = 6.788e-1;
    let posted_objective_change = 4.209e-2;
    let eval_tol = eval_options.inner_tol * (1.0 + large_scale_objective);
    assert!(
        posted_residual_plateau <= eval_tol && posted_objective_change <= eval_tol,
        "the posted saturated Newton plateau is below the spatial outer derivative accuracy target"
    );
}

pub(crate) fn outerobjective_andgradient<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    penalty_counts: &[usize],
    rho: &Array1<f64>,
    warm_start: Option<&ConstrainedWarmStart>,
) -> Result<(f64, Array1<f64>, ConstrainedWarmStart), String> {
    let (obj, grad, _, warm) = super::test_support::outerobjectivegradienthessian(
        family,
        specs,
        options,
        penalty_counts,
        rho,
        warm_start,
        EvalMode::ValueAndGradient,
    )?;
    Ok((obj, grad, warm))
}

pub(crate) struct BinomialLocationScaleWiggleOuterFixture {
    pub(crate) family: BinomialLocationScaleWiggleFamily,
    pub(crate) specs: Vec<ParameterBlockSpec>,
    pub(crate) penalty_counts: Vec<usize>,
    pub(crate) rho: Array1<f64>,
    pub(crate) options: BlockwiseFitOptions,
}

pub(crate) fn binomial_location_scale_wiggle_outer_fixture()
-> BinomialLocationScaleWiggleOuterFixture {
    let base = binomial_location_scale_base_fixture();
    let q_seed = Array1::linspace(-1.4, 1.4, base.n);
    let knots =
        gam_terms::basis::initializewiggle_knots_from_seed(q_seed.view(), 3, 4).expect("knots");
    let wiggle_block =
        gam_models::wiggle::buildwiggle_block_input_from_knots(q_seed.view(), &knots, 3, 2, false)
            .expect("wiggle block");
    let wigglespec = ParameterBlockSpec {
        name: "wiggle".to_string(),
        design: wiggle_block.design.clone(),
        offset: wiggle_block.offset.clone(),
        penalties: wiggle_block
            .penalties
            .iter()
            .map(|ps| match ps {
                gam_solve::model_types::PenaltySpec::Block {
                    local, col_range, ..
                } => PenaltyMatrix::Blockwise {
                    local: local.clone(),
                    col_range: col_range.clone(),
                    total_dim: wiggle_block.design.ncols(),
                },
                gam_solve::model_types::PenaltySpec::Dense(m)
                | gam_solve::model_types::PenaltySpec::DenseWithMean { matrix: m, .. } => {
                    PenaltyMatrix::Dense(m.clone())
                }
            })
            .collect(),
        nullspace_dims: wiggle_block.nullspace_dims.clone(),
        initial_log_lambdas: array![0.1],
        initial_beta: Some(Array1::from_elem(wiggle_block.design.ncols(), 0.03)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let family = BinomialLocationScaleWiggleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: gam_problem::InverseLink::Standard(gam_problem::StandardLink::Probit),
        threshold_design: Some(base.threshold_design),
        log_sigma_design: Some(base.log_sigma_design),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    BinomialLocationScaleWiggleOuterFixture {
        family,
        specs: vec![base.threshold_spec, base.log_sigma_spec, wigglespec],
        penalty_counts: vec![1usize, 1usize, 1usize],
        rho: array![0.05, -0.15, 0.1],
        options: BlockwiseFitOptions {
            use_remlobjective: true,
            ridge_floor: 1e-10,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        },
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockIdentityFamily;

#[test]
pub(crate) fn joint_coupled_coefficient_hessian_cost_matches_n_times_p_total_squared() {
    // Three blocks p_b = (12, 20, 8), n=200. Joint-coupled cost is
    // n·(Σp_b)² = 200·40² = 320_000. Block-diagonal default with the
    // same designs would give n·Σp_b² = 200·(144+400+64) = 121_600.
    // The cross-block fill 2·n·(p_t·p_m + p_t·p_l + p_m·p_l) =
    // 2·200·(240+96+160) = 198_400 accounts for the difference.
    let mk_spec = |p: usize| ParameterBlockSpec {
        name: "test".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (200, p),
        ))),
        offset: Array1::zeros(200),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![mk_spec(12), mk_spec(20), mk_spec(8)];
    assert_eq!(
        joint_coupled_coefficient_hessian_cost(200, &specs),
        200 * 40 * 40
    );
    assert_eq!(
        default_coefficient_hessian_cost(&specs),
        200 * (144 + 400 + 64)
    );
    assert!(
        joint_coupled_coefficient_hessian_cost(200, &specs)
            > default_coefficient_hessian_cost(&specs)
    );
}

#[test]
pub(crate) fn large_scale_exact_adaptive_hessian_order_stays_second_order() {
    let n_train = 320_000u64;
    let p = 101usize;
    let retained_rho_dim = 3usize;
    let spec = ParameterBlockSpec {
        name: "matern60".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (1, p),
        ))),
        offset: Array1::zeros(1),
        penalties: (0..retained_rho_dim)
            .map(|_| PenaltyMatrix::Dense(Array2::eye(p)))
            .collect(),
        nullspace_dims: vec![0; retained_rho_dim],
        initial_log_lambdas: Array1::zeros(retained_rho_dim),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let coefficient_hessian_cost = n_train * (p as u64) * (p as u64);

    assert_eq!(coefficient_hessian_cost, 3_264_320_000);
    assert_eq!(
        retained_rho_dim as u64 * coefficient_hessian_cost,
        9_792_960_000
    );
    assert_eq!(
        exact_outer_order_from_capability(&[spec], coefficient_hessian_cost),
        ExactOuterDerivativeOrder::Second
    );
}

#[test]
pub(crate) fn use_joint_matrix_free_path_triggers_at_each_documented_threshold() {
    // p ≥ 512 is sufficient regardless of n.
    assert!(use_joint_matrix_free_path(512, 1));
    assert!(use_joint_matrix_free_path(2048, 4));
    assert!(!use_joint_matrix_free_path(511, 1));

    // n ≥ 50_000 AND p ≥ 128: both must hold. This keeps p≈51 FLEX
    // marginal-slope large-scale fits on the bounded dense-materialized path.
    assert!(use_joint_matrix_free_path(128, 50_000));
    assert!(!use_joint_matrix_free_path(127, 50_000));
    assert!(!use_joint_matrix_free_path(128, 31_249));
    assert!(!use_joint_matrix_free_path(51, 320_000));

    // n · p ≥ 4_000_000 is the linear-work fallback, but only after the
    // same moderate-p guard; below that, materializing `p` columns is a
    // deterministic small-p bound on expensive row-kernel HVPs.
    assert!(use_joint_matrix_free_path(128, 31_250));
    assert!(!use_joint_matrix_free_path(127, 31_497));

    // Below every threshold: dense path.
    assert!(!use_joint_matrix_free_path(8, 100));
    assert!(!use_joint_matrix_free_path(64, 1000));
}

#[test]
pub(crate) fn large_scale_shape_margslope_flex_cycle0_uses_bounded_dense_route() {
    let total_p = 51;
    let total_n = 320_000;
    let max_pcg_hvps_before_fix = JOINT_PCG_MAX_ITER_MULTIPLIER * total_p;

    assert_eq!(max_pcg_hvps_before_fix, 204);
    assert!(
        !use_joint_matrix_free_path(total_p, total_n),
        "p=51/n=320k should materialize exactly 51 columns instead of risking up to {max_pcg_hvps_before_fix} expensive PCG matvecs in cycle 0"
    );
}

pub(crate) struct CountingHessianWorkspace {
    pub(crate) dense_calls: Arc<AtomicUsize>,
    pub(crate) matvec_calls: Arc<AtomicUsize>,
    pub(crate) source_preference: JointHessianSourcePreference,
}

impl ExactNewtonJointHessianWorkspace for CountingHessianWorkspace {
    fn warm_up_outer_caches_for_mode(
        &self,
        eval_mode: gam_problem::EvalMode,
    ) -> Result<(), String> {
        match eval_mode {
            gam_problem::EvalMode::ValueOnly
            | gam_problem::EvalMode::ValueAndGradient
            | gam_problem::EvalMode::ValueGradientHessian => Ok(()),
        }
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        self.dense_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Array2::eye(2)))
    }

    fn hessian_source_preference(&self) -> JointHessianSourcePreference {
        self.source_preference
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        self.matvec_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(v.clone()))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(Array1::ones(2)))
    }

    fn directional_derivative(&self, arr: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }
}

#[test]
pub(crate) fn workspace_hessian_source_prefers_dense_without_zero_matvec_probe() {
    let dense_calls = Arc::new(AtomicUsize::new(0));
    let matvec_calls = Arc::new(AtomicUsize::new(0));
    let workspace: Arc<dyn ExactNewtonJointHessianWorkspace> = Arc::new(CountingHessianWorkspace {
        dense_calls: Arc::clone(&dense_calls),
        matvec_calls: Arc::clone(&matvec_calls),
        source_preference: JointHessianSourcePreference::Dense,
    });

    let source = exact_newton_joint_hessian_source_from_workspace(
        &workspace,
        2,
        MaterializationIntent::InnerSolve,
        "counting workspace",
    )
    .expect("hessian source should build")
    .expect("hessian source should be present");

    assert_eq!(dense_calls.load(Ordering::Relaxed), 1);
    assert_eq!(matvec_calls.load(Ordering::Relaxed), 0);
    match source {
        JointHessianSource::Dense(hessian) => assert_eq!(hessian, Array2::<f64>::eye(2)),
        JointHessianSource::Operator { .. } => panic!("dense source was not preferred"),
    }
    assert_eq!(matvec_calls.load(Ordering::Relaxed), 0);
}

#[test]
pub(crate) fn workspace_hessian_source_honors_operator_preference_before_dense_probe() {
    let dense_calls = Arc::new(AtomicUsize::new(0));
    let matvec_calls = Arc::new(AtomicUsize::new(0));
    let workspace: Arc<dyn ExactNewtonJointHessianWorkspace> = Arc::new(CountingHessianWorkspace {
        dense_calls: Arc::clone(&dense_calls),
        matvec_calls: Arc::clone(&matvec_calls),
        source_preference: JointHessianSourcePreference::Operator,
    });

    let source = exact_newton_joint_hessian_source_from_workspace(
        &workspace,
        2,
        MaterializationIntent::InnerSolve,
        "operator-preferred counting workspace",
    )
    .expect("hessian source should build")
    .expect("hessian source should be present");

    assert_eq!(
        dense_calls.load(Ordering::Relaxed),
        0,
        "operator-preferred source construction must not probe hessian_dense"
    );
    match source {
        JointHessianSource::Operator { apply, .. } => {
            let v = array![3.0, -2.0];
            assert_eq!(apply(&v).expect("operator apply should succeed"), v);
            assert_eq!(matvec_calls.load(Ordering::Relaxed), 1);
        }
        JointHessianSource::Dense(_) => panic!("operator source was not preferred"),
    }
}

pub(crate) struct InnerPreludeCountingWorkspace {
    pub(crate) dense_calls: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
pub(crate) enum FusedTrialWorkspaceOutcome {
    MissingWorkspace,
    MissingLogLikelihood,
    LogLikelihoodError,
    Value(f64),
}

pub(crate) struct FusedTrialWorkspace {
    pub(crate) outcome: FusedTrialWorkspaceOutcome,
}

impl ExactNewtonJointHessianWorkspace for FusedTrialWorkspace {
    fn warm_up_outer_caches_for_mode(&self, _: EvalMode) -> Result<(), String> {
        Ok(())
    }

    fn directional_derivative(&self, _: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn joint_log_likelihood_evaluation(&self) -> Result<Option<f64>, String> {
        match self.outcome {
            FusedTrialWorkspaceOutcome::MissingLogLikelihood => Ok(None),
            FusedTrialWorkspaceOutcome::LogLikelihoodError => {
                Err("fused-trial-log-likelihood-error".to_string())
            }
            FusedTrialWorkspaceOutcome::Value(value) => Ok(Some(value)),
            FusedTrialWorkspaceOutcome::MissingWorkspace => {
                unreachable!("missing-workspace outcome never constructs a workspace")
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct FusedTrialWorkspaceFamily {
    pub(crate) advertises_log_likelihood: bool,
    pub(crate) outcome: FusedTrialWorkspaceOutcome,
}

impl CustomFamily for FusedTrialWorkspaceFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        unreachable!("the fused-trial contract test never evaluates the scalar family")
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        _: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        match self.outcome {
            FusedTrialWorkspaceOutcome::MissingWorkspace => Ok(None),
            outcome => Ok(Some(Arc::new(FusedTrialWorkspace { outcome }))),
        }
    }

    fn inner_joint_workspace_log_likelihood_available(&self, _: &[ParameterBlockSpec]) -> bool {
        self.advertises_log_likelihood
    }
}

#[test]
pub(crate) fn fused_trial_scalar_fallback_requires_absent_capability() {
    let family = FusedTrialWorkspaceFamily {
        advertises_log_likelihood: false,
        outcome: FusedTrialWorkspaceOutcome::MissingWorkspace,
    };
    assert!(
        joint_line_search_log_likelihood_with_workspace(
            &family,
            &BlockwiseFitOptions::default(),
            &[],
            &[],
        )
        .expect("absent capability is not an error")
        .is_none(),
    );
}

#[test]
pub(crate) fn fused_trial_advertised_missing_workspace_fails_closed() {
    let family = FusedTrialWorkspaceFamily {
        advertises_log_likelihood: true,
        outcome: FusedTrialWorkspaceOutcome::MissingWorkspace,
    };
    let error = joint_line_search_log_likelihood_with_workspace(
        &family,
        &BlockwiseFitOptions::default(),
        &[],
        &[],
    )
    .expect_err("an advertised fused workspace is mandatory");
    assert!(error.contains("returned no workspace"), "unexpected error: {error}");
}

#[test]
pub(crate) fn fused_trial_advertised_missing_likelihood_fails_closed() {
    let family = FusedTrialWorkspaceFamily {
        advertises_log_likelihood: true,
        outcome: FusedTrialWorkspaceOutcome::MissingLogLikelihood,
    };
    let error = joint_line_search_log_likelihood_with_workspace(
        &family,
        &BlockwiseFitOptions::default(),
        &[],
        &[],
    )
    .expect_err("an advertised fused likelihood is mandatory");
    assert!(
        error.contains("returned no log-likelihood"),
        "unexpected error: {error}",
    );
}

#[test]
pub(crate) fn fused_trial_workspace_error_is_not_scalarized() {
    let family = FusedTrialWorkspaceFamily {
        advertises_log_likelihood: true,
        outcome: FusedTrialWorkspaceOutcome::LogLikelihoodError,
    };
    assert_eq!(
        fused_first_attempt_log_likelihood(
            &family,
            &BlockwiseFitOptions::default(),
            &[],
            &[],
            0,
            true,
        )
        .expect_err("the trust-attempt gate must propagate workspace evaluation errors"),
        "fused-trial-log-likelihood-error",
    );
}

#[test]
pub(crate) fn fused_trial_returns_workspace_and_exact_likelihood_together() {
    let family = FusedTrialWorkspaceFamily {
        advertises_log_likelihood: true,
        outcome: FusedTrialWorkspaceOutcome::Value(-3.25),
    };
    let (value, _) = joint_line_search_log_likelihood_with_workspace(
        &family,
        &BlockwiseFitOptions::default(),
        &[],
        &[],
    )
    .expect("advertised workspace should evaluate")
    .expect("advertised capability must return fused evidence");
    assert_eq!(value.to_bits(), (-3.25_f64).to_bits());
}

impl ExactNewtonJointHessianWorkspace for InnerPreludeCountingWorkspace {
    fn warm_up_outer_caches_for_mode(&self, _: EvalMode) -> Result<(), String> {
        Ok(())
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        self.dense_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(array![[1.0]]))
    }

    fn directional_derivative(&self, _: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }
}

#[derive(Clone)]
pub(crate) struct InnerPreludeWorkspaceFamily {
    pub(crate) evaluations: Arc<AtomicUsize>,
    pub(crate) workspace_builds: Arc<AtomicUsize>,
    pub(crate) dense_calls: Arc<AtomicUsize>,
    pub(crate) provide_workspace: bool,
    pub(crate) advertise_workspace_gradient: bool,
}

impl CustomFamily for InnerPreludeWorkspaceFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![0.0],
                hessian: SymmetricMatrix::Dense(array![[1.0]]),
            }],
        })
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        self.workspace_builds.fetch_add(1, Ordering::Relaxed);
        if !self.provide_workspace {
            return Ok(None);
        }
        Ok(Some(Arc::new(InnerPreludeCountingWorkspace {
            dense_calls: Arc::clone(&self.dense_calls),
        })))
    }

    fn exact_newton_joint_gradient_evaluation(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood: 0.0,
            gradient: array![0.0],
        }))
    }

    fn inner_coefficient_hessian_hvp_available(&self, _: &[ParameterBlockSpec]) -> bool {
        true
    }

    fn inner_joint_workspace_gradient_available(&self, _: &[ParameterBlockSpec]) -> bool {
        self.advertise_workspace_gradient
    }

    fn joint_trust_metric_block_floor(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Array1<f64>>, String> {
        Err("inner-prelude-workspace-cycle0-reached".to_string())
    }
}

#[test]
pub(crate) fn inner_workspace_prevalidation_reuses_cycle0_hessian_without_family_replay() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let workspace_builds = Arc::new(AtomicUsize::new(0));
    let dense_calls = Arc::new(AtomicUsize::new(0));
    let family = InnerPreludeWorkspaceFamily {
        evaluations: Arc::clone(&evaluations),
        workspace_builds: Arc::clone(&workspace_builds),
        dense_calls: Arc::clone(&dense_calls),
        provide_workspace: true,
        advertise_workspace_gradient: false,
    };
    let spec = ParameterBlockSpec {
        name: "workspace".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };

    let error = inner_blockwise_fit(&family, &[spec], &[Array1::zeros(0)], &options, None)
        .expect_err("fixture stops immediately after cycle-0 consumes its Hessian source");
    assert_eq!(error, "inner-prelude-workspace-cycle0-reached");
    assert_eq!(
        evaluations.load(Ordering::Relaxed),
        0,
        "workspace curvature is authoritative; prevalidation must not replay family.evaluate",
    );
    assert_eq!(
        workspace_builds.load(Ordering::Relaxed),
        1,
        "gradient loading and cycle 0 must retain one workspace at the same beta",
    );
    assert_eq!(
        dense_calls.load(Ordering::Relaxed),
        1,
        "prevalidation must hand its exact dense source to cycle 0 instead of materializing it twice",
    );
}

#[test]
pub(crate) fn advertised_inner_workspace_missing_fails_closed_without_family_fallback() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let workspace_builds = Arc::new(AtomicUsize::new(0));
    let family = InnerPreludeWorkspaceFamily {
        evaluations: Arc::clone(&evaluations),
        workspace_builds: Arc::clone(&workspace_builds),
        dense_calls: Arc::new(AtomicUsize::new(0)),
        provide_workspace: false,
        advertise_workspace_gradient: false,
    };
    let spec = ParameterBlockSpec {
        name: "missing-workspace".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let error = inner_blockwise_fit(
        &family,
        &[spec],
        &[Array1::zeros(0)],
        &BlockwiseFitOptions::default(),
        None,
    )
    .expect_err("an advertised workspace source must not silently fall back");
    assert!(
        error.contains("requested an exact Hessian workspace, but the family returned none"),
        "unexpected missing-workspace error: {error}",
    );
    assert_eq!(workspace_builds.load(Ordering::Relaxed), 1);
    assert_eq!(
        evaluations.load(Ordering::Relaxed),
        0,
        "missing authoritative curvature must fail before family.evaluate can create a second source",
    );
}

#[test]
pub(crate) fn advertised_workspace_gradient_missing_fails_before_row_measure_fallback() {
    let evaluations = Arc::new(AtomicUsize::new(0));
    let workspace_builds = Arc::new(AtomicUsize::new(0));
    let family = InnerPreludeWorkspaceFamily {
        evaluations: Arc::clone(&evaluations),
        workspace_builds: Arc::clone(&workspace_builds),
        dense_calls: Arc::new(AtomicUsize::new(0)),
        provide_workspace: true,
        advertise_workspace_gradient: true,
    };
    let spec = ParameterBlockSpec {
        name: "missing-workspace-gradient".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let error = inner_blockwise_fit(
        &family,
        &[spec],
        &[Array1::zeros(0)],
        &BlockwiseFitOptions::default(),
        None,
    )
    .expect_err("an advertised workspace gradient must not fall back to a different row measure");
    assert!(
        error.contains(
            "advertises inner joint workspace gradients, but its workspace returned none"
        ),
        "unexpected missing-workspace-gradient error: {error}",
    );
    assert_eq!(workspace_builds.load(Ordering::Relaxed), 1);
    assert_eq!(
        evaluations.load(Ordering::Relaxed),
        0,
        "missing workspace-gradient authority must fail before family.evaluate can mix row measures",
    );
}

/// A workspace that exposes both a dense build and a matrix-free HVP and
/// refines its representation per intent (#738): matrix-free for the inner
/// solve, dense for logdet factorization. Mirrors CTN's contract.
pub(crate) struct IntentRefiningHessianWorkspace {
    pub(crate) dense_calls: Arc<AtomicUsize>,
    pub(crate) matvec_calls: Arc<AtomicUsize>,
}

impl ExactNewtonJointHessianWorkspace for IntentRefiningHessianWorkspace {
    fn warm_up_outer_caches_for_mode(
        &self,
        eval_mode: gam_problem::EvalMode,
    ) -> Result<(), String> {
        match eval_mode {
            gam_problem::EvalMode::ValueOnly
            | gam_problem::EvalMode::ValueAndGradient
            | gam_problem::EvalMode::ValueGradientHessian => Ok(()),
        }
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        self.dense_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Array2::eye(2)))
    }

    fn hessian_source_preference(&self) -> JointHessianSourcePreference {
        JointHessianSourcePreference::Operator
    }

    fn hessian_source_preference_for_intent(
        &self,
        intent: MaterializationIntent,
    ) -> JointHessianSourcePreference {
        match intent {
            MaterializationIntent::LogdetFactorization => JointHessianSourcePreference::Dense,
            MaterializationIntent::InnerSolve
            | MaterializationIntent::OuterEvaluation
            | MaterializationIntent::OuterGradient => JointHessianSourcePreference::Operator,
        }
    }

    fn hessian_matvec_available(&self) -> bool {
        true
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        self.matvec_calls.fetch_add(1, Ordering::Relaxed);
        Ok(Some(v.clone()))
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(Array1::ones(2)))
    }

    fn directional_derivative(&self, arr: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(None)
    }
}

#[test]
pub(crate) fn logdet_intent_takes_dense_while_inner_solve_takes_operator() {
    let dense_calls = Arc::new(AtomicUsize::new(0));
    let matvec_calls = Arc::new(AtomicUsize::new(0));
    let workspace: Arc<dyn ExactNewtonJointHessianWorkspace> =
        Arc::new(IntentRefiningHessianWorkspace {
            dense_calls: Arc::clone(&dense_calls),
            matvec_calls: Arc::clone(&matvec_calls),
        });

    // Logdet factorization intent: the consumer factorizes H + S_lambda,
    // so the workspace hands back the structural dense build directly,
    // probing hessian_dense and skipping the operator wrapper.
    let logdet_source = exact_newton_joint_hessian_source_from_workspace(
        &workspace,
        2,
        MaterializationIntent::LogdetFactorization,
        "intent-refining logdet",
    )
    .expect("logdet source should build")
    .expect("logdet source should be present");
    assert_eq!(dense_calls.load(Ordering::Relaxed), 1);
    assert_eq!(matvec_calls.load(Ordering::Relaxed), 0);
    match logdet_source {
        JointHessianSource::Dense(hessian) => assert_eq!(hessian, Array2::<f64>::eye(2)),
        JointHessianSource::Operator { .. } => {
            panic!("logdet intent must take the dense representation")
        }
    }

    // Inner solve intent: only H · v is applied, so the same workspace
    // hands back the matrix-free operator without touching hessian_dense.
    let inner_source = exact_newton_joint_hessian_source_from_workspace(
        &workspace,
        2,
        MaterializationIntent::InnerSolve,
        "intent-refining inner solve",
    )
    .expect("inner source should build")
    .expect("inner source should be present");
    assert_eq!(
        dense_calls.load(Ordering::Relaxed),
        1,
        "inner-solve intent must not probe hessian_dense"
    );
    match inner_source {
        JointHessianSource::Operator { apply, .. } => {
            let v = array![1.5, -4.0];
            assert_eq!(apply(&v).expect("operator apply should succeed"), v);
            assert_eq!(matvec_calls.load(Ordering::Relaxed), 1);
        }
        JointHessianSource::Dense(_) => {
            panic!("inner-solve intent must take the operator representation")
        }
    }
}

#[test]
pub(crate) fn default_coefficient_gradient_cost_is_half_of_hessian_cost() {
    // The gradient-only sweep through the inner Newton solve does
    // roughly half the per-evaluation arithmetic of the full Hessian
    // assembly path (skips K-fold pairwise B_{j,k} blocks and K-fold
    // inner derivative solves). The default trait method preserves
    // this 2× ratio; families that override `coefficient_hessian_cost`
    // (e.g. GAMLSS via `joint_coupled_coefficient_hessian_cost`)
    // automatically inherit a consistent gradient-cost scaling without
    // a per-family override.
    let mk_spec = |n: usize, p: usize| ParameterBlockSpec {
        name: "test".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (n, p),
        ))),
        offset: Array1::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![mk_spec(500, 10), mk_spec(500, 14)];
    let h_cost = default_coefficient_hessian_cost(&specs);
    let g_cost = default_coefficient_gradient_cost(&specs);
    assert_eq!(h_cost, 500 * 100 + 500 * 196);
    assert_eq!(g_cost, h_cost / 2);
}

#[test]
pub(crate) fn custom_family_default_outer_seed_config_is_tightened_for_expensive_paths() {
    let family = OneBlockIdentityFamily;

    let small = family.outer_seed_config(4);
    assert_eq!(small.max_seeds, 6);
    assert_eq!(small.seed_budget, 1);
    assert_eq!(small.screen_max_inner_iterations, 2);

    let large = family.outer_seed_config(16);
    assert_eq!(large.max_seeds, 4);
    assert_eq!(large.seed_budget, 1);
    assert_eq!(large.screen_max_inner_iterations, 2);
}

#[test]
pub(crate) fn finite_working_weight_certificate_preserves_zero_tiny_and_signed_rows_bit_exactly() {
    let weights = array![0.0, f64::from_bits(1), 1.0e-16, -1.0e-9, 0.25];
    let certified = certify_finite_working_weights(&weights).expect("finite signed weights");
    assert!(std::ptr::eq(certified, &weights));
    for (actual, expected) in certified.iter().zip(weights.iter()) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
pub(crate) fn finite_working_weight_certificate_rejects_nonfinite_rows_atomically() {
    let nan = array![0.5, f64::NAN];
    let err = certify_finite_working_weights(&nan).expect_err("NaN curvature must be rejected");
    assert!(err.contains("row 1"), "error should name the row: {err}");

    let inf = array![f64::INFINITY, 0.5];
    certify_finite_working_weights(&inf).expect_err("infinite curvature must be rejected");
}

#[test]
pub(crate) fn screened_outer_warm_start_reuses_any_matching_rho_dimension() {
    let rho_far = array![2.25, -0.5];
    let cache = Some(ConstrainedWarmStart {
        rho: array![0.0, -0.5],
        block_beta: vec![array![1.0, -1.0]],
        active_sets: vec![None],
        cached_inner: None,
    });

    let retained = screened_outer_warm_start(cache.as_ref(), &rho_far)
        .expect("matching-dimension warm starts should remain reusable");
    assert_eq!(retained.rho, array![0.0, -0.5]);
    assert_eq!(retained.block_beta[0], array![1.0, -1.0]);
    assert_eq!(retained.active_sets[0], None);
}

#[test]
pub(crate) fn cached_beta_warm_start_splits_blocks_and_validates_shape() {
    let mk_spec = |name: &str, p: usize| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (3, p),
        ))),
        offset: Array1::zeros(3),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![mk_spec("a", 2), mk_spec("b", 3)];

    let warm = constrained_warm_start_from_cached_beta(4, &specs, &array![1., 2., 3., 4., 5.])
        .expect("matching beta");
    assert_eq!(warm.rho.len(), 4);
    assert_eq!(warm.block_beta, vec![array![1., 2.], array![3., 4., 5.]]);
    assert_eq!(warm.active_sets, vec![None, None]);
    assert!(warm.cached_inner.is_none());

    let err = match constrained_warm_start_from_cached_beta(4, &specs, &array![1., 2., 3.]) {
        Ok(_) => panic!("wrong beta length should be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("cached inner beta has length 3, but custom-family blocks require length 5"),
        "{err}"
    );
}

#[test]
pub(crate) fn cached_beta_warm_start_rejects_nonfinite_entries() {
    let spec = ParameterBlockSpec {
        name: "a".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (3, 2),
        ))),
        offset: Array1::zeros(3),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let err = match constrained_warm_start_from_cached_beta(1, &[spec], &array![1.0, f64::NAN]) {
        Ok(_) => panic!("non-finite beta should be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("cached inner beta contains non-finite entries"),
        "{err}"
    );
}

#[test]
pub(crate) fn custom_outer_state_reset_preserves_seeded_cached_beta() {
    let spec = ParameterBlockSpec {
        name: "a".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::zeros(
            (3, 2),
        ))),
        offset: Array1::zeros(3),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let mut state = CustomOuterState::new(None);
    state
        .seed_cached_beta(1, &[spec], &array![4.0, -2.0])
        .expect("cached beta seed");

    state.warm_cache = None;
    state.reset();

    let warm = state
        .warm_cache
        .as_ref()
        .expect("reset should restore cached beta seed");
    assert_eq!(warm.rho.len(), 1);
    assert_eq!(warm.block_beta, vec![array![4.0, -2.0]]);
    assert!(warm.cached_inner.is_none());
}

#[test]
pub(crate) fn custom_outer_state_reset_preserves_existing_persistent_warm_start() {
    let persistent = ConstrainedWarmStart {
        rho: array![0.25],
        block_beta: vec![array![1.0, 2.0]],
        active_sets: vec![None],
        cached_inner: None,
    };
    let mut state = CustomOuterState::new(Some(persistent.clone()));

    state.warm_cache = None;
    state.reset();

    let warm = state
        .warm_cache
        .as_ref()
        .expect("reset should restore persistent warm start");
    assert_eq!(warm.rho, persistent.rho);
    assert_eq!(warm.block_beta, persistent.block_beta);
}

#[test]
pub(crate) fn public_warm_start_compatibility_checks_rho_dimension() {
    let warm = CustomFamilyWarmStart {
        inner: ConstrainedWarmStart {
            rho: array![0.0, -0.5],
            block_beta: vec![array![1.0, -1.0]],
            active_sets: vec![None],
            cached_inner: None,
        },
    };

    assert!(warm.compatible_with_rho(&array![0.75, -0.5]));
    assert!(warm.compatible_with_rho(&array![1.75, -0.5]));
    assert!(!warm.compatible_with_rho(&array![0.0]));
}

#[test]
pub(crate) fn psi_drift_deriv_workspace_preserves_block_local_operator() {
    #[derive(Clone)]
    struct ZeroFamily;

    impl CustomFamily for ZeroFamily {
        fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![],
            })
        }
    }

    struct BlockLocalPsiWorkspace;

    impl ExactNewtonJointPsiWorkspace for BlockLocalPsiWorkspace {
        fn second_order_terms(
            &self,
            _: usize,
            _: usize,
        ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
            // Default implementation ignores this parameter.
            // Default implementation ignores this parameter.
            Ok(None)
        }

        fn hessian_directional_derivative(
            &self,
            psi_index: usize,
            arr: &Array1<f64>,
        ) -> Result<Option<DriftDerivResult>, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            assert_eq!(psi_index, 0);
            Ok(Some(DriftDerivResult::Operator(Arc::new(
                BlockLocalDrift {
                    local: array![[3.0, 1.0], [1.0, 2.0]],
                    start: 1,
                    end: 3,
                    total_dim: 3,
                },
            ))))
        }
    }

    let callback = build_psi_drift_deriv_callback(
        &ZeroFamily,
        &[],
        &[],
        Arc::new(Vec::new()),
        false,
        Some(Arc::new(BlockLocalPsiWorkspace)),
    )
    .expect("non-Gaussian psi drift callback should be available");

    let result = callback(0, &array![1.0, 2.0, 3.0])
        .expect("workspace-backed psi drift derivative should be returned");

    match result {
        DriftDerivResult::Dense(_) => {
            panic!("workspace-backed block-local psi drift derivative was densified")
        }
        DriftDerivResult::Operator(op) => {
            let (local, start, end) = op
                .block_local_data()
                .expect("block-local operator metadata should be preserved");
            assert_eq!((start, end), (1, 3));
            assert_eq!(local, &array![[3.0, 1.0], [1.0, 2.0]]);
        }
    }
}

#[test]
pub(crate) fn contracted_psi_hook_declines_partial_axis_coverage_before_pair_tables_are_skipped() {
    struct PartialContractedPsiWorkspace;

    impl ExactNewtonJointPsiWorkspace for PartialContractedPsiWorkspace {
        fn second_order_terms(
            &self,
            _: usize,
            _: usize,
        ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
            // Default implementation ignores this parameter.
            // Default implementation ignores this parameter.
            Ok(None)
        }

        fn second_order_terms_contracted(
            &self,
            alpha_psi: &[f64],
        ) -> Result<Option<ExactNewtonJointPsiSecondOrderContracted>, String> {
            if alpha_psi.get(1).copied().unwrap_or(0.0) != 0.0 {
                return Ok(None);
            }
            let psi_dim = alpha_psi.len();
            Ok(Some(ExactNewtonJointPsiSecondOrderContracted {
                objective: Array1::zeros(psi_dim),
                score: Array2::zeros((psi_dim, 1)),
                hessian: (0..psi_dim)
                    .map(|_| DriftDerivResult::Dense(Array2::zeros((1, 1))))
                    .collect(),
            }))
        }

        fn hessian_directional_derivative(
            &self,
            _: usize,
            d_beta_flat: &Array1<f64>,
        ) -> Result<Option<DriftDerivResult>, String> {
            // Default implementation ignores this parameter.
            assert_eq!(d_beta_flat.len(), 1);
            Ok(None)
        }
    }

    let specs = vec![ParameterBlockSpec {
        name: "partial".to_string(),
        design: DesignMatrix::from(Array2::ones((1, 1))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let derivative_blocks = Arc::new(vec![vec![
        CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::zeros((1, 1)),
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        ),
        CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::zeros((1, 1)),
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        ),
    ]]);
    let hook = build_contracted_psi_hook(
        &specs,
        derivative_blocks,
        &array![0.0],
        &[],
        &[0],
        None,
        Some(Arc::new(PartialContractedPsiWorkspace)),
        None,
    )
    .expect("partial contracted psi hook probe should not error");

    assert!(
        hook.is_none(),
        "partial contracted psi coverage must keep the exact per-pair assembly path"
    );
}

#[test]
pub(crate) fn contracted_psi_hook_rejects_wrong_score_width_before_installing_operator_hook() {
    struct WrongScoreWidthPsiWorkspace;

    impl ExactNewtonJointPsiWorkspace for WrongScoreWidthPsiWorkspace {
        fn second_order_terms(
            &self,
            _: usize,
            _: usize,
        ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
            // Default implementation ignores this parameter.
            // Default implementation ignores this parameter.
            Ok(None)
        }

        fn second_order_terms_contracted(
            &self,
            alpha_psi: &[f64],
        ) -> Result<Option<ExactNewtonJointPsiSecondOrderContracted>, String> {
            let psi_dim = alpha_psi.len();
            Ok(Some(ExactNewtonJointPsiSecondOrderContracted {
                objective: Array1::zeros(psi_dim),
                score: Array2::zeros((psi_dim, 0)),
                hessian: (0..psi_dim)
                    .map(|_| DriftDerivResult::Dense(Array2::zeros((1, 1))))
                    .collect(),
            }))
        }

        fn hessian_directional_derivative(
            &self,
            _: usize,
            d_beta_flat: &Array1<f64>,
        ) -> Result<Option<DriftDerivResult>, String> {
            // Default implementation ignores this parameter.
            assert_eq!(d_beta_flat.len(), 1);
            Ok(None)
        }
    }

    let specs = vec![ParameterBlockSpec {
        name: "wrong-score-width".to_string(),
        design: DesignMatrix::from(Array2::ones((1, 1))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let derivative_blocks = Arc::new(vec![vec![CustomFamilyBlockPsiDerivative::new(
        None,
        Array2::zeros((1, 1)),
        Array2::zeros((1, 1)),
        None,
        None,
        None,
        None,
    )]]);

    let err = match build_contracted_psi_hook(
        &specs,
        derivative_blocks,
        &array![0.0],
        &[],
        &[0],
        None,
        Some(Arc::new(WrongScoreWidthPsiWorkspace)),
        None,
    ) {
        Ok(_) => panic!("wrong contracted score width must be rejected before hook install"),
        Err(err) => err,
    };

    assert!(
        err.contains("score=1x0") && err.contains("beta_dim=1"),
        "unexpected wrong-score-width error: {err}"
    );
}

#[test]
pub(crate) fn custom_family_outer_derivatives_respects_missing_second_order_capability() {
    #[derive(Clone)]
    struct OneBlockFirstOrderOnlyFamily;

    impl CustomFamily for OneBlockFirstOrderOnlyFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n),
                    working_weights: Array1::ones(n),
                }],
            })
        }

        fn exact_outer_derivative_order(
            &self,
            _: &[ParameterBlockSpec],
            _: &BlockwiseFitOptions,
        ) -> ExactOuterDerivativeOrder {
            ExactOuterDerivativeOrder::First
        }
    }

    let specs = vec![ParameterBlockSpec {
        name: "x".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let (gradient, hessian) = custom_family_outer_derivatives(
        &OneBlockFirstOrderOnlyFamily,
        &specs,
        &BlockwiseFitOptions::default(),
    );
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Unavailable);
}

#[derive(Clone)]
pub(crate) struct DefaultDiagonalExactHookFamily;

impl CustomFamily for DefaultDiagonalExactHookFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let eta = block_states[0].eta.clone();
        let weights = eta.mapv(|value| 2.0 + value * value);
        Ok(FamilyEvaluation {
            log_likelihood: -0.5 * eta.dot(&eta),
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: Array1::zeros(eta.len()),
                working_weights: weights,
            }],
        })
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        // Default implementation ignores this parameter.
        Ok(Some((&block_states[0].eta * d_eta) * 2.0))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let spec = default_diagonal_exact_hook_spec();
        let u_eta = spec.design.apply(u);
        let v_eta = spec.design.apply(v);
        assert_eq!(block_states[0].eta.len(), u_eta.len());
        spec.design
            .xt_diag_x_signed_op(
                FiniteSignedWeightsView::try_from_array(&((&u_eta * &v_eta) * 2.0)).unwrap(),
            )
            .map(Some)
    }
}

pub(crate) fn default_diagonal_exact_hook_spec() -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "default_exact".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.5],
            [0.0, 1.0],
            [2.0, -1.0]
        ])),
        offset: Array1::zeros(3),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(2))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.2, -0.1]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

#[test]
pub(crate) fn default_custom_family_exact_hessian_hooks_assemble_diagonal_working_sets() {
    let family = DefaultDiagonalExactHookFamily;
    let spec = default_diagonal_exact_hook_spec();
    let beta = array![0.2, -0.1];
    let eta = spec.design.apply(&beta);
    let states = vec![ParameterBlockState {
        beta: beta.clone(),
        eta: eta.clone(),
    }];

    let h = family
        .exact_newton_joint_hessian_with_specs(&states, &[spec.clone()])
        .expect("default joint Hessian hook should succeed")
        .expect("diagonal working sets should assemble an exact joint Hessian");
    let expected_h = spec
        .design
        .xt_diag_x_signed_op(
            FiniteSignedWeightsView::try_from_array(&eta.mapv(|value| 2.0 + value * value))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(h, expected_h);

    let direction = array![0.3, -0.4];
    let dh = family
        .exact_newton_joint_hessian_directional_derivative_with_specs(
            &states,
            &[spec.clone()],
            &direction,
        )
        .expect("default joint dH hook should succeed")
        .expect("diagonal weight derivative should assemble an exact joint dH");
    let d_eta = spec.design.apply(&direction);
    let expected_dh = spec
        .design
        .xt_diag_x_signed_op(
            FiniteSignedWeightsView::try_from_array(&((&eta * &d_eta) * 2.0)).unwrap(),
        )
        .unwrap();
    assert_eq!(dh, expected_dh);

    let d2h = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &direction, &beta)
        .expect("family second directional hook should succeed")
        .expect("second directional hook should be exact");
    let beta_eta = spec.design.apply(&beta);
    let expected_d2h = spec
        .design
        .xt_diag_x_signed_op(
            FiniteSignedWeightsView::try_from_array(&((&d_eta * &beta_eta) * 2.0)).unwrap(),
        )
        .unwrap();
    assert_eq!(d2h, expected_d2h);
}

#[test]
pub(crate) fn default_custom_family_exact_hessian_hooks_drive_profiled_outer_hessian() {
    let mut spec = default_diagonal_exact_hook_spec();
    spec.initial_beta = Some(Array1::zeros(2));
    let specs = [spec];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        inner_max_cycles: 1,
        ..BlockwiseFitOptions::default()
    };
    let penalty_derivatives = [vec![]];
    let rho = array![0.0];
    let result = evaluate_custom_family_joint_hyper(
        &DefaultDiagonalExactHookFamily,
        &specs,
        &options,
        &rho,
        &penalty_derivatives,
        None,
        EvalMode::ValueGradientHessian,
    )
    .expect("profiled outer Hessian should use default exact Hessian hooks");

    assert_eq!(result.gradient.len(), 1);
    let analytic = match &result.outer_hessian {
        gam_problem::HessianValue::Dense(hessian) => {
            assert_eq!(hessian.dim(), (1, 1));
            hessian[[0, 0]]
        }
        _ => panic!("outer Hessian should be analytic"),
    };

    let h = 1e-5;
    let gradient_at = |rho_value: f64| {
        evaluate_custom_family_joint_hyper(
            &DefaultDiagonalExactHookFamily,
            &specs,
            &options,
            &array![rho_value],
            &penalty_derivatives,
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("profiled outer gradient")
        .gradient[0]
    };
    let finite_difference = (gradient_at(h) - gradient_at(-h)) / (2.0 * h);
    assert!(
        (analytic - finite_difference).abs() <= 2e-3 * finite_difference.abs().max(1.0),
        "default-hook outer Hessian: analytic={analytic}, finite_difference={finite_difference}"
    );
}

#[test]
pub(crate) fn nonconverged_inner_refuses_profile_derivatives() {
    let spec = default_diagonal_exact_hook_spec();
    let result = evaluate_custom_family_joint_hyper(
        &DefaultDiagonalExactHookFamily,
        &[spec],
        &BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            compute_covariance: false,
            inner_max_cycles: 1,
            ..BlockwiseFitOptions::default()
        },
        &array![0.0],
        &[vec![]],
        None,
        EvalMode::ValueGradientHessian,
    );

    let err = match result {
        Ok(_) => panic!("non-converged inner solve must not expose derivatives"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("inner solve did not converge") && msg.contains("refusing to expose"),
        "unexpected error: {msg}"
    );
}

#[test]
pub(crate) fn custom_family_seed_screening_proxy_accepts_finite_partial_inner_fit() {
    let specs = vec![default_diagonal_exact_hook_spec()];
    let penalty_counts = validate_blockspecs(&specs).expect("valid test spec");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid label layout");
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        inner_max_cycles: 1,
        ..BlockwiseFitOptions::default()
    };

    let (score, warm_start, inner_converged) = custom_family_seed_screening_proxy_labeled(
        &DefaultDiagonalExactHookFamily,
        &specs,
        &options,
        &layout,
        &array![0.0],
        None,
        &gam_problem::RhoPrior::Flat,
    )
    .expect("screening proxy should score a finite partial inner solve");

    assert!(score.is_finite());
    assert!(
        !inner_converged,
        "one-cycle screening is expected to be a partial inner fit"
    );
    assert_eq!(warm_start.rho, array![0.0]);
    assert_eq!(warm_start.block_beta.len(), 1);
}

#[test]
pub(crate) fn custom_family_outer_derivatives_exposes_surrogate_second_order_geometry() {
    // RidgedQuadraticReml is the default objective; its analytic outer
    // Hessian is routed to ARC, which handles indefinite Hessians via
    // cubic regularization. The previous behavior forced these families
    // onto BFGS+BfgsApprox and caused benchmark hangs at iter 0.
    #[derive(Clone)]
    struct SurrogateFamily;

    impl CustomFamily for SurrogateFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n),
                    working_weights: Array1::ones(n),
                }],
            })
        }
    }

    let specs = vec![ParameterBlockSpec {
        name: "x".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        ..BlockwiseFitOptions::default()
    };
    let (gradient, hessian) = custom_family_outer_derivatives(&SurrogateFamily, &specs, &options);
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Either);
}

#[test]
pub(crate) fn custom_family_outer_derivatives_keeps_strict_second_order_geometry() {
    #[derive(Clone)]
    struct StrictFamily;

    impl CustomFamily for StrictFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n),
                    working_weights: Array1::ones(n),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::StrictPseudoLaplace
        }
    }

    let specs = vec![ParameterBlockSpec {
        name: "x".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        ..BlockwiseFitOptions::default()
    };
    let (gradient, hessian) = custom_family_outer_derivatives(&StrictFamily, &specs, &options);
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Either);
}

#[derive(Clone)]
pub(crate) struct OneBlockQuarticExactFamily {
    pub(crate) linear: f64,
    pub(crate) curvature: f64,
    pub(crate) second_scale: f64,
}

impl CustomFamily for OneBlockQuarticExactFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        // h(β) = 1 + curvature·β² genuinely depends on β; the default
        // (false for RidgedQuadraticReml) would short-circuit the joint
        // d²H aggregator to zeros and drop the per-block override below
        // before it ever reaches the outer Hessian's drift contribution.
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states[0].beta[0];
        let log_likelihood =
            self.linear * beta - 0.5 * beta * beta - self.curvature * beta.powi(4) / 12.0;
        let gradient = self.linear - beta - self.curvature * beta.powi(3) / 3.0;
        let hessian = 1.0 + self.curvature * beta * beta;
        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![gradient],
                hessian: SymmetricMatrix::Dense(array![[hessian]]),
            }],
        })
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        direction: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0);
        let beta = block_states[0].beta[0];
        Ok(Some(array![[2.0 * self.curvature * beta * direction[0]]]))
    }

    fn exact_newton_hessian_second_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        u: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_idx, 0);
        let value = 2.0 * self.curvature * self.second_scale * u[0] * v[0];
        Ok(Some(array![[value]]))
    }
}

#[test]
pub(crate) fn generic_single_block_fallback_includes_nonzero_d2h_drift() {
    let spec = ParameterBlockSpec {
        name: "quartic".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.75]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_tol: 1e-11,
        use_remlobjective: true,
        use_outer_hessian: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let penalty_counts = vec![1];
    let rho = array![0.0];

    let with_d2 = evaluate_custom_family_hyper_internal(
        &OneBlockQuarticExactFamily {
            linear: 3.0,
            curvature: 0.5,
            second_scale: 1.0,
        },
        std::slice::from_ref(&spec),
        &options,
        &penalty_counts,
        &rho,
        &[vec![]],
        None,
        gam_problem::RhoPrior::Flat,
        EvalMode::ValueGradientHessian,
    )
    .expect("single-block fallback with exact d2H should evaluate");
    let without_d2_contribution = evaluate_custom_family_hyper_internal(
        &OneBlockQuarticExactFamily {
            linear: 3.0,
            curvature: 0.5,
            second_scale: 0.0,
        },
        &[spec],
        &options,
        &penalty_counts,
        &rho,
        &[vec![]],
        None,
        gam_problem::RhoPrior::Flat,
        EvalMode::ValueGradientHessian,
    )
    .expect("single-block fallback with zero d2H should evaluate");

    let h_with = match with_d2.outer_hessian {
        gam_problem::HessianValue::Dense(hessian) => hessian,
        gam_problem::HessianValue::Operator(_) | gam_problem::HessianValue::Unavailable => {
            panic!("expected dense analytic Hessian")
        }
    };
    let h_without = match without_d2_contribution.outer_hessian {
        gam_problem::HessianValue::Dense(hessian) => hessian,
        gam_problem::HessianValue::Operator(_) | gam_problem::HessianValue::Unavailable => {
            panic!("expected dense analytic Hessian")
        }
    };
    let d2h_delta = h_with[[0, 0]] - h_without[[0, 0]];
    assert!(
        d2h_delta.abs() > 1e-8,
        "expected nonzero outer Hessian contribution from d2H; with={:?}, without={:?}",
        h_with,
        h_without
    );
}

pub(crate) fn jeffreys_seam_spec(p: usize) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "jeffreys-seam".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::eye(p))),
        offset: Array1::zeros(p),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

pub(crate) fn jeffreys_seam_state(beta: Array1<f64>) -> ParameterBlockState {
    let eta = beta.clone();
    ParameterBlockState { beta, eta }
}

/// Observed-default family for the gam#1020 seam contract: implements only
/// the observed joint Newton Hessian (and its directional derivatives) and
/// relies on the trait defaults for the Jeffreys information hooks.
#[derive(Clone)]
pub(crate) struct ObservedJeffreysSeamFamily;

impl CustomFamily for ObservedJeffreysSeamFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .eta
            .len();
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: Array1::zeros(n),
                working_weights: Array1::ones(n),
            }],
        })
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_states.len(), specs.len());
        let beta = &block_states[0].beta;
        Ok(Some(array![
            [2.0 + beta[0] * beta[0], 0.3],
            [0.3, 1.5 + beta[1] * beta[1]]
        ]))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_states.len(), specs.len());
        let beta = &block_states[0].beta;
        Ok(Some(array![
            [2.0 * beta[0] * d_beta_flat[0], 0.0],
            [0.0, 2.0 * beta[1] * d_beta_flat[1]]
        ]))
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_states.len(), specs.len());
        Ok(Some(array![
            [2.0 * d_beta_u_flat[0] * d_betav_flat[0], 0.0],
            [0.0, 2.0 * d_beta_u_flat[1] * d_betav_flat[1]]
        ]))
    }
}

/// gam#1020 acceptance: families that do NOT override the Jeffreys
/// information hooks get the OBSERVED joint Newton quantities — the seam
/// defaults are exact delegations, so behavior is unchanged.
#[test]
pub(crate) fn joint_jeffreys_information_defaults_delegate_to_observed_hessian() {
    let family = ObservedJeffreysSeamFamily;
    let specs = vec![jeffreys_seam_spec(2)];
    let states = vec![jeffreys_seam_state(array![0.4, -0.7])];
    let u = array![0.3, -0.2];
    let v = array![-0.1, 0.5];

    let observed = family
        .exact_newton_joint_hessian_with_specs(&states, &specs)
        .expect("observed H")
        .expect("observed H present");
    let info = family
        .joint_jeffreys_information_with_specs(&states, &specs)
        .expect("jeffreys info")
        .expect("jeffreys info present");
    assert_eq!(info, observed, "default Jeffreys info must be observed H");

    let observed_dot = family
        .exact_newton_joint_hessian_directional_derivative_with_specs(&states, &specs, &u)
        .expect("observed Hdot")
        .expect("observed Hdot present");
    let info_dot = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &u)
        .expect("jeffreys dI")
        .expect("jeffreys dI present");
    assert_eq!(
        info_dot, observed_dot,
        "default Jeffreys dI must be observed Hdot"
    );

    let observed_ddot = family
        .exact_newton_joint_hessian_second_directional_derivative_with_specs(
            &states, &specs, &u, &v,
        )
        .expect("observed H2dot")
        .expect("observed H2dot present");
    let info_ddot = family
        .joint_jeffreys_information_second_directional_derivative_with_specs(
            &states, &specs, &u, &v,
        )
        .expect("jeffreys d2I")
        .expect("jeffreys d2I present");
    assert_eq!(
        info_ddot, observed_ddot,
        "default Jeffreys d2I must be observed H2dot"
    );

    // Contracted hook defaults: declared unavailable and returns None, so
    // the completion keeps the pairwise H2dot fallback.
    assert!(!family.joint_jeffreys_information_contracted_trace_hessian_available());
    let weight = Array2::<f64>::eye(2);
    let contracted = family
        .joint_jeffreys_information_contracted_trace_hessian_with_specs(&states, &specs, &weight)
        .expect("contracted default");
    assert!(
        contracted.is_none(),
        "default contracted trace hook must be None"
    );

    // Observed-default families keep the matvec skip pre-checks armed.
    assert!(family.joint_jeffreys_information_matches_observed_hessian());
}

/// gam#1020: family supplying the contracted trace Hessian. The pairwise
/// second-directional path returns wildly different values so the test
/// detects which path the completion dispatched to.
#[derive(Clone)]
pub(crate) struct ContractedJeffreysSeamFamily;

impl CustomFamily for ContractedJeffreysSeamFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .eta
            .len();
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: Array1::zeros(n),
                working_weights: Array1::ones(n),
            }],
        })
    }

    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_states.len(), specs.len());
        let scale = 1.0e6 * d_beta_u_flat.dot(d_betav_flat);
        Ok(Some(scale * Array2::<f64>::eye(2)))
    }

    fn joint_jeffreys_information_contracted_trace_hessian_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        weight: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_states.len(), specs.len());
        assert_eq!(weight.dim(), (2, 2));
        Ok(Some(7.0 * Array2::<f64>::eye(2)))
    }

    fn joint_jeffreys_information_contracted_trace_hessian_available(&self) -> bool {
        true
    }
}

/// gam#1020 acceptance: the second-order completion takes the contracted
/// trace hook when the family provides one (the wide-p route), scaling it
/// by `−½·gate`; the pairwise H2dot path is not consulted.
#[test]
pub(crate) fn jeffreys_second_order_completion_prefers_contracted_hook() {
    let family = ContractedJeffreysSeamFamily;
    let specs = vec![jeffreys_seam_spec(2)];
    let states = vec![jeffreys_seam_state(Array1::zeros(2))];
    // λ_min = 1e-4 is far below the absolute conditioning gate, so the
    // gate weight is exactly 1 and the completion is −½ · contracted.
    let h_joint = array![[1.0e-4, 0.0], [0.0, 1.0]];
    let z_joint = Array2::<f64>::eye(2);
    let completion = custom_family_joint_jeffreys_second_order_completion(
        &family, &states, &specs, &h_joint, &z_joint, true,
    )
    .expect("completion")
    .expect("completion present");
    let expected = -3.5 * Array2::<f64>::eye(2);
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (completion[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "contracted completion mismatch at ({i},{j}): {} vs {}",
                completion[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

/// gam#1020: family without a contracted hook — the completion must fall
/// back to the exact pairwise second-directional path, and must return
/// `None` when the pairwise fallback is not allowed (width cap exceeded).
#[derive(Clone)]
pub(crate) struct PairwiseJeffreysSeamFamily;

impl CustomFamily for PairwiseJeffreysSeamFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .eta
            .len();
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: Array1::zeros(n),
                working_weights: Array1::ones(n),
            }],
        })
    }

    fn joint_jeffreys_information_second_directional_derivative_with_specs(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        d_beta_u_flat: &Array1<f64>,
        d_betav_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert_eq!(block_states.len(), specs.len());
        let scale = d_beta_u_flat.dot(d_betav_flat);
        Ok(Some(scale * array![[2.0, 1.0], [1.0, 3.0]]))
    }
}

#[test]
pub(crate) fn jeffreys_second_order_completion_pairwise_fallback_when_hook_absent() {
    let family = PairwiseJeffreysSeamFamily;
    let specs = vec![jeffreys_seam_spec(2)];
    let states = vec![jeffreys_seam_state(Array1::zeros(2))];
    let h_joint = array![[1.0e-4, 0.0], [0.0, 1.0]];
    let z_joint = Array2::<f64>::eye(2);

    let completion = custom_family_joint_jeffreys_second_order_completion(
        &family, &states, &specs, &h_joint, &z_joint, true,
    )
    .expect("completion")
    .expect("completion present");
    let direct =
        gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_second_order_completion(
            h_joint.view(),
            z_joint.view(),
            |u: &Array1<f64>, v: &Array1<f64>| {
                family.joint_jeffreys_information_second_directional_derivative_with_specs(
                    &states, &specs, u, v,
                )
            },
        )
        .expect("direct pairwise completion")
        .expect("direct pairwise completion present");
    assert_eq!(
        completion, direct,
        "fallback must be the exact pairwise completion"
    );
    assert!(
        completion.iter().any(|value| value.abs() > 0.0),
        "pairwise completion should be nonzero on this gated fixture"
    );

    let blocked = custom_family_joint_jeffreys_second_order_completion(
        &family, &states, &specs, &h_joint, &z_joint, false,
    )
    .expect("blocked completion");
    assert!(
        blocked.is_none(),
        "completion must decline (None) when the pairwise fallback is disallowed and no hook exists"
    );
}

#[test]
pub(crate) fn custom_family_outer_derivatives_keeps_second_order_for_large_inner_problem() {
    // Inner (n, p) scale does not block the analytic outer Hessian: the
    // outer Hessian assembled by `compute_outer_hessian` is shape
    // (K+ext_dim)×(K+ext_dim) where K = total penalties. For large inner
    // problems with modest K (the common case: n=50000, p=50, K=2) the
    // outer Hessian is tiny and must remain available so ARC can drive
    // the outer iteration. Prior versions of this test enforced an
    // inner-size cutoff that disabled the Hessian for exactly the
    // benchmark sizes (medium: n=50000,p=50; pathological: n=50000,p=80)
    // that were hanging 45-minute GH jobs on BFGS+BfgsApprox Strong Wolfe
    // failures at iter 0.
    #[derive(Clone)]
    struct StrictFamily;

    impl CustomFamily for StrictFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = block_states[0].eta.len();
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n),
                    working_weights: Array1::ones(n),
                }],
            })
        }

        fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
            ExactNewtonOuterObjective::StrictPseudoLaplace
        }
    }

    let specs = vec![ParameterBlockSpec {
        name: "x".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::<f64>::zeros((20_100, 50)),
        )),
        offset: Array1::zeros(20_100),
        penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(50))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        ..BlockwiseFitOptions::default()
    };

    let (gradient, hessian) = custom_family_outer_derivatives(&StrictFamily, &specs, &options);
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Either);
}

impl CustomFamily for OneBlockIdentityFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = block_states[0].eta.len();
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: Array1::ones(n),
                working_weights: Array1::ones(n),
            }],
        })
    }
}

#[test]
pub(crate) fn fit_custom_family_rejects_invalid_blockspec_before_output_channel_probe() {
    let spec = ParameterBlockSpec {
        name: "bad_penalty".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [2.0],
        ])),
        offset: Array1::zeros(2),
        penalties: vec![PenaltyMatrix::Dense(Array2::<f64>::eye(2))],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let err = fit_custom_family(
        &OneBlockIdentityFamily,
        &[spec],
        &BlockwiseFitOptions::default(),
    )
    .expect_err("invalid block spec should return a typed error");
    let message = err.to_string();
    assert!(
        message.contains("block 0 penalty 0 must be 1x1, got 2x2"),
        "unexpected error: {message}",
    );
}

#[derive(Clone)]
pub(crate) struct OneBlockGaussianFamily {
    pub(crate) y: Array1<f64>,
}

impl CustomFamily for OneBlockGaussianFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let eta = &block_states[0].eta;
        let resid = eta - &self.y;
        let ll = -0.5 * resid.dot(&resid);
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: self.y.clone(),
                working_weights: Array1::ones(self.y.len()),
            }],
        })
    }

    fn diagonalworking_weights_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        d_eta: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        // Default implementation ignores this parameter.
        Ok(Some(Array1::zeros(d_eta.len())))
    }

    fn diagonalworking_weights_second_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        d_eta_u: &Array1<f64>,
        arr: &Array1<f64>,
    ) -> Result<Option<Array1<f64>>, String> {
        // Default implementation ignores this parameter.
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(Array1::zeros(d_eta_u.len())))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockConstrainedExactFamily {
    pub(crate) target: f64,
    pub(crate) lower: f64,
}

impl CustomFamily for OneBlockConstrainedExactFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .first()
            .copied()
            .ok_or_else(|| "missing coefficient".to_string())?;
        let g = self.target - beta;
        let ll = -0.5 * (beta - self.target) * (beta - self.target);
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![g],
                hessian: SymmetricMatrix::Dense(array![[1.0]]),
            }],
        })
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx != 0 {
            return Ok(None);
        }
        Ok(Some(LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![self.lower],
        }))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockConstrainedNaNHessianFamily;

impl CustomFamily for OneBlockConstrainedNaNHessianFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![0.0],
                hessian: SymmetricMatrix::Dense(array![[f64::NAN]]),
            }],
        })
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx != 0 {
            return Ok(None);
        }
        Ok(Some(LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![0.0],
        }))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockConstrainedIndefiniteHessianFamily;

impl CustomFamily for OneBlockConstrainedIndefiniteHessianFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-1.0],
                hessian: SymmetricMatrix::Dense(array![[-1.0]]),
            }],
        })
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx != 0 {
            return Ok(None);
        }
        Ok(Some(LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![1.0],
        }))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockLinearLikelihoodExactFamily {
    pub(crate) score: f64,
}

impl CustomFamily for OneBlockLinearLikelihoodExactFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .first()
            .copied()
            .ok_or_else(|| "missing coefficient".to_string())?;
        Ok(FamilyEvaluation {
            log_likelihood: self.score * beta,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![self.score],
                hessian: SymmetricMatrix::Dense(array![[0.0]]),
            }],
        })
    }
}

#[derive(Clone)]
pub(crate) struct PreferJointExactFamily;

impl CustomFamily for PreferJointExactFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![0.0],
                hessian: SymmetricMatrix::Dense(array![[2.0]]),
            }],
        })
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Default implementation ignores this parameter.
        assert!(arr.iter().all(|v| !v.is_nan()));
        Err(
            "blockwise exact-newton path should not be used when joint path is available"
                .to_string(),
        )
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[2.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(array![[0.0]]))
    }
}

#[derive(Clone)]
pub(crate) struct TwoBlockJointConstrainedFamily {
    pub(crate) coupling: f64,
}

impl CustomFamily for TwoBlockJointConstrainedFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta0 = block_states[0].beta[0];
        let beta1 = block_states[1].beta[0];
        let g0 = 1.0 - beta0 - self.coupling * beta1;
        let g1 = 1.0 - beta1 - self.coupling * beta0;
        Ok(FamilyEvaluation {
            log_likelihood: -0.5
                * (beta0 * beta0 + beta1 * beta1 + 2.0 * self.coupling * beta0 * beta1)
                + beta0
                + beta1,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: array![g0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: array![g1],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                },
            ],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[1.0, self.coupling], [self.coupling, 1.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(Array2::zeros((2, 2))))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        block_spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        assert!(!block_spec.name.is_empty());
        if block_idx >= 2 {
            return Ok(None);
        }
        Ok(Some(LinearInequalityConstraints {
            a: array![[1.0]],
            b: array![0.0],
        }))
    }
}

#[derive(Clone)]
pub(crate) struct TwoBlockPersistentGradientFamily;

impl CustomFamily for TwoBlockPersistentGradientFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta0 = block_states[0].beta[0];
        let beta1 = block_states[1].beta[0];
        Ok(FamilyEvaluation {
            log_likelihood: beta0 + beta1,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: array![1.0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: array![1.0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                },
            ],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[1.0, 0.25], [0.25, 1.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(Array2::zeros((2, 2))))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }
}

#[derive(Clone)]
pub(crate) struct OneStepReturnedSaddleFamily {
    pub(crate) target: f64,
    pub(crate) evaluations: Arc<AtomicUsize>,
}

pub(crate) struct ReturnedModeSaddleWorkspace {
    pub(crate) hessian: Array2<f64>,
}

impl ExactNewtonJointHessianWorkspace for ReturnedModeSaddleWorkspace {
    fn warm_up_outer_caches_for_mode(&self, _: EvalMode) -> Result<(), String> {
        Ok(())
    }

    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.hessian.clone()))
    }
}

impl OneStepReturnedSaddleFamily {
    pub(crate) fn new(target: f64) -> Self {
        Self {
            target,
            evaluations: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(crate) fn coordinates(&self, states: &[ParameterBlockState]) -> Result<(f64, f64), String> {
        let x = states
            .first()
            .and_then(|state| state.beta.first())
            .copied()
            .ok_or_else(|| "returned-saddle fixture missing x coefficient".to_string())?;
        let y = states
            .get(1)
            .and_then(|state| state.beta.first())
            .copied()
            .ok_or_else(|| "returned-saddle fixture missing y coefficient".to_string())?;
        Ok((x, y))
    }

    pub(crate) fn hessian(&self, x: f64, y: f64) -> Array2<f64> {
        let displacement = x - self.target;
        let target_squared = self.target * self.target;
        let shape = -1.0 + 2.0 * displacement * displacement / target_squared;
        array![
            [
                1.0 + 2.0 * y * y / target_squared,
                4.0 * displacement * y / target_squared
            ],
            [4.0 * displacement * y / target_squared, shape + 6.0 * y * y],
        ]
    }
}

impl CustomFamily for OneStepReturnedSaddleFamily {
    fn evaluate(&self, states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let (x, y) = self.coordinates(states)?;
        let displacement = x - self.target;
        let target_squared = self.target * self.target;
        let shape = -1.0 + 2.0 * displacement * displacement / target_squared;
        let score_x = -displacement * (1.0 + 2.0 * y * y / target_squared);
        let score_y = -(shape * y + 2.0 * y.powi(3));
        let negative_log_likelihood =
            0.5 * displacement * displacement + 0.5 * shape * y * y + 0.5 * y.powi(4);
        let hessian = self.hessian(x, y);
        Ok(FamilyEvaluation {
            log_likelihood: -negative_log_likelihood,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: array![score_x],
                    hessian: SymmetricMatrix::Dense(array![[hessian[(0, 0)]]]),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: array![score_y],
                    hessian: SymmetricMatrix::Dense(array![[hessian[(1, 1)]]]),
                },
            ],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let (x, y) = self.coordinates(states)?;
        Ok(Some(self.hessian(x, y)))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        let (x, y) = self.coordinates(states)?;
        Ok(Some(Arc::new(ReturnedModeSaddleWorkspace {
            hessian: self.hessian(x, y),
        })))
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }
}

pub(crate) fn one_step_returned_saddle_specs() -> Vec<ParameterBlockSpec> {
    ["x", "y"]
        .into_iter()
        .map(|name| ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
            offset: array![0.0],
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
        .collect()
}

#[test]
pub(crate) fn fresh_exact_mode_curvature_certificate_detects_returned_strict_saddle() {
    let family = OneStepReturnedSaddleFamily::new(0.125);
    let specs = one_step_returned_saddle_specs();
    let options = BlockwiseFitOptions::default();
    let ranges = block_param_ranges(&specs);
    let s_lambdas = vec![Array2::zeros((1, 1)), Array2::zeros((1, 1))];
    let at_start = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
    ];
    let start_certificate = exact_joint_mode_curvature_certificate(
        &family, &at_start, &specs, &options, &ranges, &s_lambdas, 0.0, None, 2,
    )
    .expect("positive-curvature start should be certifiable");
    assert!(start_certificate.workspace.is_some());
    assert!(!start_certificate.has_resolvable_negative_curvature());

    let at_returned_beta = vec![
        ParameterBlockState {
            beta: array![family.target],
            eta: array![family.target],
        },
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
    ];
    let returned_certificate = exact_joint_mode_curvature_certificate(
        &family,
        &at_returned_beta,
        &specs,
        &options,
        &ranges,
        &s_lambdas,
        0.0,
        None,
        2,
    )
    .expect("returned strict saddle should produce an honest certificate");
    assert!(returned_certificate.workspace.is_some());
    assert!(returned_certificate.has_resolvable_negative_curvature());
    assert_eq!(returned_certificate.minimum_whitened_eigenvalue, -1.0);
}

#[test]
pub(crate) fn joint_newton_rejects_one_step_stationary_strict_saddle_at_returned_beta() {
    let family = OneStepReturnedSaddleFamily::new(0.125);
    let specs = one_step_returned_saddle_specs();
    let result = inner_blockwise_fit(
        &family,
        &specs,
        &[Array1::zeros(0), Array1::zeros(0)],
        &BlockwiseFitOptions {
            inner_max_cycles: 1,
            use_remlobjective: false,
            ..BlockwiseFitOptions::default()
        },
        None,
    );
    let error = result.expect_err(
        "a one-step Newton solve must not return a stationary strict saddle as a coefficient mode",
    );
    assert!(
        error.contains("fresh exact returned-mode curvature"),
        "unexpected returned-mode rejection: {error}",
    );
}

#[test]
pub(crate) fn joint_newton_recovers_from_returned_strict_saddle_with_remaining_cycle() {
    let family = OneStepReturnedSaddleFamily::new(0.125);
    let specs = one_step_returned_saddle_specs();
    let options = BlockwiseFitOptions {
        inner_max_cycles: 2,
        use_remlobjective: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(
        &family,
        &specs,
        &[Array1::zeros(0), Array1::zeros(0)],
        &options,
        None,
    )
    .expect("fresh negative curvature should drive the existing hard-case escape");

    assert!(result.converged);
    assert_eq!(result.cycles, 2);
    let (x, y) = family
        .coordinates(&result.block_states)
        .expect("recovered mode should retain both coordinates");
    assert!((x - family.target).abs() <= 1.0e-12, "x={x}");
    assert!(
        (y.abs() - std::f64::consts::FRAC_1_SQRT_2).abs() <= 1.0e-8,
        "y={y}",
    );

    let certificate = exact_joint_mode_curvature_certificate(
        &family,
        &result.block_states,
        &specs,
        &options,
        &block_param_ranges(&specs),
        &result.s_lambdas,
        0.0,
        None,
        2,
    )
    .expect("recovered local minimum should have certifiable exact curvature");
    assert!(!certificate.has_resolvable_negative_curvature());
}

#[test]
pub(crate) fn returned_mode_finalizer_preserves_owned_mode_without_family_replay() {
    let family = OneStepReturnedSaddleFamily::new(0.125);
    let specs = one_step_returned_saddle_specs();
    let options = BlockwiseFitOptions {
        inner_max_cycles: 2,
        use_remlobjective: false,
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    };
    let derivative_blocks: SharedDerivativeBlocks =
        Arc::new((0..specs.len()).map(|_| Vec::new()).collect::<Vec<_>>());
    let selection = evaluate_custom_family_joint_hyper_best_mode_shared(
        &family,
        &specs,
        &options,
        &Array1::zeros(0),
        derivative_blocks,
        &[None],
        EvalMode::ValueOnly,
    )
    .expect("the bounded hard case should produce one selected local mode");
    let selected_objective_bits = selection.result.objective.to_bits();
    let selected_beta_bits: Vec<Vec<u64>> = selection
        .selected_inner
        .block_states
        .iter()
        .map(|state| state.beta.iter().map(|value| value.to_bits()).collect())
        .collect();
    let evaluations_before_finalization = family.evaluations.load(Ordering::Relaxed);

    let mut raw_outer = gam_solve::rho_optimizer::OuterResult::new(
        Array1::zeros(0),
        f64::from_bits(selected_objective_bits),
        7,
        true,
        gam_solve::rho_optimizer::OuterPlan {
            solver: gam_solve::rho_optimizer::Solver::Arc,
            hessian_source: gam_solve::rho_optimizer::HessianSource::Analytic,
        },
    );
    raw_outer.final_grad_norm = Some(0.0);
    raw_outer.criterion_certificate = Some(
        gam_solve::rho_optimizer::OuterCriterionCertificate {
            stationarity: gam_solve::rho_optimizer::OuterStationarityCertificate::AnalyticGradient {
                grad_norm: 0.0,
                projected_grad_norm: 0.0,
                bound: 1.0e-8,
            },
            hessian_psd: Some(true),
            lambdas_railed: Vec::new(),
        },
    );
    let certified_outer = gam_solve::rho_optimizer::CertifiedOuterResult::try_from(raw_outer)
        .expect("the test outer optimum should carry a valid analytic certificate");

    let fit = fit_custom_family_fixed_log_lambdas_from_mode_selection(
        &family,
        &specs,
        &options,
        selection,
        &Array1::zeros(0),
        &certified_outer,
    )
    .expect("the exact selected mode should finalize without another inner solve");

    assert_eq!(
        family.evaluations.load(Ordering::Relaxed),
        evaluations_before_finalization,
        "finalization must consume the selected mode and cached Hessian without replaying the family",
    );
    assert_eq!(fit.penalized_objective.to_bits(), selected_objective_bits);
    assert_eq!(fit.outer_iterations, 7);
    assert_eq!(fit.outer_gradient_norm, Some(0.0));
    assert!(
        fit.convergence_evidence()
            .outer_certificate()
            .is_some_and(|certificate| certificate.certifies()),
    );
    assert!(fit.covariance_conditional.is_some());
    assert!(fit.geometry.is_some());
    assert_eq!(fit.block_states.len(), selected_beta_bits.len());
    for (state, expected) in fit.block_states.iter().zip(selected_beta_bits.iter()) {
        assert_eq!(
            state
                .beta
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            *expected,
        );
    }
}

/// gam#1088 fixture. A coupled two-block family whose joint Hessian carries
/// a `NaN` curvature entry — the degenerate-curvature signature seen in the
/// link-wiggle and location-scale benchmark timeouts (a collapsed/`0÷0` row
/// weight assembling into `XᵀWX`). The penalized Hessian `H_pen = H + S(λ)`
/// and its spectrum then degrade to `NaN`, so the KKT certificate is
/// structurally unreachable. The non-finite-curvature guard must detect
/// this at the head of the cycle and exit far below the budget, instead of
/// grinding the full `inner_max_cycles`.
#[derive(Clone)]
pub(crate) struct TwoBlockNonFiniteCurvatureFamily;

impl CustomFamily for TwoBlockNonFiniteCurvatureFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta0 = block_states[0].beta[0];
        let beta1 = block_states[1].beta[0];
        Ok(FamilyEvaluation {
            log_likelihood: beta0 + beta1,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: array![1.0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: array![1.0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                },
            ],
        })
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // A finite, symmetric, otherwise-PD curvature with a single NaN
        // diagonal entry: exactly the degenerate `H_pen` spectrum the guard
        // exists to catch (a real collapsed-weight curvature defect).
        Ok(Some(array![[f64::NAN, 0.25], [0.25, 1.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(Array2::zeros((2, 2))))
    }

    fn has_explicit_joint_hessian(&self) -> bool {
        true
    }
}

#[derive(Clone)]
pub(crate) struct TwoBlockJointSurrogateFamily;

impl CustomFamily for TwoBlockJointSurrogateFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n0 = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .eta
            .len();
        let n1 = block_states
            .get(1)
            .ok_or_else(|| "missing block 1".to_string())?
            .eta
            .len();
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n0),
                    working_weights: Array1::ones(n0),
                },
                BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n1),
                    working_weights: Array1::ones(n1),
                },
            ],
        })
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        _: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        Ok(Some(Array2::eye(p)))
    }

    fn exact_newton_joint_hessian_directional_derivative_with_specs(
        &self,
        _: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        Ok(Some(Array2::zeros((p, p))))
    }

    fn exact_newton_joint_hessian_second_directional_derivative_with_specs(
        &self,
        _: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        arr: &Array1<f64>,
        arr2: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        assert!(arr2.iter().all(|v| !v.is_nan()));
        let p: usize = specs.iter().map(|spec| spec.design.ncols()).sum();
        Ok(Some(Array2::zeros((p, p))))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockPseudoLaplaceExactFamily {
    pub(crate) target: f64,
}

impl CustomFamily for OneBlockPseudoLaplaceExactFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .first()
            .copied()
            .ok_or_else(|| "missing coefficient".to_string())?;
        let resid = beta - self.target;
        Ok(FamilyEvaluation {
            log_likelihood: -resid * resid,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![-2.0 * resid],
                hessian: SymmetricMatrix::Dense(array![[2.0]]),
            }],
        })
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[2.0]]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Default implementation ignores this parameter.
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(array![[0.0]]))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockExactPsiHookFamily;

impl CustomFamily for OneBlockExactPsiHookFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![0.0],
                hessian: SymmetricMatrix::Dense(array![[1.0]]),
            }],
        })
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[1.0]]))
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        _: usize,
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // Default implementation ignores this parameter.
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        _: &[ParameterBlockState],
        arr: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        assert!(arr.iter().all(|v| !v.is_nan()));
        Ok(Some(array![[0.0]]))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        _: &[Vec<CustomFamilyBlockPsiDerivative>],
        _: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        // Default implementation ignores this parameter.
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: 3.5,
            score_psi: array![0.0],
            hessian_psi: array![[0.0]],
            hessian_psi_operator: None,
        }))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockIndefinitePseudoLaplaceFamily;

impl CustomFamily for OneBlockIndefinitePseudoLaplaceFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: array![0.0],
                hessian: SymmetricMatrix::Dense(array![[-1.0]]),
            }],
        })
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[-1.0]]))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockNearlySymmetricPseudoLaplaceFamily;

impl CustomFamily for OneBlockNearlySymmetricPseudoLaplaceFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let beta = block_states
            .first()
            .ok_or_else(|| "missing block 0".to_string())?
            .beta
            .clone();
        let h = array![[2.0, 0.1], [3.0, 2.0]];
        let gradient = -h.dot(&beta);
        Ok(FamilyEvaluation {
            log_likelihood: -0.5 * beta.dot(&h.dot(&beta)),
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient,
                hessian: SymmetricMatrix::Dense(h),
            }],
        })
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }

    fn exact_newton_joint_hessian(
        &self,
        _: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(array![[2.0, 0.1], [3.0, 2.0]]))
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockAlwaysErrorFamily;

impl CustomFamily for OneBlockAlwaysErrorFamily {
    fn evaluate(&self, _: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        Err("synthetic outer objective failure: block[0] evaluate()".to_string())
    }
}

#[derive(Clone)]
pub(crate) struct OneBlockCovarianceErrorFamily;

impl CustomFamily for OneBlockCovarianceErrorFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = block_states[0].eta.len();
        Ok(FamilyEvaluation {
            log_likelihood: 0.0,
            blockworking_sets: vec![BlockWorkingSet::Diagonal {
                working_response: Array1::zeros(n),
                working_weights: Array1::ones(n),
            }],
        })
    }

    fn exact_newton_joint_hessian_with_specs(
        &self,
        _: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<Array2<f64>>, String> {
        Err("synthetic covariance assembly failure".to_string())
    }
}

#[test]
pub(crate) fn effectiveridge_is_never_below_solver_floor() {
    assert!((effective_solverridge(0.0) - 1e-15).abs() < 1e-30);
    assert!((effective_solverridge(1e-8) - 1e-8).abs() < 1e-20);
}

#[test]
pub(crate) fn objective_includes_solverridge_quadratic_term() {
    // One-parameter block with X=1, y*=1, w=1, no explicit penalties.
    // Inner solve gives beta = 1 / (1 + ridge), so objective should include
    // 0.5 * ridge * beta^2 even when no smoothing penalties are present.
    let spec = ParameterBlockSpec {
        name: "b0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        inner_tol: 0.0,
        outer_max_iter: 1,
        outer_tol: 1e-8,
        outer_rel_cost_tol: None,
        rho_lower_bound: -10.0,
        ridge_floor: 1e-4,
        ridge_policy: RidgePolicy::positive_part_approximate_objective(),
        use_remlobjective: false,
        compute_covariance: false,
        use_outer_hessian: false,
        screening_max_inner_iterations: None,
        outer_inner_max_iterations: None,
        seed_screening: false,
        early_exit_threshold: None,
        outer_score_subsample: None,
        auto_outer_subsample: false,
        outer_eval_context: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
        joint_penalties: None,
        independent_prior_factor_labels: Vec::new(),
        screen_initial_rho: true,
    };

    let result = fit_custom_family(&OneBlockIdentityFamily, &[spec], &options)
        .expect("custom family fit should succeed");
    let ridge = effective_solverridge(options.ridge_floor);
    let beta = result.block_states[0].beta[0];
    let expected_penalty = 0.5 * ridge * beta * beta;
    assert!(
        (result.penalized_objective - expected_penalty).abs() < 1e-12,
        "penalized objective should equal ridge quadratic term when ll=0 and S=0; got {}, expected {}",
        result.penalized_objective,
        expected_penalty
    );
}

#[test]
pub(crate) fn inner_block_accepts_penalty_improving_step_even_if_loglik_drops() {
    let family = OneBlockGaussianFamily { y: array![1.0] };
    let spec = ParameterBlockSpec {
        name: "b0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![],
        initial_log_lambdas: array![10.0_f64.ln()],
        initial_beta: Some(array![1.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_max_cycles: 20,
        inner_tol: 1e-10,
        outer_max_iter: 1,
        outer_tol: 1e-8,
        outer_rel_cost_tol: None,
        rho_lower_bound: -10.0,
        ridge_floor: 0.0,
        ridge_policy: RidgePolicy::positive_part_approximate_objective(),
        use_remlobjective: false,
        compute_covariance: false,
        use_outer_hessian: false,
        screening_max_inner_iterations: None,
        outer_inner_max_iterations: None,
        seed_screening: false,
        early_exit_threshold: None,
        outer_score_subsample: None,
        auto_outer_subsample: false,
        outer_eval_context: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
        joint_penalties: None,
        independent_prior_factor_labels: Vec::new(),
        screen_initial_rho: true,
    };
    let per_block_log_lambdas = vec![array![10.0_f64.ln()]];
    let inner = inner_blockwise_fit(&family, &[spec], &per_block_log_lambdas, &options, None)
        .expect("inner blockwise fit should succeed");

    let beta = inner.block_states[0].beta[0];
    assert!(
        beta < 0.5,
        "beta should shrink toward penalized mode; got {}",
        beta
    );
    assert!(
        inner.log_likelihood < -1e-8,
        "raw log-likelihood should drop for this strongly penalized move; got {}",
        inner.log_likelihood
    );
}

#[test]
pub(crate) fn exact_newton_backtracking_descent_includes_explicit_ridge() {
    let family = OneBlockLinearLikelihoodExactFamily { score: 0.5 };
    let spec = ParameterBlockSpec {
        name: "b0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![1.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        inner_tol: 0.0,
        outer_max_iter: 1,
        outer_tol: 1e-8,
        outer_rel_cost_tol: None,
        rho_lower_bound: -10.0,
        ridge_floor: 1.0,
        ridge_policy: RidgePolicy::positive_part_approximate_objective(),
        use_remlobjective: false,
        compute_covariance: false,
        use_outer_hessian: false,
        screening_max_inner_iterations: None,
        outer_inner_max_iterations: None,
        seed_screening: false,
        early_exit_threshold: None,
        outer_score_subsample: None,
        auto_outer_subsample: false,
        outer_eval_context: None,
        cache_session: None,
        cache_mirror_sessions: Vec::new(),
        joint_penalties: None,
        independent_prior_factor_labels: Vec::new(),
        screen_initial_rho: true,
    };
    let inner = inner_blockwise_fit(&family, &[spec], &[Array1::zeros(0)], &options, None)
        .expect("inner blockwise fit should succeed");

    let beta = inner.block_states[0].beta[0];
    let objective = -inner.log_likelihood + inner.penalty_value;
    assert!(
        beta < 1.0 - 1e-12,
        "ridge-aware fallback descent should shrink beta after rejecting the uphill Newton step; got {}",
        beta
    );
    assert!(
        objective < -1e-12,
        "accepted fallback step should lower the penalized objective; got {}",
        objective
    );
}

#[test]
pub(crate) fn outergradient_matches_finite_difference_for_one_block() {
    let n = 8usize;
    let y = Array1::from_vec(vec![0.4, -0.2, 0.8, 1.0, -0.5, 0.3, 0.1, -0.7]);
    let spec = ParameterBlockSpec {
        name: "b0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        )),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.2],
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: 1e-10,
        ..BlockwiseFitOptions::default()
    };
    let penalty_counts = vec![1usize];
    let rho = array![0.1];
    let (f0, g0, _) = outerobjective_andgradient(
        &OneBlockGaussianFamily { y: y.clone() },
        std::slice::from_ref(&spec),
        &options,
        &penalty_counts,
        &rho,
        None,
    )
    .expect("objective/gradient");

    let h = 1e-5;
    let rho_p = array![rho[0] + h];
    let rho_m = array![rho[0] - h];
    let (fp, _, _) = outerobjective_andgradient(
        &OneBlockGaussianFamily { y: y.clone() },
        std::slice::from_ref(&spec),
        &options,
        &penalty_counts,
        &rho_p,
        None,
    )
    .expect("objective+");
    let (fm, _, _) = outerobjective_andgradient(
        &OneBlockGaussianFamily { y },
        std::slice::from_ref(&spec),
        &options,
        &penalty_counts,
        &rho_m,
        None,
    )
    .expect("objective-");
    let gfd = (fp - fm) / (2.0 * h);
    let rel = (g0[0] - gfd).abs() / gfd.abs().max(1e-8);

    assert!(f0.is_finite());
    assert_eq!(
        g0[0].signum(),
        gfd.signum(),
        "outer gradient sign mismatch: analytic={} fd={}",
        g0[0],
        gfd
    );
    assert!(
        rel < 5e-3,
        "outer gradient mismatch: analytic={} fd={} rel={}",
        g0[0],
        gfd,
        rel
    );
}

#[test]
pub(crate) fn outergradient_prefers_joint_exact_pathwhen_available() {
    let spec = ParameterBlockSpec {
        name: "joint_exact".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: 1e-10,
        ..BlockwiseFitOptions::default()
    };
    let penalty_counts = vec![1usize];
    let rho = array![0.0];

    let result = outerobjective_andgradient(
        &PreferJointExactFamily,
        std::slice::from_ref(&spec),
        &options,
        &penalty_counts,
        &rho,
        None,
    );
    assert!(
        result.is_ok(),
        "joint exact path should be preferred over blockwise fallback: {:?}",
        result.err()
    );
}

#[test]
pub(crate) fn innerfit_uses_joint_exact_path_for_multiblock_constraints() {
    let spec0 = ParameterBlockSpec {
        name: "block0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let spec1 = ParameterBlockSpec {
        name: "block1".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        inner_tol: 1e-10,
        ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
        ..BlockwiseFitOptions::default()
    };
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];

    let result = inner_blockwise_fit(
        &TwoBlockJointConstrainedFamily { coupling: 0.25 },
        &[spec0, spec1],
        &per_block,
        &options,
        None,
    )
    .expect("joint constrained inner fit should succeed");

    assert!(
        result.converged,
        "joint constrained inner fit should converge in one cycle"
    );
    assert_eq!(result.cycles, 1);
    assert!((result.block_states[0].beta[0] - 0.8).abs() < 1e-8);
    assert!((result.block_states[1].beta[0] - 0.8).abs() < 1e-8);
    assert_eq!(result.active_sets, vec![None, None]);
}

#[test]
pub(crate) fn joint_newton_budget_exhaustion_refuses_coupled_exact_inner() {
    let spec0 = ParameterBlockSpec {
        name: "block0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let spec1 = ParameterBlockSpec {
        name: "block1".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        inner_tol: 1e-12,
        ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
        ..BlockwiseFitOptions::default()
    };
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];

    // A coupled exact-joint family whose gradient never reaches KKT (the unit
    // persistent gradient) exhausts a one-cycle budget. Per the coupled-exact
    // budget-exhaustion contract (`inner_blockwise_fit`, the `if
    // coupled_exact_joint_required` budget branch), that is NOT malformed input:
    // the inner solver returns a finite, NON-converged mode so the OUTER
    // optimizer can reject this rho and back off to a fit-able neighbour —
    // escaping the survival/location-scale flat-baseline valley — instead of
    // bubbling an `InvalidInput` through the custom-family string boundary and
    // aborting the whole fit on the first non-certifying startup rho. The outer
    // consumes `!inner.converged` (`assembly.rs`, `fit.rs`); an `Err` here would
    // defeat that rho-rejection contract.
    let result = inner_blockwise_fit(
        &TwoBlockPersistentGradientFamily,
        &[spec0, spec1],
        &per_block,
        &options,
        None,
    )
    .expect("coupled exact-joint budget exhaustion returns a non-converged mode, not an error");
    assert!(
        !result.converged,
        "a budget-exhausted coupled-exact inner mode must be reported non-converged"
    );
    assert_eq!(
        result.cycles, 1,
        "the one-cycle budget must be fully consumed before the non-converged exit"
    );
    assert!(
        result.kkt_residual.is_none(),
        "a non-converged inner mode carries no KKT certificate for the outer IFT correction"
    );
}

/// gam#1794 regression. The original report described a fit-level *wall-clock*
/// deadline that bounded a non-converging marginal-slope inner solve, claiming
/// it was armed only on the survival marginal-slope entry point and not on the
/// bernoulli/flex (BMS) marginal-slope path, so a stalled BMS/flex inner
/// joint-Newton spun unbounded until the harness killed it (TIMEOUT).
///
/// Since then, gam#2055 removed EVERY wall-clock time budget from the solver
/// (a wall clock is non-deterministic and machine-dependent, so it violates
/// the reproducibility SPEC). Both marginal-slope families now route their
/// coupled exact-joint inner solve through the SAME `inner_blockwise_fit`
/// loop, which is bounded DETERMINISTICALLY by the inner cycle budget
/// (`inner_loop_hard_ceiling = inner_max_cycles.max(200)`) plus the
/// deterministic stall early-exit guards (gam#979 / #1040 / #1088). There is
/// no per-family wall-clock arming site left to be asymmetric about
/// (`grep -rn "OUTER_WALL_CLOCK_DEADLINE\|budget_secs\|unwrap_or(300"
/// crates/` returns no hits; `outer_wall_clock_deadline_exceeded()` is an
/// inert `false` shim).
///
/// This test pins the invariant that actually protects against #1794's
/// unbounded-spin symptom on the BMS/flex-shaped path: a coupled exact-joint
/// inner solve whose gradient never reaches KKT — the never-converging
/// marginal-slope stall — is bounded by the FULL PRODUCTION cycle ceiling
/// (`DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES`, the scale of the reported
/// overrun) and RETURNS a finite, catchable, non-converged mode for outer-ρ
/// rejection instead of spinning to a harness kill. The large budget proves
/// the bound does not depend on a small `inner_max_cycles`; if the deterministic
/// ceiling were removed this test would hang instead of returning.
#[test]
pub(crate) fn bms_flex_marginal_slope_coupled_exact_inner_stall_is_deterministically_bounded_gam1794()
 {
    let spec0 = ParameterBlockSpec {
        name: "block0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let spec1 = ParameterBlockSpec {
        name: "block1".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    // The PRODUCTION ceiling — the scale of the reported #1794 overrun. The
    // deterministic bound must hold here regardless of how large the budget is;
    // a wall-clock arming asymmetry (the stale premise) or a removed ceiling
    // would let this stall spin without returning.
    let options = BlockwiseFitOptions {
        inner_max_cycles: DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
        inner_tol: 1e-12,
        ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
        ..BlockwiseFitOptions::default()
    };
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];

    // The same coupled exact-joint stall as the neighbouring one-cycle test,
    // but driven at the production ceiling. It MUST return (not hang) with a
    // bounded, non-converged mode so the outer optimizer can reject this ρ.
    let result = inner_blockwise_fit(
        &TwoBlockPersistentGradientFamily,
        &[spec0, spec1],
        &per_block,
        &options,
        None,
    )
    .expect(
        "a stalled coupled-exact marginal-slope inner solve must return a non-converged \
         mode, not spin to a harness kill",
    );
    assert!(
        !result.converged,
        "a never-KKT coupled-exact inner stall must be reported non-converged so the \
         outer optimizer rejects this ρ"
    );
    assert!(
        result.cycles >= 1 && result.cycles <= DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
        "the inner solve must be bounded by the deterministic cycle ceiling \
         (gam#1794/#2055): observed cycles={} must lie in 1..={}",
        result.cycles,
        DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
    );
    assert!(
        result.kkt_residual.is_none(),
        "a non-converged inner mode carries no KKT certificate for the outer IFT correction"
    );
}

/// gam#979 regression: the merit-descent veto on the flat-residual / slow-
/// geometric stall exit is BOUNDED, so a persistently-drifting Φ-merit can no
/// longer suppress the honest non-converged exit for the whole cycle budget.
///
/// `TwoBlockPersistentGradientFamily` has a constant unit gradient on both
/// blocks and a linear (unbounded) log-likelihood: the KKT residual is flat at
/// `‖[1,1]‖ = √2 > tol` every cycle (never a ≥10% drop), while the penalized
/// objective Φ = −loglik drifts DOWN by O(1) each cycle as the trust-region step
/// walks β along the gradient. That is exactly the survival marginal-slope
/// free-warp/gauge signature the #979 bound targets: `merit_still_descending_
/// over_window()` is TRUE forever, so before the bound it vetoed the flat-
/// residual and slow-geometric stall exits for the ENTIRE budget and the loop
/// ground to `inner_loop_hard_ceiling` (1200) on every outer ρ-eval — the ~900s
/// hang. With the bound the veto expires once the residual has been flat for
/// `RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES = 4·RESIDUAL_STALL_NO_IMPROVE_CYCLES =
/// 120` cycles and the solve exits non-converged there.
///
/// This pins the PERF backstop the neighbouring `bms_flex_..._gam1794` test does
/// NOT: that test only bounds `cycles ≤ 1200` (the hard ceiling), so a revert of
/// the #979 veto bound — which would let this stall grind all the way to 1200 —
/// still passes it. Here the budget is the full production ceiling, but the exit
/// must land near the 120-cycle veto cap, an order of magnitude below it. A
/// revert grinds to 1200 and fails `cycles < 300`; a fixture that stopped
/// exercising the descending-merit veto (exiting at the ~40-cycle min-cycle
/// window instead) fails `cycles > RESIDUAL_STALL_MIN_CYCLES`.
#[test]
pub(crate) fn persistent_merit_descent_stall_exits_at_veto_bound_not_hard_ceiling_gam979() {
    // Mirror the #1794 fixture: two scalar blocks, no penalty, so the residual
    // is the bare constant gradient and Φ = −loglik drifts down each cycle.
    let make_spec = |name: &str| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    // The full PRODUCTION ceiling — the scale of the reported #979 hang. The
    // merit-veto bound must cut the stall off an order of magnitude sooner
    // REGARDLESS of how large this budget is.
    let options = BlockwiseFitOptions {
        inner_max_cycles: DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
        inner_tol: 1e-12,
        ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
        ..BlockwiseFitOptions::default()
    };
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];

    let result = inner_blockwise_fit(
        &TwoBlockPersistentGradientFamily,
        &[make_spec("block0"), make_spec("block1")],
        &per_block,
        &options,
        None,
    )
    .expect("a merit-vetoed flat-residual stall must return a non-converged mode, not error");

    // The residual never reaches KKT, so the solve is honestly non-converged.
    assert!(
        !result.converged,
        "a never-KKT flat-residual stall must be reported non-converged so the outer \
         optimizer rejects this ρ"
    );
    // `RESIDUAL_STALL_MERIT_VETO_MAX_CYCLES = 4 * RESIDUAL_STALL_NO_IMPROVE_CYCLES = 120`
    // (kept in sync with `inner_blockwise_fit`). The exit must land near that cap.
    const MERIT_VETO_MAX_CYCLES: usize = 120;
    // (a) The core #979 guard: the drifting merit does NOT hold the veto for the
    // whole budget. A revert grinds to the 1200 hard ceiling and fails this.
    assert!(
        result.cycles < MERIT_VETO_MAX_CYCLES + 60,
        "the merit-descent veto must expire near the {MERIT_VETO_MAX_CYCLES}-cycle bound; \
         observed cycles={} (a revert of the #979 bound grinds to the {}-cycle ceiling)",
        result.cycles,
        DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
    );
    // (b) The exit is the merit-VETOED one, not an earlier unvetoed stall exit:
    // a still-descending merit vetoes every flat-residual / slow-geometric exit
    // until the bound, so the solve must run well past the 40-cycle min window.
    // If the fixture stopped producing a descending merit this trips, flagging
    // that the veto path is no longer exercised.
    assert!(
        result.cycles > 40,
        "expected the descending-merit veto to hold the exit past the min-cycle window \
         until the {MERIT_VETO_MAX_CYCLES}-cycle bound; observed cycles={} — the fixture may \
         no longer exercise the merit-veto path",
        result.cycles,
    );
    assert!(
        result.kkt_residual.is_none(),
        "a non-converged inner mode carries no KKT certificate for the outer IFT correction"
    );
}

/// gam#1088 regression. A `NaN` in the joint Hessian curvature makes
/// `H_pen = H + S(λ)` and its spectrum degenerate, so the KKT certificate
/// can never be issued. Without the non-finite-curvature guard the coupled
/// joint-Newton loop runs to the full `inner_max_cycles` ceiling (1200 in
/// production) on every outer ρ-eval, which is the multi-hour benchmark
/// timeout. The guard must detect the degenerate curvature at the head of
/// the cycle and exit FAR below the ceiling — at cycle 0 — as a non-
/// converged, structured non-budget exit so the outer optimizer rejects
/// the ρ-evaluation cleanly.
#[test]
pub(crate) fn non_finite_curvature_exits_joint_newton_far_below_budget() {
    let spec0 = ParameterBlockSpec {
        name: "block0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let spec1 = ParameterBlockSpec {
        name: "block1".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    // The PRODUCTION ceiling: the bug is that all 1200 cycles are burned.
    // The guard must make the solve exit immediately regardless of how
    // large the budget is, so we set the real ceiling here and prove the
    // exit does not depend on a small budget.
    let options = BlockwiseFitOptions {
        inner_max_cycles: DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES,
        inner_tol: 1e-12,
        ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
        ..BlockwiseFitOptions::default()
    };
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];

    let err = inner_blockwise_fit(
        &TwoBlockNonFiniteCurvatureFamily,
        &[spec0, spec1],
        &per_block,
        &options,
        None,
    )
    .expect_err("a non-finite joint Hessian must fail the coupled exact-joint inner solve");
    // gam#1088: the NaN curvature is detected and rejected at the smooth-
    // regularized logdet-Hessian boundary — BEFORE the joint-Newton loop and its
    // eigendecomposition — so the solve fails loudly essentially immediately and
    // never grinds toward the ceiling (the same fail-loudly contract pinned by
    // `exact_newton_nan_hessian_fails_loudly_before_eigendecomposition`). The
    // full production ceiling set above proves the fast exit does not depend on a
    // small budget.
    assert!(
        err.contains("smooth-regularized logdet Hessian contains non-finite entry"),
        "non-finite curvature must be rejected loudly at the logdet boundary, far \
             below the Newton budget: {err}"
    );
    assert!(
        !err.contains("exhausted the joint Newton budget"),
        "non-finite curvature must NOT consume the joint Newton budget: {err}"
    );
}

/// gam#787 binary matern centers=12 regression. Near a flat-objective
/// optimum the joint-Newton proposal shrinks to the step-tol floor while
/// `predicted_reduction = rhs·δ − ½δᵀHδ` becomes round-off-signed. The
/// `predicted_reduction ≤ 0` branch must NOT fire the preconditioned-descent
/// substitution there (it would replace the tiny KKT-polishing step with an
/// objective-descent step that catapults the residual off the near-converged
/// iterate). `joint_proposal_at_step_floor` is the suppression gate.
#[test]
pub(crate) fn joint_proposal_at_step_floor_suppresses_descent_substitution_near_optimum() {
    // The exact c12 cycle-10 operating point: proposal_inf=1.413e-5,
    // step_tol=1.355e-5 (proposal a hair = 1.04× above tol). The iterate is
    // polishing KKT, so a pred≤0 here is round-off — the gate must fire.
    assert!(
        joint_proposal_at_step_floor(1.413e-5, 1.355e-5),
        "a proposal within 4× step_tol is at the convergence floor; \
             the descent substitution must be suppressed"
    );
    // Exactly at the 4× band edge: still at the floor.
    assert!(joint_proposal_at_step_floor(4.0 * 1.355e-5, 1.355e-5));
    // A genuinely large proposal (model-invalid direction far from the
    // optimum) is NOT at the floor — the descent substitution must still run.
    assert!(
        !joint_proposal_at_step_floor(1.182e-2, 1.355e-5),
        "an O(1e-2) proposal is far above the step floor; the \
             preconditioned-descent fallback must remain active there"
    );
    // Non-finite inputs never certify the floor (so the substitution path
    // keeps its existing non-finite handling).
    assert!(!joint_proposal_at_step_floor(f64::NAN, 1.0e-5));
    assert!(!joint_proposal_at_step_floor(1.0e-6, f64::INFINITY));
}

/// Independent derivation and direct numerical proof of the
/// ρ ≈ 2 inner-PIRLS pathology pinned by the large-scale saturated-probit
/// failure trace.
///
/// # Mechanism
///
/// Inner Newton on the penalized objective `f(β) = -ℓ(β) + ½βᵀSβ`
/// uses two different ridge values:
///   * **APPLY** path (`apply_joint_penalized_hessian_into`, called
///     inside `joint_quadratic_predicted_reduction`) uses
///     `joint_solver_diagonal_ridge`, which equals
///     `joint_mode_diagonal_ridge + JOINT_TRACE_STABILITY_RIDGE +
///     stabilizing_shift`, where the stabilizing shift is whatever
///     positive quantity `stabilized_joint_solver_diagonal_ridge`
///     adds to lift a negative-eigenvalue joint Hessian above the
///     SPD floor.
///   * **TRIAL OBJECTIVE** path (`total_quadratic_penalty`) uses
///     only `joint_mode_diagonal_ridge` (= `effective_solverridge`),
///     which is the true penalty in the objective `f` and does NOT
///     include the stabilizing shift.
///
/// Let `Δ = joint_solver_diagonal_ridge - joint_mode_diagonal_ridge`
/// (the gap between the SOLVE / APPLY matrix and the TRUE Hessian).
/// For a Newton step `δ = (H_NLL + S + joint_solver_diagonal_ridge·I)⁻¹·rhs`,
/// the Newton identity gives `δᵀ·H_used·δ = rhs·δ`, so:
///
///     predicted = rhs·δ − ½·δᵀ·H_used·δ = ½·rhs·δ
///     actual    = rhs·δ − ½·δᵀ·H_true·δ
///               = rhs·δ − ½·(δᵀ·H_used·δ − Δ·‖δ‖²)
///               = ½·rhs·δ + ½·Δ·‖δ‖²
///     ρ = actual / predicted = 1 + Δ·‖δ‖² / (rhs·δ)
///
/// When `δ ∈ null(H_true)` (e.g. the marginal-block cancellation
/// direction from `marginal_block_hessian_cancels_in_saturated_regime`
/// combined with an unpenalized direction in the smoothing penalty's
/// null space), `H_true·δ = 0`, so `H_used·δ = Δ·δ` and therefore
/// `rhs = Δ·δ`, giving `rhs·δ = Δ·‖δ‖²`. Substituting:
///
///     ρ = 1 + Δ·‖δ‖² / (Δ·‖δ‖²) = 2  EXACTLY.
///
/// This is independent of `Δ`, of the data size, and of `‖δ‖` — it
/// is a structural consequence of "SOLVE/APPLY add a stabilizing
/// shift that TRIAL OBJECTIVE doesn't see" combined with "Newton
/// step lies in the null space of the true Hessian".
///
/// # Test
///
/// We construct a 2D synthetic case with H_NLL indefinite (one
/// negative eigenvalue, mimicking the entry-survival concave term),
/// `S = 0`, and `joint_mode_diagonal_ridge = 0` (i.e. the policy
/// does NOT include the ridge in the objective). The stabilizing
/// shift lifts the negative eigenvalue to the SPD floor; the Newton
/// step lies in the formerly-near-null direction; predicted and
/// actual are computed by the exact same routines the inner solver
/// uses; ρ comes out to exactly 2.0 to floating-point precision.
#[test]
pub(crate) fn ridge_stabilization_gap_produces_exact_rho_two_in_null_direction() {
    // Synthetic 3D joint Hessian with the structure of the
    // saturated-probit failure case at large scale:
    //   - dim 0: indefinite contribution (eigenvalue −1) from the
    //     concave entry-survival term `+w·log Φ(−η₀)`. This triggers
    //     the SPD stabilizer in the solver.
    //   - dim 1: positive contribution (+1) from a non-saturated
    //     coefficient direction.
    //   - dim 2: ZERO from the marginal-block Hessian cancellation
    //     proven separately in `marginal_block_hessian_cancels_in_saturated_regime`.
    //     This is the saturating direction that sits in null(H_true).
    let h_nll = array![[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    let source = JointHessianSource::Dense(h_nll.clone());
    let ranges = vec![(0, 3)];
    // Smoothing penalty `S` is zero in the saturating direction
    // (dim 2) — mirrors the duchon-smooth polynomial null space
    // containing constants/linears.
    let s_lambdas = vec![Array2::<f64>::zeros((3, 3))];

    // Stabilized solver ridge: should add ~1.0 to lift the
    // -1 eigenvalue to the SPD floor (~ridge_floor).
    let base = JOINT_TRACE_STABILITY_RIDGE;
    let ridge_floor = 1.0e-12_f64;
    let joint_mode_diagonal_ridge = 0.0_f64; // policy: ridge NOT in objective
    // `stabilized_joint_solver_diagonal_ridge` consults the family only
    // for `use_exact_newton_strict_spd`, which defaults to false; we
    // simulate that branch by computing the live PSD-penalized shift with
    // the same source matrix.
    let mut lhs = h_nll.clone();
    add_joint_penalty_to_matrix(&mut lhs, &ranges, &s_lambdas, base, None);
    let shift = exact_newton_stabilizing_shift_psd_penalized(&lhs, &lhs, ridge_floor)
        .expect("indefinite Hessian must yield a positive stabilizing shift");
    assert!(
        shift > 0.9,
        "shift should lift the -1 eigenvalue; got {shift}"
    );
    let joint_solver_diagonal_ridge = base + shift;
    let big_delta = joint_solver_diagonal_ridge - joint_mode_diagonal_ridge;

    // True Hessian (what TRIAL OBJECTIVE sees):
    //   H_true = H_NLL + S + joint_mode_diagonal_ridge·I
    //          = diag(-1, 1, 0)
    //   ⇒ dim 2 is a null direction of H_true.
    // Used Hessian (what SOLVE / APPLY uses):
    //   H_used = H_NLL + S + joint_solver_diagonal_ridge·I
    //          = diag(-1+Δ, 1+Δ, Δ)   where Δ ≈ 1.0
    //   ⇒ dim 2 has curvature Δ (purely from the stabilizing shift,
    //     which fires because dim 0 is negative).
    // rhs aimed entirely in dim 2 puts the Newton step in null(H_true).
    let rhs = array![0.0_f64, 0.0, 1.0];
    let h_used_22 = 0.0 + joint_solver_diagonal_ridge;
    let delta = array![0.0, 0.0, rhs[2] / h_used_22];

    // Compute hpen_delta via the SAME helper the inner solver uses.
    let mut hpen_delta = Array1::<f64>::zeros(3);
    apply_joint_penalized_hessian_into(
        &source,
        &ranges,
        &s_lambdas,
        joint_solver_diagonal_ridge,
        &delta,
        &mut hpen_delta,
        None,
    )
    .expect("apply joint penalized hessian must succeed");

    // Predicted = the exact formula the inner solver uses.
    let predicted = joint_quadratic_predicted_reduction(&rhs, &hpen_delta, &delta);

    // Actual (true) reduction: f(β=0) − f(β+δ) for the true objective
    //   f(β) = ½·βᵀ·H_NLL·β + ½·βᵀ·S·β + ½·joint_mode_diagonal_ridge·‖β‖² + bᵀ·β
    // taking β_start = 0 and using the Newton identity for the truth:
    //   actual = rhs·δ − ½·δᵀ·H_true·δ
    // where H_true = H_NLL + S + joint_mode_diagonal_ridge·I.
    let mut h_true_delta = Array1::<f64>::zeros(3);
    apply_joint_penalized_hessian_into(
        &source,
        &ranges,
        &s_lambdas,
        joint_mode_diagonal_ridge,
        &delta,
        &mut h_true_delta,
        None,
    )
    .expect("apply true (un-stabilized) hessian must succeed");
    let actual = rhs.dot(&delta) - 0.5 * delta.dot(&h_true_delta);

    let rho = actual / predicted;

    // ρ must be EXACTLY 2 to floating-point precision (not just "close to 2").
    // This is the structural fingerprint of the SOLVE/APPLY-vs-OBJECTIVE
    // ridge-stabilization gap in the saturated regime.
    assert!(
        (rho - 2.0).abs() <= 1e-10,
        "ρ should be EXACTLY 2 when Newton step lies in null(H_true) with stabilizing-shift gap; got {rho}",
    );

    // Sanity: the identity rhs·δ = Δ·‖δ‖² must hold (this is the
    // mathematical core of why ρ = 2 specifically and not 1.5 or 3).
    let rhs_dot_delta = rhs.dot(&delta);
    let delta_sq_times_big_delta = big_delta * delta.dot(&delta);
    assert!(
        (rhs_dot_delta - delta_sq_times_big_delta).abs() <= 1e-10 * rhs_dot_delta.abs(),
        "Newton-identity null-space condition: rhs·δ ({rhs_dot_delta}) should equal Δ·‖δ‖² ({delta_sq_times_big_delta})",
    );

    // And ρ = 2 holds AT ALL MAGNITUDES of δ — verify by scaling rhs:
    for scale in [0.001_f64, 0.029, 1.0, 988.0] {
        let scaled_rhs = &rhs * scale;
        let scaled_delta = &delta * scale;
        let mut scaled_hpen = Array1::<f64>::zeros(3);
        apply_joint_penalized_hessian_into(
            &source,
            &ranges,
            &s_lambdas,
            joint_solver_diagonal_ridge,
            &scaled_delta,
            &mut scaled_hpen,
            None,
        )
        .expect("apply scaled");
        let scaled_predicted =
            joint_quadratic_predicted_reduction(&scaled_rhs, &scaled_hpen, &scaled_delta);
        let mut scaled_h_true_delta = Array1::<f64>::zeros(3);
        apply_joint_penalized_hessian_into(
            &source,
            &ranges,
            &s_lambdas,
            joint_mode_diagonal_ridge,
            &scaled_delta,
            &mut scaled_h_true_delta,
            None,
        )
        .expect("apply scaled true");
        let scaled_actual =
            scaled_rhs.dot(&scaled_delta) - 0.5 * scaled_delta.dot(&scaled_h_true_delta);
        let scaled_rho = scaled_actual / scaled_predicted;
        assert!(
            (scaled_rho - 2.0).abs() <= 1e-10,
            "ρ invariance under step rescaling broke at scale {scale}: got {scaled_rho}",
        );
    }
}

/// gam#979 survival marginal-slope flex non-convergence (the constrained
/// joint-Newton feasibility reroute). When a trust-region trial step crosses a
/// BINDING monotonicity row — the current iterate sits on the cone face
/// (slack≈0) and the step has negative drift on that row — the two feasibility
/// mechanisms behave very differently:
///
///   * the global fraction-to-boundary scalar `α = slack / −drift` (what
///     `apply_joint_feasibility_limit` applied to the WHOLE joint step) is ~0 on
///     a binding row, so it crushes the ENTIRE step — including its components
///     orthogonal to the binding row — to a microscopic fraction. β then crawls
///     ~α·‖δ‖ per cycle and the inner joint-Newton grinds its budget without
///     converging (the survival hang);
///   * the strict-interior cone projection keeps the step's components in the
///     unconstrained directions and only corrects the binding direction, so the
///     realized step retains O(1) magnitude.
///
/// This pins that contrast on a one-row binding cone: the projection's step is
/// orders of magnitude larger than the α-crushed step, which is the whole reason
/// the constrained path now routes feasibility through the projection.
#[test]
pub(crate) fn cone_projection_preserves_step_where_alpha_crush_collapses_it() {
    use gam_problem::LinearInequalityConstraints;
    use gam_solve::active_set::project_point_strictly_into_feasible_cone;
    // One monotonicity row `a·β ≥ 0` with a = [1, 0]; the current iterate
    // β = [0, 0] sits exactly on it (slack = 0). The Newton trial step wants to
    // move DOWN on the binding coordinate (δ_0 = −1, would violate) and freely on
    // the orthogonal coordinate (δ_1 = +5, unconstrained).
    let a = array![[1.0_f64, 0.0]];
    let b = Array1::<f64>::zeros(1);
    let constraints =
        LinearInequalityConstraints::new(a, b).expect("test constraint shape invariant");

    let beta = array![0.0_f64, 0.0];
    let trial_step = array![-1.0_f64, 5.0];
    let trial_point = &beta + &trial_step;

    // ── Old mechanism: global fraction-to-boundary α ────────────────────────
    // slack = a·β − b = 0; drift = a·δ = −1 (< 0) ⇒ α = slack/−drift = 0. The
    // whole joint step is scaled by α, so BOTH components collapse.
    let slack = constraints.a.row(0).dot(&beta) - constraints.b[0];
    let drift = constraints.a.row(0).dot(&trial_step);
    assert!(drift < 0.0, "binding-row drift must be negative");
    let alpha = (slack / -drift).clamp(0.0, 1.0);
    let alpha_step_norm = {
        let s = &trial_step * alpha;
        s.dot(&s).sqrt()
    };
    assert!(
        alpha_step_norm < 1e-6,
        "α-crush must collapse the whole step on a binding row; got |step|={alpha_step_norm:.3e}"
    );

    // ── New mechanism: strict-interior cone projection ──────────────────────
    // Projects the trial point onto `β_0 ≥ 0`; the orthogonal component
    // (β_1 = 5) is preserved, the binding component is clipped to ~0.
    let projected = project_point_strictly_into_feasible_cone(&trial_point, &constraints)
        .expect("cone projection of the trial point must succeed");
    let projected_step = &projected - &beta;
    let projected_step_norm = projected_step.dot(&projected_step).sqrt();

    // The unconstrained coordinate's full motion survives the projection.
    assert!(
        (projected[1] - 5.0).abs() < 1e-9,
        "unconstrained coordinate must keep its full motion; got {:.6}",
        projected[1]
    );
    // The realized step magnitude is O(1) — orders of magnitude above the
    // α-crushed step (which would have frozen the solve).
    assert!(
        projected_step_norm > 4.9,
        "cone projection must preserve the unconstrained step magnitude; got |step|={projected_step_norm:.3e}"
    );
    assert!(
        projected_step_norm > 1e6 * alpha_step_norm,
        "projection step ({projected_step_norm:.3e}) must dwarf the α-crushed step ({alpha_step_norm:.3e})"
    );
    // The projected point is feasible (binding coordinate ≥ 0).
    assert!(
        projected[0] >= -1e-9,
        "projected binding coordinate must be feasible; got {:.3e}",
        projected[0]
    );
}

/// gam#979 gated QP-feasibility reroute discriminator. The constrained
/// joint-Newton path only bypasses the global α-crush when the α it WOULD apply
/// falls below `JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD`. This test pins that
/// discriminator on `compute_joint_feasibility_alpha`:
///   * a HEALTHY step (α = 1.0, no binding constraint) is at/above the threshold
///     ⇒ the legacy truncate + α path runs UNCHANGED (byte-identical numerics —
///     the guarantee for every currently-converging arm, e.g. binary BMS), and
///   * a PATHOLOGICAL step (α far below the threshold, a binding row crushing the
///     whole step) is detected as the crush case ⇒ the magnitude-preserving cone
///     projection is used instead.
#[test]
pub(crate) fn joint_feasibility_alpha_gate_discriminates_healthy_from_crush() {
    // Minimal family supplying a controllable per-block feasibility α via
    // `max_feasible_step_size`. α is the configured value (or `None` ⇒ no limit).
    #[derive(Clone)]
    struct AlphaFamily {
        alpha: Option<f64>,
    }
    impl CustomFamily for AlphaFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            // Single-block fixture: the engine always passes exactly one block.
            assert_eq!(block_states.len(), 1);
            Ok(FamilyEvaluation {
                log_likelihood: 0.0,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![0.0],
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                }],
            })
        }
        fn max_feasible_step_size(
            &self,
            block_states: &[ParameterBlockState],
            idx: usize,
            arr: &Array1<f64>,
        ) -> Result<Option<f64>, String> {
            // The configured α is returned regardless of the proposed step;
            // assert the engine hands us a well-formed single-block query.
            assert!(idx < block_states.len());
            assert!(!arr.is_empty());
            Ok(self.alpha)
        }
    }

    let states = vec![ParameterBlockState {
        beta: array![0.0],
        eta: array![0.0],
    }];
    let ranges = vec![(0usize, 1usize)];
    let step = array![1.0_f64];

    // No feasibility limit ⇒ α = 1.0 (fully feasible). At/above the threshold:
    // the legacy path runs unchanged.
    let healthy = AlphaFamily { alpha: None };
    let (alpha_healthy, _) =
        compute_joint_feasibility_alpha(&healthy, &states, &ranges, &step).unwrap();
    assert_eq!(alpha_healthy, 1.0, "no constraint ⇒ α = 1.0");
    assert!(
        alpha_healthy >= JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
        "healthy α must NOT trip the crush bypass (legacy path stays byte-identical)"
    );

    // A moderate limit just above the threshold is still NOT a crush: legacy
    // α-scaling applies, no reroute.
    let moderate = AlphaFamily {
        alpha: Some(2.0 * JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD),
    };
    let (alpha_moderate, _) =
        compute_joint_feasibility_alpha(&moderate, &states, &ranges, &step).unwrap();
    assert!(
        alpha_moderate >= JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
        "moderate α (2× threshold) must stay on the legacy path"
    );

    // The survival pathology: α ≈ 1e-4 on a binding monotone row. Below the
    // threshold ⇒ the bypass fires and the cone projection takes over.
    let crush = AlphaFamily { alpha: Some(1e-4) };
    let (alpha_crush, limiting) =
        compute_joint_feasibility_alpha(&crush, &states, &ranges, &step).unwrap();
    assert!(
        alpha_crush < JOINT_FEASIBILITY_ALPHA_CRUSH_THRESHOLD,
        "the survival pathology (α≈1e-4) must trip the crush bypass; got α={alpha_crush:.3e}"
    );
    assert_eq!(limiting, Some(0), "the binding block must be reported");
}

/// gam#979 (per-block exact-Newton arm; the bernoulli marginal-slope binary
/// path). The per-block left-hand side is `lhs = H_data + S` with `S ⪰ 0` an
/// over-smoothed block penalty. A naive Gershgorin bound on the *penalized*
/// matrix `lhs` (computed inline below) reads a spurious huge-negative `λ_min`
/// because `S`'s large off-diagonals are balanced by equally large diagonals →
/// adds a giant ridge → collapses every per-block Newton step (the survival-hang
/// fingerprint). The PSD-penalized variant
/// [`stabilize_exact_newton_penalized_lhs_in_place`] must bound the shift by the
/// DATA Hessian's curvature (`exact_newton_stabilizing_shift_psd_penalized`)
/// instead, leaving the step well-scaled.
#[test]
pub(crate) fn per_block_penalized_shift_stays_data_scaled_under_oversmoothed_penalty() {
    // Data Hessian with one NEGATIVE eigenvalue along (1,−1,0) (the concave
    // entry-survival term makes the per-block data Hessian indefinite away from
    // the optimum). Crucially that negative direction lies in `ker(S)` of the
    // over-smoothed penalty below, so the penalty does NOT lift it — the
    // penalized matrix `lhs` stays genuinely indefinite, the no-shift Cholesky
    // FAILS, and the shift branch (with its Gershgorin bound) actually runs. On
    // a PD matrix the shared fast path returns `None` before Gershgorin is ever
    // consulted, which is exactly why the bug only bites an indefinite cycle.
    //
    // `h_data = I − 1.4·(1,−1,0)(1,−1,0)ᵀ/2`: eigenvalue 1 − 1.4 = −0.4 along
    // (1,−1,0)/√2, +1.0 on the orthogonal complement (curvature scale ≈ 1).
    let h_data = array![
        [1.0 - 0.7, 0.7, 0.0],
        [0.7, 1.0 - 0.7, 0.0],
        [0.0, 0.0, 1.0],
    ];

    // Heavily over-smoothed PSD penalty: a rank-1 `λ·vvᵀ` with `v = (1,1,1)` and
    // `λ ≈ 1e7`. It is PSD (single eigenvalue 3λ along (1,1,1); zero on the
    // orthogonal complement, which CONTAINS the data's negative direction
    // (1,−1,0)). Its large off-diagonals `λ·v_i v_j` are balanced by equally
    // large diagonals `λ·v_i²`, so the matrix is exactly PSD — but the per-row
    // Gershgorin `diag − radius = λ(v_i² − v_i·Σ_{j≠i}|v_j|) = λ(1 − 2) = −λ` is
    // hugely negative. This is the exact shape that fools plain Gershgorin into
    // reading a spurious ~−λ `λ_min` even though `S` adds NO indefiniteness.
    let lam = 1.0e7_f64;
    let s = &array![[1.0_f64, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],] * lam;

    let lhs = &h_data + &s;

    // Sanity: the penalized matrix is genuinely indefinite (Cholesky fails), so
    // the shift branch runs rather than the PD fast path.
    assert!(
        lhs.cholesky(Side::Lower).is_err(),
        "penalized lhs must stay indefinite along ker(S) ∋ (1,−1,0) so the shift branch engages"
    );

    let ridge_floor = 1.0e-12_f64;

    // ── Naive penalized-Gershgorin shift (the bug) ──────────────────────────
    // Gershgorin lower bound on the PENALIZED matrix `lhs = H_data + S`:
    //   min_i (lhs_ii − Σ_{j≠i} |lhs_ij|).
    // The over-smoothed `S` drives this hugely negative (~−2λ), so the lifting
    // shift `floor − g` is ~λ-scale — the spurious ridge that froze the survival
    // per-block Newton. We compute it directly here (rather than via the now
    // deleted plain-Gershgorin wrapper) so the contrast is explicit and the
    // test does not depend on the data-bounded fast path's Cholesky outcome.
    let p = lhs.nrows();
    let naive_gershgorin_min = (0..p)
        .map(|i| {
            let radius: f64 = (0..p).filter(|&j| j != i).map(|j| lhs[[i, j]].abs()).sum();
            lhs[[i, i]] - radius
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        naive_gershgorin_min < -1.0e6,
        "over-smoothed penalty must make the naive penalized-Gershgorin bound spuriously huge-negative; got {naive_gershgorin_min:.3e}"
    );
    let naive_shift = ridge_floor.max(1e-15) - naive_gershgorin_min;
    assert!(
        naive_shift > 1.0e6,
        "naive penalized-Gershgorin shift should read the spurious ~λ ridge; got {naive_shift:.3e}",
    );

    // ── PSD-penalized shift (the fix): Gershgorin bounded by the data Hessian.
    let psd_shift =
        exact_newton_stabilizing_shift_psd_penalized(&lhs, &h_data, ridge_floor).unwrap_or(0.0);

    // The data Hessian's most-negative eigenvalue is −0.4, so the data-bounded
    // shift stays O(data scale) (a few units), NOT the ~1e7 penalty scale.
    assert!(
        psd_shift < 10.0,
        "PSD-penalized shift must stay O(data scale), NOT the ~{lam:.0e} penalty scale; got {psd_shift:.3e}",
    );
    // And it must lift the genuine data indefiniteness (it is positive).
    assert!(
        psd_shift > 0.0,
        "PSD-penalized shift must still lift the data Hessian's negative eigenvalue; got {psd_shift:.3e}"
    );
    // Concretely: the data-bounded shift is ≥ 5 orders of magnitude smaller than
    // the spurious naive one.
    assert!(
        psd_shift * 1.0e5 < naive_shift,
        "PSD-penalized shift ({psd_shift:.3e}) must be ≥1e5× smaller than the spurious naive shift ({naive_shift:.3e})",
    );

    // And the shift restores positive (semi)definiteness: by Weyl,
    // `λ_min(lhs + δI) ≥ λ_min(H_data) + δ ≥ λ_min(H_data) − gershgorin_min(H_data) ≥ 0`,
    // because `λ_min(H_data) ≥ gershgorin_min(H_data)`. So the shift covers the
    // data Hessian's most-negative eigenvalue. Verify the data-Gershgorin bound
    // the shift is built from is at least as negative as `H_data`'s true λ_min,
    // and that `lhs + (δ + margin)·I` is PD (the floor makes the borderline
    // λ_min = 0 case strictly PD downstream; we add a tiny margin here so the
    // numerical Cholesky is unambiguous).
    let data_gershgorin_min = (0..p)
        .map(|i| {
            let radius: f64 = (0..p)
                .filter(|&j| j != i)
                .map(|j| h_data[[i, j]].abs())
                .sum();
            h_data[[i, i]] - radius
        })
        .fold(f64::INFINITY, f64::min);
    assert!(
        psd_shift >= -data_gershgorin_min - 1e-9,
        "PSD-penalized shift ({psd_shift:.3e}) must cover the data-Gershgorin bound ({data_gershgorin_min:.3e})"
    );
    let mut stabilized = lhs.clone();
    for d in 0..stabilized.nrows() {
        stabilized[[d, d]] += psd_shift + 1e-6;
    }
    assert!(
        stabilized.cholesky(Side::Lower).is_ok(),
        "stabilized penalized lhs must be PD after the PSD-penalized shift",
    );
}

#[test]
pub(crate) fn joint_solver_ridge_stabilizes_dense_indefinite_coupled_hessian() {
    let family = TwoBlockJointConstrainedFamily { coupling: 2.0 };
    let source = JointHessianSource::Dense(array![[1.0, 2.0], [2.0, 1.0]]);
    let ranges = vec![(0, 1), (1, 2)];
    let s_lambdas = vec![Array2::zeros((1, 1)), Array2::zeros((1, 1))];
    let ridge = stabilized_joint_solver_diagonal_ridge(
        &family,
        &source,
        &ranges,
        &s_lambdas,
        JOINT_TRACE_STABILITY_RIDGE,
        1e-12,
        None,
    );

    assert!(
        ridge > 1.0,
        "dense joint solver ridge should lift the negative eigenvalue; got {ridge}"
    );
    let mut stabilized = match source {
        JointHessianSource::Dense(matrix) => matrix,
        JointHessianSource::Operator { .. } => {
            panic!("dense joint solver fixture must use a dense Hessian source")
        }
    };
    add_joint_penalty_to_matrix(&mut stabilized, &ranges, &s_lambdas, ridge, None);
    let min_eval = 0.5
        * (stabilized[[0, 0]] + stabilized[[1, 1]]
            - ((stabilized[[0, 0]] - stabilized[[1, 1]]).powi(2)
                + 4.0 * stabilized[[0, 1]].powi(2))
            .sqrt());
    assert!(
        min_eval > 0.0,
        "stabilized dense joint Hessian should be SPD; min_eval={min_eval}"
    );
}

#[test]
pub(crate) fn outergradient_uses_joint_surrogate_formultiblock_diagonal_family() {
    let spec0 = ParameterBlockSpec {
        name: "block0".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let spec1 = ParameterBlockSpec {
        name: "block1".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: 1e-10,
        outer_max_iter: 1,
        ..BlockwiseFitOptions::default()
    };
    let penalty_counts = vec![1usize, 1usize];
    let rho = array![0.0, 0.0];

    let result = outerobjective_andgradient(
        &TwoBlockJointSurrogateFamily,
        &[spec0, spec1],
        &options,
        &penalty_counts,
        &rho,
        None,
    );
    assert!(
        result.is_ok(),
        "default joint multi-block surrogate path should succeed without blockwise dW callbacks: {:?}",
        result.err()
    );
}

#[test]
pub(crate) fn exact_newton_pseudo_laplace_objective_uses_logdet_h_without_logdet_s() {
    let spec = ParameterBlockSpec {
        name: "pseudo_laplace".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: CUSTOM_FAMILY_RIDGE_FLOOR,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let fit = fit_custom_family(
        &OneBlockPseudoLaplaceExactFamily { target: 1.5 },
        &[spec],
        &options,
    )
    .expect("pseudo-laplace exact-newton fit");
    let expected = 0.5 * 2.0_f64.ln();
    assert!(
        (fit.penalized_objective - expected).abs() < 1e-8,
        "pseudo-Laplace objective mismatch: got {}, expected {}",
        fit.penalized_objective,
        expected
    );
}

#[test]
pub(crate) fn exact_newton_joint_psi_hook_can_supply_fixed_beta_termswithout_quadratic_spsi() {
    let spec = ParameterBlockSpec {
        name: "psi_hook".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let deriv = CustomFamilyBlockPsiDerivative {
        penalty_index: None,
        x_psi: Array2::zeros((1, 1)),
        s_psi: Array2::zeros((1, 1)),
        s_psi_components: None,
        s_psi_penalty_components: None,
        x_psi_psi: None,
        s_psi_psi: None,
        s_psi_psi_components: None,
        s_psi_psi_penalty_components: None,
        implicit_operator: None,
        implicit_axis: 0,
        implicit_group_id: None,
    };
    let result = evaluate_custom_family_joint_hyper(
        &OneBlockExactPsiHookFamily,
        &[spec],
        &BlockwiseFitOptions {
            use_remlobjective: true,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        },
        &Array1::zeros(0),
        &[vec![deriv]],
        None,
        EvalMode::ValueAndGradient,
    )
    .expect("joint hyper eval with exact joint psi hook");
    assert_eq!(result.gradient.len(), 1);
    assert!(
        (result.gradient[0] - 3.5).abs() < 1e-12,
        "expected family-supplied joint psi term, got {}",
        result.gradient[0]
    );
}

#[test]
pub(crate) fn pseudo_laplace_exact_newton_rejects_indefinite_hessian() {
    // #748: an indefinite joint coefficient Hessian (here a 1×1 block with
    // H=-1) is a real defect — a mis-signed / non-convex curvature, or a β
    // that is not at the inner block optimum. The strict pseudo-Laplace
    // REML logdet must REJECT such a ρ-trial, not mask it. The earlier path
    // returned `log|H + δI|` with δ escalated to 10 (so H+δI=[[9]],
    // logdet=log 9) and let the fit "succeed" — but the analytic REML
    // gradient still used `tr((H+S_λ)⁻¹·)` on the un-ridged H, so value and
    // gradient described two different objectives. Rejecting is the honest
    // signal: the outer optimizer steps back instead of optimizing a biased,
    // δ-shifted surface. The fit therefore now ERRORS where it formerly
    // returned a masked result.
    let spec = ParameterBlockSpec {
        name: "indefinite".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let result = fit_custom_family(
        &OneBlockIndefinitePseudoLaplaceFamily,
        &[spec],
        &BlockwiseFitOptions {
            use_remlobjective: true,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        },
    );
    let err = result
            .expect_err(
                "strict pseudo-Laplace must reject the indefinite Hessian H=[[-1]], not δ-ridge mask it",
            )
            .to_string();
    assert!(
        err.contains("indefinite") || err.contains("below -tol"),
        "rejection error should name the indefiniteness; got: {err}",
    );
}

#[test]
pub(crate) fn auto_determinant_mode_is_exact_full_logdet_policy() {
    let h = array![[6.0, 0.8, 0.1], [0.8, 4.5, 0.4], [0.1, 0.4, 3.2]];
    let exact = stable_logdet_with_ridge_policy(&h, 1e-8, RidgePolicy::exact_full_objective())
        .expect("exact logdet");
    let auto = stable_logdet_with_ridge_policy(&h, 1e-8, RidgePolicy::exact_full_objective())
        .expect("auto logdet");
    assert!((auto - exact).abs() < 1e-12, "auto={auto}, exact={exact}");
}

#[test]
pub(crate) fn indefinite_hessian_uses_smooth_regularized_logdet() {
    // Indefinite Hessian: eigenvalues {-1, 2}.
    //
    // Old behaviour: silently drop the -1 direction from logdet, warn,
    // and after enough repeats escalate to an EFS abort (first-order
    // fallback marker).
    //
    // New behaviour: every eigenvalue contributes via the smooth
    // regularizer r_ε(σ) = ½(σ + √(σ² + 4ε²)).  No direction is ignored,
    // no escalation, and the logdet matches what the downstream
    // `DenseSpectralOperator` gradient computes — eliminating the
    // cost/gradient mismatch that broke BFGS line search.
    let h = array![[-1.0, 0.0], [0.0, 2.0]];
    let logdet = stable_logdet_with_ridge_policy(
        &h,
        1e-12,
        RidgePolicy::positive_part_approximate_objective(),
    )
    .expect("smooth-regularized logdet must be finite for indefinite H");
    assert!(
        logdet.is_finite(),
        "smooth-regularized logdet should be finite, got {logdet}"
    );
    // Reference value using the same formula directly on the eigenvalues
    // of H + ridge·I (ridge = 1e-12 here).  Since ε ≫ ridge (spectral_epsilon
    // floors at √(eps_mach) ≈ 1.5e-8 for p=2), the ridge contribution is
    // absorbed into ε and the expected value is Σ log r_ε(σ_j).
    let eps = spectral_epsilon(&[-1.0_f64, 2.0]).max(1e-12_f64.max(1e-14));
    // A + ridge·I has eigenvalues shifted by 1e-12, negligible relative to ε.
    let expected: f64 = [-1.0_f64 + 1e-12, 2.0 + 1e-12]
        .iter()
        .map(|&s| spectral_regularize(s, eps).ln())
        .sum();
    assert!(
        (logdet - expected).abs() < 1e-10,
        "logdet={logdet}, expected={expected}"
    );
}

#[test]
pub(crate) fn pseudo_laplace_exact_newton_symmetrizes_nearly_symmetrichessian() {
    let spec = ParameterBlockSpec {
        name: "nearly_symmetric".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0, 0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let fit = fit_custom_family(
        &OneBlockNearlySymmetricPseudoLaplaceFamily,
        &[spec],
        &BlockwiseFitOptions {
            use_remlobjective: true,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        },
    )
    .expect("nearly symmetric pseudo-laplace Hessian should be accepted after symmetrization");
    assert!(
        fit.penalized_objective.is_finite(),
        "expected finite pseudo-laplace objective, got {}",
        fit.penalized_objective
    );
}

#[test]
pub(crate) fn outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active() {
    let BinomialLocationScaleWiggleOuterFixture {
        family,
        specs,
        penalty_counts,
        rho,
        options: base_options,
    } = binomial_location_scale_wiggle_outer_fixture();
    // FD/analytic noise floor below is `EPS·|cost|/h`, valid only when PIRLS
    // converges to f64 precision; HardPseudo + σ_min~1e-10 amplifies the
    // default 1e-6 inner residual into ~1e-7 cost slack that lifts both
    // estimators above the machine-precision floor.
    let options = BlockwiseFitOptions {
        inner_tol: 1e-12,
        inner_max_cycles: 500,
        ..base_options
    };

    let (f0, g0, _) =
        outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho, None)
            .expect("objective/gradient");
    assert!(f0.is_finite());
    assert_eq!(g0.len(), rho.len());

    let h = 1e-5;
    for k in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[k] += h;
        rho_m[k] -= h;
        let (fp, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho_p, None)
                .expect("objective+");
        let (fm, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho_m, None)
                .expect("objective-");
        let gfd = (fp - fm) / (2.0 * h);

        // Noise floor for FD-vs-analytic comparisons.
        //
        // At a rank-deficient optimum (σ_min(H) ≲ ε_machine) the outer
        // REML gradient is a DIFFERENCE of two nearly-equal O(1)
        // quantities — ½ λ_k (H⁺[k,k] − S⁺[k,k]) — so the true gradient
        // is very close to zero.  The FD estimator `(f_p − f_m)/(2h)`
        // then measures cost-sum round-off: at f64 precision each cost
        // value carries an uncertainty of ~EPS · |cost|, and the
        // symmetric FD inflates that by 1/(2h), producing a noise floor
        // of roughly `EPS · |cost| / h` on |gfd|.  Below that floor
        // neither `|gfd|`, `|g0|`, nor `sign(gfd)` reflect the true
        // derivative — they reflect arithmetic noise.
        //
        // Concretely: for this test `|cost| ~ 6`, `h = 1e-5`, so the
        // floor is ~1.3e-10 (≈ f64::EPSILON · 6 / 1e-5).  We round up
        // to a problem-scale-derived value and treat pairs where BOTH
        // |g0| and |gfd| lie below the floor as a pass (the assertion
        // is making a claim about the TRUE derivative, and a true
        // derivative strictly less than noise is indistinguishable
        // from zero — sign is not a correctness property there).
        let cost_magnitude = f0.abs().max(1.0);
        let noise_floor = (10.0 * f64::EPSILON * cost_magnitude / h).max(1e-9);
        let both_in_noise = g0[k].abs() < noise_floor && gfd.abs() < noise_floor;

        if !both_in_noise {
            assert_eq!(
                g0[k].signum(),
                gfd.signum(),
                "outer LAML gradient sign mismatch at {}: analytic={} fd={} noise_floor={:.3e}",
                k,
                g0[k],
                gfd,
                noise_floor,
            );
            let rel = (g0[k] - gfd).abs() / gfd.abs().max(noise_floor);
            assert!(
                rel < 2e-2,
                "outer LAML gradient mismatch at {}: analytic={} fd={} rel={} noise_floor={:.3e}",
                k,
                g0[k],
                gfd,
                rel,
                noise_floor,
            );
        }
    }
}

#[test]
pub(crate) fn rho_only_outer_objective_matches_joint_hyper_when_psi_is_empty() {
    let BinomialLocationScaleWiggleOuterFixture {
        family,
        specs,
        penalty_counts,
        rho,
        options,
    } = binomial_location_scale_wiggle_outer_fixture();

    let (outer_obj, outer_grad, outer_hessian, _) =
        super::test_support::outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho,
            None,
            EvalMode::ValueGradientHessian,
        )
        .expect("rho-only outer objective");
    let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
    let joint_result = evaluate_custom_family_joint_hyper(
        &family,
        &specs,
        &options,
        &rho,
        &derivative_blocks,
        None,
        EvalMode::ValueGradientHessian,
    )
    .expect("joint hyper objective with empty psi");

    assert!(
        (outer_obj - joint_result.objective).abs() < 1e-12,
        "objective mismatch: rho-only={} joint={}",
        outer_obj,
        joint_result.objective
    );
    assert_eq!(outer_grad.len(), joint_result.gradient.len());
    let max_grad_diff = outer_grad
        .iter()
        .zip(joint_result.gradient.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_grad_diff < 1e-12,
        "gradient mismatch: max diff={}",
        max_grad_diff
    );

    let outer_hessian = outer_hessian.expect("rho-only outer Hessian");
    let joint_hessian = joint_result
        .outer_hessian
        .materialize_dense()
        .expect("joint outer Hessian should materialize")
        .expect("joint outer Hessian");
    assert_eq!(outer_hessian.dim(), joint_hessian.dim());
    let max_hessian_diff = outer_hessian
        .iter()
        .zip(joint_hessian.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_hessian_diff < 1e-12,
        "outer Hessian mismatch: max diff={}",
        max_hessian_diff
    );
}

/// Shared probit binomial-location-scale outer-derivative test fixture:
/// builds the (threshold, log_sigma) block specs, family, penalty counts,
/// and outer options that every `outer_laml*_binomial_location_scale_*`
/// finite-difference test constructs identically apart from `y` and the
/// two block initial betas.
pub(crate) fn binomial_location_scale_outer_fixture(
    y: Array1<f64>,
    threshold_initial_beta: f64,
    log_sigma_initial_beta: f64,
) -> (
    BinomialLocationScaleFamily,
    Vec<ParameterBlockSpec>,
    Vec<usize>,
    BlockwiseFitOptions,
) {
    let n = y.len();
    let weights = Array1::from_elem(n, 1.0);
    let thresholdspec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        )),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![threshold_initial_beta]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let log_sigmaspec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        )),
        offset: Array1::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![log_sigma_initial_beta]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let threshold_design = thresholdspec.design.clone();
    let log_sigma_design = log_sigmaspec.design.clone();
    let family = BinomialLocationScaleFamily {
        y,
        weights,
        link_kind: gam_problem::InverseLink::Standard(gam_problem::StandardLink::Probit),
        threshold_design: Some(threshold_design),
        log_sigma_design: Some(log_sigma_design),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let specs = vec![thresholdspec, log_sigmaspec];
    let penalty_counts = vec![1usize, 1usize];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: 1e-10,
        outer_max_iter: 1,
        ..BlockwiseFitOptions::default()
    };
    (family, specs, penalty_counts, options)
}

#[test]
pub(crate) fn outer_lamlgradient_diagonal_binomial_location_scale_matchesfd() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let (family, specs, penalty_counts, options) =
        binomial_location_scale_outer_fixture(y, 0.0, 0.0);
    let rho = array![0.0, 0.0];

    let (f0, g0, _) =
        outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho, None)
            .expect("objective/gradient");
    assert!(f0.is_finite());
    assert_eq!(g0.len(), rho.len());

    let h = 1e-5;
    for k in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[k] += h;
        rho_m[k] -= h;
        let (fp, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho_p, None)
                .expect("objective+");
        let (fm, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho_m, None)
                .expect("objective-");
        let gfd = (fp - fm) / (2.0 * h);
        let abs = (g0[k] - gfd).abs();
        let rel = abs / gfd.abs().max(1e-8);
        if abs >= 2e-3 {
            assert_eq!(
                g0[k].signum(),
                gfd.signum(),
                "outer diagonal LAML gradient sign mismatch at {}: analytic={} fd={}",
                k,
                g0[k],
                gfd
            );
        }
        assert!(
            abs < 2e-3 || rel < 2e-3,
            "outer diagonal LAML gradient mismatch at {}: analytic={} fd={} abs={} rel={}",
            k,
            g0[k],
            gfd,
            abs,
            rel
        );
    }
}

#[test]
pub(crate) fn outer_lamlgradient_diagonal_binomial_location_scale_hard_case_matchesfd() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let (family, specs, penalty_counts, options) =
        binomial_location_scale_outer_fixture(y, 0.2, -0.1);
    let rho = array![0.15, -0.25];

    let (f0, g0, _) =
        outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho, None)
            .expect("objective/gradient");
    assert!(f0.is_finite());
    assert_eq!(g0.len(), rho.len());

    let h = 1e-5;
    for k in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[k] += h;
        rho_m[k] -= h;
        let (fp, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho_p, None)
                .expect("objective+");
        let (fm, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &penalty_counts, &rho_m, None)
                .expect("objective-");
        let gfd = (fp - fm) / (2.0 * h);
        let abs = (g0[k] - gfd).abs();
        let rel = abs / gfd.abs().max(1e-8);
        if abs >= 2e-3 {
            assert_eq!(
                g0[k].signum(),
                gfd.signum(),
                "outer diagonal hard-case LAML gradient sign mismatch at {}: analytic={} fd={}",
                k,
                g0[k],
                gfd
            );
        }
        assert!(
            abs < 2e-3 || rel < 2e-3,
            "outer diagonal hard-case LAML gradient mismatch at {}: analytic={} fd={} abs={} rel={}",
            k,
            g0[k],
            gfd,
            abs,
            rel
        );
    }
}

#[test]
pub(crate) fn outer_lamlhessian_joint_exact_binomial_location_scale_matchesfd() {
    // Asymmetric y (6 ones / 4 zeros). A balanced 5/5 vector forces
    // β̂_threshold = 0 by probit-link symmetry, which makes the joint
    // observed Hessian block-diagonal in (threshold, log_sigma) at the
    // inner mode. The outer LAML Hessian off-diagonals are then ~1e-11,
    // below the central-FD noise floor (≈ pirls_tol / h) at h=1e-5, so
    // FD-vs-analytic agreement cannot be enforced. Asymmetric y gives
    // β̂_threshold ≠ 0, coupling the (β_0, β_1) blocks through the
    // observed-information weights and making all four entries validatable.
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
    let (family, specs, penalty_counts, options) =
        binomial_location_scale_outer_fixture(y, 0.15, -0.05);
    let rho = array![0.1, -0.2];

    let (_, _, h0_opt, _) = super::test_support::outerobjectivegradienthessian(
        &family,
        &specs,
        &options,
        &penalty_counts,
        &rho,
        None,
        EvalMode::ValueGradientHessian,
    )
    .expect("objective/gradient/hessian");
    let h0 = h0_opt.expect("analytic outer Hessian should be available");
    assert_eq!(h0.nrows(), rho.len());
    assert_eq!(h0.ncols(), rho.len());

    let h = 1e-5;
    for l in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[l] += h;
        rho_m[l] -= h;
        let (_, gp, _, _) = super::test_support::outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho_p,
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("objective/gradient +");
        let (_, gm, _, _) = super::test_support::outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho_m,
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("objective/gradient -");

        for k in 0..rho.len() {
            let hfd = (gp[k] - gm[k]) / (2.0 * h);
            let abs_err = (h0[[k, l]] - hfd).abs();
            let rel = (h0[[k, l]] - hfd).abs() / hfd.abs().max(1e-7);
            if h0[[k, l]].abs().max(hfd.abs()) > 1e-10 {
                assert_eq!(
                    h0[[k, l]].signum(),
                    hfd.signum(),
                    "outer Hessian sign mismatch at ({k},{l}): analytic={} fd={}",
                    h0[[k, l]],
                    hfd
                );
            }
            assert!(
                abs_err < 1e-8 || rel < 2e-2,
                "outer Hessian mismatch at ({k},{l}): analytic={} fd={} abs={} rel={}",
                h0[[k, l]],
                hfd,
                abs_err,
                rel
            );
        }
    }

    for i in 0..h0.nrows() {
        for j in 0..i {
            let asym = (h0[[i, j]] - h0[[j, i]]).abs();
            assert!(
                asym < 1e-8,
                "outer Hessian not symmetric at ({i},{j}): {asym}"
            );
        }
    }
}

#[test]
pub(crate) fn block_solve_sparse_matches_dense() {
    let x_dense = array![
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0],
        [0.0, 6.0, 0.0]
    ];
    let y_star = array![1.0, -1.0, 0.5, 2.0];
    let w = array![1.0, 0.5, 2.0, 1.5];
    let s_lambda = Array2::<f64>::eye(3) * 0.1;

    let mut triplets = Vec::new();
    for i in 0..x_dense.nrows() {
        for j in 0..x_dense.ncols() {
            let v = x_dense[[i, j]];
            if v != 0.0 {
                triplets.push(Triplet::new(i, j, v));
            }
        }
    }
    let x_sparse = SparseColMat::try_new_from_triplets(4, 3, &triplets)
        .expect("sparse matrix build should succeed");

    let beta_dense = solve_blockweighted_system(
        &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_dense.clone())),
        &y_star,
        &w,
        &s_lambda,
        1e-12,
        RidgePolicy::positive_part_approximate_objective(),
    )
    .expect("dense solve should succeed");

    let beta_sparse = solve_blockweighted_system(
        &DesignMatrix::from(x_sparse),
        &y_star,
        &w,
        &s_lambda,
        1e-12,
        RidgePolicy::positive_part_approximate_objective(),
    )
    .expect("sparse solve should succeed");

    for j in 0..beta_dense.len() {
        assert!(
            (beta_dense[j] - beta_sparse[j]).abs() < 1e-10,
            "dense/sparse mismatch at {}: {} vs {}",
            j,
            beta_dense[j],
            beta_sparse[j]
        );
    }
}

#[test]
pub(crate) fn outer_lamlhessian_joint_exact_binomial_location_scale_hard_case_matchesfd() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let (family, specs, penalty_counts, options) =
        binomial_location_scale_outer_fixture(y, 0.2, -0.1);
    let rho = array![0.15, -0.25];

    let (_, _, h0_opt, _) = super::test_support::outerobjectivegradienthessian(
        &family,
        &specs,
        &options,
        &penalty_counts,
        &rho,
        None,
        EvalMode::ValueGradientHessian,
    )
    .expect("objective/gradient/hessian");
    let h0 = h0_opt.expect("analytic outer Hessian should be available");
    assert_eq!(h0.nrows(), rho.len());
    assert_eq!(h0.ncols(), rho.len());

    let h = 1e-5;
    for l in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[l] += h;
        rho_m[l] -= h;
        let (_, gp, _, _) = super::test_support::outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho_p,
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("objective/gradient +");
        let (_, gm, _, _) = super::test_support::outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho_m,
            None,
            EvalMode::ValueAndGradient,
        )
        .expect("objective/gradient -");

        for k in 0..rho.len() {
            let hfd = (gp[k] - gm[k]) / (2.0 * h);
            let abs_err = (h0[[k, l]] - hfd).abs();
            let rel = abs_err / hfd.abs().max(1e-7);
            if h0[[k, l]].abs().max(hfd.abs()) > 1e-10 {
                assert_eq!(
                    h0[[k, l]].signum(),
                    hfd.signum(),
                    "hard-case outer Hessian sign mismatch at ({k},{l}): analytic={} fd={}",
                    h0[[k, l]],
                    hfd
                );
            }
            assert!(
                abs_err < 1e-8 || rel < 2e-2,
                "hard-case outer Hessian mismatch at ({k},{l}): analytic={} fd={} abs={} rel={}",
                h0[[k, l]],
                hfd,
                abs_err,
                rel
            );
        }
    }
}

#[test]
pub(crate) fn block_solve_falls_backwhen_llt_rejects_indefinite_system() {
    let x_dense = array![[1.0, 0.0], [0.0, 0.0]];
    let y_star = array![2.0, 0.0];
    let w = array![1.0, 1.0];
    let s_lambda = array![[0.0, 0.0], [0.0, -1e-12]];

    let beta = solve_blockweighted_system(
        &DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_dense)),
        &y_star,
        &w,
        &s_lambda,
        1e-12,
        RidgePolicy::positive_part_approximate_objective(),
    )
    .expect("fallback solve should succeed");

    assert!(beta.iter().all(|v| v.is_finite()));
    assert!(
        (beta[0] - 2.0).abs() < 1e-10,
        "unexpected solved coefficient"
    );
    assert!(
        beta[1].abs() < 1e-8,
        "null-space coefficient should stay near zero"
    );
}

#[test]
pub(crate) fn exact_newton_block_enforces_linear_constraints() {
    let spec = ParameterBlockSpec {
        name: "exact_block".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![1.5]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let family = OneBlockConstrainedExactFamily {
        target: 0.0,
        lower: 1.0,
    };
    let fit = fit_custom_family(&family, &[spec], &BlockwiseFitOptions::default())
        .expect("constrained exact-newton fit");
    let beta = fit.block_states[0].beta[0];
    assert!(
        (beta - 1.0).abs() < 1e-8,
        "expected constrained optimum at lower bound, got {beta}"
    );
}

#[test]
pub(crate) fn extract_simple_lower_bounds_accepts_axis_aligned_rows() {
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]],
        b: array![0.25, 1.0, 1.5],
    };
    let bounds = extract_simple_lower_bounds(&constraints, 2)
        .expect("lower-bound extraction should succeed")
        .expect("axis-aligned rows should map to lower bounds");
    assert_relative_eq!(bounds.lower_bounds[0], 0.5, epsilon = 1e-12);
    assert_relative_eq!(bounds.lower_bounds[1], 0.5, epsilon = 1e-12);
    assert_eq!(bounds.coeff_to_row, vec![Some(2), Some(1)]);
}

#[test]
pub(crate) fn extract_simple_lower_bounds_rejects_coupled_rows() {
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 1.0]],
        b: array![0.0],
    };
    assert!(
        extract_simple_lower_bounds(&constraints, 2)
            .expect("lower-bound extraction should not error on valid shapes")
            .is_none(),
        "coupled rows must stay on the generic linear-constraint path"
    );
}

#[test]
pub(crate) fn constrained_exact_newton_indefinite_hessian_uses_stabilized_delta_solve() {
    let spec = ParameterBlockSpec {
        name: "exact_block".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![1.5]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let states = vec![ParameterBlockState {
        beta: array![1.5],
        eta: array![1.5],
    }];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0]],
        b: array![1.0],
    };
    let hessian = SymmetricMatrix::Dense(array![[-1.0]]);
    let updater = ExactNewtonBlockUpdater {
        gradient: &array![-1.0],
        hessian: &hessian,
    };
    let s_lambda = Array2::zeros((1, 1));
    let update = updater
        .compute_update_step(&BlockUpdateContext {
            family: &OneBlockConstrainedIndefiniteHessianFamily,
            states: &states,
            spec: &spec,
            block_idx: 0,
            s_lambda: &s_lambda,
            options: &BlockwiseFitOptions::default(),
            linear_constraints: Some(&constraints),
            cached_active_set: None,
        })
        .expect("indefinite constrained exact-newton update should be stabilized");
    assert_relative_eq!(update.beta_new_raw[0], 1.0, epsilon = 1e-12);
    assert_eq!(update.active_set, Some(vec![0]));
}

#[test]
pub(crate) fn quadratic_linear_constraints_release_positive_kkt_systemmultiplier() {
    // max ll with exact Newton equivalent to minimizing
    // 0.5 * x^2 - rhs*x with rhs=1 under 0 <= x <= 0.1.
    // At x=0, active-set KKT solve gives lambda_sys=+1 for the lower bound,
    // which must be released (lambda_true = -lambda_sys).
    let hessian = array![[1.0]];
    let rhs = array![1.0];
    let beta_start = array![0.0];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0], [-1.0]],
        b: array![0.0, -0.1],
    };

    let (beta, active) =
        solve_quadratic_with_linear_constraints(&hessian, &rhs, &beta_start, &constraints, None)
            .expect("constrained quadratic solve should succeed");

    assert!(
        (beta[0] - 0.1).abs() <= 1e-10,
        "expected constrained optimum at upper bound 0.1, got {}",
        beta[0]
    );
    assert_eq!(active.len(), 1);
}

#[test]
pub(crate) fn quadratic_linear_constraints_ignore_near_tangential_inactiverows() {
    let hessian = array![[1.0, 0.0], [0.0, 1.0]];
    let rhs = array![1.0, 0.0];
    let beta_start = array![0.0, 0.0];
    let constraints = LinearInequalityConstraints {
        a: array![[-1e-16, 1.0]],
        b: array![-1.0],
    };

    let (beta, active) =
        solve_quadratic_with_linear_constraints(&hessian, &rhs, &beta_start, &constraints, None)
            .expect("near-tangential inactive row should not block the quadratic step");

    assert!(
        (beta[0] - 1.0).abs() <= 1e-12,
        "expected unconstrained x-solution of 1.0, got {}",
        beta[0]
    );
    assert!(
        beta[1].abs() <= 1e-12,
        "expected zero y-solution, got {}",
        beta[1]
    );
    assert!(active.is_empty(), "no row should become active");
}

#[test]
pub(crate) fn quadratic_linear_constraints_projectwarm_activerows_back_to_boundary() {
    let hessian = array![[2.0]];
    let rhs = array![0.0];
    let beta_start = array![1e-9];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0]],
        b: array![0.0],
    };

    let (beta, active) = solve_quadratic_with_linear_constraints(
        &hessian,
        &rhs,
        &beta_start,
        &constraints,
        Some(&[0]),
    )
    .expect("constrained quadratic solve should project back to the boundary");

    assert_relative_eq!(beta[0], 0.0, epsilon = 1e-14);
    assert_eq!(active, vec![0]);
}

#[test]
pub(crate) fn quadratic_linear_constraints_handles_near_dependent_rows() {
    // Three constraints in R^2 where the third is nearly a linear
    // combination of the first two, making the naive KKT system
    // ill-conditioned.  The rank-reducing compression should drop
    // the dependent row and the QP should converge cleanly.
    //
    //   x1 >= 0,  x2 >= 0,  x1 + x2 + eps >= 0   (eps ≈ 0)
    //
    // Minimize 0.5 * ||x - [−1, −1]||^2  =>  optimum at origin.
    let hessian = Array2::eye(2);
    let rhs = array![-1.0, -1.0]; // gradient points toward (−1,−1)
    let beta_start = array![0.0, 0.0];
    let eps = 1e-14;
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 0.0], [0.0, 1.0], [1.0 + eps, 1.0]],
        b: array![0.0, 0.0, 0.0],
    };

    let (beta, active) = solve_quadratic_with_linear_constraints(
        &hessian,
        &rhs,
        &beta_start,
        &constraints,
        Some(&[0, 1, 2]), // all three active
    )
    .expect("near-dependent constraint QP should converge");

    assert!(
        beta[0].abs() <= 1e-10 && beta[1].abs() <= 1e-10,
        "expected optimum at origin, got ({}, {})",
        beta[0],
        beta[1]
    );
    assert!(
        active.len() <= 2,
        "at most 2 independent constraints should remain active, got {}",
        active.len()
    );
}

#[test]
pub(crate) fn quadratic_linear_constraints_release_merged_constraint_group_by_id() {
    // Two redundant lower-bound rows compress into one active KKT row.
    // Releasing that merged row must drop both original constraint ids,
    // not transient positions in the active vector.
    let hessian = array![[1.0]];
    let rhs = array![1.0];
    let beta_start = array![0.0];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0], [2.0], [-1.0]],
        b: array![0.0, 0.0, -0.1],
    };

    let (beta, active) = solve_quadratic_with_linear_constraints(
        &hessian,
        &rhs,
        &beta_start,
        &constraints,
        Some(&[0, 1]),
    )
    .expect("merged active constraint group should release cleanly");

    assert!(
        (beta[0] - 0.1).abs() <= 1e-10,
        "expected constrained optimum at upper bound 0.1, got {}",
        beta[0]
    );
    assert_eq!(active, vec![2]);
}

#[test]
pub(crate) fn quadratic_linear_constraints_release_merged_group_with_unsorted_active_positions() {
    let hessian = array![[1.0]];
    let rhs = array![1.0];
    let beta_start = array![0.0];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0], [2.0], [-1.0]],
        b: array![0.0, 0.0, -0.1],
    };

    let (beta, active) = solve_quadratic_with_linear_constraints(
        &hessian,
        &rhs,
        &beta_start,
        &constraints,
        Some(&[2, 0, 1]),
    )
    .expect("merged active group release should handle unsorted active positions");

    assert!(
        (beta[0] - 0.1).abs() <= 1e-10,
        "expected constrained optimum at upper bound 0.1, got {}",
        beta[0]
    );
    assert_eq!(active, vec![2]);
}

#[test]
pub(crate) fn quadratic_linear_constraints_accept_boundary_kkt_after_rank_reduction() {
    let hessian = array![[2.0]];
    let rhs = array![0.0];
    let beta_start = array![1e-9];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0], [1.0 + 1e-13], [2.0], [3.0]],
        b: array![0.0, 0.0, 0.0, 0.0],
    };

    let (beta, active) = solve_quadratic_with_linear_constraints(
        &hessian,
        &rhs,
        &beta_start,
        &constraints,
        Some(&[0, 1, 2, 3]),
    )
    .expect("degenerate boundary KKT point should be accepted");

    assert_relative_eq!(beta[0], 0.0, epsilon = 1e-14);
    assert!(
        active.len() <= 1,
        "rank-reduced boundary solution should keep at most one representative, got {:?}",
        active
    );
}

#[test]
pub(crate) fn quadratic_linear_constraints_singular_kkt_uses_pseudoinverse_fallback() {
    let hessian = Array2::<f64>::zeros((2, 2));
    let rhs = array![0.0, 0.0];
    let beta_start = array![0.0, 0.0];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 1.0]],
        b: array![0.0],
    };

    let (beta, active) = solve_quadratic_with_linear_constraints(
        &hessian,
        &rhs,
        &beta_start,
        &constraints,
        Some(&[0]),
    )
    .expect("singular KKT system should fall back to a finite pseudoinverse solve");

    assert!(beta.iter().all(|value| value.is_finite()));
    assert_relative_eq!(beta[0], 0.0, epsilon = 1e-14);
    assert_relative_eq!(beta[1], 0.0, epsilon = 1e-14);
    assert_eq!(active, vec![0]);
}

#[test]
pub(crate) fn rank_reduce_drops_exactly_dependent_row() {
    // Row 3 = Row 1 + Row 2 exactly. Rank reduction should drop it.
    let a = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],];
    let b = array![0.0, 0.0, 0.0];
    let member_constraint_ids = vec![vec![0], vec![1], vec![2]];
    let (a_out, b_out, member_constraint_ids_out, _) =
        gam_solve::active_set::rank_reduce_rows_pivoted_qr_with_dependence(
            a,
            b,
            member_constraint_ids,
        );
    assert_eq!(
        a_out.nrows(),
        2,
        "should keep 2 independent rows, got {}",
        a_out.nrows()
    );
    assert_eq!(b_out.len(), 2);
    // The third constraint id should have been merged into one of the first two rows.
    let total_constraint_ids: usize = member_constraint_ids_out.iter().map(|g| g.len()).sum();
    assert_eq!(
        total_constraint_ids, 3,
        "all original constraint ids must be preserved"
    );
}

#[test]
pub(crate) fn rank_reduce_preserves_full_rank_matrix() {
    let a = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0],];
    let b = array![0.0, 0.0, 0.0];
    let member_constraint_ids = vec![vec![0], vec![1], vec![2]];
    let (a_out, b_out, member_constraint_ids_out, _) =
        gam_solve::active_set::rank_reduce_rows_pivoted_qr_with_dependence(
            a,
            b,
            member_constraint_ids,
        );
    // All three rows are independent in R^2 (but we only have rank 2).
    // The first two span R^2, so row 3 = row 1 + row 2 is dependent.
    assert_eq!(a_out.nrows(), 2);
    assert_eq!(b_out.len(), 2);
    let total_constraint_ids: usize = member_constraint_ids_out.iter().map(|g| g.len()).sum();
    assert_eq!(total_constraint_ids, 3);
}

#[test]
pub(crate) fn constrained_exact_newton_nan_hessian_returns_feasible_noop_instead_of_failing() {
    let spec = ParameterBlockSpec {
        name: "exact_block".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let states = vec![ParameterBlockState {
        beta: array![0.0],
        eta: array![0.0],
    }];
    let constraints = LinearInequalityConstraints {
        a: array![[1.0]],
        b: array![0.0],
    };
    let hessian = SymmetricMatrix::Dense(array![[f64::NAN]]);
    let updater = ExactNewtonBlockUpdater {
        gradient: &array![0.0],
        hessian: &hessian,
    };
    let s_lambda = Array2::zeros((1, 1));
    let update = updater
        .compute_update_step(&BlockUpdateContext {
            family: &OneBlockConstrainedNaNHessianFamily,
            states: &states,
            spec: &spec,
            block_idx: 0,
            s_lambda: &s_lambda,
            options: &BlockwiseFitOptions::default(),
            linear_constraints: Some(&constraints),
            cached_active_set: None,
        })
        .expect("constrained exact-newton NaN Hessian should produce a no-op update");
    assert_relative_eq!(update.beta_new_raw[0], 0.0, epsilon = 1e-14);
    assert_eq!(update.active_set, Some(vec![0]));
}

#[test]
pub(crate) fn outerobjective_failure_context_is_preserved() {
    // One penalty forces the outer rho optimizer to run, which should now preserve
    // the real evaluation error instead of returning an opaque line-search failure.
    let spec = ParameterBlockSpec {
        name: "err_block".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
        nullspace_dims: vec![],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        outer_max_iter: 3,
        ..BlockwiseFitOptions::default()
    };
    let err = match fit_custom_family(&OneBlockAlwaysErrorFamily, &[spec], &options) {
        Ok(_) => panic!("fit should fail when family evaluate always errors"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains(
            "last objective error: synthetic outer objective failure: block[0] evaluate()"
        ),
        "expected preserved root-cause context in error, got: {err}"
    );
}

#[test]
pub(crate) fn fit_fails_when_requested_covariance_cannot_be_computed() {
    let spec = ParameterBlockSpec {
        name: "cov_block".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0],
            [1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let options = BlockwiseFitOptions {
        use_remlobjective: false,
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    };
    let err = match fit_custom_family(&OneBlockCovarianceErrorFamily, &[spec], &options) {
        Ok(_) => panic!("fit should fail when covariance computation fails"),
        Err(e) => e,
    };
    assert!(
        err.to_string()
            .contains("synthetic covariance assembly failure"),
        "expected covariance root cause in fit error, got: {err}"
    );
}

// Exact analytic Hessians must be finite. Non-finite Hessians are rejected
// loudly instead of being masked by a surrogate update.

/// A QuadraticReml family whose log_sigma block returns a Hessian containing
/// NaN, simulating what happens when exp(eta_sigma) overflows during
/// location-scale fitting.
#[derive(Clone)]
pub(crate) struct TwoBlockNaNHessianFamily;

impl CustomFamily for TwoBlockNaNHessianFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n0 = block_states[0].eta.len();
        let p1 = block_states[1].beta.len();
        // Block 0 (mu): well-behaved diagonal working set.
        // Block 1 (log_sigma): ExactNewton with NaN in the Hessian,
        // simulating overflow from extreme coefficients.
        let mut hessian = Array2::<f64>::eye(p1);
        hessian[[0, 0]] = f64::NAN; // overflow poison
        Ok(FamilyEvaluation {
            log_likelihood: -0.5 * block_states[0].eta.iter().map(|&v| v * v).sum::<f64>(),
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n0),
                    working_weights: Array1::ones(n0),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: Array1::zeros(p1),
                    hessian: SymmetricMatrix::Dense(hessian),
                },
            ],
        })
    }
}

/// Same two-block layout but with finite Hessians — the control group.
#[derive(Clone)]
pub(crate) struct TwoBlockFiniteHessianFamily;

impl CustomFamily for TwoBlockFiniteHessianFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n0 = block_states[0].eta.len();
        let p1 = block_states[1].beta.len();
        let beta1 = &block_states[1].beta;
        let resid1: f64 = beta1.iter().map(|&b| b * b).sum();
        Ok(FamilyEvaluation {
            log_likelihood: -0.5 * block_states[0].eta.iter().map(|&v| v * v).sum::<f64>()
                - 0.5 * resid1,
            blockworking_sets: vec![
                BlockWorkingSet::Diagonal {
                    working_response: Array1::zeros(n0),
                    working_weights: Array1::ones(n0),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: -beta1.clone(),
                    hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                },
            ],
        })
    }
}

/// Same NaN-Hessian family but with PseudoLaplace objective, which takes
/// the strict-SPD path and skips the eigendecomposition in compute_update_step.
#[derive(Clone)]
pub(crate) struct TwoBlockNaNHessianPseudoLaplaceFamily;

impl CustomFamily for TwoBlockNaNHessianPseudoLaplaceFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        TwoBlockNaNHessianFamily.evaluate(block_states)
    }

    fn exact_newton_outerobjective(&self) -> ExactNewtonOuterObjective {
        ExactNewtonOuterObjective::StrictPseudoLaplace
    }
}

pub(crate) fn make_two_block_specs(n: usize) -> Vec<ParameterBlockSpec> {
    vec![
        ParameterBlockSpec {
            name: "mu".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::from_elem((n, 1), 1.0),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::from_elem((n, 2), 1.0),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(array![0.0, 0.0]),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ]
}

#[test]
pub(crate) fn exact_newton_nan_hessian_fails_loudly_before_eigendecomposition() {
    // Exact Newton Hessians are part of the mathematical contract.  A
    // NaN in a block Hessian means the family derivative is invalid; we
    // should reject it at the logdet boundary instead of hiding it behind
    // a conservative eigendecomposition fallback.
    let specs = make_two_block_specs(4);
    let per_block_log_lambdas = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(
        &TwoBlockNaNHessianFamily,
        &specs,
        &per_block_log_lambdas,
        &options,
        None,
    );
    let err = result.expect_err("NaN exact Hessian must fail loudly");
    assert!(
        err.contains("smooth-regularized logdet Hessian contains non-finite entry"),
        "expected explicit non-finite Hessian error, got: {err}"
    );
}

#[test]
pub(crate) fn exact_newton_finite_hessian_succeeds_where_nan_hessian_fails() {
    // SUFFICIENCY (control): The identical two-block structure with a
    // finite Hessian succeeds, proving that NaN in the Hessian is the
    // specific trigger — not the block layout, penalty structure, or
    // solver configuration.
    let specs = make_two_block_specs(4);
    let per_block_log_lambdas = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(
        &TwoBlockFiniteHessianFamily,
        &specs,
        &per_block_log_lambdas,
        &options,
        None,
    );
    assert!(
        result.is_ok(),
        "inner fit should succeed with finite Hessian: {:?}",
        result.err()
    );
}

#[test]
pub(crate) fn checked_penalizedobjective_rejects_non_finite_values() {
    let err = checked_penalizedobjective(-1.0, 0.5, f64::NAN, "test objective")
        .expect_err("non-finite objective should fail loudly");
    assert!(
        err.contains("non-finite penalized objective"),
        "unexpected error: {err}"
    );
}

#[test]
pub(crate) fn exact_newton_dh_closure_rejects_non_finite_directional_derivative() {
    #[derive(Clone)]
    struct OneBlockNonFiniteJointDhFamily;

    impl CustomFamily for OneBlockNonFiniteJointDhFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let beta = block_states
                .first()
                .ok_or_else(|| "missing block 0".to_string())?
                .beta
                .clone();
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * beta.dot(&beta),
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: beta.mapv(|v| -v),
                    hessian: SymmetricMatrix::Dense(array![[1.0]]),
                }],
            })
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[1.0]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            arr: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            assert!(arr.iter().all(|v| !v.is_nan()));
            Ok(Some(array![[f64::NAN]]))
        }
    }

    let family = OneBlockNonFiniteJointDhFamily;
    let specs = vec![ParameterBlockSpec {
        name: "beta".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::from_elem((2, 1), 1.0),
        )),
        offset: Array1::zeros(2),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let states = vec![ParameterBlockState {
        beta: array![0.0],
        eta: Array1::zeros(2),
    }];
    let synced_states = Arc::new(
        synchronized_states_from_flat_beta(&family, &specs, &states, &array![0.0])
            .expect("sync states for exact_newton_dh_closure"),
    );
    let compute_dh = exact_newton_dh_closure(&family, synced_states, &specs, 1, false, 1.0, None);
    let err = compute_dh(&array![1.0]).expect_err("non-finite dH should fail loudly");
    assert!(err.contains("non-finite"), "unexpected error: {err}");
}

#[test]
pub(crate) fn nan_propagating_min_detects_nan_eigenvalues() {
    // Verify the fix: our NaN-propagating min correctly detects
    // NaN eigenvalues, unlike f64::min which silently ignored them.
    let mut mat = Array2::<f64>::eye(3);
    mat[[1, 0]] = f64::NAN;
    mat[[0, 1]] = f64::NAN;

    use gam_linalg::faer_ndarray::FaerEigh;
    match FaerEigh::eigh(&mat, faer::Side::Lower) {
        Err(_) => {
            // eigh failed — the fallback chain in compute_update_step
            // now catches this and applies a conservative ridge.
        }
        Ok((evals, _)) => {
            // NaN-propagating fold (matches the production code):
            let new_min = evals.iter().copied().fold(f64::INFINITY, |a, b| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.min(b)
                }
            });
            assert!(
                !new_min.is_finite(),
                "NaN-propagating min should detect NaN eigenvalues, got {new_min}"
            );
        }
    }
}

#[test]
pub(crate) fn multiblock_generic_outer_fallback_returns_error_instead_of_panicking() {
    let family = TwoBlockFiniteHessianFamily;
    let specs = make_two_block_specs(4);
    let penalty_counts = vec![0usize, 0usize];
    let rho = Array1::zeros(0);
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        outer_max_iter: 1,
        ..BlockwiseFitOptions::default()
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::test_support::outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &penalty_counts,
            &rho,
            None,
            EvalMode::ValueGradientHessian,
        )
    }));

    let outcome = result.expect("multi-block outer fallback must return an error, not panic");
    let err = match outcome {
        Ok(_) => panic!("multi-block family without a joint path should fail loudly"),
        Err(err) => err.to_string(),
    };
    assert!(
        err.contains("multi-block families must provide a joint outer path"),
        "unexpected error: {err}"
    );
}

#[test]
pub(crate) fn pseudo_laplace_path_skips_eigendecomposition_avoiding_nan_crash() {
    // SUFFICIENCY: The PseudoLaplace path takes strict_solve_spd instead
    // of eigendecomposition-based ridging.  It will still fail (the Hessian
    // is NaN so the solve produces garbage), but the failure is NOT the
    // eigendecomposition NoConvergence error — it's a different error
    // downstream.  This proves the eigendecomposition call is the unique
    // failure point for QuadraticReml families.
    let specs = make_two_block_specs(4);
    let per_block_log_lambdas = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(
        &TwoBlockNaNHessianPseudoLaplaceFamily,
        &specs,
        &per_block_log_lambdas,
        &options,
        None,
    );
    // The PseudoLaplace path may fail for other reasons (NaN in solve),
    // but it must NOT fail with the eigendecomposition error.
    match result {
        Ok(_) => {} // Acceptable — strict_solve_spd might produce NaN
        // betas which don't trigger a hard error.
        Err(ref msg) => {
            assert!(
                !msg.contains("exact-newton eigendecomposition failed"),
                "PseudoLaplace path should NOT hit eigendecomposition; \
                     got eigendecomposition error anyway: {msg}"
            );
        }
    }
}

/// Regression check: when `strict_solve_spd_with_lm_continuation` is given a
/// strongly negative-definite matrix whose `|λ_min|` exceeds the LM δ-ridge
/// schedule's terminal δ (≈ ε · trace_scale · 10¹⁶), the bare schedule can't
/// rescue Cholesky and the terminal eigen-floor fallback must return a
/// finite solution equal to `Q diag(1/Λ̃) Qᵀ rhs`, with
/// `Λ̃_i = max(Λ_i, ε λ_max)`.
///
/// We also exercise the schedule-success path with a milder matrix to lock
/// in that the eigen-floor doesn't perturb the LM-δ output for cases the
/// schedule can already handle.
#[test]
pub(crate) fn strict_solve_spd_falls_back_to_eigen_floor_on_indefinite_matrix() {
    // δ schedule from `delta0 = max(ε·tr/p, 1e-12)`, growth 10×, 16 steps.
    // With `tr = 4·1e30` we get `delta0 ≈ ε·1e30 ≈ 2.2e14`; terminal δ at
    // escalation 16 is `2.2e14 · 1e16 = 2.2e30`. Set `λ_min ≈ -1e32` to
    // outpace the schedule and force the eigen-floor branch.
    let p = 4usize;
    let mut h = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        h[[i, i]] = -1e32 - (i as f64) * 1e30;
    }
    h[[0, 1]] = 5e29;
    h[[1, 0]] = 5e29;
    let rhs = Array1::from_vec(vec![1e30, -5e29, 2.5e29, 7.5e29]);

    let (x, stats) = strict_solve_spd_with_lm_continuation(&h, &rhs)
        .expect("eigen-floor fallback must succeed on the negative-definite matrix");
    assert!(
        stats.escalations > 16,
        "expected eigen-floor terminal fallback (escalations > MAX_ESCALATIONS), got {}",
        stats.escalations,
    );
    for &v in x.iter() {
        assert!(
            v.is_finite(),
            "eigen-floor solve returned non-finite component {v}"
        );
    }

    // Reconstruct the analytic floored solve and compare component-wise.
    let mut sym = h.clone();
    symmetrize_dense_in_place(&mut sym);
    let (evals, evecs) = FaerEigh::eigh(&sym, Side::Lower).expect("eigh");
    let max_abs_eval = evals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let eps_floor = (CUSTOM_FAMILY_EVAL_FLOOR * max_abs_eval).max(1e-300);
    let mut want = Array1::<f64>::zeros(p);
    for k in 0..p {
        let mut q_t_rhs = 0.0;
        for i in 0..p {
            q_t_rhs += evecs[[i, k]] * rhs[i];
        }
        let scaled = q_t_rhs / evals[k].max(eps_floor);
        for i in 0..p {
            want[i] += evecs[[i, k]] * scaled;
        }
    }
    for i in 0..p {
        let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
        assert!(
            (want[i] - x[i]).abs() <= tol,
            "eigen-floor solve component {i}: want={:.6e}, got={:.6e}",
            want[i],
            x[i],
        );
    }
}

// ---------- eta_backup heterogeneous-shape regression tests ----------
//
// Regression note: a previous `inner_blockwise_fit` implementation
// reused a single `eta_backup` buffer across blocks during line search.
// With heterogeneous eta lengths (e.g. survival time block = 3n,
// threshold/log-sigma = n), that buffer could be left at the wrong
// shape for the next block update and trigger an ndarray broadcast
// panic:
//   "could not broadcast array from shape: [n] to: [3n]"

/// Minimal two-block family where block 0 has design nrows=3n and
/// block 1 has design nrows=n. Both use ExactNewton. Block 0's
/// gradient is nonzero so the Newton step exceeds tol and exercises
/// the line-search path that previously mishandled heterogeneous
/// eta buffer shapes.
#[derive(Clone)]
pub(crate) struct HeterogeneousEtaLengthFamily {
    pub(crate) n: usize,
}

impl CustomFamily for HeterogeneousEtaLengthFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = self.n;
        let eta0 = &block_states[0].eta;
        let eta1 = &block_states[1].eta;
        assert_eq!(eta0.len(), 3 * n, "block 0 eta must be 3n");
        assert_eq!(eta1.len(), n, "block 1 eta must be n");
        let p0 = block_states[0].beta.len();
        let p1 = block_states[1].beta.len();
        // Simple quadratic log-likelihood so optimum is at beta=0.
        let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
        // Nonzero gradient drives a real step in both blocks.
        let grad0 = &(-&block_states[0].beta) + &Array1::from_elem(p0, 0.1);
        let grad1 = &(-&block_states[1].beta) + &Array1::from_elem(p1, 0.1);
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: vec![
                BlockWorkingSet::ExactNewton {
                    gradient: grad0,
                    hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                },
                BlockWorkingSet::ExactNewton {
                    gradient: grad1,
                    hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                },
            ],
        })
    }
}

pub(crate) fn make_heterogeneous_eta_specs(n: usize) -> Vec<ParameterBlockSpec> {
    let p0 = 2;
    let p1 = 2;
    vec![
        ParameterBlockSpec {
            name: "big_block".to_string(),
            // 3n rows — mimics survival time block stacking
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::from_elem((3 * n, p0), 1.0),
            )),
            offset: Array1::zeros(3 * n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::from_elem(p0, 1.0)),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "small_block".to_string(),
            // n rows — mimics threshold/log-sigma block
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::from_elem((n, p1), 1.0),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::from_elem(p1, 1.0)),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ]
}

/// Regression guard: blocks with identical eta lengths never exercised
/// the old heterogeneous-shape failure mode.
#[test]
pub(crate) fn uniform_eta_lengths_do_not_panic() {
    let n = 10;
    #[derive(Clone)]
    struct UniformEtaFamily;
    impl CustomFamily for UniformEtaFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let p0 = block_states[0].beta.len();
            let p1 = block_states[1].beta.len();
            let eta0 = &block_states[0].eta;
            let eta1 = &block_states[1].eta;
            let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: &(-&block_states[0].beta) + &Array1::from_elem(p0, 0.1),
                        hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: &(-&block_states[1].beta) + &Array1::from_elem(p1, 0.1),
                        hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                    },
                ],
            })
        }
    }
    // Both blocks have n rows — no shape mismatch possible.
    let specs = vec![
        ParameterBlockSpec {
            name: "block_a".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::from_elem((n, 2), 1.0),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::from_elem(2, 1.0)),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "block_b".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::from_elem((n, 2), 1.0),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: Some(Array1::from_elem(2, 1.0)),
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 3,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    // Must NOT panic — uniform eta lengths keep eta_backup
    // compatible with every block's eta after mem::swap.
    let result = inner_blockwise_fit(&UniformEtaFamily, &specs, &per_block, &options, None);
    assert!(
        result.is_ok(),
        "uniform eta lengths should not panic: {result:?}"
    );
}

/// Regression guard: heterogeneous eta lengths (3n vs n) must not
/// prevent the inner fit from completing. Older code could panic with
/// "could not broadcast array from shape: [n] to: [3n]" due to the
/// eta_backup swap bug.
#[test]
pub(crate) fn heterogeneous_eta_lengths_inner_fit_completes() {
    let n = 10;
    let family = HeterogeneousEtaLengthFamily { n };
    let specs = make_heterogeneous_eta_specs(n);
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 3,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
    assert!(result.is_ok(), "inner fit should complete: {result:?}");
}

/// SUFFICIENCY (single-cycle): even one inner cycle must complete
/// without panic when blocks have heterogeneous eta lengths.
#[test]
pub(crate) fn heterogeneous_eta_single_cycle_completes() {
    let n = 10;
    let family = HeterogeneousEtaLengthFamily { n };
    let specs = make_heterogeneous_eta_specs(n);
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
    assert!(
        result.is_ok(),
        "single-cycle inner fit should complete: {result:?}"
    );
}

/// Regression guard: when all blocks have step <= tol, the line-search
/// path is skipped for every block, so this case should remain safe
/// even with heterogeneous eta lengths.
#[test]
pub(crate) fn heterogeneous_eta_no_panic_when_all_blocks_converged() {
    let n = 10;
    #[derive(Clone)]
    struct AllConvergedFamily {
        pub(crate) n: usize,
    }
    impl CustomFamily for AllConvergedFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = self.n;
            let eta0 = &block_states[0].eta;
            let eta1 = &block_states[1].eta;
            assert_eq!(eta0.len(), 3 * n);
            assert_eq!(eta1.len(), n);
            let p0 = block_states[0].beta.len();
            let p1 = block_states[1].beta.len();
            let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![
                    BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(p0),
                        hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: Array1::zeros(p1),
                        hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                    },
                ],
            })
        }
    }
    let mut specs = make_heterogeneous_eta_specs(n);
    specs[0].initial_beta = Some(Array1::zeros(2));
    specs[1].initial_beta = Some(Array1::zeros(2));
    let family = AllConvergedFamily { n };
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    // All blocks converged → step=0 → `continue` before swap →
    // eta_backup never participates → no broadcast panic.
    let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
    assert!(
        result.is_ok(),
        "should not panic when all blocks are converged: {result:?}"
    );
}

/// Regression guard: even when only the second (smaller) block takes
/// a step, the fit must complete. Earlier code could still panic here
/// after reusing an oversized eta_backup buffer across blocks.
#[test]
pub(crate) fn heterogeneous_eta_completes_when_only_small_block_steps() {
    let n = 10;
    #[derive(Clone)]
    struct OnlySmallBlockStepsFamily {
        pub(crate) n: usize,
    }
    impl CustomFamily for OnlySmallBlockStepsFamily {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            let n = self.n;
            let eta0 = &block_states[0].eta;
            let eta1 = &block_states[1].eta;
            assert_eq!(eta0.len(), 3 * n);
            assert_eq!(eta1.len(), n);
            let p0 = block_states[0].beta.len();
            let p1 = block_states[1].beta.len();
            let ll = -0.5 * eta0.dot(eta0) - 0.5 * eta1.dot(eta1);
            Ok(FamilyEvaluation {
                log_likelihood: ll,
                blockworking_sets: vec![
                    BlockWorkingSet::ExactNewton {
                        // Block 0: converged, step=0
                        gradient: Array1::zeros(p0),
                        hessian: SymmetricMatrix::Dense(Array2::eye(p0)),
                    },
                    BlockWorkingSet::ExactNewton {
                        // Block 1: nontrivial step
                        gradient: &(-&block_states[1].beta) + &Array1::from_elem(p1, 0.1),
                        hessian: SymmetricMatrix::Dense(Array2::eye(p1)),
                    },
                ],
            })
        }
    }
    let mut specs = make_heterogeneous_eta_specs(n);
    specs[0].initial_beta = Some(Array1::zeros(2)); // block 0 at optimum
    let family = OnlySmallBlockStepsFamily { n };
    let per_block = vec![Array1::zeros(0), Array1::zeros(0)];
    let options = BlockwiseFitOptions {
        inner_max_cycles: 1,
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let result = inner_blockwise_fit(&family, &specs, &per_block, &options, None);
    assert!(
        result.is_ok(),
        "fit should complete when only small block steps: {result:?}"
    );
}

/// Direct test of the KKT-aware projection in
/// `projected_stationarity_inf_norm`.
///
/// Contract:
///   (i)   with no constraints, returns the plain inf-norm of the residual;
///   (ii)  at an active lower bound with multiplier-signed residual
///         (`β_j == lb_j` and `residual_j > 0`) the coordinate is skipped;
///   (iii) at an active lower bound with wrong-signed residual
///         (`residual_j < 0`) the coordinate still contributes;
///   (iv)  interior coordinates always contribute regardless of
///         residual sign.
///
/// This pins the exact convergence semantics that the joint-Newton loop
/// relies on: a genuine constrained-KKT optimum must score zero, while
/// infeasibility and interior non-stationarity remain observable.
#[test]
pub(crate) fn projected_stationarity_inf_norm_respects_kkt_multipliers() {
    // Test (i): no constraints → plain inf-norm.
    let beta = array![1.0, 2.0, -0.5];
    let residual = array![0.3, -0.1, 0.2];
    let inf_nocon = projected_stationarity_inf_norm(&residual, &beta, None, None);
    assert_relative_eq!(inf_nocon, 0.3_f64, epsilon = 1e-12);

    // Test (ii): β_j at its lower bound with residual_j > 0 is a KKT
    // multiplier; projection drops it, so only the interior entry (-0.1)
    // contributes.
    let beta_active = array![0.0, 2.0];
    let residual_active = array![0.5, -0.1];
    let constraints_lb0 = LinearInequalityConstraints {
        a: array![[1.0, 0.0], [0.0, 1.0]],
        b: array![0.0, f64::NEG_INFINITY], // only β_0 has a finite lower bound
    };
    // Build a minimal single-row constraint first (β_0 ≥ 0) so the
    // "active lower bound + positive residual" branch of the projection
    // is exercised in isolation.  β_1 is left unconstrained relative to
    // this single-row constraint matrix (it's not pinned by any row),
    // so its contribution (|-0.1| = 0.1) stays in the inf-norm.
    let single = LinearInequalityConstraints {
        a: array![[1.0, 0.0]],
        b: array![0.0],
    };
    let inf_projected =
        projected_stationarity_inf_norm(&residual_active, &beta_active, Some(&single), None);
    assert_relative_eq!(inf_projected, 0.1_f64, epsilon = 1e-12);
    let vec_projected = projected_linear_constraint_stationarity_vector(
        &residual_active,
        &beta_active,
        &single,
        None,
    )
    .expect("active lower-bound projection should succeed");
    assert_relative_eq!(vec_projected[0], 0.0_f64, epsilon = 1e-10);
    assert_relative_eq!(vec_projected[1], -0.1_f64, epsilon = 1e-12);

    // Also verify the per-coord handling of an explicitly-unconstrained
    // row (b = -inf) in the two-row form: β_0 has a finite lower bound
    // of 0 (from row 0), β_1 gets lb = -inf (from row 1 via b/a), which
    // `lb.is_finite() == false` routes to the "no lower bound" branch of
    // the projection.  The active-bound drop still fires on coord 0, so
    // the result matches the single-row case: 0.1.  This documents that
    // the projection's per-coord `lb.is_finite()` gate is what makes the
    // unconstrained-coord case work — NOT rejection of the whole
    // constraint set by `extract_simple_lower_bounds`.
    let inf_with_two_row = projected_stationarity_inf_norm(
        &residual_active,
        &beta_active,
        Some(&constraints_lb0),
        None,
    );
    assert_relative_eq!(inf_with_two_row, 0.1_f64, epsilon = 1e-12);

    // Test (iii): β_j at its bound but residual points the WRONG way
    // (residual_j < 0 means the KKT dual feasibility λ_j ≥ 0 is violated
    // — i.e. the bound should release).  Keep that coordinate in the
    // norm so the optimizer does not declare convergence on an infeasible
    // multiplier.
    let beta_wrong_sign = array![0.0];
    let residual_wrong_sign = array![-0.2];
    let single1 = LinearInequalityConstraints {
        a: array![[1.0]],
        b: array![0.0],
    };
    let inf_wrong_sign = projected_stationarity_inf_norm(
        &residual_wrong_sign,
        &beta_wrong_sign,
        Some(&single1),
        None,
    );
    assert_relative_eq!(inf_wrong_sign, 0.2_f64, epsilon = 1e-12);

    // Test (iv): an interior coordinate with a valid lower bound keeps
    // contributing to the norm, whatever the residual sign.
    let beta_interior = array![1.5];
    let residual_interior = array![0.4];
    let inf_interior =
        projected_stationarity_inf_norm(&residual_interior, &beta_interior, Some(&single1), None);
    assert_relative_eq!(inf_interior, 0.4_f64, epsilon = 1e-12);

    // #1793/#1040: monotone derivative rows have `b = 0`, so a numerically
    // pinned baseline-hazard row can sit a few 1e-3 inside the feasible cone
    // after repeated basis projections even though the residual is entirely a
    // valid KKT multiplier.  The projection must treat that row as an active
    // candidate and let the nonnegative cone solve remove the multiplier.
    let beta_nearly_pinned = array![0.004];
    let residual_multiplier = array![2.0];
    let near_pinned = projected_stationarity_inf_norm(
        &residual_multiplier,
        &beta_nearly_pinned,
        Some(&single1),
        None,
    );
    assert_relative_eq!(near_pinned, 0.0_f64, epsilon = 1e-10);

    // But a row clearly in the interior remains visible, so the wider
    // near-active band cannot erase ordinary interior non-stationarity.
    let beta_clearly_interior = array![0.02];
    let interior_residual = projected_stationarity_inf_norm(
        &residual_multiplier,
        &beta_clearly_interior,
        Some(&single1),
        None,
    );
    assert_relative_eq!(interior_residual, 2.0_f64, epsilon = 1e-12);
}

/// Pins the constrained-stationary certificate semantics.
///
/// The certificate combines three local signals from the most recent
/// accepted Newton step:
///
///   1. `linearized_rel = ‖g + Hδ‖∞ / (1 + ‖g‖∞)` ≥ 0.5
///      — the linear solve refused to neutralise most of `g`; the
///        unreduced component lives in the constraint-active subspace
///        and IS a Lagrange multiplier, not a defect of the solve.
///
///   2. `scalar_model_relative_error()` ≤ 1e-3
///      — the local quadratic Newton model agrees with the observed
///        objective change to roundoff, proving the Hessian+gradient
///        are correct at this β.  Rules out genuine model mismatch
///        masquerading as a multiplier.
///
///   3. `|Δobjective|` ≤ `objective_tol`
///      — the objective has ceased moving.
///
/// Reproduces the large-scale survival-marginal-slope failure numerics:
/// `old_kkt ≈ 8.6e5`, `linearized_next ≈ 8.6e5`, `actual ≈ pred ≈ 1.6e-2`.
#[test]
pub(crate) fn joint_newton_math_constrained_stationary_signature_matches_aou_failure() {
    let math = JointNewtonMathDiagnostic {
        old_kkt_inf: 8.613e5,
        linearized_next_kkt_inf: 8.580e5,
        predicted_reduction: 1.589e-2,
        actual_reduction: 1.589e-2,
        trust_ratio: 1.000,
        step_inf: 1.270e-2,
        proposal_inf: 1.270e-2,
    };
    // (1) The linearized solve neutralised <1% of g — Lagrange multiplier
    // pattern, not a defect of the solve.
    let linearized_rel = math.linearized_next_kkt_inf / (1.0 + math.old_kkt_inf);
    assert!(
        linearized_rel >= 0.5,
        "large-scale exit has linearized_rel = {:.3e}, must be >= 0.5 for the \
             constrained-stationary certificate to fire",
        linearized_rel,
    );
    // (2) Scalar Newton model is correct to roundoff — Hessian+gradient OK.
    let relerr = math.scalar_model_relative_error();
    assert!(
        relerr <= 1e-3,
        "large-scale exit has scalar_model_relerr = {:.3e}, must be <= 1e-3 \
             (model agrees with actual ⇒ residual is a real multiplier)",
        relerr,
    );
    // (3) Objective change at obj_tol scale. At |obj| ~ 3.5e5 and
    // inner_tol ~ 1e-6, obj_tol ≈ 0.348, and observed Δobj ≈ 1.6e-2.
    let objective_change = 1.589e-2_f64;
    let objective_tol = 1e-6 * (1.0 + 3.484783e5_f64);
    assert!(
        objective_change <= objective_tol,
        "large-scale exit has |Δobj| = {:.3e}, must be <= obj_tol {:.3e}",
        objective_change,
        objective_tol,
    );
}

/// Reproduces the post-diagnostic large-scale trace: the scalar Newton model
/// and objective plateau tests alone look like a constrained-stationary
/// point, but the projected KKT residual is hundreds of times above
/// tolerance and the accepted Newton step is still macroscopic. That is
/// not a terminal certificate; it is a normal in-progress Newton cycle.
#[test]
pub(crate) fn constrained_stationary_certificate_keeps_iterating_when_step_is_large() {
    let math = JointNewtonMathDiagnostic {
        old_kkt_inf: 2.708e4,
        linearized_next_kkt_inf: 2.707e4,
        predicted_reduction: 3.421e-1,
        actual_reduction: 3.421e-1,
        trust_ratio: 1.0,
        step_inf: 2.891e-2,
        proposal_inf: 2.891e-2,
    };
    let objective_change = 3.421e-1;
    let objective_tol = 3.479e-1;
    let residual = 8.102;
    let residual_tol = 2.707e-2;
    let step_tol = 1.2e-5;

    // These are the three non-step conditions that made 0.1.126 reject a
    // seed as soon as objective change touched tolerance.
    let linearized_rel = math.linearized_next_kkt_inf / (1.0 + math.old_kkt_inf);
    assert!(linearized_rel >= 0.5);
    assert!(math.scalar_model_relative_error() <= 1e-3);
    assert!(objective_change <= objective_tol);
    assert!(math.step_inf > step_tol);

    // The projected residual still rules out accepting convergence, but
    // the large step rules out terminal refusal. The loop must continue.
    assert!(residual > residual_tol);
    assert_eq!(
        constrained_stationary_certificate_decision(
            &math,
            objective_change,
            objective_tol,
            step_tol,
            None,
            residual,
            residual_tol,
        ),
        ConstrainedStationaryCertificate::NotCandidate,
    );
}

#[test]
pub(crate) fn residual_steady_geometric_descent_distinguishes_converging_from_plateau() {
    use std::collections::VecDeque;
    // gam#787 duchon centers≥20: the logslope block converged geometrically
    // (~0.33×/cycle) but `linearized_rel ≥ 0.5` + flat objective routed it
    // into the plateau-refusal break a few cycles short of tol. The
    // steady-descent guard must keep it iterating.
    let converging: VecDeque<f64> = [6.985e-4, 2.388e-4, 7.987e-5, 2.597e-5]
        .into_iter()
        .collect();
    assert!(
        residual_in_steady_geometric_descent(&converging),
        "a steadily ~0.33x/cycle descending residual must be recognized as converging"
    );
    // A genuine multiplier/null plateau: residual flat/oscillating above tol.
    let plateau: VecDeque<f64> = [2.066e0, 2.063e0, 2.066e0, 2.063e0].into_iter().collect();
    assert!(
        !residual_in_steady_geometric_descent(&plateau),
        "a flat/oscillating residual plateau must NOT be treated as converging"
    );
    // A single lucky drop inside an otherwise flat window must not qualify.
    let noisy: VecDeque<f64> = [2.0e0, 2.0e0, 1.0e-3].into_iter().collect();
    assert!(
        !residual_in_steady_geometric_descent(&noisy),
        "a single-cycle drop must not be mistaken for steady descent"
    );
    // Too few cycles to judge steadiness.
    let short: VecDeque<f64> = [1.0e-3, 3.0e-4].into_iter().collect();
    assert!(
        !residual_in_steady_geometric_descent(&short),
        "fewer than the window of cycles must not assert steady descent"
    );
}

#[test]
pub(crate) fn constrained_stationary_certificate_refuses_only_when_step_is_exhausted() {
    let math = JointNewtonMathDiagnostic {
        old_kkt_inf: 2.708e4,
        linearized_next_kkt_inf: 2.707e4,
        predicted_reduction: 3.421e-1,
        actual_reduction: 3.421e-1,
        trust_ratio: 1.0,
        step_inf: 2.891e-7,
        proposal_inf: 2.891e-7,
    };
    let objective_change = 3.421e-1;
    let objective_tol = 3.479e-1;
    let step_tol = 1.0e-6;
    let residual_tol = 2.707e-2;

    // Inside the certification band (`residual <= 4x residual_tol`, the
    // documented gam#797 conditioning/round-off allowance) a fully
    // stationary iterate is accepted.
    assert_eq!(
        constrained_stationary_certificate_decision(
            &math,
            objective_change,
            objective_tol,
            step_tol,
            None,
            residual_tol,
            residual_tol,
        ),
        ConstrainedStationaryCertificate::Accept,
    );
    assert_eq!(
        constrained_stationary_certificate_decision(
            &math,
            objective_change,
            objective_tol,
            step_tol,
            None,
            // Still within 4x: a residual a hair above 1x must remain
            // accepted, because the active-projected residual genuinely
            // floors just above the scale-relative tolerance.
            residual_tol + 1.0e-12,
            residual_tol,
        ),
        ConstrainedStationaryCertificate::Accept,
    );
    // Beyond the 4x band the residual is too large to be a mere
    // conditioning floor: the certificate must refuse the phantom
    // multiplier rather than fake convergence.
    assert_eq!(
        constrained_stationary_certificate_decision(
            &math,
            objective_change,
            objective_tol,
            step_tol,
            None,
            4.0 * residual_tol + 1.0e-6,
            residual_tol,
        ),
        ConstrainedStationaryCertificate::RefusePhantomMultiplier,
    );
}

/// Negative case: a genuine non-stationary state must NOT trigger
/// the certificate. We construct numbers where the linear solve
/// successfully neutralises g (linearized_rel small) — meaning Newton
/// is making real progress on an unconstrained problem — and verify
/// the certificate does NOT fire.
#[test]
pub(crate) fn joint_newton_math_unconstrained_progress_does_not_match_certificate() {
    let math = JointNewtonMathDiagnostic {
        // Unconstrained Newton: linear solve reduces ‖g‖ by O(1e-12).
        old_kkt_inf: 1.0e3,
        linearized_next_kkt_inf: 1.0e-9,
        predicted_reduction: 5.0e-1,
        actual_reduction: 5.0e-1,
        trust_ratio: 1.0,
        step_inf: 1.0e-12,
        proposal_inf: 1.0e-12,
    };
    assert_eq!(
        constrained_stationary_certificate_decision(
            &math, 1.0e-12, 1.0e-8, 1.0e-8, None, 1.0e-12, 1.0e-8,
        ),
        ConstrainedStationaryCertificate::NotCandidate,
        "objective and step exhaustion must not certify a Newton step whose linearized residual is genuinely falling"
    );
}

#[test]
pub(crate) fn projected_stationarity_inf_norm_projects_coupled_linear_kkt_multipliers() {
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 1.0]],
        b: array![1.0],
    };
    let beta_active = array![0.25, 0.75];

    let residual_valid_multiplier = array![3.0, 3.0];
    let inf_valid = projected_stationarity_inf_norm(
        &residual_valid_multiplier,
        &beta_active,
        Some(&constraints),
        None,
    );
    assert_relative_eq!(inf_valid, 0.0_f64, epsilon = 1e-10);
    let vec_valid = projected_linear_constraint_stationarity_vector(
        &residual_valid_multiplier,
        &beta_active,
        &constraints,
        None,
    )
    .expect("coupled active projection should succeed");
    assert_relative_eq!(vec_valid[0], 0.0_f64, epsilon = 1e-10);
    assert_relative_eq!(vec_valid[1], 0.0_f64, epsilon = 1e-10);

    let residual_wrong_sign = array![-3.0, -3.0];
    let inf_wrong = projected_stationarity_inf_norm(
        &residual_wrong_sign,
        &beta_active,
        Some(&constraints),
        None,
    );
    assert_relative_eq!(inf_wrong, 3.0_f64, epsilon = 1e-12);

    let beta_interior = array![0.75, 0.75];
    let inf_interior = projected_stationarity_inf_norm(
        &residual_valid_multiplier,
        &beta_interior,
        Some(&constraints),
        None,
    );
    assert_relative_eq!(inf_interior, 3.0_f64, epsilon = 1e-12);
}

#[test]
pub(crate) fn joint_stationarity_from_gradient_projects_coupled_linear_constraints() {
    let spec = ParameterBlockSpec {
        name: "coupled".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let state = ParameterBlockState {
        beta: array![0.25, 0.75],
        eta: array![0.25, 0.75],
    };
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 1.0]],
        b: array![1.0],
    };
    let s_lambdas = vec![Array2::<f64>::zeros((2, 2))];

    // residual = S beta - gradient = [4, 4] = A_active^T lambda,
    // lambda=4.  This is a valid constrained KKT point and must not be
    // reported as a large free-gradient residual.
    let residual_multiplier = array![4.0, 4.0];
    let gradient = -&residual_multiplier;
    let projected = exact_newton_joint_stationarity_inf_norm_from_gradient(
        &gradient,
        &[state.clone()],
        std::slice::from_ref(&spec),
        &s_lambdas,
        0.0,
        RidgePolicy::exact_full_objective(),
        &[Some(constraints.clone())],
        None,
        None,
    )
    .expect("stationarity projection should succeed");
    assert_relative_eq!(projected, 0.0_f64, epsilon = 1e-10);
    let kkt_residual = exact_newton_joint_projected_kkt_residual_for_ift_from_gradient(
        &gradient,
        std::slice::from_ref(&spec),
        &[state.clone()],
        &s_lambdas,
        0.0,
        RidgePolicy::exact_full_objective(),
        &[Some(constraints.clone())],
        None,
        None,
    )
    .expect("KKT residual assembly should succeed")
    .expect("exact-gradient path should produce residual");
    assert_relative_eq!(kkt_residual.as_array()[0], 0.0_f64, epsilon = 1e-10);
    assert_relative_eq!(kkt_residual.as_array()[1], 0.0_f64, epsilon = 1e-10);

    // Wrong-signed normal residual means the active constraint wants to
    // release. That is not convergence and must remain visible.
    let wrong_signed_gradient = residual_multiplier;
    let unprojected = exact_newton_joint_stationarity_inf_norm_from_gradient(
        &wrong_signed_gradient,
        &[state],
        &[spec],
        &s_lambdas,
        0.0,
        RidgePolicy::exact_full_objective(),
        &[Some(constraints)],
        None,
        None,
    )
    .expect("stationarity projection should succeed");
    assert_relative_eq!(unprojected, 4.0_f64, epsilon = 1e-12);
}

/// gam#979: the stationarity certificate must project out the KKT multipliers of
/// ACTIVE SIMPLE LOWER BOUNDS (the box-bound analog of the linear-constraint
/// projection) — but ONLY when the sign is valid. A coordinate at its lower
/// bound whose objective-gradient residual pushes INTO the bound (residual ≥ 0,
/// a nonnegative multiplier) is at a valid constrained optimum and is projected
/// to zero; a coordinate at its bound whose residual is NEGATIVE (wants to LEAVE
/// the bound) is NOT optimal and MUST remain in the residual so the certificate
/// still refuses. This regresses the survival marginal-slope hang, where the
/// monotone-baseline-hazard block sat at its ≥0 bound with a large valid
/// multiplier that the (linear-only) projection left in the residual, mis-
/// refusing a genuinely-optimal constrained iterate.
#[test]
pub(crate) fn stationarity_projects_valid_lower_bound_multiplier_but_keeps_wrong_sign_979() {
    // One block, width 2, β pinned at its lower bound [0, 0].
    let spec = ParameterBlockSpec {
        name: "box-bound".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.0, 0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let state = ParameterBlockState {
        beta: array![0.0, 0.0],
        eta: array![0.0, 0.0],
    };
    let s_lambdas = vec![Array2::<f64>::zeros((2, 2))];
    let lower_bounds = array![0.0_f64, 0.0_f64];

    // residual = Sβ − gradient = −gradient (S=0). Coord 0: residual=+626 (valid
    // nonneg lower-bound multiplier, the bound holds) → projected out. Coord 1:
    // residual=−5 (wants to LEAVE the bound) → NOT optimal, must survive.
    let gradient = array![-626.0_f64, 5.0_f64];
    let inf = exact_newton_joint_stationarity_inf_norm_from_gradient(
        &gradient,
        std::slice::from_ref(&state),
        std::slice::from_ref(&spec),
        &s_lambdas,
        0.0,
        RidgePolicy::exact_full_objective(),
        &[None],
        None,
        Some(&lower_bounds),
    )
    .expect("box-bound stationarity projection should succeed");
    // The valid +626 multiplier is projected out; the wrong-signed −5 remains.
    assert_relative_eq!(inf, 5.0_f64, epsilon = 1e-12);

    // Without the bounds (None) the certificate has no box path and reads the
    // raw 626 — the pre-fix behaviour that mis-refused the constrained optimum.
    let inf_no_bounds = exact_newton_joint_stationarity_inf_norm_from_gradient(
        &gradient,
        std::slice::from_ref(&state),
        std::slice::from_ref(&spec),
        &s_lambdas,
        0.0,
        RidgePolicy::exact_full_objective(),
        &[None],
        None,
        None,
    )
    .expect("stationarity projection should succeed");
    assert_relative_eq!(inf_no_bounds, 626.0_f64, epsilon = 1e-12);
}

#[test]
pub(crate) fn kkt_residual_uses_cached_joint_gradient_without_re_evaluating_family() {
    let spec = ParameterBlockSpec {
        name: "cached-gradient".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let state = ParameterBlockState {
        beta: array![2.0, -1.0],
        eta: array![2.0, -1.0],
    };
    let s_lambda = Array2::<f64>::eye(2);
    let expected_residual = array![0.25, -0.5];
    let cached_gradient = s_lambda.dot(&state.beta) - &expected_residual;

    let residual = exact_newton_joint_kkt_residual_for_ift_from_cached_gradient(
        &OneBlockAlwaysErrorFamily,
        std::slice::from_ref(&spec),
        std::slice::from_ref(&state),
        std::slice::from_ref(&s_lambda),
        0.0,
        RidgePolicy::exact_full_objective(),
        None,
        Some(&cached_gradient),
        None,
    )
    .expect("cached gradient path should not call family.evaluate()")
    .expect("cached gradient should produce a KKT residual");

    assert_relative_eq!(
        residual.as_array()[0],
        expected_residual[0],
        epsilon = 1e-12
    );
    assert_relative_eq!(
        residual.as_array()[1],
        expected_residual[1],
        epsilon = 1e-12
    );
}

#[test]
pub(crate) fn projected_stationarity_vector_uses_penalized_residual_not_raw_score() {
    let spec = ParameterBlockSpec {
        name: "score-cancellation".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0]
        ])),
        offset: array![0.0, 0.0],
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let state = ParameterBlockState {
        beta: array![10.0, -4.0],
        eta: array![10.0, -4.0],
    };
    let s_lambda = array![[2.0, 0.0], [0.0, 3.0]];
    let gradient = array![19.5, -12.25];

    let residual = exact_newton_joint_projected_stationarity_vector_from_gradient(
        &gradient,
        std::slice::from_ref(&state),
        std::slice::from_ref(&spec),
        std::slice::from_ref(&s_lambda),
        0.0,
        RidgePolicy::exact_full_objective(),
        &[None],
        None,
        None,
    )
    .expect("projected stationarity residual should assemble");

    assert_relative_eq!(residual[0], 0.5, epsilon = 1e-12);
    assert_relative_eq!(residual[1], 0.25, epsilon = 1e-12);
}

#[test]
pub(crate) fn zero_psi_derivative_operator_acts_as_zero_map() {
    let n = 17usize;
    let p = 5usize;
    let op = ZeroPsiDerivativeOperator::new(n, p);

    assert_eq!(op.n_data(), n);
    assert_eq!(op.p_out(), p);

    let u = Array1::from_iter((0..p).map(|k| 1.0 + k as f64));
    let v = Array1::from_iter((0..n).map(|k| 1.0 - 0.5 * k as f64));

    let fwd = op.forward_mul(0, &u.view()).expect("forward_mul");
    assert_eq!(fwd.len(), n);
    assert!(fwd.iter().all(|x| *x == 0.0));

    let trn = op.transpose_mul(0, &v.view()).expect("transpose_mul");
    assert_eq!(trn.len(), p);
    assert!(trn.iter().all(|x| *x == 0.0));

    let fwd2 = op
        .forward_mul_second_diag(0, &u.view())
        .expect("forward_mul_second_diag");
    assert_eq!(fwd2.len(), n);
    assert!(fwd2.iter().all(|x| *x == 0.0));

    let trn2 = op
        .transpose_mul_second_diag(0, &v.view())
        .expect("transpose_mul_second_diag");
    assert_eq!(trn2.len(), p);
    assert!(trn2.iter().all(|x| *x == 0.0));

    let fwd_cross = op
        .forward_mul_second_cross(0, 1, &u.view())
        .expect("forward_mul_second_cross");
    assert_eq!(fwd_cross.len(), n);
    assert!(fwd_cross.iter().all(|x| *x == 0.0));

    let trn_cross = op
        .transpose_mul_second_cross(0, 1, &v.view())
        .expect("transpose_mul_second_cross");
    assert_eq!(trn_cross.len(), p);
    assert!(trn_cross.iter().all(|x| *x == 0.0));

    let chunk = op.row_chunk_first(0, 3..7).expect("row_chunk_first");
    assert_eq!(chunk.dim(), (4, p));
    assert!(chunk.iter().all(|x| *x == 0.0));

    let chunk_diag = op
        .row_chunk_second_diag(0, 0..n)
        .expect("row_chunk_second_diag");
    assert_eq!(chunk_diag.dim(), (n, p));
    assert!(chunk_diag.iter().all(|x| *x == 0.0));

    let chunk_cross = op
        .row_chunk_second_cross(0, 1, 1..3)
        .expect("row_chunk_second_cross");
    assert_eq!(chunk_cross.dim(), (2, p));
    assert!(chunk_cross.iter().all(|x| *x == 0.0));

    let mut row = Array1::from_elem(p, 9.5);
    op.row_vector_first_into(0, 4, row.view_mut())
        .expect("row_vector_first_into");
    assert!(row.iter().all(|x| *x == 0.0));

    // The operator must not advertise dense materialization — production
    // hot paths rely on this to avoid forming an (n, p) buffer.
    assert!(op.as_materializable().is_none());
}

/// At large scale (n=320 000, p=101) a dense `Array2::zeros((n, p))`
/// for an unused ψ-derivative slot consumes ≈ 0.24 GiB; the spatial-
/// adaptive baseline used to allocate one per ψ coordinate (≈ 1.4 GiB
/// of guaranteed-zero memory at six coords). Replacing the dense zero
/// matrix with a `(0, 0)` shape sentinel — without an implicit
/// operator — must still resolve to `PsiDesignMap::Zero` so callers
/// see exact-zero semantics with O(1) memory.
#[test]
pub(crate) fn spatial_adaptive_zero_xpsi_uses_zero_map_without_dense_allocation() {
    let n = 320_000usize;
    let p = 101usize;
    let deriv = CustomFamilyBlockPsiDerivative {
        penalty_index: None,
        x_psi: Array2::<f64>::zeros((0, 0)),
        s_psi: Array2::<f64>::zeros((0, 0)),
        s_psi_components: None,
        s_psi_penalty_components: None,
        x_psi_psi: None,
        s_psi_psi: None,
        s_psi_psi_components: None,
        s_psi_psi_penalty_components: None,
        implicit_operator: None,
        implicit_axis: 0,
        implicit_group_id: None,
    };
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let map = resolve_custom_family_x_psi_map(
        &deriv,
        n,
        p,
        0..n,
        "spatial-adaptive zero sentinel",
        &policy,
    )
    .expect("resolve x_psi map for (0, 0)-sentinel deriv");
    match map {
        PsiDesignMap::Zero { nrows, ncols } => {
            assert_eq!(nrows, n);
            assert_eq!(ncols, p);
        }
        other => panic!(
            "(0, 0) x_psi sentinel must resolve to PsiDesignMap::Zero, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

#[test]
pub(crate) fn zero_psi_derivative_operator_resolves_to_zero_design_map() {
    let n = 12usize;
    let p = 4usize;
    let zero_op: Arc<dyn CustomFamilyPsiDerivativeOperator> =
        Arc::new(ZeroPsiDerivativeOperator::new(n, p));
    let deriv = CustomFamilyBlockPsiDerivative {
        penalty_index: None,
        x_psi: Array2::<f64>::zeros((0, 0)),
        s_psi: Array2::<f64>::zeros((0, 0)),
        s_psi_components: None,
        s_psi_penalty_components: None,
        x_psi_psi: None,
        s_psi_psi: None,
        s_psi_psi_components: None,
        s_psi_psi_penalty_components: None,
        implicit_operator: Some(Arc::clone(&zero_op)),
        implicit_axis: 0,
        implicit_group_id: None,
    };
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let map = resolve_custom_family_x_psi_map(&deriv, n, p, 0..n, "zero", &policy)
        .expect("resolve x_psi map");
    let u = Array1::from_iter((0..p).map(|k| 1.0 + k as f64));
    let fwd = map.forward_mul(u.view()).expect("forward_mul map");
    assert_eq!(fwd.len(), n);
    assert!(fwd.iter().all(|x| *x == 0.0));

    let chunk = map.row_chunk(2..5).expect("row_chunk map");
    assert_eq!(chunk.dim(), (3, p));
    assert!(chunk.iter().all(|x| *x == 0.0));

    let map_second =
        resolve_custom_family_x_psi_psi_map(&deriv, &deriv, 0, n, p, 0..n, "zero", &policy)
            .expect("resolve x_psi_psi map");
    let fwd_second = map_second
        .forward_mul(u.view())
        .expect("forward_mul second");
    assert_eq!(fwd_second.len(), n);
    assert!(fwd_second.iter().all(|x| *x == 0.0));
}

#[test]
pub(crate) fn rowwise_kronecker_psi_row_chunks_are_window_consistent() {
    let first = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let second_diag = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]];
    let second_cross = array![[-1.0, 0.25], [-1.5, 0.5], [-2.0, 0.75]];
    let base = build_embedded_dense_psi_operator(
        &first,
        &second_diag,
        Some(&vec![(1, second_cross.clone())]),
        0..2,
        2,
        0,
    )
    .expect("embedded dense base");
    let time_a = Arc::new(array![[1.0, 0.0], [0.5, 1.0], [1.5, -0.5]]);
    let time_b = Arc::new(array![[0.25, 2.0], [-1.0, 0.75], [0.0, 1.25]]);
    let op = build_rowwise_kronecker_psi_operator(base, vec![time_a, time_b])
        .expect("rowwise kronecker psi operator");
    let mat = op
        .as_materializable()
        .expect("rowwise operator dense reference");
    let rows = 1..5;

    let first_dense = mat.materialize_first(0).expect("dense first");
    let first_chunk = op.row_chunk_first(0, rows.clone()).expect("chunk first");
    assert_eq!(
        first_chunk,
        first_dense.slice(ndarray::s![rows.clone(), ..]).to_owned()
    );

    let diag_full = op
        .row_chunk_second_diag(0, 0..op.n_data())
        .expect("full row-chunk diag");
    let diag_chunk = op
        .row_chunk_second_diag(0, rows.clone())
        .expect("chunk diag");
    assert_eq!(
        diag_chunk,
        diag_full.slice(ndarray::s![rows.clone(), ..]).to_owned()
    );

    let cross_full = op
        .row_chunk_second_cross(0, 1, 0..op.n_data())
        .expect("full row-chunk cross");
    let cross_chunk = op
        .row_chunk_second_cross(0, 1, rows.clone())
        .expect("chunk cross");
    assert_eq!(
        cross_chunk,
        cross_full.slice(ndarray::s![rows, ..]).to_owned()
    );
}

#[test]
pub(crate) fn joint_trust_region_radius_update_accept_reject_logic() {
    let accepted = update_joint_trust_region_radius(1.0, 1.0, 2.0, 2.0, 1.0);
    assert!(accepted.accepted);
    assert!((accepted.rho - 1.0).abs() < 1.0e-12);
    assert!((accepted.radius - 2.0).abs() < 1.0e-12);
    assert_eq!(accepted.decision.label(), "grow_at_boundary");

    let rejected = update_joint_trust_region_radius(1.0, 0.5, -0.1, 2.0, 1.0);
    assert!(!rejected.accepted);
    assert!(rejected.rho < 0.0);
    assert!((rejected.radius - 0.25).abs() < 1.0e-12);
    assert_eq!(rejected.decision.label(), "shrink_reject");

    let rejected_inside_radius = update_joint_trust_region_radius(1.0, 1.0e-3, -0.1, 2.0, 1.0);
    assert!(!rejected_inside_radius.accepted);
    assert!(
        rejected_inside_radius.radius < 1.0e-3,
        "a rejected in-radius step must be outside the next trust region"
    );
    assert!((rejected_inside_radius.radius - 5.0e-4).abs() < 1.0e-12);
    assert_eq!(rejected_inside_radius.decision.label(), "shrink_reject");

    let poor = update_joint_trust_region_radius(1.0, 0.5, 0.1, 1.0, 1.0);
    assert!(poor.accepted);
    assert!((poor.rho - 0.1).abs() < 1.0e-12);
    assert!((poor.radius - 0.25).abs() < 1.0e-12);
    assert_eq!(poor.decision.label(), "shrink_marginal_accept");
}

/// gam#979: the coupled marginal↔logslope inner joint-Newton must NOT grind its
/// full cycle budget on a near-singular system whose trust region has collapsed
/// to the absolute `1e-12` floor with every line-search attempt rejected
/// (`phantom_multiplier_with_well_conditioned_H`). The deterministic guard fires
/// after `JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES` consecutive
/// all-reject-at-floor cycles — well before any realistic `inner_max_cycles`.
///
/// This replays the exact counter bookkeeping the loop runs (the same
/// `joint_trust_radius_at_absolute_floor` / `joint_collapsed_floor_all_reject_exit`
/// predicates the production loop calls) over a deterministic per-cycle reject
/// stream, and asserts (a) the guard fires before the budget, (b) it fires at the
/// threshold cycle (not earlier — so a transient reject does not abort a fit), and
/// (c) it never fires while the radius is above the floor (a progressing fit).
#[test]
pub(crate) fn joint_newton_collapsed_trust_region_all_reject_exits_before_grinding_budget() {
    // A representative inner budget. The guard must exit FAR below it.
    let inner_max_cycles = 200usize;
    assert!(
        JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES < inner_max_cycles / 4,
        "the collapsed-floor guard must exit well before the inner cycle budget"
    );

    // The absolute floor `update_joint_trust_region_radius` clamps to — driven
    // by repeated rejection (it ratchets the radius down by 0.25 each reject).
    let mut radius = 1.0_f64;
    for _ in 0..200 {
        let rejected = update_joint_trust_region_radius(radius, 0.5 * radius, -1.0, 2.0, 1.0);
        assert!(
            !rejected.accepted,
            "a genuine objective increase must reject"
        );
        radius = rejected.radius;
    }
    assert!(
        joint_trust_radius_at_absolute_floor(radius),
        "sustained rejection must collapse the radius to its absolute 1e-12 floor"
    );
    assert_eq!(
        update_joint_trust_region_radius(radius, 0.5 * radius, -1.0, 2.0, 1.0)
            .decision
            .label(),
        "reject_floor",
        "at the floor a rejected step must be classified reject_floor"
    );

    // Replay the loop's per-cycle bookkeeping: every cycle is fully rejected at
    // the collapsed floor. The guard must fire (and the loop break) at exactly
    // the threshold cycle, never grinding to `inner_max_cycles`.
    let mut consecutive = 0usize;
    let mut exit_cycle: Option<usize> = None;
    for cycle in 0..inner_max_cycles {
        let all_attempts_rejected = true;
        let at_floor = joint_trust_radius_at_absolute_floor(radius);
        let all_attempts_rejected_at_floor = all_attempts_rejected && at_floor;
        if all_attempts_rejected_at_floor {
            consecutive += 1;
        } else {
            consecutive = 0;
        }
        if joint_collapsed_floor_all_reject_exit(consecutive, all_attempts_rejected_at_floor) {
            exit_cycle = Some(cycle);
            break;
        }
    }
    let exit_cycle = exit_cycle.expect("collapsed-floor guard must exit, not grind to budget");
    assert_eq!(
        exit_cycle,
        JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES - 1,
        "guard must fire exactly at the threshold cycle (0-indexed)"
    );
    assert!(
        exit_cycle + 1 < inner_max_cycles,
        "guard must exit ({} cycles) well before the {} budget",
        exit_cycle + 1,
        inner_max_cycles
    );

    // Normal-path preservation: a progressing fit keeps the radius above the
    // floor, so neither the per-cycle predicate nor the streak ever trips —
    // even after far more than the threshold number of rejected (but not
    // floored) cycles.
    let progressing_radius = 1e-3_f64;
    assert!(
        !joint_trust_radius_at_absolute_floor(progressing_radius),
        "an above-floor radius must not count as the absolute floor"
    );
    let mut progressing_consecutive = 0usize;
    for _ in 0..(4 * JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES) {
        let all_attempts_rejected = true;
        let at_floor = joint_trust_radius_at_absolute_floor(progressing_radius);
        let all_attempts_rejected_at_floor = all_attempts_rejected && at_floor;
        if all_attempts_rejected_at_floor {
            progressing_consecutive += 1;
        } else {
            progressing_consecutive = 0;
        }
        assert!(
            !joint_collapsed_floor_all_reject_exit(
                progressing_consecutive,
                all_attempts_rejected_at_floor
            ),
            "the collapsed-floor guard must never fire while the radius is above the floor"
        );
    }

    // And: even AT the floor, a single accepted (non-reject) cycle resets the
    // streak, so the guard cannot fire on a fit that is still making progress.
    let mut streak = JOINT_COLLAPSED_FLOOR_ALL_REJECT_MAX_CYCLES - 1;
    // An accepted cycle => `all_attempts_rejected_at_floor` is false => reset.
    let accepted_cycle_all_reject_at_floor = false;
    streak = if accepted_cycle_all_reject_at_floor {
        streak + 1
    } else {
        0
    };
    assert!(
        !joint_collapsed_floor_all_reject_exit(streak, accepted_cycle_all_reject_at_floor),
        "an accepted cycle must reset the streak and suppress the guard"
    );
}

#[test]
pub(crate) fn joint_trust_region_noise_floor_accepts_round_off_negative_actual() {
    // Near-converged iterate at large objective scale: both the
    // model-predicted decrease and the realized objective change are
    // below the noise floor. Round-off can flip the sign of `actual`;
    // the principled response is to accept (rho ≈ 1) rather than
    // declare failure on the sign of noise. Mirrors the noise-floor
    // branch in `src/solver/pirls.rs`.
    let objective_scale = 1.66e5;
    let noise_floor = objective_scale * 1e-14;
    let predicted = noise_floor * 0.1;
    let actual = -noise_floor * 0.5;
    let update = update_joint_trust_region_radius(1.0, 0.05, actual, predicted, objective_scale);
    assert!(
        update.accepted,
        "sub-noise-floor sign flip must not reject as failure"
    );
    assert!((update.rho - 1.0).abs() < 1.0e-12);
}

#[test]
pub(crate) fn joint_trust_region_noise_floor_rejects_genuine_increase() {
    // Genuine objective increase clearly beyond the noise floor must
    // still be rejected even when predicted_reduction is sub-floor:
    // this is real model failure, not round-off.
    let objective_scale = 1.66e5;
    let noise_floor = objective_scale * 1e-14;
    let predicted = noise_floor * 0.1;
    let actual = -1.0;
    let update = update_joint_trust_region_radius(1.0, 0.5, actual, predicted, objective_scale);
    assert!(
        !update.accepted,
        "objective increase beyond noise must reject"
    );
    assert!(update.rho.is_infinite() && update.rho < 0.0);
}

#[test]
pub(crate) fn joint_objective_roundoff_slack_accepts_large_scale_wobble() {
    let old_objective = 1.218530e5;
    let trial_objective = old_objective + 2.183e-10;
    assert!(
        trial_objective
            <= old_objective + joint_objective_roundoff_slack(old_objective, trial_objective),
        "sub-nanounit objective wobble at large scale should not burn all trust attempts"
    );
}

#[test]
pub(crate) fn joint_objective_floor_only_accepts_sub_tolerance_model_steps() {
    let old_objective = 1.218942e5_f64;
    let objective_tol = 1e-6 * (1.0 + old_objective.abs());
    let actual_reduction = -3.783e-10;
    let predicted_reduction = 9.481e-15;
    let trial_objective = old_objective - actual_reduction;
    assert!(
        joint_objective_floor_reached(
            old_objective,
            trial_objective,
            actual_reduction,
            predicted_reduction,
            objective_tol,
        ),
        "the repeated large-scale roundoff wobble should terminate immediately"
    );

    assert!(
        !joint_objective_floor_reached(
            old_objective,
            old_objective + 2.0,
            -2.0,
            predicted_reduction,
            objective_tol,
        ),
        "real objective increases must still be rejected"
    );
    assert!(
        !joint_objective_floor_reached(
            old_objective,
            trial_objective,
            actual_reduction,
            10.0 * objective_tol,
            objective_tol,
        ),
        "non-negligible predicted progress must not be hidden by the floor exit"
    );
    // A positive-but-noise-level `actual_reduction` must NOT trigger the
    // floor (asymmetric guard). At rank-deficient optima the outer-gradient
    // FD identity (`outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active`,
    // inner_tol=1e-12) relies on the trust-region loop running the same
    // number of attempts at neighbouring λ probes; accepting positive-noise
    // reductions exits a cycle earlier on the probe where round-off
    // happened to land positive and decorrelates the null-space drift.
    let positive_noise_actual = 3.783e-10_f64;
    let positive_noise_trial = old_objective - positive_noise_actual;
    assert!(
        !joint_objective_floor_reached(
            old_objective,
            positive_noise_trial,
            positive_noise_actual,
            predicted_reduction,
            objective_tol,
        ),
        "positive-noise reductions must NOT trigger the floor; symmetric exit breaks rank-deficient FD identity"
    );
}

#[test]
pub(crate) fn joint_inner_convergence_rejects_objective_flat_non_kkt_stall() {
    // Direct reproduction of the bad 0.1.79 log shape:
    //
    //   obj=4.472714e5 Δobj=5.381e-2 |δ|∞=2.794e-2
    //   residual=5.980e1 tol=4.473e-1
    //
    // The objective and step are both flat at this scale, but the KKT
    // residual is 134x tolerance. Accepting this as an inner optimum makes
    // the envelope-theorem outer gradient invalid, which is what surfaced
    // as outer BFGS objective stalls with |g|≈1e14-1e16.
    let objective = 4.472714e5_f64;
    let inner_tol = 1.0e-6_f64;
    let objective_change = 5.381e-2_f64;
    let accepted_step_inf = 2.794e-2_f64;
    let residual = 5.980e1_f64;
    let residual_tol = inner_tol * (1.0 + objective);
    let step_tol = 1.242e-3_f64;
    let objective_tol = residual_tol;
    let old_flat_step_predicate = objective_change <= objective_tol
        && accepted_step_inf <= objective_tol.sqrt().max(step_tol);

    assert!(
        old_flat_step_predicate,
        "the historical objective-flat/step-flat predicate would have accepted this stalled inner solve"
    );
    assert!(
        !joint_inner_kkt_converged(residual, residual_tol),
        "inner convergence must require KKT residual <= tolerance"
    );
    assert!(
        !joint_inner_kkt_converged(1.5 * residual_tol, residual_tol),
        "near-miss residual slack would still invalidate the outer envelope gradient"
    );
}

#[test]
pub(crate) fn joint_trust_region_block_metric_does_not_starve_unrelated_blocks() {
    const TIME_W: usize = 12;
    const MARG_W: usize = 11;
    const LOG_W: usize = 10;
    const P: usize = TIME_W + MARG_W + LOG_W;

    let mut h = Array2::<f64>::zeros((P, P));
    let mut g = Array1::<f64>::zeros(P);
    h[[0, 0]] = 2.24e8;
    g[0] = -5.6e8;
    for i in 1..TIME_W {
        h[[i, i]] = 1.0 + 0.3 * i as f64;
        g[i] = -0.3 - 0.07 * i as f64;
    }
    for j in 0..MARG_W {
        let idx = TIME_W + j;
        h[[idx, idx]] = 1.2 + 0.2 * j as f64;
        g[idx] = -0.9;
    }
    let log0 = TIME_W + MARG_W;
    h[[log0, log0]] = 1.0e-5;
    g[log0] = -2.173;
    for k in 1..LOG_W {
        let idx = log0 + k;
        h[[idx, idx]] = 1.5 + 0.1 * k as f64;
        g[idx] = -0.4;
    }

    let mut newton = Array1::<f64>::zeros(P);
    for i in 0..P {
        newton[i] = -g[i] / h[[i, i]];
    }

    let mut raw_global = newton.clone();
    let raw_norm = raw_global.iter().map(|v| v * v).sum::<f64>().sqrt();
    if raw_norm.is_finite() && raw_norm > 20.0 {
        raw_global.mapv_inplace(|v| v * (20.0 / raw_norm));
    }
    let raw_linearized = (&g + &h.dot(&raw_global))
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        / (1.0 + g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    assert!(
        raw_linearized > 0.99,
        "raw concatenated L2 truncation should reproduce the starvation mechanism"
    );

    let ranges = vec![(0, TIME_W), (TIME_W, TIME_W + MARG_W), (TIME_W + MARG_W, P)];
    let metric_diag = h.diag().to_owned();
    let full_block_norms = joint_trust_region_block_metric_norms(&newton, &ranges, &metric_diag);
    let mut block_metric = newton.clone();
    let block_radii = vec![full_block_norms[0], full_block_norms[1], 20.0];
    truncate_joint_step_to_block_metric_radii(
        &mut block_metric,
        &ranges,
        &metric_diag,
        &block_radii,
    );
    let block_linearized = (&g + &h.dot(&block_metric))
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        / (1.0 + g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    assert!(
        block_linearized < 1.0e-6,
        "block-local curvature metric must let the time block neutralize its KKT defect; got {block_linearized:.3e}"
    );
}

#[test]
pub(crate) fn shrink_active_joint_block_trust_radii_strictly_decreases_max_radius() {
    // Regression for the joint-Newton fully-rejected stall. Before the
    // fix, when a boundary block's radius was already at the 1e-12 floor
    // and an interior block held the max, `shrink_active_joint_block_trust_radii`
    // returned the same `max(block_radii)` on every call — the trust
    // region never actually shrank, the dogleg recomputed an identical
    // joint δ, and the inner solver burned `inner_loop_hard_ceiling`
    // cycles before the 8-cycle stall guard finally bailed it out. The
    // fix must guarantee that every call strictly decreases the joint
    // trust radius until the floor.
    let mut block_radii = vec![1.0, 1.0e-12];
    // Boundary block (#1) sits at the radius floor with step at boundary;
    // interior block (#0) has step well inside its radius. Before the
    // fix: only block #1 participates, its radius re-clamps to 1e-12,
    // returned max stays at 1.0 — byte-identical to the previous call.
    let block_step_norms = vec![1.0e-3, 1.0e-12];
    let old_max = block_radii.iter().copied().fold(0.0_f64, f64::max);
    let new_max = shrink_active_joint_block_trust_radii(&mut block_radii, &block_step_norms, 0.25);
    assert!(
        new_max < old_max,
        "joint trust radius must strictly decrease when a step is rejected (was {old_max:.3e}, now {new_max:.3e})"
    );
    // Interior block must have shrunk below its current step norm so the
    // next dogleg step is forced strictly smaller in that block.
    assert!(
        block_radii[0] < block_step_norms[0],
        "interior block radius must drop below its step norm to force a strictly smaller next step (radius {:.3e}, step {:.3e})",
        block_radii[0],
        block_step_norms[0]
    );
}

#[test]
pub(crate) fn shrink_active_joint_block_trust_radii_decreases_max_when_max_held_by_interior_block()
{
    // Production stall (Rust CI Test job ~2-hour hang, cycles
    // 117..305+ all logging
    // `r=1.562e-2 (held) decision=shrink_reject |δ|=1.562e-2`
    // identically): the Moré–Sorensen inner trust-region step
    // (`spectrum.trust_region_step(joint_trust_radius)`) uses the
    // SCALAR `joint_trust_radius = max(block_radii)` as its trust
    // constraint. When a boundary block hits its per-block radius
    // (and shrinks) while an interior block holds the joint MAX
    // radius — but the boundary block is NOT yet at the floor, so
    // the `all_boundary_blocks_at_floor` carve-out doesn't fire —
    // only the boundary block participates, the interior max-holder
    // keeps its radius, `max(block_radii)` is held, MS re-computes
    // the byte-identical rejected step, and the inner Newton loop
    // stalls at `inner_loop_hard_ceiling`. The fix makes the
    // max-holder participate even when it's an interior block, so
    // the scalar joint radius strictly decreases on every rejected
    // attempt until the floor (where the `FULLY_REJECTED_STALL_MAX_CYCLES`
    // guard bails cleanly).
    let mut block_radii = vec![1.562e-2, 1.562e-2];
    // Block 0: step at per-block boundary (the boundary block).
    // Block 1: interior step well below its radius.
    // Both blocks share the joint max radius 1.562e-2 — the MS step
    // is constrained by that scalar value.
    let block_step_norms = vec![1.562e-2, 1.0e-6];
    let old_max = block_radii.iter().copied().fold(0.0_f64, f64::max);
    let new_max = shrink_active_joint_block_trust_radii(&mut block_radii, &block_step_norms, 0.25);
    assert!(
        new_max < old_max,
        "joint trust radius (= scalar Moré–Sorensen constraint) must \
             strictly decrease on rejection even when the max is held by \
             an interior block (was {old_max:.3e}, now {new_max:.3e})"
    );
}

#[test]
pub(crate) fn shrink_active_joint_block_trust_radii_pulls_radius_below_step_norm() {
    // The accept-path radius update (`update_joint_trust_region_radius`)
    // pulls the new radius below `0.5 * step_norm` on rejection so the
    // next step is provably smaller; the reject-path block shrink must
    // do the same. Otherwise an interior block with `step_norm <<
    // factor * radius` re-takes the identical Newton step on the next
    // dogleg attempt and the trust-region globalization is degenerate.
    let mut block_radii = vec![1.0];
    let block_step_norms = vec![1.0e-3];
    let new_max = shrink_active_joint_block_trust_radii(&mut block_radii, &block_step_norms, 0.25);
    assert!(
        new_max <= 0.5 * block_step_norms[0],
        "shrunken radius must be ≤ 0.5 · step_norm to force a strictly smaller next step (was {new_max:.3e}, step {:.3e})",
        block_step_norms[0]
    );
}

#[test]
pub(crate) fn blockwise_trust_region_uses_penalized_metric_not_raw_coefficient_size() {
    let spec = ParameterBlockSpec {
        name: "single_block".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::<f64>::zeros((1, 3)),
        )),
        offset: Array1::zeros(1),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let h: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0e-10]];
    let work = BlockWorkingSet::ExactNewton {
        gradient: array![0.0, 0.0, 0.0],
        hessian: SymmetricMatrix::Dense(h.clone()),
    };
    let s_lambda = Array2::<f64>::zeros((3, 3));
    let raw_delta: Array1<f64> = array![2.0, -1.0, 2.0e5];
    let raw_inf = raw_delta.iter().fold(0.0_f64, |m, v| {
        let value: f64 = *v;
        m.max(value.abs())
    });
    let radius = 20.0_f64;

    let raw_inf_scaled = &raw_delta * (radius / raw_inf);
    assert!(
        raw_inf_scaled[0].abs() < 1.0e-3,
        "the old raw coefficient cap would starve ordinary coordinates inside the block"
    );

    let (metric_delta, metric_norm) = truncate_block_step_to_metric_radius(
        &spec,
        &work,
        &s_lambda,
        raw_delta,
        radius,
        0.0,
        RidgePolicy::positive_part_approximate_objective(),
    )
    .expect("block metric truncation should succeed");
    assert!(
        metric_norm < radius,
        "the near-null coordinate is large in beta-space but small in the block's penalized-Hessian metric"
    );
    assert!(
        (metric_delta[0] - 2.0).abs() < 1.0e-12
            && (metric_delta[1] + 1.0).abs() < 1.0e-12
            && (metric_delta[2] - 2.0e5).abs() < 1.0e-6,
        "blockwise trust regions must size steps in objective curvature units, not raw coefficient units"
    );
}

#[test]
pub(crate) fn blockwise_trust_region_never_reverts_to_raw_beta_norm_on_indefinite_curvature() {
    let spec = ParameterBlockSpec {
        name: "single_block".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::<f64>::zeros((1, 3)),
        )),
        offset: Array1::zeros(1),
        penalties: vec![],
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let h: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0e-8]];
    let work = BlockWorkingSet::ExactNewton {
        gradient: array![0.0, 0.0, 0.0],
        hessian: SymmetricMatrix::Dense(h),
    };
    let s_lambda = Array2::<f64>::zeros((3, 3));
    let raw_delta: Array1<f64> = array![2.0, -1.0, 2.0e5];
    let radius = 20.0_f64;

    let old_quadratic = raw_delta.dot(&array![2.0, -1.0, -2.0e-3]);
    assert!(
        old_quadratic < 0.0,
        "fixture must hit the historical non-SPD branch"
    );

    let (metric_delta, metric_norm) = truncate_block_step_to_metric_radius(
        &spec,
        &work,
        &s_lambda,
        raw_delta,
        radius,
        0.0,
        RidgePolicy::positive_part_approximate_objective(),
    )
    .expect("block metric truncation should succeed");
    assert!(
        metric_norm < radius,
        "indefinite curvature must still use the positive penalized diagonal metric, not raw beta length"
    );
    assert!(
        (metric_delta[0] - 2.0).abs() < 1.0e-12
            && (metric_delta[1] + 1.0).abs() < 1.0e-12
            && (metric_delta[2] - 2.0e5).abs() < 1.0e-6,
        "non-SPD local curvature must not resurrect coefficient-space trust-region scaling"
    );
}

#[test]
pub(crate) fn joint_trust_region_rosenbrock_like_quadratic_is_armijo_safe() {
    // Local Rosenbrock-at-the-valley quadratic in variables (x, y):
    // f ≈ 0.5 * [dx, dy]' H [dx, dy], H = [[802, -400], [-400, 200]].
    // Add a tiny ridge to make the test SPD and use a gradient whose full
    // Newton step crosses the radius, exercising truncation before the
    // objective is evaluated.
    let h = array![[802.0, -400.0], [-400.0, 200.1]];
    let unconstrained = array![1.0, 1.0];
    let gradient = -h.dot(&unconstrained);
    let rhs = -&gradient;
    let mut step = unconstrained.clone();
    let unconstrained_norm = unconstrained.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(unconstrained_norm > 0.25);
    step.mapv_inplace(|v| v * (0.25 / unconstrained_norm));
    let step_norm = step.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(step_norm <= 0.25 + 1.0e-12);

    let h_step = h.dot(&step);
    let predicted = joint_quadratic_predicted_reduction(&rhs, &h_step, &step);
    let old_objective = 0.0;
    let trial_objective = gradient.dot(&step) + 0.5 * step.dot(&h_step);
    let actual = old_objective - trial_objective;
    assert!(predicted > 0.0);
    assert!((predicted - actual).abs() < 1.0e-10);

    let update =
        update_joint_trust_region_radius(0.25, step_norm, actual, predicted, old_objective);
    assert!(update.accepted);
    assert!(trial_objective < old_objective);
}

// Inline RED REPRO moved to tests/joint_newton_isotropic_tr_starvation.rs
// so it survives in-progress refactors of the surrounding test
// support module (this `mod tests { }` currently does not compile due
// to `gam_test_support::*` / `test_outerobjective_andgradient` WIP).

/// Synthetic 3-block fixture where the joint penalized Hessian is
/// rank-deficient inside block 2 (block-diagonal H with two
/// well-conditioned 3x3 identity blocks and a rank-1 third block; all
/// s_lambdas are zero so the penalty does not lift the deficiency).
/// The gradient is concentrated on block 2's null directions so the
/// stationarity residual is dominated by block 2. The report must
/// (a) classify the refusal as `RankDeficientHPen`, (b) record
/// nullity > 0, and (c) name block 2 as the carrying block.
#[test]
pub(crate) fn kkt_refusal_report_classifies_rank_deficient_hpen_third_block() {
    let block_widths = [3usize, 3, 3];
    let total_p: usize = block_widths.iter().sum();
    let block_count = block_widths.len();

    let mut specs: Vec<ParameterBlockSpec> = Vec::with_capacity(block_count);
    let mut states: Vec<ParameterBlockState> = Vec::with_capacity(block_count);
    let mut s_lambdas: Vec<Array2<f64>> = Vec::with_capacity(block_count);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(block_count);
    let names = ["block_a", "block_b", "block_c_rank_deficient"];
    let mut offset = 0usize;
    for (b, &width) in block_widths.iter().enumerate() {
        let start = offset;
        let end = start + width;
        offset = end;
        ranges.push((start, end));
        specs.push(ParameterBlockSpec {
            name: names[b].to_string(),
            design: DesignMatrix::from(Array2::<f64>::zeros((1, width))),
            offset: Array1::zeros(1),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        });
        states.push(ParameterBlockState {
            beta: Array1::zeros(width),
            eta: Array1::zeros(1),
        });
        s_lambdas.push(Array2::<f64>::zeros((width, width)));
    }

    // Block-diagonal H: I(3) ⊕ I(3) ⊕ e0 e0ᵀ (third block rank 1, nullity 2).
    let mut h = Array2::<f64>::zeros((total_p, total_p));
    for i in 0..3 {
        h[[i, i]] = 1.0;
        h[[3 + i, 3 + i]] = 1.0;
    }
    h[[6, 6]] = 1.0;

    let source = JointHessianSource::Dense(h);

    // Concentrate the gradient on block 2's null directions (rows 7,8).
    // With s_lambdas all zero and β=0, the stationarity residual equals
    // -gradient, so block 2 carries the dominant residual mass.
    let mut joint_grad = Array1::<f64>::zeros(total_p);
    joint_grad[7] = 5.0;
    joint_grad[8] = 3.0;
    joint_grad[0] = 1.0e-6;

    let cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; block_count];
    let block_constraints: Vec<Option<LinearInequalityConstraints>> = vec![None; block_count];

    let math = JointNewtonMathDiagnostic {
        old_kkt_inf: 5.0,
        linearized_next_kkt_inf: 4.9,
        predicted_reduction: 1.0e-4,
        actual_reduction: 1.0e-4,
        trust_ratio: 1.0,
        step_inf: 1.0e-9,
        proposal_inf: 1.0e-3,
    };

    let residual_tol = 1.0e-6;
    let projected_residual_inf = 5.0;

    let report = compute_kkt_refusal_report(
        42,
        &states,
        &specs,
        &s_lambdas,
        &ranges,
        Some(&joint_grad),
        &cached_active_sets,
        &block_constraints,
        Some(&source),
        total_p,
        0.0,
        RidgePolicy::exact_full_objective(),
        1.0e-9,
        1.0e-3,
        1.0,
        residual_tol,
        1.0e-6,
        1.0e-6,
        1.0e-8,
        projected_residual_inf,
        Some(&math),
    );

    assert_eq!(
        report.diagnosis,
        KktRefusalDiagnosis::RankDeficientHPen,
        "block-2 rank-1 H_pen with zero s_lambdas must classify as RankDeficientHPen, got {:?}",
        report.diagnosis,
    );
    assert!(
        report.hpen_nullity_at_rank_tol > 0,
        "rank-1 block embedded in 9x9 block-diagonal H must register nullity > 0, got {}",
        report.hpen_nullity_at_rank_tol,
    );
    assert_eq!(
        report.block_carrying_residual,
        Some(2),
        "block 2 must carry the largest |∇L − Sβ|∞ component; got {:?}, residuals={:?}",
        report.block_carrying_residual,
        report.block_residual_inf,
    );
    assert_eq!(report.block_names.len(), block_count);
    assert_eq!(
        report.block_names[2], "block_c_rank_deficient",
        "carrying-block name should be the third block",
    );
    assert!(
        report
            .format_structured_log(residual_tol)
            .contains("rank_deficient_H_pen"),
        "structured log must surface the diagnosis label",
    );
    assert!(
        report
            .format_bubbled_error()
            .contains("block_c_rank_deficient"),
        "bubbled error must name the carrying block by spec.name",
    );
    assert!(
        report
            .format_bubbled_error()
            .contains("structural or numerical null direction"),
        "rank-deficient refusals should no longer emit the old polynomial-only guidance",
    );
}

/// Round-trip: every variant's `as_str()` output, when embedded in the
/// `diagnosis: <label>` slot of the bubbled-error format, must parse
/// back via `parse_from_error`. seed-accounting's `InnerStatus`
/// classifier reads diagnoses out of bubbled error strings via that
/// parser; if a variant's label diverges between formatter and parser
/// the classifier silently falls back to "unknown" and the early-exit
/// canary degrades to a generic non-converged result.
#[test]
pub(crate) fn kkt_refusal_diagnosis_string_round_trip_through_bubbled_error_parser() {
    for diagnosis in [
        KktRefusalDiagnosis::RankDeficientHPen,
        KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH,
        KktRefusalDiagnosis::ActiveSetIncomplete,
        KktRefusalDiagnosis::AliasingDetectedAtFit,
    ] {
        let label = diagnosis.as_str();
        // Mimic the trailing slot exactly as `format_bubbled_error`
        // emits it (label at the very end after `; diagnosis: `).
        let synthetic_error = format!(
            "coupled exact-joint inner solve exited the joint Newton path before convergence \
                 — cycle=7 cert REFUSED: residual=1.0e-2 > tol=1.0e-6; \
                 diagnosis: {label}"
        );
        let parsed = KktRefusalDiagnosis::parse_from_error(&synthetic_error);
        assert_eq!(
            parsed,
            Some(diagnosis),
            "label '{label}' must round-trip through parse_from_error; got {:?}",
            parsed,
        );
    }
}

#[test]
pub(crate) fn kkt_refusal_guidance_distinguishes_marginal_slope_coupling_from_polynomial_nullspace()
{
    let phantom = KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH.guidance();
    assert!(phantom.contains("marginal/logslope coupling"));
    assert!(phantom.contains("rather than a"));
    assert!(phantom.contains("Matérn/Duchon polynomial-nullspace failure"));

    let active = KktRefusalDiagnosis::ActiveSetIncomplete.guidance();
    assert!(active.contains("active-set certification failure"));
    assert!(active.contains("not a polynomial-nullspace diagnosis"));

    let alias = KktRefusalDiagnosis::AliasingDetectedAtFit.guidance();
    assert!(alias.contains("drop or reparameterize"));
}

/// Regression canary: a synthetic 3-block fixture chosen to mimic the
/// large-scale rank-deficient-H_pen failure mode — block-diagonal H with
/// a fully degenerate third block and zero s_lambdas — must classify
/// as `RankDeficientHPen` with nullity matching the structural rank
/// deficiency. When `nullspace-lead`'s smooth-construction
/// reparameterization lands and absorbs polynomial null spaces into
/// the parametric block, the SAME fixture (rewritten with a
/// full-rank reparameterized basis) should fit cleanly with no
/// refusal. That follow-up half is wired below behind `#[ignore]`
/// per the lead's note; the diagnosis half here is active so the
/// canary fires today on the failure mode the rework targets.
#[test]
pub(crate) fn rank_deficient_hpen_canary_fires_on_large_scale_shaped_failure() {
    let block_widths = [4usize, 4, 4];
    let total_p: usize = block_widths.iter().sum();
    let block_count = block_widths.len();

    let mut specs: Vec<ParameterBlockSpec> = Vec::with_capacity(block_count);
    let mut states: Vec<ParameterBlockState> = Vec::with_capacity(block_count);
    let mut s_lambdas: Vec<Array2<f64>> = Vec::with_capacity(block_count);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(block_count);
    let names = ["location_block", "scale_block", "marginal_slope_block"];
    let mut offset = 0usize;
    for (b, &width) in block_widths.iter().enumerate() {
        let start = offset;
        let end = start + width;
        offset = end;
        ranges.push((start, end));
        specs.push(ParameterBlockSpec {
            name: names[b].to_string(),
            design: DesignMatrix::from(Array2::<f64>::zeros((1, width))),
            offset: Array1::zeros(1),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        });
        states.push(ParameterBlockState {
            beta: Array1::zeros(width),
            eta: Array1::zeros(1),
        });
        s_lambdas.push(Array2::<f64>::zeros((width, width)));
    }

    // H = I(4) ⊕ I(4) ⊕ 0 — the third block is the marginal-slope
    // pathology: zero Hessian curvature on a 4-D null space the
    // penalty does not constrain (s_lambdas are zero everywhere).
    let mut h = Array2::<f64>::zeros((total_p, total_p));
    for i in 0..4 {
        h[[i, i]] = 1.0;
        h[[4 + i, 4 + i]] = 1.0;
    }
    // Marginal-slope block left as the zero matrix → nullity = 4.

    let source = JointHessianSource::Dense(h);

    // Gradient mass concentrated on the marginal-slope block. With
    // β=0 and S=0, the stationarity residual on that block equals
    // −gradient there, so the carrying block is unambiguous.
    let mut joint_grad = Array1::<f64>::zeros(total_p);
    joint_grad[8] = 4.2;
    joint_grad[9] = 1.7;
    joint_grad[10] = -2.5;
    joint_grad[11] = 0.9;

    let cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; block_count];
    let block_constraints: Vec<Option<LinearInequalityConstraints>> = vec![None; block_count];
    let math = JointNewtonMathDiagnostic {
        old_kkt_inf: 4.2,
        linearized_next_kkt_inf: 4.2,
        predicted_reduction: 0.0,
        actual_reduction: 0.0,
        trust_ratio: 0.0,
        step_inf: 0.0,
        proposal_inf: 1.0e-3,
    };

    let report = compute_kkt_refusal_report(
        123,
        &states,
        &specs,
        &s_lambdas,
        &ranges,
        Some(&joint_grad),
        &cached_active_sets,
        &block_constraints,
        Some(&source),
        total_p,
        0.0,
        RidgePolicy::exact_full_objective(),
        0.0,
        1.0e-3,
        1.0,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        0.0,
        4.2,
        Some(&math),
    );

    assert_eq!(
        report.diagnosis,
        KktRefusalDiagnosis::RankDeficientHPen,
        "large-scale-shaped marginal-slope failure must classify as RankDeficientHPen \
             (this is the canary nullspace-lead's smooth-construction rework targets)",
    );
    assert!(
        report.hpen_nullity_at_rank_tol >= 4,
        "fully degenerate marginal-slope block (4 zero eigenvalues) must contribute \
             nullity >= 4; got {}",
        report.hpen_nullity_at_rank_tol,
    );
    assert_eq!(
        report.block_carrying_residual,
        Some(2),
        "marginal_slope_block (idx 2) must carry the residual; got {:?}, residuals={:?}",
        report.block_carrying_residual,
        report.block_residual_inf,
    );
    let bubbled = report.format_bubbled_error();
    assert_eq!(
        KktRefusalDiagnosis::parse_from_error(&bubbled),
        Some(KktRefusalDiagnosis::RankDeficientHPen),
        "canary's bubbled-error string must parse back via the classifier's parser",
    );
    assert!(
        bubbled.contains("marginal-slope fits can also expose callback-owned weak directions"),
        "BMS-shaped refusal should mention the callback-owned weak-direction mechanism"
    );
}

/// Post-fix half of the canary: once `nullspace-lead`'s smooth
/// reparameterization absorbs polynomial null spaces into the
/// parametric block, the marginal-slope synthetic above (rewritten
/// to use a full-rank reparameterized basis with the absorbed null
/// columns moved into a separate identifiable block) should fit
/// without any cert refusal.
#[test]
pub(crate) fn rank_deficient_hpen_canary_disappears_after_nullspace_absorption() {
    let block_widths = [4usize, 4, 4];
    let total_p: usize = block_widths.iter().sum();
    let block_count = block_widths.len();

    let mut specs: Vec<ParameterBlockSpec> = Vec::with_capacity(block_count);
    let mut states: Vec<ParameterBlockState> = Vec::with_capacity(block_count);
    let mut s_lambdas: Vec<Array2<f64>> = Vec::with_capacity(block_count);
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(block_count);
    let names = ["location_block", "scale_block", "marginal_slope_block"];
    let mut offset = 0usize;
    for (b, &width) in block_widths.iter().enumerate() {
        let start = offset;
        let end = start + width;
        offset = end;
        ranges.push((start, end));
        specs.push(ParameterBlockSpec {
            name: names[b].to_string(),
            design: DesignMatrix::from(Array2::<f64>::zeros((1, width))),
            offset: Array1::zeros(1),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        });
        states.push(ParameterBlockState {
            beta: Array1::zeros(width),
            eta: Array1::zeros(1),
        });
        s_lambdas.push(Array2::<f64>::zeros((width, width)));
    }

    // Full-rank H across all three blocks — the post-absorption
    // shape: the polynomial null space has been moved out of the
    // smooth and the remaining basis is fully identified by the
    // likelihood Hessian.
    let h = Array2::<f64>::eye(total_p);
    let source = JointHessianSource::Dense(h);
    let joint_grad = Array1::<f64>::zeros(total_p);
    let cached_active_sets: Vec<Option<Vec<usize>>> = vec![None; block_count];
    let block_constraints: Vec<Option<LinearInequalityConstraints>> = vec![None; block_count];
    let math = JointNewtonMathDiagnostic {
        old_kkt_inf: 0.0,
        linearized_next_kkt_inf: 0.0,
        predicted_reduction: 0.0,
        actual_reduction: 0.0,
        trust_ratio: 1.0,
        step_inf: 0.0,
        proposal_inf: 0.0,
    };

    let report = compute_kkt_refusal_report(
        0,
        &states,
        &specs,
        &s_lambdas,
        &ranges,
        Some(&joint_grad),
        &cached_active_sets,
        &block_constraints,
        Some(&source),
        total_p,
        0.0,
        RidgePolicy::exact_full_objective(),
        0.0,
        0.0,
        1.0,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        0.0,
        0.0,
        Some(&math),
    );

    assert_eq!(
        report.hpen_nullity_at_rank_tol, 0,
        "post-absorption: full-rank H_pen must register nullity 0",
    );
    assert_ne!(
        report.diagnosis,
        KktRefusalDiagnosis::RankDeficientHPen,
        "post-absorption: the rank-deficiency diagnosis must no longer fire",
    );
}

/// Pins the structural effective-df machinery to the exact trace identity
///
/// ```text
/// Σ_j γ_j/(γ_j + λ) = tr{ G (G + λ S)⁻¹ }
/// ```
///
/// on a NON-commuting Gram/penalty pair, where the historical Rayleigh-quotient
/// implementation (diagonal of B only) gave the wrong answer. With
/// `S = diag(1, 4)` and `G = [[1, 0.8], [0.8, 1]]` the true generalized
/// eigenvalues are eig(D^{-1/2} Uᵀ G U D^{-1/2}) ≈ [0.0767072, 1.1732928],
/// whereas the Rayleigh quotients are [1, 0.25]; only the former reproduce the
/// trace identity, and they disagree at λ = 1 (≈0.6111 vs the buggy 0.7000).
#[test]
pub(crate) fn structural_edf_matches_trace_identity_noncommuting_pair() {
    // Penalty S = diag(1, 4).
    let s = array![[1.0, 0.0], [0.0, 4.0]];
    // Design with Gram G = XᵀX = [[1, 0.8], [0.8, 1]]. Use the symmetric
    // square root G^{1/2} so that XᵀX = G exactly:
    //   G = 1.8·v1v1ᵀ + 0.2·v2v2ᵀ, v1=[1,1]/√2, v2=[1,-1]/√2.
    let off = 0.5 * (1.8_f64.sqrt() - 0.2_f64.sqrt());
    let diag = 0.5 * (1.8_f64.sqrt() + 0.2_f64.sqrt());
    let x = array![[diag, off], [off, diag]];
    let design = DesignMatrix::from(x);
    let penalty = PenaltyMatrix::Dense(s.clone());

    let gammas = design_penalty_range_gammas(&design, &penalty)
        .expect("2x2 full-rank p×p pair must yield generalized eigenvalues");
    assert_eq!(gammas.len(), 2, "range(S) is full rank ⇒ two γ_j");

    // Reference: G = XᵀX, and tr(G (G+λS)⁻¹) computed via the closed-form
    // 2×2 inverse of M = G + λ S (det/adjugate), independent of the helper.
    let g = array![[1.0, 0.8], [0.8, 1.0]];
    let trace_g_minv = |lambda: f64| -> f64 {
        let m00 = g[(0, 0)] + lambda * s[(0, 0)];
        let m01 = g[(0, 1)] + lambda * s[(0, 1)];
        let m10 = g[(1, 0)] + lambda * s[(1, 0)];
        let m11 = g[(1, 1)] + lambda * s[(1, 1)];
        let det = m00 * m11 - m01 * m10;
        // M⁻¹ = (1/det) [[m11, -m01], [-m10, m00]];
        // tr(G M⁻¹) = (1/det) · [ G00·m11 - G01·m10 - G10·m01 + G11·m00 ].
        (g[(0, 0)] * m11 - g[(0, 1)] * m10 - g[(1, 0)] * m01 + g[(1, 1)] * m00) / det
    };

    for &lambda in &[1.0_f64, 0.3] {
        let rho = lambda.ln();
        let edf = unit_weight_term_edf(&gammas, rho).expect("finite canonical log strength");
        let trace = trace_g_minv(lambda);
        assert!(
            (edf - trace).abs() < 1e-9,
            "structural edf {edf} must equal tr(G(G+λS)⁻¹) {trace} at λ={lambda}",
        );
    }

    // Sanity: the buggy Rayleigh quotients [1, 0.25] would give 0.7 at λ=1,
    // which the trace identity (≈0.6111) rejects — guard against regression
    // to the diagonal-only computation.
    let edf_at_one =
        unit_weight_term_edf(&gammas, 0.0_f64).expect("zero is a canonical log strength");
    assert!(
        (edf_at_one - 0.611111_f64).abs() < 1e-5,
        "edf at λ=1 must be ≈0.6111 (true), not 0.7000 (Rayleigh-quotient bug): got {edf_at_one}",
    );
}

#[test]
pub(crate) fn per_penalty_edf_uses_realized_penalty_rank_2288() {
    // Two blocks × two penalties, with distinct ranks, strengths, diagonal
    // weights, and Hessian scales. Every flattened output is intentionally
    // different: a positional permutation, a reset of the global penalty
    // cursor at a block boundary, or use of containing-block width as rank can
    // no longer pass accidentally.
    let specs = vec![
        ParameterBlockSpec {
            name: "first_two_penalty_block".to_string(),
            design: DesignMatrix::from(Array2::<f64>::zeros((2, 3))),
            offset: Array1::zeros(2),
            penalties: vec![
                PenaltyMatrix::Dense(array![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],]),
                PenaltyMatrix::Dense(array![[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],]),
            ],
            // Empty is deliberate: canonicalization clears pre-transform
            // nullities, so rank must come from the realized roots.
            nullspace_dims: vec![],
            initial_log_lambdas: array![1.0_f64.ln(), 0.5_f64.ln()],
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "second_two_penalty_block".to_string(),
            design: DesignMatrix::from(Array2::<f64>::zeros((2, 3))),
            offset: Array1::zeros(2),
            penalties: vec![
                PenaltyMatrix::Dense(array![[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],]),
                PenaltyMatrix::Dense(array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0],]),
            ],
            nullspace_dims: vec![],
            initial_log_lambdas: array![0.25_f64.ln(), 2.0_f64.ln()],
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    let h = Array2::from_diag(&array![4.0, 5.0, 10.0, 8.0, 10.0, 20.0]);
    let lambdas = array![1.0, 0.5, 0.25, 2.0];

    let (edf_total, edf_by_penalty, block_edf, penalty_trace) =
        custom_family_blockwise_edf(&h, &specs, &lambdas.view()).expect("exact composed EDF");

    // Independent diagonal oracle:
    //   λ tr(H⁻¹S) = [1/4, (1/2)(2/5+3/10), (1/4)(4/8), 2(1/10+2/20)].
    let expected_trace = [0.25, 0.35, 0.125, 0.4];
    let expected_edf = [0.75, 1.65, 0.875, 1.6];
    assert_eq!(penalty_trace.len(), expected_trace.len());
    assert_eq!(edf_by_penalty.len(), expected_edf.len());
    for (&actual, &expected) in penalty_trace.iter().zip(expected_trace.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-12);
    }
    for (&actual, &expected) in edf_by_penalty.iter().zip(expected_edf.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-12);
    }
    assert_relative_eq!(block_edf[0], 2.4, epsilon = 1e-12);
    assert_relative_eq!(block_edf[1], 2.475, epsilon = 1e-12);
    assert_relative_eq!(edf_total, 4.875, epsilon = 1e-12);
}

/// Structural edf with a penalty NULLSPACE coupled to the range through the
/// design Gram: the γ_j must come from the Schur complement
/// `A_rr − A_r0 A₀₀⁺ A₀r` on the penalty quotient, not from `A_rr` alone.
///
/// `S = diag(0, 1, 1)` and `G = [[1, 1, 0], [1, 1+ε, 0], [0, 0, 1]]`: the
/// unpenalized null coordinate absorbs the shared curvature between
/// coordinates 0 and 1 at every λ, so the quotient eigenvalues are (ε, 1) —
/// keeping `A_rr` alone would claim (1+ε, 1) and overstate the λ-resistant
/// df. Verified against the exact trace identity
/// `tr{G (G+λS)⁻¹} = rank(A₀₀) + Σ_j γ_j/(γ_j+λ)` with `rank(A₀₀) = 1`.
#[test]
pub(crate) fn structural_edf_quotients_nullspace_range_coupling() {
    let eps = 0.01_f64;
    // XᵀX = [[1,1,0],[1,1+ε,0],[0,0,1]] via the Cholesky factor
    // L = [[1,0,0],[1,√ε,0],[0,0,1]], X = Lᵀ.
    let x = array![[1.0, 1.0, 0.0], [0.0, eps.sqrt(), 0.0], [0.0, 0.0, 1.0]];
    let design = DesignMatrix::from(x);
    let s = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let penalty = PenaltyMatrix::Dense(s.clone());

    let mut gammas = design_penalty_range_gammas(&design, &penalty)
        .expect("rank-2 penalty range must yield generalized eigenvalues");
    gammas.sort_by(|a, b| a.partial_cmp(b).expect("finite γ"));
    assert_eq!(gammas.len(), 2, "range(S) has rank 2");
    assert!(
        (gammas[0] - eps).abs() < 1e-9,
        "coupled range direction must carry quotient curvature ε={eps}, got {}",
        gammas[0]
    );
    assert!(
        (gammas[1] - 1.0).abs() < 1e-9,
        "uncoupled range direction keeps curvature 1, got {}",
        gammas[1]
    );

    // Exact trace identity: tr(G (G+λS)⁻¹) = 1 + Σ_j γ_j/(γ_j+λ), the leading
    // 1 being the null-block rank (fitted unpenalized at every λ). Reference
    // computed with a dense solve independent of the helper.
    let g = array![[1.0, 1.0, 0.0], [1.0, 1.0 + eps, 0.0], [0.0, 0.0, 1.0]];
    for &lambda in &[0.25_f64, 1.0, 7.5] {
        let m = &g + &(&s * lambda);
        let m_inv = {
            use gam_linalg::faer_ndarray::FaerCholesky;
            let chol = m.cholesky(faer::Side::Lower).expect("G+λS is SPD here");
            let mut ident = Array2::<f64>::eye(3);
            chol.solve_mat_in_place(&mut ident);
            ident
        };
        let trace: f64 = g.dot(&m_inv).diag().iter().sum();
        let edf = 1.0
            + unit_weight_term_edf(&gammas, lambda.ln())
                .expect("test lambdas have canonical log strengths");
        assert!(
            (edf - trace).abs() < 1e-9,
            "quotient edf {edf} must equal tr(G(G+λS)⁻¹) {trace} at λ={lambda}",
        );
    }
}

#[test]
fn unit_weight_structural_edf_rejects_noncanonical_log_strengths() {
    let gammas = [0.25, 1.0, 4.0];
    for rho in [
        gam_problem::LOG_STRENGTH_MIN - 1.0,
        gam_problem::LOG_STRENGTH_MAX + 1.0,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NAN,
    ] {
        assert!(unit_weight_term_edf(&gammas, rho).is_err());
    }
}

/// gam#1854 / gam#1395: the multinomial Firth/Jeffreys separation fallback assembles
/// the outer joint Hessian `H_unpen + S_λ + scale·H_Φ` and, for small systems
/// (`total <= JOINT_LOGDET_GUARD_MAX_DIM`), realizes its `0.5·log|H|` Laplace term
/// through `BlockCoupledOperator::from_joint_hessian_with_mode` →
/// `DenseSpectralOperator::from_symmetric_with_mode` → `eigh(Side::Lower)`. That
/// eigensolver reads ONLY the lower triangle and ASSUMES the input is symmetric.
///
/// On the near-separation Firth path the divided-difference `H_Φ` (plus its
/// second-order completion) carries an `O(1e10)` curvature scale, so reduction-order
/// floating-point noise desyncs the assembled matrix's mirror entries by an amount
/// that is *large in absolute terms*. Reading the raw lower triangle then yields a
/// materially different spectrum — and logdet — than the symmetrized matrix. The
/// gam#1395 ground-truth guard in `joint_outer_evaluate` reconstructs the SAME matrix
/// but symmetrizes it first, so an unsymmetrized assembly makes the assembled-vs-
/// reference logdet diverge and the guard `assert!` fires (caught by the fallback's
/// `catch_unwind` and degraded to the clean separation error — the #1854 symptom).
///
/// The fix symmetrizes the assembled joint Hessian in place before constructing the
/// `BlockCoupledOperator`, mirroring the guard's ground truth and the matrix-free
/// dense-assemble path. This test pins that invariant at the operator boundary that
/// the guard compares across: on a symmetric input the `BlockCoupledOperator` and the
/// guard's `DenseSpectralOperator` realize the identical logdet (the guard's apples-to-
/// apples assumption), while the RAW asymmetric matrix — the pre-symmetrization state —
/// diverges by FAR more than the guard tolerance. That divergence is exactly why the
/// symmetrization is load-bearing; removing it re-opens the #1854 guard trip.
#[test]
fn multinomial_firth_joint_hessian_logdet_needs_symmetrization_1854() {
    let mode = PseudoLogdetMode::Smooth;

    // A 3×3 SPD joint-Hessian stand-in with an `O(1e10)` curvature scale (the
    // near-separation Firth regime) and a large lower-/upper-triangle desync on one
    // off-diagonal pair, standing in for reduction-order f.p. noise on a huge entry.
    let mut raw = Array2::<f64>::zeros((3, 3));
    raw[[0, 0]] = 1.0e10;
    raw[[1, 1]] = 5.0;
    raw[[2, 2]] = 2.0;
    raw[[0, 1]] = 0.0; // upper mirror entry
    raw[[1, 0]] = 2.0e5; // lower mirror entry — desynced from the upper one

    // Guard ground truth: symmetrize first, then the dense spectral operator.
    let mut symmetric = raw.clone();
    symmetrize_dense_in_place(&mut symmetric);
    let reference = DenseSpectralOperator::from_symmetric_with_mode(&symmetric, mode)
        .expect("reference eigendecomposition of the symmetrized joint Hessian");
    let reference_logdet = reference.logdet();
    assert!(
        reference_logdet.is_finite(),
        "reference logdet must be finite: {reference_logdet}"
    );

    // Post-fix assembly route: `BlockCoupledOperator` on the SAME symmetrized matrix.
    let assembled = BlockCoupledOperator::from_joint_hessian_with_mode(&symmetric, mode)
        .expect("assembled BlockCoupledOperator on the symmetrized joint Hessian");
    let assembled_logdet = assembled.logdet();

    // Guard tolerance, verbatim from `joint_outer_evaluate`'s gam#1395 check.
    let total = 3usize;
    let tol = 1e-7 * (total as f64) * (1.0 + reference_logdet.abs());

    // Apples-to-apples: on a symmetric input the two operator routes realize the
    // identical logdet, so the guard passes. This is the property the symmetrization
    // restores.
    assert!(
        (assembled_logdet - reference_logdet).abs() <= tol,
        "symmetrized assembly must match the gam#1395 reference logdet within guard \
         tolerance: assembled={assembled_logdet:.9e} reference={reference_logdet:.9e} \
         tol={tol:.3e}"
    );

    // Load-bearing check: feeding the RAW asymmetric matrix (the pre-symmetrization
    // state) to the same operator route makes `eigh(Side::Lower)` read the desynced
    // lower triangle, diverging from the guard's reference by FAR more than the guard
    // tolerance — i.e. skipping the symmetrization trips the gam#1395 guard exactly as
    // reported in #1854.
    let unsymmetrized = BlockCoupledOperator::from_joint_hessian_with_mode(&raw, mode)
        .expect("BlockCoupledOperator on the raw asymmetric joint Hessian");
    let unsymmetrized_logdet = unsymmetrized.logdet();
    assert!(
        (unsymmetrized_logdet - reference_logdet).abs() > 1.0e3 * tol,
        "raw asymmetric assembly must diverge from the reference (symmetrization is \
         load-bearing): raw={unsymmetrized_logdet:.9e} reference={reference_logdet:.9e} \
         tol={tol:.3e}"
    );
}
