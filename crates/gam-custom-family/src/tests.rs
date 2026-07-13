//! Unit tests for the custom-family blockwise carrier. Declared from `mod.rs`
//! as `#[cfg(test)] mod tests;`; reaches the FD helper via `super::test_support`.

use super::*;

pub(crate) fn test_design_hyper_layout(
    design_derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
) -> CustomFamilyHyperLayout {
    let axis_count: usize = design_derivative_blocks.iter().map(Vec::len).sum();
    CustomFamilyHyperLayout::new(
        design_derivative_blocks,
        Vec::new(),
        Array1::zeros(axis_count),
    )
    .expect("test design-hyper layout must satisfy the typed axis contract")
}

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
                working: Some(WorkingGeometry {
                    working_weights: Array1::ones(2),
                    working_response: Array1::zeros(2),
                }),
            }),
            precomputed_edf: Some((1.0, Vec::new(), vec![1.0], Vec::new())),
            joint_log_lambdas: None,
        },
        &[spec],
    )
    .expect("stacked solver eta should assemble against canonical geometry rows");

    assert_eq!(fit.block_states[0].eta.len(), 6);
    assert_eq!(
        fit.geometry
            .as_ref()
            .unwrap()
            .working
            .as_ref()
            .unwrap()
            .working_weights
            .len(),
        2,
    );
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
        working: Some(WorkingGeometry {
            working_weights: array![1.0],
            working_response: array![0.0],
        }),
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
        terminal_working_sets: None,
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
        terminal_working_sets: None,
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
            terminal_working_sets: None,
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
        terminal_working_sets: None,
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
struct IntentRefiningHessianWorkspace {
    dense_calls: Arc<AtomicUsize>,
    matvec_calls: Arc<AtomicUsize>,
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
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);
    let rho = array![0.0];
    let result = evaluate_custom_family_joint_hyper(
        &DefaultDiagonalExactHookFamily,
        &specs,
        &options,
        &rho,
        &hyper_layout,
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
            &hyper_layout,
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
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);
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
        &hyper_layout,
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
struct OneBlockQuarticExactFamily {
    linear: f64,
    curvature: f64,
    second_scale: f64,
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
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);

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
        &hyper_layout,
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
        &hyper_layout,
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
struct PairwiseJeffreysSeamFamily;

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

pub(crate) fn certified_test_outer(
    theta: Array1<f64>,
    objective: f64,
) -> gam_solve::rho_optimizer::CertifiedOuterResult {
    assert!(!theta.is_empty(), "certificate fixture requires one real outer coordinate");
    let dimension = theta.len();
    let cost_center = theta.clone();
    let gradient_center = theta.clone();
    let problem = gam_solve::rho_optimizer::OuterProblem::new(dimension)
        .with_gradient(gam_problem::Derivative::Analytic)
        .with_hessian(gam_problem::DeclaredHessianForm::Dense)
        .with_disable_fixed_point(true)
        .with_fallback_policy(gam_solve::rho_optimizer::FallbackPolicy::Disabled)
        .with_initial_rho(theta)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..gam_problem::SeedConfig::default()
        });
    let mut outer_objective = problem.build_objective(
        (),
        move |_: &mut (), point: &Array1<f64>| {
            let displacement = point - &cost_center;
            Ok(objective + displacement.dot(&displacement))
        },
        move |_: &mut (), point: &Array1<f64>| {
            let displacement = point - &gradient_center;
            Ok(gam_problem::OuterEval {
                cost: objective + displacement.dot(&displacement),
                gradient: displacement.mapv(|value| 2.0 * value),
                hessian: gam_problem::HessianValue::Dense(Array2::from_shape_fn(
                    (dimension, dimension),
                    |(row, column)| if row == column { 2.0 } else { 0.0 },
                )),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<
            fn(
                &mut (),
                &Array1<f64>,
            ) -> Result<gam_problem::EfsEval, gam_problem::EstimationError>,
        >,
    );
    problem
        .run_certified(&mut outer_objective, "custom-family certificate fixture")
        .expect("a real convex outer solve should issue the test certificate")
}

#[test]
pub(crate) fn owned_mode_outer_finalizer_rejects_certified_objective_mismatch() {
    let specs = vec![ParameterBlockSpec {
        name: "certified_fixed_rho".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let selected_theta = array![0.0];
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);
    let owned = evaluate_custom_family_joint_hyper_owned(
        &OneBlockIdentityFamily,
        &specs,
        &options,
        &selected_theta,
        &hyper_layout,
        None,
        EvalMode::ValueOnly,
    )
    .expect("the coefficient mode should evaluate before certificate binding");
    let certified_outer = certified_test_outer(selected_theta.clone(), 123.0);

    let error = fit_custom_family_fixed_log_lambdas_from_owned_mode(
        &OneBlockIdentityFamily,
        &specs,
        &options,
        owned.mode,
        &selected_theta,
        &certified_outer,
    )
    .expect_err("a different coefficient objective cannot inherit the outer certificate");

    assert!(
        error
            .to_string()
            .contains("does not belong to the certified outer optimum"),
        "unexpected error: {error}",
    );
}

#[test]
pub(crate) fn terminal_mode_binding_rejects_gradient_substitution() {
    let specs = vec![ParameterBlockSpec {
        name: "terminal_gradient_identity".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let theta = array![0.0];
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);
    let owned = evaluate_custom_family_joint_hyper_owned(
        &OneBlockIdentityFamily,
        &specs,
        &options,
        &theta,
        &hyper_layout,
        None,
        EvalMode::ValueOnly,
    )
    .expect("terminal coefficient mode fixture");
    let objective = owned.result.objective;
    let certified_outer = certified_test_outer(theta.clone(), objective);
    let substituted = CustomFamilyTerminalMode {
        theta,
        objective,
        // The certified fixture owns an exact zero terminal gradient. Keeping
        // theta/objective/mode identical while substituting only this vector
        // must still fail closed.
        gradient: array![1.0],
        mode: owned.mode,
    };

    let error = match bind_certified_custom_family_terminal_mode(substituted, &certified_outer) {
        Ok(_) => panic!("a different terminal gradient cannot inherit the outer certificate"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("gradient does not bitwise match"),
        "unexpected error: {error}",
    );
}

#[test]
pub(crate) fn labeled_terminal_mode_keeps_one_outer_rho_for_two_physical_penalties() {
    let shared = "shared_precision";
    let specs = vec![ParameterBlockSpec {
        name: "tied_terminal_rho".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![
            PenaltyMatrix::Dense(array![[1.0]]).with_precision_label(shared),
            PenaltyMatrix::Dense(array![[2.0]]).with_precision_label(shared),
        ],
        nullspace_dims: vec![0, 0],
        initial_log_lambdas: array![0.25, 0.25],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        use_remlobjective: false,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let penalty_counts = validate_blockspecs(&specs).expect("valid tied penalties");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid tied layout");
    assert_eq!(layout.initial_rho, array![0.25]);
    let theta = array![-0.375];
    let eval = outerobjectivegradienthessian_labeled(
        &OneBlockIdentityFamily,
        &specs,
        &options,
        &layout,
        &theta,
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueAndGradient,
    )
    .expect("tied labeled outer evaluation");
    let physical = split_labeled_log_lambdas(&theta, &layout).expect("physical expansion");
    assert_eq!(physical, vec![array![-0.375, -0.375]]);
    assert_eq!(
        eval.warm_start.rho, theta,
        "pullback must restore the semantic labeled coordinate on the warm cache",
    );
    let persistent = constrained_warm_start_from_inner(&theta, &eval.inner);
    assert_eq!(
        persistent.rho, theta,
        "persistent custom-family warm starts are keyed by outer/labeled rho, not physical slots",
    );

    let objective = eval.objective;
    let gradient = eval.gradient.clone();
    let mode = CustomFamilyOwnedMode {
        objective,
        rho: theta.clone(),
        inner: eval.inner,
    };
    let mut state = CustomOuterState::new(None);
    state.install_terminal_mode(&theta, objective, &gradient, mode);
    let terminal = state
        .terminal_mode
        .take()
        .expect("the non-Clone mode must move exactly once into terminal ownership");
    assert_eq!(terminal.mode.rho, theta);
    assert_eq!(terminal.theta.len(), 1);
}

#[test]
pub(crate) fn owned_joint_penalty_geometry_uses_terminal_workspace_without_family_replay() {
    #[derive(Clone)]
    struct CountingJointQuadratic {
        evaluations: Arc<AtomicUsize>,
    }

    impl CustomFamily for CountingJointQuadratic {
        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            let beta = &block_states[0].beta;
            Ok(FamilyEvaluation {
                log_likelihood: -0.5 * beta.dot(beta),
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: -beta,
                    hessian: SymmetricMatrix::Dense(Array2::eye(2)),
                }],
            })
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(Array2::eye(2)))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(Array2::zeros((2, 2))))
        }

        fn has_explicit_joint_hessian(&self) -> bool {
            true
        }
    }

    let family = CountingJointQuadratic {
        evaluations: Arc::new(AtomicUsize::new(0)),
    };
    let specs = vec![ParameterBlockSpec {
        name: "joint_terminal_geometry".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(Array2::eye(2))),
        offset: Array1::zeros(2),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(array![0.25, -0.5]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let joint_spec = gam_problem::JointPenaltySpec {
        label: Some("joint_precision".to_string()),
        matrix: Array2::eye(2),
        initial_log_lambda: 0.0,
        nullspace_dim: 0,
    };
    let layout = penalty_label_layout_with_joint(&specs, vec![0], vec![joint_spec])
        .expect("valid joint-penalty layout");
    let theta = array![std::f64::consts::LN_2];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    };
    let evaluated = outerobjectivegradienthessian_labeled(
        &family,
        &specs,
        &options,
        &layout,
        &theta,
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueAndGradient,
    )
    .expect("joint-penalty terminal evaluation");
    let workspace = evaluated
        .inner
        .joint_workspace
        .as_ref()
        .expect("terminal mode must retain returned-beta Hessian workspace");
    let evaluations_before_assembly = family.evaluations.load(Ordering::Relaxed);
    let source = exact_newton_joint_hessian_source_from_workspace(
        workspace,
        2,
        MaterializationIntent::LogdetFactorization,
        "joint-penalty terminal test Hessian",
    )
    .expect("terminal Hessian source")
    .expect("terminal Hessian source must be present");
    let hessian = materialize_joint_hessian_source(
        &source,
        2,
        "joint-penalty terminal test Hessian materialization",
    )
    .expect("terminal Hessian materialization");
    let bundle = gam_problem::JointPenaltyBundle::new(
        Arc::new(layout.joint_specs.clone()),
        layout.joint_log_lambdas(&theta),
        2,
    )
    .expect("rho-specific joint bundle");
    let assembly_options = BlockwiseFitOptions {
        joint_penalties: Some(Arc::new(bundle)),
        ..options.clone()
    };
    let per_block = split_labeled_log_lambdas(&theta, &layout).expect("empty block rho layout");
    let covariance = compute_joint_covariance_required(
        &family,
        &specs,
        &evaluated.inner.block_states,
        &per_block,
        &assembly_options,
        Some(&hessian),
    )
    .expect("joint terminal covariance")
    .expect("covariance requested");
    let geometry = compute_joint_geometry(
        &family,
        &specs,
        &evaluated.inner.block_states,
        &per_block,
        &assembly_options,
        Some(&hessian),
        evaluated.inner.terminal_working_sets.as_deref(),
    )
    .expect("joint terminal geometry");
    assert_eq!(
        family.evaluations.load(Ordering::Relaxed),
        evaluations_before_assembly,
        "terminal Hessian materialization, covariance, and geometry decision must not call family.evaluate",
    );
    let expected_covariance = Array2::eye(2) / 3.0;
    assert!(
        geometry.working.is_none(),
        "exact joint coefficient curvature has no single truthful IRLS row measure",
    );
    assert_eq!(
        geometry.penalized_hessian.as_array(),
        &(Array2::eye(2) * 3.0),
        "Exact-Newton terminal geometry must retain the joint penalized precision",
    );
    assert!(
        covariance
            .iter()
            .zip(expected_covariance.iter())
            .all(|(actual, expected)| (actual - expected).abs() <= 1.0e-12),
    );
}

#[test]
pub(crate) fn owned_mode_finalizer_preserves_prior_and_active_jeffreys_without_replay() {
    #[derive(Clone)]
    struct ActiveJeffreysQuadraticFamily {
        evaluations: Arc<AtomicUsize>,
    }

    impl CustomFamily for ActiveJeffreysQuadraticFamily {
        fn joint_jeffreys_term_required(&self) -> bool {
            true
        }

        fn evaluate(
            &self,
            block_states: &[ParameterBlockState],
        ) -> Result<FamilyEvaluation, String> {
            self.evaluations.fetch_add(1, Ordering::Relaxed);
            let beta = block_states[0].beta[0];
            Ok(FamilyEvaluation {
                log_likelihood: -0.25 * beta * beta,
                blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                    gradient: array![-0.5 * beta],
                    hessian: SymmetricMatrix::Dense(array![[0.5]]),
                }],
            })
        }

        fn exact_newton_joint_hessian(
            &self,
            _: &[ParameterBlockState],
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.5]]))
        }

        fn exact_newton_joint_hessian_directional_derivative(
            &self,
            _: &[ParameterBlockState],
            _: &Array1<f64>,
        ) -> Result<Option<Array2<f64>>, String> {
            Ok(Some(array![[0.0]]))
        }

        fn has_explicit_joint_hessian(&self) -> bool {
            true
        }
    }

    let family = ActiveJeffreysQuadraticFamily {
        evaluations: Arc::new(AtomicUsize::new(0)),
    };
    let specs = vec![ParameterBlockSpec {
        name: "active_jeffreys".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        compute_covariance: false,
        ..BlockwiseFitOptions::default()
    };
    let rho = array![0.0];
    let penalty_counts = validate_blockspecs(&specs).expect("valid test block");
    let layout = penalty_label_layout_with_joint(&specs, penalty_counts, Vec::new())
        .expect("valid labeled layout");
    let flat = outerobjectivegradienthessian_labeled(
        &family,
        &specs,
        &options,
        &layout,
        &rho,
        None,
        &gam_problem::RhoPrior::Flat,
        EvalMode::ValueOnly,
    )
    .expect("the flat active-Jeffreys profile should evaluate");
    let prior = gam_problem::RhoPrior::Normal { mean: 1.0, sd: 2.0 };
    let profiled = outerobjectivegradienthessian_labeled(
        &family,
        &specs,
        &options,
        &layout,
        &rho,
        Some(&flat.warm_start),
        &prior,
        EvalMode::ValueOnly,
    )
    .expect("the prior-bearing active-Jeffreys profile should evaluate");
    let (prior_cost, _, _) = rho_prior_cost_gradient_hessian(&prior, &rho)
        .expect("normal rho prior should evaluate");
    assert_eq!(
        profiled.objective.to_bits(),
        (flat.objective + prior_cost).to_bits(),
        "the owned objective must include the active labeled rho prior exactly once",
    );
    let beta = profiled
        .warm_start
        .block_beta
        .first()
        .expect("profiled mode must retain beta")
        .to_owned();
    let states = vec![ParameterBlockState {
        beta: beta.clone(),
        eta: specs[0].design.apply(&beta),
    }];
    let (phi, _, _) = custom_family_outer_jeffreys_hphi(
        &family,
        &states,
        &specs,
        &block_param_ranges(&specs),
    )
    .expect("Jeffreys profile probe")
    .expect("absolute curvature below one must arm the Jeffreys profile");
    assert_ne!(phi.to_bits(), 0.0_f64.to_bits());
    let evaluations_before_finalization = family.evaluations.load(Ordering::Relaxed);

    let objective = profiled.objective;
    let inner_recomposition = inner_penalized_objective(
        &profiled.inner,
        include_exact_newton_logdet_h(&family, &options),
        include_exact_newton_logdet_s(&family, &options),
        "prior-bearing terminal test mode",
    )
    .expect("owned-inner objective probe");
    assert_ne!(
        inner_recomposition.to_bits(),
        flat.objective.to_bits(),
        "active evaluator-side Jeffreys/Firth augmentation must not be recoverable by reconstructing from inner summary fields",
    );
    let terminal = CustomFamilyTerminalMode {
        theta: rho.clone(),
        objective,
        gradient: profiled.gradient,
        mode: CustomFamilyOwnedMode {
            objective,
            rho: rho.clone(),
            inner: profiled.inner,
        },
    };
    let certified_outer = certified_test_outer(rho, objective);
    let bound_mode = bind_certified_custom_family_terminal_mode(terminal, &certified_outer)
        .expect("the prior-bearing terminal identity must bind without replay");

    assert_eq!(
        bound_mode.objective.to_bits(),
        certified_outer.final_value().to_bits(),
        "the public fit objective must be the complete certified REML/LAML objective",
    );
    assert_eq!(
        certified_outer.final_value().to_bits(),
        objective.to_bits(),
        "the optimizer certificate must retain the prior-bearing outer objective",
    );
    assert_eq!(
        family.evaluations.load(Ordering::Relaxed),
        evaluations_before_finalization,
        "terminal identity binding must not call the family evaluator again",
    );
}

#[test]
pub(crate) fn failed_terminal_probe_clears_stale_owned_mode() {
    let specs = vec![ParameterBlockSpec {
        name: "failed_terminal_probe".to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]])),
        offset: array![0.0],
        penalties: vec![PenaltyMatrix::Dense(array![[1.0]])],
        nullspace_dims: vec![0],
        initial_log_lambdas: array![0.0],
        initial_beta: Some(array![0.0]),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }];
    let theta = array![0.0];
    let hyper_layout = test_design_hyper_layout(vec![vec![]]);
    let evaluated = evaluate_custom_family_joint_hyper_owned(
        &OneBlockIdentityFamily,
        &specs,
        &BlockwiseFitOptions {
            use_remlobjective: false,
            compute_covariance: false,
            ..BlockwiseFitOptions::default()
        },
        &theta,
        &hyper_layout,
        None,
        EvalMode::ValueOnly,
    )
    .expect("stale terminal mode fixture");
    let objective = evaluated.result.objective;
    let mode = CustomFamilyOwnedMode {
        objective,
        rho: theta.clone(),
        inner: evaluated.mode.inner,
    };
    let mut state = CustomOuterState::new(None);
    state.install_terminal_mode(&theta, objective, &array![0.0], mode);
    assert!(state.terminal_mode.is_some());

    // This is the transaction boundary used immediately before every
    // derivative-bearing outer probe. A subsequent Err/infeasible return does
    // not reinstall anything, so the previous successful basin cannot leak
    // into certified assembly.
    state.begin_terminal_evaluation();
    assert!(
        state.terminal_mode.is_none(),
        "a failed derivative probe must leave no stale terminal mode",
    );
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
        .mode
        .inner
        .block_states
        .iter()
        .map(|state| state.beta.iter().map(|value| value.to_bits()).collect())
        .collect();
    let evaluations_before_finalization = family.evaluations.load(Ordering::Relaxed);

    let selected_theta = Array1::zeros(1);
    let certified_outer =
        certified_test_outer(selected_theta.clone(), f64::from_bits(selected_objective_bits));

    let fit = fit_custom_family_fixed_log_lambdas_from_mode_selection(
        &family,
        &specs,
        &options,
        selection,
        &selected_theta,
        &certified_outer,
    )
    .expect("the exact selected mode should finalize without another inner solve");

    assert_eq!(
        family.evaluations.load(Ordering::Relaxed),
        evaluations_before_finalization,
        "finalization must consume the selected mode and cached Hessian without replaying the family",
    );
    assert_eq!(fit.penalized_objective.to_bits(), selected_objective_bits);
    assert_eq!(fit.outer_iterations, certified_outer.iterations());
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

#[test]
pub(crate) fn returned_mode_finalizer_rejects_different_certified_theta() {
    let family = OneStepReturnedSaddleFamily::new(0.125);
    let specs = one_step_returned_saddle_specs();
    let options = BlockwiseFitOptions {
        inner_max_cycles: 2,
        use_remlobjective: false,
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    };
    let selection = evaluate_custom_family_joint_hyper_best_mode_shared(
        &family,
        &specs,
        &options,
        &Array1::zeros(0),
        Arc::new((0..specs.len()).map(|_| Vec::new()).collect()),
        &[None],
        EvalMode::ValueOnly,
    )
    .expect("the bounded hard case should select a mode");
    let objective = selection.result.objective;
    let certified_outer = certified_test_outer(array![2.0], objective);
    let error = fit_custom_family_fixed_log_lambdas_from_mode_selection(
        &family,
        &specs,
        &options,
        selection,
        &array![1.0],
        &certified_outer,
    )
    .expect_err("a certificate at a different full theta cannot mint the fit");
    assert!(
        error.to_string().contains("full hyperparameter vector"),
        "unexpected error: {error}",
    );
}

#[test]
pub(crate) fn returned_mode_finalizer_rejects_different_certified_objective() {
    let family = OneStepReturnedSaddleFamily::new(0.125);
    let specs = one_step_returned_saddle_specs();
    let options = BlockwiseFitOptions {
        inner_max_cycles: 2,
        use_remlobjective: false,
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    };
    let selection = evaluate_custom_family_joint_hyper_best_mode_shared(
        &family,
        &specs,
        &options,
        &Array1::zeros(0),
        Arc::new((0..specs.len()).map(|_| Vec::new()).collect()),
        &[None],
        EvalMode::ValueOnly,
    )
    .expect("the bounded hard case should select a mode");
    let selected_theta = Array1::zeros(1);
    let certified_outer =
        certified_test_outer(selected_theta.clone(), selection.result.objective + 1.0);
    let error = fit_custom_family_fixed_log_lambdas_from_mode_selection(
        &family,
        &specs,
        &options,
        selection,
        &selected_theta,
        &certified_outer,
    )
    .expect_err("a different certified objective cannot mint the selected-mode fit");
    assert!(
        error.to_string().contains("does not belong to the certified outer optimum"),
        "unexpected error: {error}",
    );
}

mod inner_solver_numerics;
