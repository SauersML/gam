//! Tests for the survival marginal-slope family (relocated verbatim).

use super::*;
use crate::custom_family::{CustomFamily, ExactOuterDerivativeOrder};
use crate::matrix::{DenseDesignMatrix, SymmetricMatrix};
use approx::assert_relative_eq;
use faer::sparse::{SparseColMat, Triplet};
use ndarray::array;

/// Local scalar closeness assertion used throughout this module's exactness
/// gates. Asserts `|lhs - rhs| <= tol`, reporting both operands on failure.
fn assert_close(lhs: f64, rhs: f64, tol: f64, label: &str) {
    assert!(
        (lhs - rhs).abs() <= tol,
        "{label} mismatch: lhs={lhs:.12e}, rhs={rhs:.12e}, tol={tol:.3e}"
    );
}

/// Pin the survival cross-block W metric to the survival row Hessian
/// formula `u2_eta1 = (1-d)·w·k2(-η, w·(1-d)) + d·w` rather than the
/// Bernoulli probit proxy `φ²/(Φ·(1-Φ))`. Three rows exercise the
/// censored-only, event-only, and weighted-censored branches; the
/// expected values are recomputed in-test from
/// `signed_probit_neglog_derivatives_up_to_fourth` to bind the metric
/// to its source-of-truth derivative routine.
#[test]
fn survival_pilot_irls_row_metric_matches_u2_eta1_formula() {
    let eta = ndarray::Array1::from(vec![0.3_f64, -0.7, 1.2]);
    let weights = ndarray::Array1::from(vec![1.0_f64, 2.5, 0.5]);
    let event = ndarray::Array1::from(vec![0.0_f64, 1.0, 0.0]);
    let computed = super::survival_pilot_irls_row_metric_at_eta(&eta, &weights, &event)
        .expect("survival pilot W metric");
    assert_eq!(computed.len(), 3);
    for i in 0..3 {
        let d = event[i];
        let w = weights[i];
        let e = eta[i];
        let (_, k2, _, _) = super::signed_probit_neglog_derivatives_up_to_fourth(-e, w * (1.0 - d))
            .expect("probit k2");
        let expected = k2 + w * d;
        approx::assert_relative_eq!(computed[i], expected, max_relative = 1e-12);
    }
    // Censored row (d=0): metric must be strictly positive at finite η —
    // the Mills-ratio second derivative of -log Φ(-η) is positive on ℝ.
    assert!(
        computed[0] > 0.0,
        "censored row metric must be > 0, got {}",
        computed[0],
    );
    // Event row (d=1): the formula collapses to w·d exactly; the censored
    // branch contributes zero because the k2 weight `w·(1-d)` is zero.
    approx::assert_relative_eq!(computed[1], weights[1], max_relative = 1e-12);
}

/// Length-mismatch contract: the metric helper must reject misaligned
/// inputs instead of producing a truncated W vector that silently
/// passes the cross-block routine but breaks the W-inner product.
#[test]
fn survival_pilot_irls_row_metric_rejects_length_mismatch() {
    let eta = ndarray::Array1::from(vec![0.0_f64, 1.0]);
    let weights = ndarray::Array1::from(vec![1.0_f64]); // wrong length
    let event = ndarray::Array1::from(vec![0.0_f64, 0.0]);
    let result = super::survival_pilot_irls_row_metric_at_eta(&eta, &weights, &event);
    match result {
        Ok(_) => panic!("expected length-mismatch error, got Ok"),
        Err(msg) => assert!(
            msg.contains("length mismatch"),
            "expected 'length mismatch' in error, got: {msg}",
        ),
    }
}

#[test]
fn smgs_rawstack_reduction_rejects_required_channel_deletion() {
    assert_eq!(
        super::smgs_deleted_required_channel_reason(12, 7, 7, 12, 7, 0),
        Some("logslope"),
        "the #808 clustered-PC rawstack map must not delete the log-slope channel"
    );
    assert_eq!(
        super::smgs_deleted_required_channel_reason(12, 7, 7, 12, 6, 7),
        None,
        "partial reductions that preserve all physical channels remain valid"
    );
    assert_eq!(
        super::smgs_deleted_required_channel_reason(12, 7, 7, 0, 7, 7),
        Some("time"),
        "the baseline/time channel is also required"
    );
}

fn empty_termspec() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: vec![],
        random_effect_terms: vec![],
        smooth_terms: vec![],
    }
}

fn unit_score_covariance() -> MarginalSlopeCovariance {
    MarginalSlopeCovariance::Diagonal(array![1.0])
}

// Mirrors the production single-z convention produced by
// `combine_logslope_surface_designs` for `designs.len() == 1`, where the
// emitted ranges vector is `vec![0..ncols]`. Test fixtures use empty
// logslope designs (`n × 0`), so the single placeholder range is `0..0`.
fn empty_logslope_surface_ranges() -> Vec<std::ops::Range<usize>> {
    let placeholder = 0..0;
    vec![placeholder]
}

fn base_time_block() -> TimeBlockInput {
    TimeBlockInput {
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Array1::zeros(1),
        offset_exit: Array1::zeros(1),
        derivative_offset_exit: Array1::from_elem(
            1,
            DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
        ),
        time_monotonicity:
            crate::families::survival::location_scale::TimeBlockMonotonicity::EnforcedByRowConstraint,
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: None,
        initial_beta: Some(Array1::zeros(1)),
    }
}

#[test]
fn time_nullspace_shrinkage_adds_precision_for_uncontrolled_time_direction() {
    let mut block = TimeBlockInput {
        design_entry: DesignMatrix::from(Array2::zeros((3, 2))),
        design_exit: DesignMatrix::from(Array2::zeros((3, 2))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((3, 2))),
        penalties: vec![array![[1.0, 0.0], [0.0, 0.0]]],
        nullspace_dims: vec![1],
        initial_beta: Some(Array1::zeros(2)),
        ..base_time_block()
    };

    assert!(
        install_time_nullspace_shrinkage_penalty(&mut block)
            .expect("time nullspace shrinkage should build"),
        "expected a shrinkage penalty to be appended",
    );
    assert_eq!(block.penalties.len(), 2);
    assert_eq!(block.nullspace_dims, vec![1, 0]);
    assert!((block.penalties[1][[0, 0]]).abs() <= 1e-12);
    assert!((block.penalties[1][[1, 1]] - 1.0).abs() <= 1e-12);
}

#[test]
fn time_nullspace_shrinkage_is_noop_for_full_rank_time_penalty() {
    let mut block = TimeBlockInput {
        design_entry: DesignMatrix::from(Array2::zeros((3, 2))),
        design_exit: DesignMatrix::from(Array2::zeros((3, 2))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((3, 2))),
        penalties: vec![Array2::<f64>::eye(2)],
        nullspace_dims: vec![0],
        initial_beta: Some(Array1::zeros(2)),
        ..base_time_block()
    };

    assert!(
        !install_time_nullspace_shrinkage_penalty(&mut block)
            .expect("full-rank time penalty should be accepted"),
        "full-rank time penalties should not get another penalty",
    );
    assert_eq!(block.penalties.len(), 1);
    assert_eq!(block.nullspace_dims, vec![0]);
}

fn sparse_design(dense: &Array2<f64>) -> DesignMatrix {
    let mut triplets = Vec::<Triplet<usize, usize, f64>>::new();
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            let value = dense[[i, j]];
            if value != 0.0 {
                triplets.push(Triplet::new(i, j, value));
            }
        }
    }
    let sparse = SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
        .expect("assemble sparse design");
    DesignMatrix::Sparse(crate::matrix::SparseDesignMatrix::new(sparse))
}

/// Build an n-row closed-form survival family with empty time/marginal/
/// logslope blocks (so q0/q1/qd1 come from offsets only). No flex
/// deviations are configured, so `log_likelihood_only` takes the
/// closed-form fast path.
fn make_closed_form_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    // Pseudo-random rows uncorrelated with row parity, so an even-only
    // subsample is representative for the Horvitz-Thompson rescaling
    // check.
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_entry: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_exit: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
    );
    // qd1 must remain strictly above the derivative guard.
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
    SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        // Empty time/marginal/logslope designs: `n_rows × 0` so the
        // closed-form q geometry is driven entirely by offsets.
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(Array2::zeros((n, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((n, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    }
}

fn closed_form_block_states(
    family: &SurvivalMarginalSlopeFamily,
    g: f64,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    vec![
        // Time block: empty beta; per-row eta entries unused (designs
        // are zero-column).
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        // Marginal block: empty beta.
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        // Log-slope block: empty beta with per-row eta = g (constant
        // log-slope across rows).
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::from_elem(n, g),
        },
    ]
}

#[test]
fn survival_primary_g_fourth_cell_partials_are_zero() {
    let family = make_closed_form_test_family(1);
    let primary = flex_primary_slices(&family);
    let score_span = exact_kernel::LocalSpanCubic {
        left: -1.0,
        right: 1.0,
        c0: 0.2,
        c1: -0.1,
        c2: 0.05,
        c3: -0.02,
    };
    let link_span = exact_kernel::LocalSpanCubic {
        left: -0.5,
        right: 0.5,
        c0: 0.1,
        c1: 0.3,
        c2: -0.2,
        c3: 0.4,
    };
    let fixed = family
        .denested_cell_primary_fixed_partials(&primary, 0.2, 0.7, score_span, link_span, 0.0, 0.0)
        .expect("primary fixed partials");
    let (_, dc_daab, dc_dabb, dc_dbbb) = exact_kernel::denested_cell_third_partials(link_span);

    assert_eq!(fixed.coeff_aau[primary.g], dc_daab);
    assert_eq!(fixed.coeff_abu[primary.g], dc_dabb);
    assert_eq!(fixed.coeff_bbu[primary.g], dc_dbbb);
    assert!(dc_daab.iter().any(|value| *value != 0.0));
    assert!(dc_dabb.iter().any(|value| *value != 0.0));
    assert!(dc_dbbb.iter().any(|value| *value != 0.0));
    assert_eq!(fixed.coeff_aaau[primary.g], [0.0; 4]);
    assert_eq!(fixed.coeff_aabu[primary.g], [0.0; 4]);
    assert_eq!(fixed.coeff_abbu[primary.g], [0.0; 4]);
    assert_eq!(fixed.coeff_bbbu[primary.g], [0.0; 4]);
}

#[test]
fn survival_log_likelihood_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let baseline = family
        .log_likelihood_only(&states)
        .expect("baseline ll (no subsample)");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full_mask = family
        .log_likelihood_only_with_options(&states, &opts_full)
        .expect("ll with mask=full");

    let rel = ((with_full_mask - baseline) / baseline.abs().max(1.0)).abs();
    assert!(
        rel < 1e-12,
        "subsample(mask=full) {} differs from baseline {} by rel {}",
        with_full_mask,
        baseline,
        rel
    );
}

#[test]
fn survival_log_likelihood_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .log_likelihood_only_with_options(&states, &opts_half)
        .expect("ll with mask=even");

    // Raw even-row sum: same mask but weight_scale = 1.0.
    let mut opts_even_unscaled = BlockwiseFitOptions::default();
    opts_even_unscaled.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::with_uniform_weight(even_mask, m, 0, 1.0),
    ));
    let raw_even_sum = family
        .log_likelihood_only_with_options(&states, &opts_even_unscaled)
        .expect("raw even-row ll sum");

    let expected_scaled = (n as f64 / m as f64) * raw_even_sum;
    let rel = ((scaled - expected_scaled) / expected_scaled.abs().max(1.0)).abs();
    assert!(
        rel < 1e-12,
        "scaled {} != 2*even_sum {} (rel {})",
        scaled,
        expected_scaled,
        rel
    );

    // Horvitz-Thompson check: 2 * Σ_even ≈ full-data sum.
    let baseline = family.log_likelihood_only(&states).expect("baseline ll");
    let ht_rel = ((scaled - baseline) / baseline.abs().max(1.0)).abs();
    assert!(
        ht_rel < 0.05,
        "Horvitz-Thompson scaled {} not near baseline {} (rel {})",
        scaled,
        baseline,
        ht_rel
    );
}

#[test]
fn poly_mul_treats_empty_inputs_as_zero_polynomials() {
    assert!(poly_mul(&[], &[1.0, 2.0]).is_empty());
    assert!(poly_mul(&[1.0, 2.0], &[]).is_empty());
    assert_eq!(
        poly_mul(&[1.0, 2.0], &[3.0, 4.0]).as_slice(),
        &[3.0, 10.0, 8.0][..]
    );
}

fn dummy_blockspec(cols: usize) -> ParameterBlockSpec {
    // `validate_blockspecs` enforces unique block names so coefficient
    // labels stay unambiguous. A monotonic per-process counter keeps
    // each call's name distinct even when multiple specs are stacked
    // into one `Vec`.
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let idx = SEQ.fetch_add(1, Ordering::Relaxed);
    ParameterBlockSpec {
        name: format!("dummy_{idx}"),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::zeros((1, cols)))),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: Some(Array1::zeros(cols)),
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn dummy_penalized_blockspec(cols: usize, penalties: usize) -> ParameterBlockSpec {
    let mut spec = dummy_blockspec(cols);
    spec.penalties = (0..penalties)
        .map(|_| PenaltyMatrix::Dense(Array2::eye(cols)))
        .collect();
    spec.nullspace_dims = vec![0; penalties];
    spec.initial_log_lambdas = Array1::zeros(penalties);
    spec
}

fn test_deviation_runtime() -> DeviationRuntime {
    build_score_warp_deviation_block_from_seed(
        &array![-1.0, 0.0, 1.0],
        &DeviationBlockConfig {
            degree: 3,
            num_internal_knots: 1,
            penalty_order: 2,
            penalty_orders: vec![1, 2, 3],
            double_penalty: false,
            monotonicity_eps: 1e-4,
        },
    )
    .expect("build test deviation runtime")
    .runtime
}

fn max_abs_diff_vec(lhs: &Array1<f64>, rhs: &Array1<f64>) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs_diff_mat(lhs: &Array2<f64>, rhs: &Array2<f64>) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn assert_blockwise_matches_joint_principal_blocks(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
) {
    let eval = family
        .evaluate_blockwise_exact_newton(block_states)
        .expect("blockwise exact-newton evaluation");
    let (joint_ll, joint_gradient, joint_hessian) = family
        .evaluate_exact_newton_joint_dense(block_states)
        .expect("joint dense exact-newton evaluation");
    let slices = block_slices(family, block_states);
    let mut block_ranges = vec![
        slices.time.clone(),
        slices.marginal.clone(),
        slices.logslope.clone(),
    ];
    if let Some(range) = slices.score_warp.clone() {
        block_ranges.push(range);
    }
    if let Some(range) = slices.link_dev.clone() {
        block_ranges.push(range);
    }

    assert!((eval.log_likelihood - joint_ll).abs() <= 1e-10);
    assert_eq!(eval.blockworking_sets.len(), block_ranges.len());
    for (work, range) in eval.blockworking_sets.iter().zip(block_ranges.iter()) {
        let BlockWorkingSet::ExactNewton { gradient, hessian } = work else {
            panic!("expected exact-newton block working set");
        };
        let expected_gradient = joint_gradient.slice(s![range.clone()]).to_owned();
        let expected_hessian = joint_hessian
            .slice(s![range.clone(), range.clone()])
            .to_owned();
        assert!(
            max_abs_diff_vec(gradient, &expected_gradient) <= 1e-10,
            "gradient block mismatch"
        );
        assert!(
            max_abs_diff_mat(&hessian.to_dense(), &expected_hessian) <= 1e-10,
            "hessian block mismatch"
        );
    }
}

fn test_family(
    score_warp: Option<DeviationRuntime>,
    link_dev: Option<DeviationRuntime>,
) -> SurvivalMarginalSlopeFamily {
    SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::from_elem(1, 1e-6)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 2))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 3))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp,
        link_dev,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    }
}

#[test]
fn validate_spec_rejects_coordinate_cone_without_guard_offset() {
    let spec = SurvivalMarginalSlopeTermSpec {
        age_entry: array![0.0, 0.0],
        age_exit: array![1.0, 1.0],
        event_target: array![0.0, 1.0],
        weights: array![1.0, 1.0],
        z: array![-1.0, 1.0].insert_axis(Axis(1)),
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec: empty_termspec(),
        marginal_offset: Array1::zeros(2),
        frailty: FrailtySpec::None,
        derivative_guard: 1e-4,
        time_block: TimeBlockInput {
            design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
            offset_entry: Array1::zeros(2),
            offset_exit: Array1::zeros(2),
            derivative_offset_exit: Array1::zeros(2),
            time_monotonicity: crate::families::survival::location_scale::TimeBlockMonotonicity::EnforcedByCoordinateCone,
            ..base_time_block()
        },
        timewiggle_block: None,
        logslopespec: empty_termspec(),
        logslopespecs: None,
        logslope_offset: Array1::zeros(2),
        score_warp: None,
        link_dev: None,
        score_influence_jacobian: None,
        latent_z_policy: LatentZPolicy::default(),
    };

    let err = validate_spec(&spec).expect_err("coordinate cone without guard offset should fail");
    assert!(
        err.contains("coordinate-cone time block requires derivative offset >= guard"),
        "unexpected error: {err}"
    );
}

#[test]
fn validate_spec_rejects_learnable_gaussian_shift_sigma() {
    let spec = SurvivalMarginalSlopeTermSpec {
        age_entry: array![0.0, 0.0],
        age_exit: array![1.0, 1.0],
        event_target: array![0.0, 1.0],
        weights: array![1.0, 1.0],
        z: array![-1.0, 1.0].insert_axis(Axis(1)),
        base_link: InverseLink::Standard(StandardLink::Probit),
        marginalspec: empty_termspec(),
        marginal_offset: Array1::zeros(2),
        frailty: FrailtySpec::GaussianShift { sigma_fixed: None },
        derivative_guard: 1e-4,
        time_block: base_time_block(),
        timewiggle_block: None,
        logslopespec: empty_termspec(),
        logslopespecs: None,
        logslope_offset: Array1::zeros(2),
        score_warp: None,
        link_dev: None,
        score_influence_jacobian: None,
        latent_z_policy: LatentZPolicy::default(),
    };

    let err = validate_spec(&spec).expect_err("learnable GaussianShift sigma should be rejected");
    assert!(err.contains("learnable GaussianShift sigma is not implemented"));
}

#[test]
fn block_slices_handles_link_only_survival_flex_layout() {
    let link_runtime = test_deviation_runtime();
    let family = test_family(None, Some(link_runtime.clone()));
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(2),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(3),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];

    let slices = block_slices(&family, &block_states);
    assert!(slices.score_warp.is_none());
    assert_eq!(
        slices.link_dev.as_ref().expect("link-only slice").len(),
        link_runtime.basis_dim()
    );
    assert_eq!(slices.total, 1 + 2 + 3 + link_runtime.basis_dim());
}

// ── Single-source block-layout parity (#428) ─────────────────────────
//
// `HessBlock` + `BlockHessianAccumulator::block_view` are the one place
// the 5×5 block layout and its transpose relationships live. Every
// assembler (`to_dense`, `diagonal`, operator `mul_vec`, operator
// `bilinear`) is driven by them. These tests pin that machinery against a
// *separately hand-written* fifteen-block scatter so a layout or
// transpose regression in the shared helpers cannot hide behind itself.

/// Build contiguous block slices for the given per-block widths. A flex
/// block with width 0 is absent (`None`) and consumes no coordinates,
/// exactly mirroring how `block_slices` lays out optional deviation blocks.
fn parity_make_slices(
    pt: usize,
    pm: usize,
    pg: usize,
    ph: usize,
    pw: usize,
    pi: usize,
) -> BlockSlices {
    let mut cursor = 0usize;
    let mut take = |n: usize| {
        let r = cursor..cursor + n;
        cursor += n;
        r
    };
    let time = take(pt);
    let marginal = take(pm);
    let logslope = take(pg);
    let score_warp = (ph > 0).then(|| take(ph));
    let link_dev = (pw > 0).then(|| take(pw));
    let influence = (pi > 0).then(|| take(pi));
    let total = cursor;
    BlockSlices {
        time,
        marginal,
        logslope,
        score_warp,
        link_dev,
        influence,
        total,
    }
}

/// A genuinely-symmetric block (diagonal blocks of a Hessian are symmetric
/// by construction), with a per-block `tag` so cross-block contamination
/// is detectable.
fn parity_sym(n: usize, tag: f64) -> Array2<f64> {
    Array2::from_shape_fn((n, n), |(i, j)| {
        tag + (i as f64 + j as f64) * 0.5 + (i as f64) * (j as f64) * 0.0625
    })
}

/// A general (non-symmetric) off-diagonal block, distinct per `tag` and
/// deliberately asymmetric in (i, j) so a missing/extra transpose shows up.
fn parity_gen(rows: usize, cols: usize, tag: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        tag + (i as f64) * 0.5 + (j as f64) * 0.125
    })
}

fn parity_filled_accumulator(
    pt: usize,
    pm: usize,
    pg: usize,
    ph: usize,
    pw: usize,
    pi: usize,
) -> BlockHessianAccumulator {
    BlockHessianAccumulator {
        h_tt: parity_sym(pt, 1.0),
        h_mm: parity_sym(pm, 2.0),
        h_gg: parity_sym(pg, 3.0),
        h_hh: parity_sym(ph, 4.0),
        h_ww: parity_sym(pw, 5.0),
        h_ii: parity_sym(pi, 6.0),
        h_tm: parity_gen(pt, pm, 10.0),
        h_tg: parity_gen(pt, pg, 11.0),
        h_th: parity_gen(pt, ph, 12.0),
        h_tw: parity_gen(pt, pw, 13.0),
        h_ti: parity_gen(pt, pi, 20.0),
        h_mg: parity_gen(pm, pg, 14.0),
        h_mh: parity_gen(pm, ph, 15.0),
        h_mw: parity_gen(pm, pw, 16.0),
        h_mi: parity_gen(pm, pi, 21.0),
        h_gh: parity_gen(pg, ph, 17.0),
        h_gw: parity_gen(pg, pw, 18.0),
        h_gi: parity_gen(pg, pi, 22.0),
        h_hw: parity_gen(ph, pw, 19.0),
        h_hi: parity_gen(ph, pi, 23.0),
        h_wi: parity_gen(pw, pi, 24.0),
    }
}

/// Independent dense oracle: scatter the fifteen stored blocks by hand,
/// placing each off-diagonal block in its upper position and its explicit
/// transpose in the mirror position. Deliberately avoids `block_view`,
/// `range_of`, and `for_each_offdiagonal_pair`.
fn parity_reference_dense(acc: &BlockHessianAccumulator, sl: &BlockSlices) -> Array2<f64> {
    let mut out = Array2::zeros((sl.total, sl.total));
    out.slice_mut(s![sl.time.clone(), sl.time.clone()])
        .assign(&acc.h_tt);
    out.slice_mut(s![sl.marginal.clone(), sl.marginal.clone()])
        .assign(&acc.h_mm);
    out.slice_mut(s![sl.logslope.clone(), sl.logslope.clone()])
        .assign(&acc.h_gg);
    if let Some(h) = &sl.score_warp {
        out.slice_mut(s![h.clone(), h.clone()]).assign(&acc.h_hh);
    }
    if let Some(w) = &sl.link_dev {
        out.slice_mut(s![w.clone(), w.clone()]).assign(&acc.h_ww);
    }
    if let Some(i) = &sl.influence {
        out.slice_mut(s![i.clone(), i.clone()]).assign(&acc.h_ii);
    }
    let mut place =
        |r: std::ops::Range<usize>, c: std::ops::Range<usize>, m: ArrayView2<'_, f64>| {
            out.slice_mut(s![r.clone(), c.clone()]).assign(&m);
            out.slice_mut(s![c, r]).assign(&m.t());
        };
    place(sl.time.clone(), sl.marginal.clone(), acc.h_tm.view());
    place(sl.time.clone(), sl.logslope.clone(), acc.h_tg.view());
    place(sl.marginal.clone(), sl.logslope.clone(), acc.h_mg.view());
    if let Some(h) = &sl.score_warp {
        place(sl.time.clone(), h.clone(), acc.h_th.view());
        place(sl.marginal.clone(), h.clone(), acc.h_mh.view());
        place(sl.logslope.clone(), h.clone(), acc.h_gh.view());
    }
    if let Some(w) = &sl.link_dev {
        place(sl.time.clone(), w.clone(), acc.h_tw.view());
        place(sl.marginal.clone(), w.clone(), acc.h_mw.view());
        place(sl.logslope.clone(), w.clone(), acc.h_gw.view());
    }
    if let (Some(h), Some(w)) = (&sl.score_warp, &sl.link_dev) {
        place(h.clone(), w.clone(), acc.h_hw.view());
    }
    if let Some(i) = &sl.influence {
        place(sl.time.clone(), i.clone(), acc.h_ti.view());
        place(sl.marginal.clone(), i.clone(), acc.h_mi.view());
        place(sl.logslope.clone(), i.clone(), acc.h_gi.view());
        if let Some(h) = &sl.score_warp {
            place(h.clone(), i.clone(), acc.h_hi.view());
        }
        if let Some(w) = &sl.link_dev {
            place(w.clone(), i.clone(), acc.h_wi.view());
        }
    }
    out
}

const PARITY_LAYOUTS: [(usize, usize, usize, usize, usize, usize); 7] = [
    (2, 3, 2, 4, 3, 0), // full flex, no absorber
    (2, 3, 2, 0, 0, 0), // rigid: no flex blocks
    (2, 3, 2, 4, 0, 0), // score-warp only
    (2, 3, 2, 0, 3, 0), // link-deviation only
    (2, 3, 2, 0, 0, 5), // absorber only (no flex)
    (2, 3, 2, 4, 3, 5), // full flex + absorber (#461)
    (2, 3, 2, 4, 0, 5), // score-warp + absorber
];

#[test]
fn block_to_dense_matches_hand_scatter_bit_exact() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let got = acc.to_dense(&sl);
        let want = parity_reference_dense(&acc, &sl);
        assert_eq!(
            got,
            want,
            "to_dense diverged from hand scatter for layout {:?}",
            (pt, pm, pg, ph, pw)
        );
    }
}

#[test]
fn block_diagonal_matches_dense_diagonal_bit_exact() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let got = acc.diagonal(&sl);
        let want = parity_reference_dense(&acc, &sl).diag().to_owned();
        assert_eq!(
            got,
            want,
            "diagonal diverged from dense diagonal for layout {:?}",
            (pt, pm, pg, ph, pw)
        );
    }
}

#[test]
fn block_operator_matvec_matches_dense_gemv() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let dense = parity_reference_dense(&acc, &sl);
        let v = Array1::from_shape_fn(sl.total, |i| (i as f64 * 0.37).sin());
        let op = acc.into_operator(sl.clone());
        let got = op.mul_vec(&v);
        let want = dense.dot(&v);
        assert_relative_eq!(
            got.as_slice().unwrap(),
            want.as_slice().unwrap(),
            max_relative = 1e-12,
            epsilon = 1e-12
        );
    }
}

#[test]
fn block_operator_bilinear_matches_dense_quadratic_form() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let dense = parity_reference_dense(&acc, &sl);
        let v = Array1::from_shape_fn(sl.total, |i| (i as f64 * 0.37).sin());
        let u = Array1::from_shape_fn(sl.total, |i| (i as f64 * 0.53).cos());
        let want = v.dot(&dense.dot(&u));
        let op = acc.into_operator(sl.clone());
        let got = op.bilinear(&v, &u);
        assert_relative_eq!(got, want, max_relative = 1e-12, epsilon = 1e-12);
    }
}

#[test]
fn block_operator_dense_matches_accumulator_dense_bit_exact() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        let direct = acc.to_dense(&sl);
        let op = acc.into_operator(sl.clone());
        let via_op = op.to_dense();
        assert_eq!(
            direct,
            via_op,
            "operator to_dense diverged from accumulator to_dense for layout {:?}",
            (pt, pm, pg, ph, pw)
        );
    }
}

#[test]
fn block_view_is_transpose_symmetric_across_present_pairs() {
    for (pt, pm, pg, ph, pw, pi) in PARITY_LAYOUTS {
        let sl = parity_make_slices(pt, pm, pg, ph, pw, pi);
        let acc = parity_filled_accumulator(pt, pm, pg, ph, pw, pi);
        for a in HessBlock::ALL {
            if sl.range_of(a).is_none() {
                continue;
            }
            for b in HessBlock::ALL {
                if sl.range_of(b).is_none() {
                    continue;
                }
                let ab = acc.block_view(a, b).to_owned();
                let ba_t = acc.block_view(b, a).t().to_owned();
                assert_eq!(
                    ab, ba_t,
                    "block_view({a:?},{b:?}) != block_view({b:?},{a:?})^T"
                );
            }
        }
    }
}

#[test]
fn exact_flex_row_matches_rigid_closed_form_without_deviations() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.7]),
        z: Arc::new(array![0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![0.2]),
        offset_exit: Arc::new(array![0.4]),
        derivative_offset_exit: Arc::new(array![0.8]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.6],
        },
    ];
    let q_geom = family
        .row_dynamic_q_geometry(0, &block_states)
        .expect("row geometry");
    let primary = flex_primary_slices(&family);
    let (nll_exact, grad_exact, hess_exact) = family
        .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
        .expect("exact flex row");
    let (nll_rigid, grad_rigid, hess_rigid) = row_primary_closed_form(
        q_geom.q0,
        q_geom.q1,
        q_geom.qd1,
        block_states[2].eta[0],
        family.z[[0, 0]],
        family.weights[0],
        family.event[0],
        family.derivative_guard,
        family.probit_frailty_scale(),
    )
    .expect("rigid row");

    assert!((nll_exact - nll_rigid).abs() < 1e-10);
    for idx in 0..N_PRIMARY {
        assert!((grad_exact[idx] - grad_rigid[idx]).abs() < 1e-8);
    }
    for i in 0..N_PRIMARY {
        for j in 0..N_PRIMARY {
            assert!((hess_exact[[i, j]] - hess_rigid[i][j]).abs() < 1e-7);
        }
    }
}

#[test]
fn row_primary_closed_form_rejects_negative_infinite_signed_margin() {
    let err = row_primary_closed_form(f64::INFINITY, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1e-6, 1.0)
        .expect_err("exact closed-form row should reject -inf signed margins");
    assert!(err.contains("non-finite signed margin"));
}

/// Mechanism-of-ρ=2 proof for the inner-PIRLS pathology on large-scale
/// saturated probit fits.
///
/// `add_pullback_primary_hessian` (line 9753) sums `h[0,0] + h[1,1]`
/// over rows into the marginal-block joint Hessian (β_marg enters both
/// `q0` and `q1` with the SAME `marginal_design` row, so the Jacobian
/// pullback adds the q0 and q1 second derivatives at the same slot).
///
/// Mathematical facts pinned here:
///   - Censored rows: `h[0,0] + h[1,1] = 0` to ULP at every η ≥ 0.
///     The entry term `+w·log Φ(−η₀)` (concave, curvature −w·c²) and
///     the exit term `−w·log Φ(−η₁)` (convex, curvature +w·c²) have
///     equal-and-opposite second derivatives when q₀ = q₁.
///   - Event rows: residual `+w·c²·(1/η² + O(1/η⁴))` from the Mills
///     asymptotic. The event-density term `+w·η₁²/2` contributes
///     exactly `+w·c²`; the entry-survival term contributes
///     `−w·c²·(1 − 1/η²)`. Sum = `+w·c²/η²`.
///
/// At large-scale saturation (η ~ 988), the marginal-block joint Hessian
/// collapses to `O(w/η²) = O(1e-6)` per event row from the likelihood
/// side; censored contributions are 0 to ULP. The Newton step in that
/// block is then dominated by the smoothing penalty `S_marg`. When
/// the saturating direction lies in the null space of `S_marg`
/// (typical for the duchon-smooth's polynomial null space), the
/// effective curvature drops to the f64 ridge floor, the inner
/// Newton step is set by ridge alone, and `actual = rhs·δ` while
/// `predicted = ½·rhs·δ` — yielding ρ ≡ 2 to floating-point precision
/// as observed.
#[test]
fn marginal_block_hessian_cancels_in_saturated_regime() {
    let probit_scale = 1.0_f64;
    let w = 1.0_f64;
    let derivative_guard = 1e-6;
    let qd1 = 1.0_f64;
    let g = 0.0_f64;
    let z = 0.0_f64;

    // Censored rows, q0 = q1 = η, at a wide range of saturations:
    // cancellation must be ULP-exact for every η.
    for &eta in &[0.5_f64, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0, 500.0, 988.0] {
        let (_nll, _grad, hess) =
            row_primary_closed_form(eta, eta, qd1, g, z, w, 0.0, derivative_guard, probit_scale)
                .expect("rigid censored row");
        let sum = hess[0][0] + hess[1][1];
        assert!(
            sum.abs() <= 1e-12 * (hess[0][0].abs() + hess[1][1].abs()).max(1.0),
            "censored cancellation broke at η={eta}: h[0,0]={:.3e} h[1,1]={:.3e} sum={:.3e}",
            hess[0][0],
            hess[1][1],
            sum,
        );
    }

    // Event rows, q0 = q1 = η, deep saturation: residual scales as
    // 1/η² by Mills asymptotic M(−η) = η + 1/η + O(1/η³).
    for &eta in &[40.0_f64, 100.0, 500.0, 988.0] {
        let (_nll, _grad, hess) =
            row_primary_closed_form(eta, eta, qd1, g, z, w, 1.0, derivative_guard, probit_scale)
                .expect("rigid event row");
        let sum = hess[0][0] + hess[1][1];
        let bound = 2.0 / (eta * eta);
        assert!(
            sum > 0.0 && sum <= bound,
            "event cancellation residual at η={eta}: sum={:.3e} expected (0, {:.3e}]",
            sum,
            bound,
        );
    }

    // Cross-check at η = 988 (the user's large-scale saturation):
    // both kinds of rows hit the predicted floor exactly.
    let (_, _, ev) =
        row_primary_closed_form(988.0, 988.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1e-6, 1.0).unwrap();
    let (_, _, ce) =
        row_primary_closed_form(988.0, 988.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1e-6, 1.0).unwrap();
    let ev_sum = ev[0][0] + ev[1][1];
    let ce_sum = ce[0][0] + ce[1][1];
    assert!(
        ev_sum > 0.0 && ev_sum < 2.0e-6,
        "event saturated h[0,0]+h[1,1] = {ev_sum:.3e}, expected ~1/988² ≈ 1e-6",
    );
    assert_eq!(
        ce_sum, 0.0,
        "censored saturated h[0,0]+h[1,1] must be EXACTLY 0, got {ce_sum:.3e}",
    );
}

#[test]
fn row_primary_closed_form_rejects_nan_signed_margin() {
    let err = row_primary_closed_form(f64::NAN, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1e-6, 1.0)
        .expect_err("exact closed-form row should reject NaN signed margins");
    assert!(err.contains("non-finite signed margin"));
}

#[test]
fn rigid_row_kernel_propagates_invalid_nonfinite_signed_margin_errors() {
    let mut family = test_family(None, None);
    family.offset_entry = Arc::new(array![f64::INFINITY]);
    family.offset_exit = Arc::new(array![0.0]);
    family.derivative_offset_exit = Arc::new(array![1.0]);
    family.event = Arc::new(array![1.0]);

    let kernel = SurvivalMarginalSlopeRowKernel::new(
        family,
        vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(3),
                eta: Array1::zeros(1),
            },
        ],
    );

    let err =
        <SurvivalMarginalSlopeRowKernel as crate::families::row_kernel::RowKernel<4>>::row_kernel(
            &kernel, 0,
        )
        .expect_err("row kernel should propagate exact probit boundary failures");
    assert!(err.contains("non-finite signed margin"));
}

#[test]
fn rigid_row_kernel_propagates_nan_signed_margin_errors() {
    let mut family = test_family(None, None);
    family.offset_entry = Arc::new(array![f64::NAN]);
    family.offset_exit = Arc::new(array![0.0]);
    family.derivative_offset_exit = Arc::new(array![1.0]);
    family.event = Arc::new(array![1.0]);

    let kernel = SurvivalMarginalSlopeRowKernel::new(
        family,
        vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(3),
                eta: Array1::zeros(1),
            },
        ],
    );

    let err =
        <SurvivalMarginalSlopeRowKernel as crate::families::row_kernel::RowKernel<4>>::row_kernel(
            &kernel, 0,
        )
        .expect_err("row kernel should propagate NaN probit boundary failures");
    assert!(err.contains("non-finite signed margin"));
}

/// The single-expression Taylor-jet tower (#932) of the rigid K=1
/// survival marginal-slope row NLL, written ONCE over `Tower4<4>`
/// primaries `(q0, q1, qd1, g)`. It reuses the family's OWN hand-certified
/// `[f64; 5]` special-function derivative stacks (`unary_derivatives_sqrt`
/// / `_neglog_phi` / `_log_normal_pdf` / `_log`) through
/// `Tower4::compose_unary`, so no probit/log primitive is re-derived here:
/// the tower mechanizes only the Leibniz / Faà di Bruno composition that
/// `row_primary_closed_form` previously coded by hand (where #736's
/// cross-block sign flip lived). The value channel of the returned tower
/// is the row NLL expression, so every derivative channel is exact by
/// construction.
struct SurvivalMarginalSlopeRigidNllProgram {
    primaries: Vec<[f64; 4]>,
    z: Vec<f64>,
    w: Vec<f64>,
    d: Vec<f64>,
    probit_scale: f64,
}

impl crate::families::jet_tower::RowNllProgram<4> for SurvivalMarginalSlopeRigidNllProgram {
    fn n_rows(&self) -> usize {
        self.primaries.len()
    }

    fn primaries(&self, row: usize) -> Result<[f64; 4], String> {
        self.primaries
            .get(row)
            .copied()
            .ok_or_else(|| format!("rigid nll program: row {row} out of range"))
    }

    fn row_nll(
        &self,
        row: usize,
        p: &[crate::families::jet_tower::Tower4<4>; 4],
    ) -> Result<crate::families::jet_tower::Tower4<4>, String> {
        use crate::families::jet_tower::Tower4;
        let z = *self
            .z
            .get(row)
            .ok_or_else(|| format!("rigid nll program: z row {row} out of range"))?;
        let w = self.w[row];
        let d = self.d[row];
        let s_f = self.probit_scale;
        let q0 = p[0];
        let q1 = p[1];
        let qd1 = p[2];
        let g = p[3];

        // c(g) = sqrt(1 + (s_f g)^2)  — K=1 covariance_ones = 1, exactly the
        // shared MultiDirJet `one_plus_b2 -> sqrt` composition.
        let observed_g = g * s_f;
        let one_plus_b2 = observed_g * observed_g + 1.0;
        let c = one_plus_b2.compose_unary(unary_derivatives_sqrt(one_plus_b2.v));

        let eta0 = q0 * c + observed_g * z;
        let eta1 = q1 * c + observed_g * z;
        let ad1 = qd1 * c;

        // Entry survival: +w logΦ(-η0) = -1 * (-w logΦ(-η0)).
        let neg_eta0 = -eta0;
        let entry = neg_eta0
            .compose_unary(unary_derivatives_neglog_phi(neg_eta0.v, w))
            .scale(-1.0);
        // Exit survival: (1-d) * (-w logΦ(-η1)) carried with weight w(1-d).
        let neg_eta1 = -eta1;
        let exit = neg_eta1.compose_unary(unary_derivatives_neglog_phi(neg_eta1.v, w * (1.0 - d)));
        // Event density: -w d logφ(η1).
        let event_density = if d > 0.0 {
            eta1.compose_unary(unary_derivatives_log_normal_pdf(eta1.v))
                .scale(-w * d)
        } else {
            Tower4::<4>::zero()
        };
        // Time derivative: -w d log(ad1).
        let time_deriv = if d > 0.0 {
            ad1.compose_unary(unary_derivatives_log(ad1.v))
                .scale(-w * d)
        } else {
            Tower4::<4>::zero()
        };

        Ok(exit + entry + event_density + time_deriv)
    }
}

/// Build a rigid K=1 survival marginal-slope family whose entry/exit/
/// derivative/marginal/logslope designs are all nontrivial dense blocks,
/// so every one of the four primaries `(q0, q1, qd1, g)` is exercised
/// when the kernel reads its designs. `n` rows, `event` per row.
fn oracle_rigid_family(
    n: usize,
    z: &[f64],
    weights: &[f64],
    event: &[f64],
    gaussian_frailty_sd: Option<f64>,
) -> SurvivalMarginalSlopeFamily {
    let z_col = Array2::from_shape_fn((n, 1), |(r, _)| z[r]);
    // Distinct entry/exit/derivative rows so q0 != q1 != qd1.
    let design_entry = Array2::from_shape_fn((n, 1), |(r, _)| {
        0.4 + 0.13 * (r as f64) - 0.05 * (r as f64).cos()
    });
    let design_exit = Array2::from_shape_fn((n, 1), |(r, _)| {
        0.9 + 0.07 * (r as f64) + 0.04 * (r as f64).sin()
    });
    // Strictly positive derivative-exit design so qd1 > 0 (monotone).
    let design_deriv = Array2::from_shape_fn((n, 1), |(r, _)| 1.2 + 0.21 * (r as f64).abs().sqrt());
    SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(Array1::from(event.to_vec())),
        weights: Arc::new(Array1::from(weights.to_vec())),
        z: Arc::new(z_col),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd,
        derivative_guard: 1e-8,
        design_entry: DesignMatrix::from(design_entry),
        design_exit: DesignMatrix::from(design_exit),
        design_derivative_exit: DesignMatrix::from(design_deriv),
        offset_entry: Arc::new(Array1::from_shape_fn(n, |r| 0.05 * (r as f64) - 0.2)),
        offset_exit: Arc::new(Array1::from_shape_fn(n, |r| 0.15 - 0.03 * (r as f64))),
        derivative_offset_exit: Arc::new(Array1::from_elem(n, 0.0)),
        marginal_design: DesignMatrix::from(Array2::zeros((n, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((n, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    }
}

/// #932 universal oracle on the ONLY production `RowKernel` impl.
///
/// Audits every channel the production `SurvivalMarginalSlopeRowKernel`
/// emits — value / gradient / Hessian / `row_third_contracted(dir)` /
/// `row_fourth_contracted(u, v)` — against the single-expression
/// `RowNllProgram<4>`-derived tower truth, over several fixture rows
/// (mixed event/censored, with and without Gaussian frailty so the probit
/// scale ≠ 1) and several random direction vectors. The cross blocks that
/// #736's sign flip corrupted are contracted explicitly. Agreement here is
/// the proof the production kernel is correct; the planted-flip test in
/// `jet_tower` proves the same harness is loud on disagreement.
#[test]
fn rigid_row_kernel_agrees_with_jet_tower_program_all_channels() {
    use crate::families::jet_tower::{KernelChannels, evaluate_program, verify_kernel_channels};
    use crate::families::row_kernel::RowKernel;

    let n = 7;
    let z = [0.4, -1.1, 0.0, 0.7, -0.3, 1.6, -1.4];
    let weights = [1.0, 0.8, 1.3, 0.9, 1.1, 0.7, 1.4];
    // Mix exact events and right-censored rows; the last two rows push
    // eta into opposite normal-tail regimes while retaining finite truth.
    let event = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    // logslope eta (g) per row, fed through block 2.
    let g_eta = array![0.2, -0.5, 0.35, -0.15, 0.6, 0.45, -0.55];
    // marginal eta (block 1) — additive index shared by η0 and η1.
    let marginal_eta = array![0.1, -0.2, 0.05, 0.12, -0.08, 6.2, -7.4];

    // Deterministic pseudo-random direction vectors (no RNG dependency).
    let dirs: [[f64; 4]; 3] = [
        [0.7, -1.3, 0.5, 0.9],
        [-0.4, 0.6, -1.1, 0.3],
        [1.2, 0.2, -0.7, -0.5],
    ];

    for frailty in [None, Some(0.6_f64)] {
        let family = oracle_rigid_family(n, &z, &weights, &event, frailty);
        let probit_scale = family.probit_frailty_scale();
        let beta_time = array![0.85]; // single time coefficient
        let block_states = vec![
            ParameterBlockState {
                beta: beta_time.clone(),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: marginal_eta.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: g_eta.clone(),
            },
        ];

        let kernel = SurvivalMarginalSlopeRowKernel::new(family.clone(), block_states.clone());

        // Build the tower program's primaries exactly as `row_kernel` reads
        // them (designs · beta_time + offsets + shared marginal/logslope eta).
        let mut primaries = Vec::with_capacity(n);
        for row in 0..n {
            let q0 = family.design_entry.dot_row(row, &beta_time)
                + family.offset_entry[row]
                + marginal_eta[row];
            let q1 = family.design_exit.dot_row(row, &beta_time)
                + family.offset_exit[row]
                + marginal_eta[row];
            let qd1 = family.design_derivative_exit.dot_row(row, &beta_time)
                + family.derivative_offset_exit[row];
            let g = g_eta[row];
            primaries.push([q0, q1, qd1, g]);
        }

        let program = SurvivalMarginalSlopeRigidNllProgram {
            primaries,
            z: z.to_vec(),
            w: weights.to_vec(),
            d: event.to_vec(),
            probit_scale,
        };

        for row in 0..n {
            let tower = evaluate_program(&program, row).expect("tower evaluation");

            let (value, gradient, hessian) =
                RowKernel::row_kernel(&kernel, row).expect("production kernel value/grad/hess");

            let third: Vec<([f64; 4], [[f64; 4]; 4])> = dirs
                .iter()
                .map(|dir| {
                    let claim = RowKernel::row_third_contracted(&kernel, row, dir)
                        .expect("production kernel third");
                    (*dir, claim)
                })
                .collect();

            let fourth: Vec<([f64; 4], [f64; 4], [[f64; 4]; 4])> = dirs
                .iter()
                .enumerate()
                .map(|(i, u)| {
                    let v = dirs[(i + 1) % dirs.len()];
                    let claim = RowKernel::row_fourth_contracted(&kernel, row, u, &v)
                        .expect("production kernel fourth");
                    (*u, v, claim)
                })
                .collect();

            let claims = KernelChannels {
                value,
                gradient,
                hessian,
                third,
                fourth,
            };

            verify_kernel_channels(&tower, &claims, 1e-9).unwrap_or_else(|e| {
                    panic!(
                        "frailty {frailty:?} row {row}: production RowKernel disagrees with #932 jet-tower truth: {e}"
                    )
                });
        }
    }
}

#[test]
fn exact_flex_row_value_matches_rigid_with_zero_score_and_link_coefficients() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![0.9]),
        z: Arc::new(array![-0.35].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![-0.1]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.6]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.45],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let q_geom = family
        .row_dynamic_q_geometry(0, &block_states)
        .expect("row geometry");
    let primary = flex_primary_slices(&family);
    let (nll_exact, grad_exact, hess_exact) = family
        .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
        .expect("exact flex row");
    let (nll_rigid, grad_rigid, hess_rigid) = row_primary_closed_form(
        q_geom.q0,
        q_geom.q1,
        q_geom.qd1,
        block_states[2].eta[0],
        family.z[[0, 0]],
        family.weights[0],
        family.event[0],
        family.derivative_guard,
        family.probit_frailty_scale(),
    )
    .expect("rigid row");

    assert!((nll_exact - nll_rigid).abs() < 1e-10);
    assert!((grad_exact[primary.q0] - grad_rigid[0]).abs() < 1e-8);
    assert!((grad_exact[primary.q1] - grad_rigid[1]).abs() < 1e-8);
    assert!((grad_exact[primary.qd1] - grad_rigid[2]).abs() < 1e-8);
    assert!((grad_exact[primary.g] - grad_rigid[3]).abs() < 1e-8);
    assert!((hess_exact[[primary.q0, primary.q0]] - hess_rigid[0][0]).abs() < 1e-7);
    assert!((hess_exact[[primary.q0, primary.g]] - hess_rigid[0][3]).abs() < 1e-7);
    assert!((hess_exact[[primary.q1, primary.q1]] - hess_rigid[1][1]).abs() < 1e-7);
    assert!((hess_exact[[primary.q1, primary.g]] - hess_rigid[1][3]).abs() < 1e-7);
    assert!((hess_exact[[primary.qd1, primary.qd1]] - hess_rigid[2][2]).abs() < 1e-7);
    assert!((hess_exact[[primary.g, primary.g]] - hess_rigid[3][3]).abs() < 1e-7);
}

/// gam#932/#979: INDEPENDENT witness for the survival marginal-slope FLEX
/// higher-order tower (`row_flex_primary_{third,fourth}_contracted_exact`).
///
/// The production CPU↔GPU parity tests
/// (`block10_cpu_oracle_{third,fourth}_contraction_matches_family_shared_fixtures`)
/// feed BOTH sides the SAME `flex_primary_timepoint_jets_for_test` inputs the
/// family produces, so a shared-input bug in the flex calculus passes both — they
/// are not an independent witness. The rigid K=1 path is independently guarded by
/// `SurvivalMarginalSlopeRigidNllProgram` (a single-expression `Tower4` algebra
/// re-derivation, no shared jet code); the flex path was not.
///
/// This closes that gap on the `(q0, q1, qd1, g)` primary block: at ZERO
/// deviation coefficients the flex de-nested calibration / cell-moment / intercept
/// machinery is fully exercised (it does NOT short-circuit to the rigid kernel —
/// it runs the partition with zero-coefficient cubics), and its third/fourth
/// directional contractions over `(q0, q1, qd1, g)` MUST equal the independent
/// rigid `Tower4` truth. A planted cross-block sign flip must leave the band,
/// proving resolving power (the #736 genus the shared-input parity cannot catch).
#[test]
fn flex_contracted_tower_matches_independent_rigid_tower_and_catches_sign_flip() {
    use crate::families::jet_tower::{derived_fourth_contracted, derived_third_contracted};

    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    // Several fixture rows: events + censored, distinct q-geometry, frailty off
    // (probit scale = 1; the rigid program and the flex path share the closed
    // form there) and a non-zero logslope g so the c(g) coupling is live.
    struct Fix {
        event: f64,
        weight: f64,
        z: f64,
        q0: f64,
        q1: f64,
        qd1: f64,
        g: f64,
    }
    let fixtures = [
        Fix {
            event: 1.0,
            weight: 0.75,
            z: -0.2,
            q0: -0.4,
            q1: 0.6,
            qd1: 0.85,
            g: 0.32,
        },
        Fix {
            event: 0.0,
            weight: 1.35,
            z: -1.15,
            q0: -1.35,
            q1: -0.9,
            qd1: 0.42,
            g: -0.55,
        },
        Fix {
            event: 1.0,
            weight: 0.9,
            z: 0.7,
            q0: 0.15,
            q1: 1.05,
            qd1: 0.6,
            g: 0.45,
        },
    ];

    for fix in &fixtures {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![fix.event]),
            weights: Arc::new(array![fix.weight]),
            z: Arc::new(array![fix.z].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![fix.q0]),
            offset_exit: Arc::new(array![fix.q1]),
            derivative_offset_exit: Arc::new(array![fix.qd1]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        // ZERO deviation coefficients: the flex calculus runs in full, but the
        // primary NLL reduces to the rigid closed form so the rigid Tower4 is the
        // exact independent truth on the (q0,q1,qd1,g) block.
        let block_states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![fix.g],
            },
            ParameterBlockState {
                beta: Array1::zeros(score_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(link_runtime.basis_dim()),
                eta: Array1::zeros(1),
            },
        ];

        let primary = flex_primary_slices(&family);
        let p = primary.total;
        // The four time/marginal/logslope primaries occupy the leading slots.
        let block_idx = [primary.q0, primary.q1, primary.qd1, primary.g];

        // Independent rigid Tower4 program at the SAME primaries.
        let program = SurvivalMarginalSlopeRigidNllProgram {
            primaries: vec![[fix.q0, fix.q1, fix.qd1, fix.g]],
            z: vec![fix.z],
            w: vec![fix.weight],
            d: vec![fix.event],
            probit_scale: family.probit_frailty_scale(),
        };

        // Distinct 4-vector directions confined to (q0,q1,qd1,g); embed each into
        // the full p-vector at the leading slots for the flex call.
        let dirs4: [[f64; 4]; 3] = [
            [0.7, -1.3, 0.5, 0.9],
            [-0.4, 0.6, -1.1, 0.3],
            [1.2, 0.2, -0.7, -0.5],
        ];
        let embed = |d4: &[f64; 4]| -> Array1<f64> {
            let mut full = Array1::zeros(p);
            for (k, &slot) in block_idx.iter().enumerate() {
                full[slot] = d4[k];
            }
            full
        };

        // ── Third-order: D_dir H over the (q0,q1,qd1,g) block ───────────────
        for d4 in &dirs4 {
            let flex_full = family
                .row_flex_primary_third_contracted_exact(0, &block_states, &embed(d4))
                .expect("flex third contracted at zero deviation");
            let rigid = derived_third_contracted(&program, 0, d4).expect("rigid third");
            let scale = rigid
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(1.0);
            for (u, &bu) in block_idx.iter().enumerate() {
                for (v, &bv) in block_idx.iter().enumerate() {
                    let got = flex_full[[bu, bv]];
                    let want = rigid[u][v];
                    assert!(
                        (got - want).abs() <= 1e-7 * scale,
                        "third[{u},{v}] flex {got:+.9e} != independent rigid tower {want:+.9e} (z={}, event={})",
                        fix.z,
                        fix.event
                    );
                }
            }
        }

        // Planted sign-flip tripwire on a representative third-order cross block.
        {
            let d4 = &dirs4[0];
            let flex_full = family
                .row_flex_primary_third_contracted_exact(0, &block_states, &embed(d4))
                .expect("flex third contracted (tripwire)");
            let rigid = derived_third_contracted(&program, 0, d4).expect("rigid third (tripwire)");
            // (q0, g) cross block — the marginal↔logslope coupling.
            let want = rigid[0][3];
            let scale = want.abs().max(1.0);
            if want.abs() > 1e-6 {
                let flipped = -flex_full[[primary.q0, primary.g]];
                assert!(
                    (flipped - want).abs() > 1e-7 * scale,
                    "independent rigid tower failed to reject a planted (q0,g) sign flip: flipped {flipped:+.9e} vs truth {want:+.9e}"
                );
            }
        }

        // ── Fourth-order: D_u D_v H over the (q0,q1,qd1,g) block ─────────────
        let quad_pairs = [(0usize, 1usize), (1, 2), (2, 0)];
        for &(iu, iv) in &quad_pairs {
            let du = &dirs4[iu];
            let dv = &dirs4[iv];
            let flex_full = family
                .row_flex_primary_fourth_contracted_exact(0, &block_states, &embed(du), &embed(dv))
                .expect("flex fourth contracted at zero deviation");
            let rigid = derived_fourth_contracted(&program, 0, du, dv).expect("rigid fourth");
            let scale = rigid
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(1.0);
            for (u, &bu) in block_idx.iter().enumerate() {
                for (v, &bv) in block_idx.iter().enumerate() {
                    let got = flex_full[[bu, bv]];
                    let want = rigid[u][v];
                    assert!(
                        (got - want).abs() <= 1e-6 * scale,
                        "fourth[{u},{v}] flex {got:+.9e} != independent rigid tower {want:+.9e} (z={}, event={})",
                        fix.z,
                        fix.event
                    );
                }
            }
        }
    }
}

/// gam#932/#979: INDEPENDENT finite-difference witness for the survival
/// marginal-slope FLEX higher-order tower with NON-ZERO deviation coefficients —
/// the part Arm A (`flex_contracted_tower_matches_independent_rigid_tower_*`)
/// cannot reach, and the part most likely to harbor a shared-input bug.
///
/// The witness re-derives the scalar survival flex row NLL FROM SCRATCH over the
/// primary vector `p = [q0, q1, qd1, g, β_h..., β_w...]`:
///   * the deviation index `index(a, g, z) = a + g·warp(z) + linkdev(a + g·z)`
///     is reconstructed from the RAW basis matrices (`DeviationRuntime::design`),
///     dotted with the coefficients — no production jet / cell-moment code;
///   * the per-timepoint intercept `a(q)` is the calibration root
///     `∫ Φ(−index(a,g,z))·φ(z) dz = Φ(−q)`, solved by an independent secant on
///     a fine composite-Simpson quadrature of the latent normal, with the density
///     normalization `D = |∂F/∂a|` from the same quadrature;
///   * the NLL is assembled from the closed-form survival pieces
///     `w·[logΦ(−η0) − (1−d)logΦ(−η1) − d·logφ(η1) − d·log χ1 − d·logφ(q1)
///        + d·log D1 − d·log qd1]`.
///
/// To DE-RISK the witness (a witness-side re-derivation error would masquerade as
/// a production disagreement), the witness scalar NLL is first self-validated
/// against the production `row_neglog_flex_value` at the SAME non-zero β; only
/// then is it finite-differenced (Richardson O(h⁴)) and compared to the
/// production `row_flex_primary_{third,fourth}_contracted_exact`. A planted sign
/// flip on a deviation-touching cross block must leave the band.
#[test]
fn flex_contracted_tower_matches_independent_fd_witness_nonzero_deviation() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();

    // No frailty ⇒ probit scale = 1 ⇒ the calibration measure is the standard
    // normal latent and the index is unscaled — the regime the witness derives.
    let z_row = 0.3_f64;
    let q0v = -0.25_f64;
    let q1v = 0.7_f64;
    let qd1v = 0.9_f64;
    let gv = 0.4_f64;
    let weight = 0.85_f64;
    let event = 1.0_f64;

    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![event]),
        weights: Arc::new(array![weight]),
        z: Arc::new(array![z_row].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![q0v]),
        offset_exit: Arc::new(array![q1v]),
        derivative_offset_exit: Arc::new(array![qd1v]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let primary = flex_primary_slices(&family);
    let p = primary.total;
    let h_range = primary.h.clone().expect("score-warp primary range");
    let w_range = primary.w.clone().expect("link-dev primary range");

    // Small, distinct non-zero deviation coefficients so every basis column
    // carries signal into the derivative chain.
    let beta_h0: Vec<f64> = (0..h_dim)
        .map(|k| 0.04 * ((k as f64 + 1.3).sin()))
        .collect();
    let beta_w0: Vec<f64> = (0..w_dim)
        .map(|k| 0.035 * ((k as f64 + 0.7).cos()))
        .collect();

    // ── Independent basis evaluations (raw design rows · β) ──────────────────
    let warp_eval = |beta_h: &[f64], z: f64| -> f64 {
        let row = score_runtime
            .design(&array![z])
            .expect("score-warp basis row");
        row.row(0).iter().zip(beta_h).map(|(b, c)| b * c).sum()
    };
    let linkdev_eval = |beta_w: &[f64], u: f64| -> f64 {
        let row = link_runtime.design(&array![u]).expect("link-dev basis row");
        row.row(0).iter().zip(beta_w).map(|(b, c)| b * c).sum()
    };
    // Survival deviation index inside Φ: a + g·z + g·warp(z) + linkdev(a + g·z).
    // The rigid slope term `g·z` is explicit: the score-warp value basis is a
    // pure DEVIATION (identity excluded — deviation_runtime.rs documents "zero
    // coefficients mean the identity map"), and production adds the rigid `b·z`
    // separately in the denested cell coefficient. Omitting it here left an
    // odd-in-z residual that the φ(z)-weighted calibration mostly absorbed into
    // the intercept, surfacing as a ~8.4e-4 self-validation gap.
    let index = |a: f64, g: f64, beta_h: &[f64], beta_w: &[f64], z: f64| -> f64 {
        a + g * z + g * warp_eval(beta_h, z) + linkdev_eval(beta_w, a + g * z)
    };
    // Witness-exact standard-normal primitives (`libm::erfc`, no piecewise
    // rational approximation). The intercept density `d1 = |∂F/∂a|` is taken by
    // a central FD with ε = 1e-6, which divides any error in `F` by 2ε = 2e-6;
    // production's `normal_cdf` carries an ~5e-12 oscillating approximation
    // error, so `F(a±ε)` inherit independent ~5e-12 perturbations whose FD is
    // ~4e-6 — exactly the residual `d1` gap (and hence the `ln d1` term left a
    // ~1.3e-5 self-validation gap, #979). Routing the calibration and the
    // survival log-CDF through ulp-accurate `erfc` removes that amplified noise.
    fn wnorm_cdf(x: f64) -> f64 {
        0.5 * libm::erfc(-x / std::f64::consts::SQRT_2)
    }
    fn wnorm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
    // ── Like-for-like model representation: production's own denested cells ──
    // Production does NOT Simpson-integrate the symbolic composed `index` over a
    // uniform latent grid. It partitions z into cells at every score-warp knot and
    // every link-knot crossing `z=(τ−a)/g`, and on each cell represents the index
    // as the EXACT piecewise cubic `cell.eta(z)=c0+c1 z+c2 z²+c3 z³`. Because the
    // deviation bases are piecewise-cubic, `cell.eta(z)` equals the symbolic
    // `index(a,g,z)` pointwise — but a single uniform Simpson grid that straddles
    // those interior breakpoints integrates a piecewise-cubic integrand with nodes
    // off the breaks, leaving a ~4e-6 aliasing error in `d1` (the #979 value gap).
    // The witness here re-derives production's SAME moment integral by an
    // INDEPENDENT quadrature: composite Simpson applied PER production cell (so every
    // breakpoint is an exact node, and each subinterval integrand is a smooth product
    // of a single cubic with φ). Infinite tail cells are clamped to ±8 (φ decay).
    //
    // `index(a,g,z)` is retained above and used for the observed-node eta/χ, where a
    // single cell covers z_row and the symbolic form coincides with `cell.eta`.
    let denested_cells = |a: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| {
        let beta_h_arr = Array1::from(beta_h.to_vec());
        let beta_w_arr = Array1::from(beta_w.to_vec());
        family
            .denested_partition_cells(a, g, Some(&beta_h_arr), Some(&beta_w_arr))
            .expect("production denested partition cells")
    };
    // Composite Simpson of `f` over a single (possibly infinite) cell, clamped to
    // [-8, 8]; `sub` even subintervals so every cell is integrated to round-off for
    // the smooth single-cubic·φ integrand.
    let cell_simpson = |left: f64, right: f64, sub: usize, f: &dyn Fn(f64) -> f64| -> f64 {
        let lo = left.max(-8.0_f64);
        let hi = right.min(8.0_f64);
        if !(hi > lo) {
            return 0.0;
        }
        let m = sub; // even
        let h = (hi - lo) / m as f64;
        let mut acc = 0.0_f64;
        for k in 0..=m {
            let z = lo + h * k as f64;
            let coef = if k == 0 || k == m {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            acc += coef * f(z);
        }
        acc * h / 3.0
    };
    // Calibration F(a) = ∑cells ∫_cell Φ(−eta_cell(z)) φ(z) dz − Φ(−q), Simpson per
    // production cell on its exact representation of the index.
    let calibration = |a: f64, q: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| -> f64 {
        let cells = denested_cells(a, g, beta_h, beta_w);
        let mut acc = 0.0_f64;
        for partition_cell in &cells {
            let cell = partition_cell.cell;
            acc += cell_simpson(cell.left, cell.right, 1024, &|z| {
                wnorm_cdf(-cell.eta(z)) * wnorm_pdf(z)
            });
        }
        acc - wnorm_cdf(-q)
    };
    // Intercept root + density normalization D = |∂F/∂a|.
    let solve_intercept = |q: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| -> (f64, f64) {
        let f = |a: f64| calibration(a, q, g, beta_h, beta_w);
        // Monotone in a; secant from two seeds around the rigid closed form.
        let c = (1.0 + g * g).sqrt();
        let mut a0 = q * c - 0.5;
        let mut a1 = q * c + 0.5;
        let mut f0 = f(a0);
        for _ in 0..200 {
            let f1 = f(a1);
            if (f1 - f0).abs() <= f64::MIN_POSITIVE {
                break;
            }
            let a2 = a1 - f1 * (a1 - a0) / (f1 - f0);
            a0 = a1;
            f0 = f1;
            a1 = a2;
            if (a1 - a0).abs() <= 1e-13 {
                break;
            }
        }
        let a = a1;
        // Density D = |∂F/∂a| = ∑cells ∫_cell φ(eta_cell(z))·(∂eta_cell/∂a)(z)·φ(z) dz,
        // Simpson per production cell. On each cell `∂eta/∂a` is the EXACT cubic in z
        // whose coefficients production derives from the same score/link spans via
        // `denested_cell_coefficient_partials` (this equals `1 + linkdev'(a+g·z)` on
        // the cell, but taken from production's own cell algebra so the witness
        // re-derives production's density with no symbolic-vs-cell aliasing — the
        // ~4e-6 `d1` value gap of #979). Integrating per cell puts every breakpoint on
        // a Simpson node, so the single-cubic·φ integrand resolves to round-off.
        let cells = denested_cells(a, g, beta_h, beta_w);
        let mut acc = 0.0_f64;
        for partition_cell in &cells {
            let cell = partition_cell.cell;
            let (dc_da, _) = exact_kernel::denested_cell_coefficient_partials(
                partition_cell.score_span,
                partition_cell.link_span,
                a,
                g,
            );
            acc += cell_simpson(cell.left, cell.right, 1024, &|z| {
                let deta_da = dc_da[0] + dc_da[1] * z + dc_da[2] * z * z + dc_da[3] * z * z * z;
                wnorm_pdf(cell.eta(z)) * deta_da * wnorm_pdf(z)
            });
        }
        let d = acc.abs();
        (a, d)
    };
    // χ1 = ∂η1/∂a at the observed node (FD of the observed index in a).
    let observed_eta_chi = |a: f64, g: f64, beta_h: &[f64], beta_w: &[f64]| -> (f64, f64) {
        let eta = index(a, g, beta_h, beta_w, z_row);
        let eps = 1e-6;
        let chi = (index(a + eps, g, beta_h, beta_w, z_row)
            - index(a - eps, g, beta_h, beta_w, z_row))
            / (2.0 * eps);
        (eta, chi)
    };
    // Independent scalar survival flex row NLL over the primary vector.
    let witness_nll = |pv: &[f64]| -> f64 {
        let q0 = pv[primary.q0];
        let q1 = pv[primary.q1];
        let qd1 = pv[primary.qd1];
        let g = pv[primary.g];
        let beta_h: Vec<f64> = h_range.clone().map(|i| pv[i]).collect();
        let beta_w: Vec<f64> = w_range.clone().map(|i| pv[i]).collect();
        let (a0, _) = solve_intercept(q0, g, &beta_h, &beta_w);
        let (a1, d1) = solve_intercept(q1, g, &beta_h, &beta_w);
        let (eta0, _) = observed_eta_chi(a0, g, &beta_h, &beta_w);
        let (eta1, chi1) = observed_eta_chi(a1, g, &beta_h, &beta_w);
        let log_surv0 = wnorm_cdf(-eta0).ln();
        let log_surv1 = wnorm_cdf(-eta1).ln();
        let tau_ln = std::f64::consts::TAU.ln();
        let log_phi_eta1 = -0.5 * (eta1 * eta1 + tau_ln);
        let log_phi_q1 = -0.5 * (q1 * q1 + tau_ln);
        weight
            * (log_surv0
                - (1.0 - event) * log_surv1
                - event * log_phi_eta1
                - event * chi1.ln()
                - event * log_phi_q1
                + event * d1.ln()
                - event * qd1.ln())
    };

    // Base primary point with NON-ZERO deviation coefficients.
    let mut p0 = vec![0.0_f64; p];
    p0[primary.q0] = q0v;
    p0[primary.q1] = q1v;
    p0[primary.qd1] = qd1v;
    p0[primary.g] = gv;
    for (k, i) in h_range.clone().enumerate() {
        p0[i] = beta_h0[k];
    }
    for (k, i) in w_range.clone().enumerate() {
        p0[i] = beta_w0[k];
    }

    // ── De-risk: witness scalar NLL must match the production scalar value ───
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![gv],
        },
        ParameterBlockState {
            beta: Array1::from(beta_h0.clone()),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::from(beta_w0.clone()),
            eta: array![0.0],
        },
    ];
    let prod_value = family
        .row_neglog_flex_value(0, &block_states)
        .expect("production flex row value");
    let wit_value = witness_nll(&p0);
    assert!(
        (prod_value - wit_value).abs() <= 1e-7 * prod_value.abs().max(1.0),
        "witness re-derivation disagrees with production scalar NLL: witness {wit_value:+.10e} vs production {prod_value:+.10e} \
         (this is a witness-side error to fix BEFORE trusting the FD jets, NOT a production bug)"
    );

    // ── Richardson central differences of the witness scalar NLL ────────────
    let central = |axes: &[(usize, usize)], h: f64| -> f64 {
        fn stencil(order: usize) -> &'static [(i64, f64)] {
            match order {
                1 => &[(-1, -0.5), (1, 0.5)],
                2 => &[(-1, 1.0), (0, -2.0), (1, 1.0)],
                3 => &[(-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)],
                4 => &[(-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)],
                _ => &[(0, 1.0)],
            }
        }
        fn walk(
            stencils: &[(usize, &'static [(i64, f64)])],
            h: f64,
            coeff_acc: f64,
            point: &mut Vec<f64>,
            f: &dyn Fn(&[f64]) -> f64,
        ) -> f64 {
            match stencils.split_first() {
                None => coeff_acc * f(point),
                Some((&(idx, st), rest)) => {
                    let mut acc = 0.0;
                    let saved = point[idx];
                    for &(off, c) in st {
                        point[idx] = saved + (off as f64) * h;
                        acc += walk(rest, h, coeff_acc * c, point, f);
                    }
                    point[idx] = saved;
                    acc
                }
            }
        }
        let mut total_order = 0usize;
        let stencils: Vec<(usize, &'static [(i64, f64)])> = axes
            .iter()
            .map(|&(idx, ord)| {
                total_order += ord;
                (idx, stencil(ord))
            })
            .collect();
        let mut point = p0.clone();
        let raw = walk(&stencils, h, 1.0, &mut point, &witness_nll);
        raw / h.powi(total_order as i32)
    };
    let central_rich = |axes: &[(usize, usize)], h: f64| -> f64 {
        let coarse = central(axes, h);
        let fine = central(axes, h * 0.5);
        (4.0 * fine - coarse) / 3.0
    };

    // The deviation axes the witness must witness (q, g, and a β_h / β_w coord).
    let q0i = primary.q0;
    let gi = primary.g;
    let hi0 = h_range.start;
    let wi0 = w_range.start;

    // Embed a unit direction along a single primary axis.
    let unit = |idx: usize| -> Array1<f64> {
        let mut d = Array1::zeros(p);
        d[idx] = 1.0;
        d
    };

    let q_geom = family
        .row_dynamic_q_geometry(0, &block_states)
        .expect("diagnostic q geometry");
    let (_, _, prod_hess) = family
        .compute_row_flex_primary_gradient_hessian_exact(0, &block_states, &q_geom, &primary)
        .expect("diagnostic production flex hessian");
    let witness_h_gw = central_rich(&[(gi, 1), (wi0, 1)], 6e-3);
    eprintln!(
        "#932 diagnostic base hess[g,w0]: production {:+.6e} witness {:+.6e}",
        prod_hess[[gi, wi0]],
        witness_h_gw
    );
    {
        let beta_h_arr = Array1::from(beta_h0.clone());
        let beta_w_arr = Array1::from(beta_w0.clone());
        let tp_diag = |label: &str, q: f64, q_index: usize| {
            let (a, d) = family
                .solve_row_survival_intercept_with_slot(
                    q,
                    gv,
                    Some(&beta_h_arr),
                    Some(&beta_w_arr),
                    Some((
                        0,
                        if q_index == primary.q0 {
                            SurvivalInterceptSlotKind::Entry
                        } else {
                            SurvivalInterceptSlotKind::Exit
                        },
                    )),
                )
                .expect("diagnostic intercept");
            let cached = family
                .build_cached_partition(&primary, a, gv, Some(&beta_h_arr), Some(&beta_w_arr))
                .expect("diagnostic cached partition");
            let base = family
                .compute_survival_timepoint_exact_from_cached(
                    0,
                    &primary,
                    q,
                    q_index,
                    a,
                    gv,
                    d,
                    Some(&beta_h_arr),
                    Some(&beta_w_arr),
                    0.0,
                    q_index == primary.q1,
                    &cached,
                )
                .expect("diagnostic base timepoint");
            let ext = family
                .compute_survival_timepoint_directional_exact_from_cached(
                    0,
                    &primary,
                    q,
                    q_index,
                    a,
                    gv,
                    Some(&beta_h_arr),
                    Some(&beta_w_arr),
                    &cached,
                    &unit(gi),
                    q_index == primary.q1,
                )
                .expect("diagnostic directional timepoint");
            let base_at = |s: f64| -> SurvivalFlexTimepointExact {
                let g = gv + s;
                let (a, d) = family
                    .solve_row_survival_intercept_with_slot(
                        q,
                        g,
                        Some(&beta_h_arr),
                        Some(&beta_w_arr),
                        Some((
                            0,
                            if q_index == primary.q0 {
                                SurvivalInterceptSlotKind::Entry
                            } else {
                                SurvivalInterceptSlotKind::Exit
                            },
                        )),
                    )
                    .expect("diagnostic perturbed intercept");
                let cached = family
                    .build_cached_partition(&primary, a, g, Some(&beta_h_arr), Some(&beta_w_arr))
                    .expect("diagnostic perturbed cached partition");
                family
                    .compute_survival_timepoint_exact_from_cached(
                        0,
                        &primary,
                        q,
                        q_index,
                        a,
                        g,
                        d,
                        Some(&beta_h_arr),
                        Some(&beta_w_arr),
                        0.0,
                        q_index == primary.q1,
                        &cached,
                    )
                    .expect("diagnostic perturbed base timepoint")
            };
            let fd = |sel: &dyn Fn(&SurvivalFlexTimepointExact) -> f64, h: f64| -> f64 {
                let coarse = (sel(&base_at(h)) - sel(&base_at(-h))) / (2.0 * h);
                let fine = (sel(&base_at(h * 0.5)) - sel(&base_at(-h * 0.5))) / h;
                (4.0 * fine - coarse) / 3.0
            };
            eprintln!(
                "#932 {label} [g,w0] eta_uv {:+.6e} fd {:+.6e}; chi_uv {:+.6e} fd {:+.6e}; d_uv {:+.6e} fd {:+.6e}; base eta_uv {:+.6e} chi_uv {:+.6e} d_uv {:+.6e}",
                ext.eta_uv_dir[[gi, wi0]],
                fd(&|b| b.eta_uv[[gi, wi0]], 2e-3),
                ext.chi_uv_dir[[gi, wi0]],
                fd(&|b| b.chi_uv[[gi, wi0]], 2e-3),
                ext.d_uv_dir[[gi, wi0]],
                fd(&|b| b.d_uv[[gi, wi0]], 2e-3),
                base.eta_uv[[gi, wi0]],
                base.chi_uv[[gi, wi0]],
                base.d_uv[[gi, wi0]],
            );
        };
        tp_diag("entry", q0v, primary.q0);
        tp_diag("exit", q1v, primary.q1);
    }

    // ── Third order: production D_dir H[u,v] vs witness ∂³ along (u,v,dir) ───
    // Contract along the logslope axis g; check cross blocks touching the
    // deviation coordinates (the channels Arm A cannot reach).
    let third = family
        .row_flex_primary_third_contracted_exact(0, &block_states, &unit(gi))
        .expect("production third contracted (nonzero deviation)");
    let third_checks = [(q0i, hi0), (gi, wi0), (hi0, wi0), (q0i, wi0)];
    for &(u, v) in &third_checks {
        let want = central_rich(&[(u, 1), (v, 1), (gi, 1)], 6e-3);
        let got = third[[u, v]];
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= 5e-3 * scale + 1e-6,
            "third[{u},{v}] (contract g) production {got:+.6e} != independent FD witness {want:+.6e}"
        );
    }
    // Planted sign-flip tripwire on a deviation-touching cross block.
    {
        let want = central_rich(&[(q0i, 1), (wi0, 1), (gi, 1)], 6e-3);
        if want.abs() > 1e-5 {
            let flipped = -third[[q0i, wi0]];
            assert!(
                (flipped - want).abs() > 5e-3 * want.abs().max(1.0) + 1e-6,
                "independent FD witness failed to reject a planted (q0, β_w) sign flip: flipped {flipped:+.6e} vs witness {want:+.6e}"
            );
        }
    }

    // ── Fourth order: production D_u D_v H[p,q] vs witness ∂⁴ ────────────────
    let fourth = family
        .row_flex_primary_fourth_contracted_exact(0, &block_states, &unit(gi), &unit(wi0))
        .expect("production fourth contracted (nonzero deviation)");
    let fourth_checks = [(q0i, gi), (q0i, hi0), (gi, hi0)];
    for &(u, v) in &fourth_checks {
        let want = central_rich(&[(u, 1), (v, 1), (gi, 1), (wi0, 1)], 9e-3);
        let got = fourth[[u, v]];
        let scale = want.abs().max(1.0);
        assert!(
            (got - want).abs() <= 2e-2 * scale + 1e-5,
            "fourth[{u},{v}] (contract g,β_w) production {got:+.6e} != independent FD witness {want:+.6e}"
        );
    }
}

#[test]
fn debug_flex_directional_quantities_fd_localize() {
    // FD-localize WHICH directional timepoint quantity (eta_uv_dir / chi_uv_dir
    // / d_uv_dir) disagrees with a central difference of its base counterpart
    // along `dir`. Fixture is aligned to
    // `flex_contracted_tower_matches_independent_fd_witness_nonzero_deviation`
    // so the printed per-(u,v) gaps map directly onto the failing third[g,w0]
    // (contract g). Prints per-quantity, per-(u,v) absolute gaps. Pure
    // diagnostic: no assertions, so it is always green; read --nocapture
    // stderr for the localization.
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let event = 1.0_f64;
    let weight = 0.85_f64;
    let z_row = 0.3_f64;
    let q0v = -0.25_f64;
    let q1v = 0.7_f64;
    let qd1v = 0.9_f64;
    let gv = 0.4_f64;

    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![event]),
        weights: Arc::new(array![weight]),
        z: Arc::new(array![z_row].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![q0v]),
        offset_exit: Arc::new(array![q1v]),
        derivative_offset_exit: Arc::new(array![qd1v]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let primary = flex_primary_slices(&family);
    let p = primary.total;
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();
    // NON-ZERO deviations so the link curvature is live (mirrors the
    // flex_..._fd_witness test that still fails third[3,6]=[g,w0]).
    let beta_h: Array1<f64> =
        Array1::from_iter((0..h_dim).map(|k| 0.04 * ((k as f64 + 1.3).sin())));
    let beta_w: Array1<f64> =
        Array1::from_iter((0..w_dim).map(|k| 0.035 * ((k as f64 + 0.7).cos())));

    // Contract along e_g ONLY (matches the failing third[3,6] check), so the
    // FD perturbs g alone.
    let w_range = primary.w.clone().unwrap();
    let mut dir = Array1::<f64>::zeros(p);
    dir[primary.g] = 1.0;

    // Base + directional at the EXIT timepoint.
    let (a1, d1) = family
        .solve_row_survival_intercept_with_slot(
            q1v,
            gv,
            Some(&beta_h),
            Some(&beta_w),
            Some((0, SurvivalInterceptSlotKind::Exit)),
        )
        .expect("exit intercept");
    let cached = family
        .build_cached_partition(&primary, a1, gv, Some(&beta_h), Some(&beta_w))
        .expect("exit cached partition");
    let base = family
        .compute_survival_timepoint_exact_from_cached(
            0,
            &primary,
            q1v,
            primary.q1,
            a1,
            gv,
            d1,
            Some(&beta_h),
            Some(&beta_w),
            0.0,
            true,
            &cached,
        )
        .expect("exit base");
    let ext = family
        .compute_survival_timepoint_directional_exact_from_cached(
            0,
            &primary,
            q1v,
            primary.q1,
            a1,
            gv,
            Some(&beta_h),
            Some(&beta_w),
            &cached,
            &dir,
            true,
        )
        .expect("exit directional");

    // FD of base eta_uv/chi_uv/d_uv along dir: perturb (q1,g) by ±h·dir, re-solve a.
    let base_at = |s: f64| -> SurvivalFlexTimepointExact {
        let q = q1v + s * dir[primary.q1];
        let g = gv + s * dir[primary.g];
        let (a, d) = family
            .solve_row_survival_intercept_with_slot(
                q,
                g,
                Some(&beta_h),
                Some(&beta_w),
                Some((0, SurvivalInterceptSlotKind::Exit)),
            )
            .expect("perturbed exit intercept");
        family
            .compute_survival_timepoint_exact(
                0,
                &primary,
                q,
                primary.q1,
                a,
                g,
                d,
                Some(&beta_h),
                Some(&beta_w),
                0.0,
                true,
            )
            .expect("perturbed exit base")
    };
    let fd = |sel: &dyn Fn(&SurvivalFlexTimepointExact) -> f64, h: f64| -> f64 {
        let coarse = (sel(&base_at(h)) - sel(&base_at(-h))) / (2.0 * h);
        let fine = (sel(&base_at(h * 0.5)) - sel(&base_at(-h * 0.5))) / h;
        (4.0 * fine - coarse) / 3.0
    };

    let g = primary.g;
    let q1 = primary.q1;
    let w0 = w_range.start;
    let probe = [(q1, q1), (g, q1), (g, g), (g, w0), (w0, w0)];
    for &(u, v) in &probe {
        let eta_fd = fd(&|b| b.eta_uv[[u, v]], 2e-3);
        let chi_fd = fd(&|b| b.chi_uv[[u, v]], 2e-3);
        let d_fd = fd(&|b| b.d_uv[[u, v]], 2e-3);
        // Convergence check: a genuine bug holds the gap as h shrinks; a
        // knot-crossing kink artifact moves the FD value as h changes.
        let d_fd_fine = fd(&|b| b.d_uv[[u, v]], 8e-4);
        eprintln!(
            "[{u},{v}] eta_uv_dir prod {:+.6e} fd {:+.6e} gap {:.2e} | chi_uv_dir prod {:+.6e} fd {:+.6e} gap {:.2e} | d_uv_dir prod {:+.6e} fd {:+.6e} gap {:.2e} (fine {:+.6e} gap {:.2e})",
            ext.eta_uv_dir[[u, v]],
            eta_fd,
            (ext.eta_uv_dir[[u, v]] - eta_fd).abs(),
            ext.chi_uv_dir[[u, v]],
            chi_fd,
            (ext.chi_uv_dir[[u, v]] - chi_fd).abs(),
            ext.d_uv_dir[[u, v]],
            d_fd,
            (ext.d_uv_dir[[u, v]] - d_fd).abs(),
            d_fd_fine,
            (ext.d_uv_dir[[u, v]] - d_fd_fine).abs(),
        );
    }
    // d_u_dir localization too — INCLUDING w0 (the untested deviation index).
    for &u in &[q1, g, w0] {
        let d_u_fd = fd(&|b| b.d_u[u], 4e-3);
        let d_u_fd_fine = fd(&|b| b.d_u[u], 1e-3);
        eprintln!(
            "[{u}] d_u_dir prod {:+.6e} fd {:+.6e} gap {:.2e} (fine {:+.6e} gap {:.2e})",
            ext.d_u_dir[u],
            d_u_fd,
            (ext.d_u_dir[u] - d_u_fd).abs(),
            d_u_fd_fine,
            (ext.d_u_dir[u] - d_u_fd_fine).abs()
        );
    }
    // Per-term d_uv_dir localization at the (w0,w0) probe block: FD each base
    // term integral T_i (from the directional struct's debug_d_uv_terms.0) and
    // compare to the analytic dir term integral T_i_dir (debug_d_uv_terms.1).
    {
        let dir_terms_base = |s: f64| -> [f64; 5] {
            let q = q1v + s * dir[primary.q1];
            let g = gv + s * dir[primary.g];
            let (a, _d) = family
                .solve_row_survival_intercept_with_slot(
                    q,
                    g,
                    Some(&beta_h),
                    Some(&beta_w),
                    Some((0, SurvivalInterceptSlotKind::Exit)),
                )
                .expect("perturbed dir intercept");
            let e = family
                .build_cached_partition(&primary, a, g, Some(&beta_h), Some(&beta_w))
                .and_then(|cached| {
                    family.compute_survival_timepoint_directional_exact_from_cached(
                        0,
                        &primary,
                        q,
                        primary.q1,
                        a,
                        g,
                        Some(&beta_h),
                        Some(&beta_w),
                        &cached,
                        &dir,
                        true,
                    )
                })
                .expect("perturbed dir");
            e.debug_d_uv_terms.expect("debug terms").0
        };
        let dir_terms = ext.debug_d_uv_terms.expect("debug terms").1;
        let h = 2e-3_f64;
        let plus = dir_terms_base(h);
        let minus = dir_terms_base(-h);
        for i in 0..5 {
            let fd_i = (plus[i] - minus[i]) / (2.0 * h);
            eprintln!(
                "[w0,w0] term t{}: T_i_dir(analytic)={:+.6e} D_dir(T_i)(fd)={:+.6e} gap={:.2e}",
                i + 1,
                dir_terms[i],
                fd_i,
                (dir_terms[i] - fd_i).abs()
            );
        }
    }
    // Scalar/first-order base quantities to localize the eta_uv_dir error:
    // chi_dir (=D_dir chi), eta_dir (=D_dir eta), and D_dir eta_u[u].
    eprintln!(
        "scalar: chi base {:+.6e} D_dir(chi) fd {:+.6e} | eta base {:+.6e} D_dir(eta) fd {:+.6e}",
        base.chi,
        fd(&|b| b.chi, 4e-3),
        base.eta,
        fd(&|b| b.eta, 4e-3),
    );
    for &u in &[q1, g, w0] {
        eprintln!(
            "[{u}] eta_u base {:+.6e} D_dir(eta_u) prod {:+.6e} fd {:+.6e} gap {:.2e} | chi_u base {:+.6e} D_dir(chi_u) prod {:+.6e} fd {:+.6e} gap {:.2e}",
            base.eta_u[u],
            ext.eta_u_dir[u],
            fd(&|b| b.eta_u[u], 4e-3),
            (ext.eta_u_dir[u] - fd(&|b| b.eta_u[u], 4e-3)).abs(),
            base.chi_u[u],
            ext.chi_u_dir[u],
            fd(&|b| b.chi_u[u], 4e-3),
            (ext.chi_u_dir[u] - fd(&|b| b.chi_u[u], 4e-3)).abs(),
        );
    }
    // Direct cross-check: FD of base eta_uv[q1,q1] sanity (recompute from a
    // wider/narrower h to confirm the harness isn't aliasing the q-self term).
    eprintln!(
        "[q1,q1] eta_uv base {:+.6e} | eta_uv(+h) {:+.6e} eta_uv(-h) {:+.6e}",
        base.eta_uv[[q1, q1]],
        base_at(4e-3).eta_uv[[q1, q1]],
        base_at(-4e-3).eta_uv[[q1, q1]],
    );
}

/// gam#932/#979 coupling probe: is the production BASE Hessian cross
/// `H[g, w0]` / `H[w0, w0]` (the marginal↔logslope coupling the #979 inner
/// Newton uses) consistent with an INDEPENDENT finite-difference of the
/// production gradient? If the base Hessian is wrong, that is the #979 root and
/// the #932 directional third inherits it. Uses the SAME nonzero-deviation
/// fixture as `flex_contracted_tower_matches_independent_fd_witness_nonzero_deviation`.
#[test]
#[ignore = "debug FD-of-gradient base-Hessian probe for #932/#979 marginal/logslope coupling"]
fn debug_flex_base_hessian_vs_gradient_fd() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();
    let z_row = 0.3_f64;
    let q0v = -0.25_f64;
    let q1v = 0.7_f64;
    let qd1v = 0.9_f64;
    let gv = 0.4_f64;
    let weight = 0.85_f64;
    let event = 1.0_f64;
    let beta_h0: Vec<f64> = (0..h_dim)
        .map(|k| 0.04 * ((k as f64 + 1.3).sin()))
        .collect();
    let beta_w0: Vec<f64> = (0..w_dim)
        .map(|k| 0.035 * ((k as f64 + 0.7).cos()))
        .collect();

    let make = |g: f64, beta_h: &[f64], beta_w: &[f64]| {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![event]),
            weights: Arc::new(array![weight]),
            z: Arc::new(array![z_row].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![q0v]),
            offset_exit: Arc::new(array![q1v]),
            derivative_offset_exit: Arc::new(array![qd1v]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let bs = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![g],
            },
            ParameterBlockState {
                beta: Array1::from(beta_h.to_vec()),
                eta: array![0.0],
            },
            ParameterBlockState {
                beta: Array1::from(beta_w.to_vec()),
                eta: array![0.0],
            },
        ];
        (family, bs)
    };

    let (family, bs) = make(gv, &beta_h0, &beta_w0);
    let primary = flex_primary_slices(&family);
    let g = primary.g;
    let w0 = primary.w.clone().unwrap().start;
    let q1 = primary.q1;

    // Production gradient + Hessian at the base point.
    let grad_hess = |g: f64, beta_h: &[f64], beta_w: &[f64]| -> (Array1<f64>, Array2<f64>) {
        let (fam, bsl) = make(g, beta_h, beta_w);
        let qg = fam.row_dynamic_q_geometry(0, &bsl).unwrap();
        let pr = flex_primary_slices(&fam);
        let (_, grad, hess) = fam
            .compute_row_flex_primary_gradient_hessian_exact(0, &bsl, &qg, &pr)
            .unwrap();
        (grad, hess)
    };
    let (_g0, hess0) = grad_hess(gv, &beta_h0, &beta_w0);

    // Independent FD of the gradient along g to get H[*, g].
    let grad_at_g = |s: f64| -> Array1<f64> { grad_hess(gv + s, &beta_h0, &beta_w0).0 };
    let fd_col_g = |h: f64| -> Array1<f64> {
        let coarse = (&grad_at_g(h) - &grad_at_g(-h)) / (2.0 * h);
        let fine = (&grad_at_g(h * 0.5) - &grad_at_g(-h * 0.5)) / h;
        (&fine * 4.0 - &coarse) / 3.0
    };
    // Independent FD of the gradient along w0 to get H[*, w0].
    let grad_at_w0 = |s: f64| -> Array1<f64> {
        let mut bw = beta_w0.clone();
        bw[0] += s;
        grad_hess(gv, &beta_h0, &bw).0
    };
    let fd_col_w0 = |h: f64| -> Array1<f64> {
        let coarse = (&grad_at_w0(h) - &grad_at_w0(-h)) / (2.0 * h);
        let fine = (&grad_at_w0(h * 0.5) - &grad_at_w0(-h * 0.5)) / h;
        (&fine * 4.0 - &coarse) / 3.0
    };
    let hg = fd_col_g(2e-3);
    let hw = fd_col_w0(2e-3);
    for &(u, name) in &[(g, "g"), (w0, "w0"), (q1, "q1")] {
        eprintln!(
            "BASE-HESS col-g [{name}]: production H[{u},{g}]={:+.8e} fd_grad={:+.8e} gap={:.2e}",
            hess0[[u, g]],
            hg[u],
            (hess0[[u, g]] - hg[u]).abs()
        );
    }
    for &(u, name) in &[(g, "g"), (w0, "w0"), (q1, "q1")] {
        eprintln!(
            "BASE-HESS col-w0 [{name}]: production H[{u},{w0}]={:+.8e} fd_grad={:+.8e} gap={:.2e}",
            hess0[[u, w0]],
            hw[u],
            (hess0[[u, w0]] - hw[u]).abs()
        );
    }
    eprintln!(
        "symmetry check: H[g,w0]={:+.8e} H[w0,g]={:+.8e} fd col-g[w0]={:+.8e} fd col-w0[g]={:+.8e}",
        hess0[[g, w0]],
        hess0[[w0, g]],
        hg[w0],
        hw[g]
    );

    // VALUE-FD gradient: differentiate the row NLL scalar VALUE directly to get
    // the TRUE gradient[g] / grad[w0], pinning WHICH production gradient
    // component is wrong (the Hessian asymmetry above proves one of them is).
    let (grad0, _) = grad_hess(gv, &beta_h0, &beta_w0);
    let value_at = |g: f64, beta_w: &[f64]| -> f64 {
        let (fam, bsl) = make(g, &beta_h0, beta_w);
        fam.row_neglog_flex_value(0, &bsl).unwrap()
    };
    let fd_val_g = |h: f64| -> f64 {
        let c = (value_at(gv + h, &beta_w0) - value_at(gv - h, &beta_w0)) / (2.0 * h);
        let f = (value_at(gv + h * 0.5, &beta_w0) - value_at(gv - h * 0.5, &beta_w0)) / h;
        (4.0 * f - c) / 3.0
    };
    let fd_val_w0 = |h: f64| -> f64 {
        let mut bp = beta_w0.clone();
        let mut bm = beta_w0.clone();
        bp[0] += h;
        bm[0] -= h;
        let c = (value_at(gv, &bp) - value_at(gv, &bm)) / (2.0 * h);
        let mut bp2 = beta_w0.clone();
        let mut bm2 = beta_w0.clone();
        bp2[0] += h * 0.5;
        bm2[0] -= h * 0.5;
        let f = (value_at(gv, &bp2) - value_at(gv, &bm2)) / h;
        (4.0 * f - c) / 3.0
    };
    eprintln!(
        "VALUE-FD grad[g]: production={:+.8e} fd_value={:+.8e} gap={:.2e}",
        grad0[g],
        fd_val_g(2e-3),
        (grad0[g] - fd_val_g(2e-3)).abs()
    );
    eprintln!(
        "VALUE-FD grad[w0]: production={:+.8e} fd_value={:+.8e} gap={:.2e}",
        grad0[w0],
        fd_val_w0(2e-3),
        (grad0[w0] - fd_val_w0(2e-3)).abs()
    );
}

/// gam#932/#979: the logslope first-sensitivity must include the Leibniz
/// boundary term for density-normalization cells whose link-knot crossings move
/// with g.
#[test]
fn flex_logslope_first_sensitivity_matches_fd() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();
    let z_row = 0.3_f64;
    let q0v = -0.25_f64;
    let q1v = 0.7_f64;
    let qd1v = 0.9_f64;
    let gv = 0.4_f64;
    let weight = 0.85_f64;
    let event = 1.0_f64;
    let beta_h0: Vec<f64> = (0..h_dim)
        .map(|k| 0.04 * ((k as f64 + 1.3).sin()))
        .collect();
    let beta_w0: Vec<f64> = (0..w_dim)
        .map(|k| 0.035 * ((k as f64 + 0.7).cos()))
        .collect();

    let make = |g: f64| {
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![event]),
            weights: Arc::new(array![weight]),
            z: Arc::new(array![z_row].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![q0v]),
            offset_exit: Arc::new(array![q1v]),
            derivative_offset_exit: Arc::new(array![qd1v]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        family
    };
    let family = make(gv);
    let primary = flex_primary_slices(&family);
    let g = primary.g;
    let bh = Array1::from(beta_h0.clone());
    let bw = Array1::from(beta_w0.clone());

    // Exit-timepoint base struct with eta_u/chi_u/d_u and scalars eta/chi/d.
    let base_at = |gg: f64| -> SurvivalFlexTimepointExact {
        let fam = make(gg);
        let (a1, d1) = fam
            .solve_row_survival_intercept_with_slot(
                q1v,
                gg,
                Some(&bh),
                Some(&bw),
                Some((0, SurvivalInterceptSlotKind::Exit)),
            )
            .expect("exit intercept");
        fam.compute_survival_timepoint_exact(
            0,
            &primary,
            q1v,
            primary.q1,
            a1,
            gg,
            d1,
            Some(&bh),
            Some(&bw),
            0.0,
            true,
        )
        .expect("exit base")
    };
    let b0 = base_at(gv);
    let fd = |sel: &dyn Fn(&SurvivalFlexTimepointExact) -> f64, h: f64| -> f64 {
        let c = (sel(&base_at(gv + h)) - sel(&base_at(gv - h))) / (2.0 * h);
        let f = (sel(&base_at(gv + h * 0.5)) - sel(&base_at(gv - h * 0.5))) / h;
        (4.0 * f - c) / 3.0
    };
    let fd_d = fd(&|b| b.d, 2e-3);
    let fd_eta = fd(&|b| b.eta, 2e-3);
    let fd_chi = fd(&|b| b.chi, 2e-3);
    assert!(
        (b0.d_u[g] - fd_d).abs() <= 5e-4 * fd_d.abs().max(1.0),
        "d_u[g] production={:+.8e} fd={:+.8e}",
        b0.d_u[g],
        fd_d
    );
    assert!(
        (b0.eta_u[g] - fd_eta).abs() <= 5e-4 * fd_eta.abs().max(1.0),
        "eta_u[g] production={:+.8e} fd={:+.8e}",
        b0.eta_u[g],
        fd_eta
    );
    assert!(
        (b0.chi_u[g] - fd_chi).abs() <= 5e-4 * fd_chi.abs().max(1.0),
        "chi_u[g] production={:+.8e} fd={:+.8e}",
        b0.chi_u[g],
        fd_chi
    );
}

#[test]
fn link_flex_family_supports_second_order_exact_outer_path() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::ones(1)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let specs = vec![
        dummy_blockspec(1),
        dummy_blockspec(0),
        dummy_blockspec(score_runtime.basis_dim()),
        dummy_blockspec(link_runtime.basis_dim()),
    ];
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
}

#[test]
fn timewiggle_scorewarp_family_supports_second_order_exact_outer_path() {
    let score_runtime = test_deviation_runtime();
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 5))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 5))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 5))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::ones(1)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let specs = vec![
        dummy_blockspec(5),
        dummy_blockspec(0),
        dummy_blockspec(score_runtime.basis_dim()),
    ];
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
}

#[test]
fn timewiggle_time_jacobian_nonzero_at_zero_beta_linearization() {
    // Regression: when `timewiggle(...)` disables the base time basis the
    // time block's coefficient slots are zero placeholder columns. The
    // identifiability canonicaliser linearises every block at β=0 by
    // calling `effective_jacobian_at` with `beta = &[]`. Previously the
    // timewiggle callback inferred block existence from the (empty)
    // coefficient slice, drove `beta_tw` empty, and returned an all-zero
    // time Jacobian — so the compiler reported "block 0 fully aliased:
    // structural residual Gram has no positive eigenspace". The true
    // derivative ∂q/∂β_tw[j] = B_j(h) at β=0 is the wiggle basis value and
    // is nonzero.
    let (knots, degree, p_tw) = standard_test_time_wiggle();
    assert!(p_tw > 0);
    let n = 3usize;
    // Base time basis disabled: p_base = 0, every column is a wiggle slot,
    // densified as zeros (the placeholder tail the workflow appends).
    let zeros = Arc::new(Array2::<f64>::zeros((n, p_tw)));
    // Pilot coordinates in the interior of the knot span [0, 1].
    let offset_entry = Arc::new(array![0.2, 0.4, 0.6]);
    let offset_exit = Arc::new(array![0.5, 0.7, 0.9]);
    let offset_deriv = Arc::new(array![1.0, 1.0, 1.0]);
    let jac_cb = SmsTimewiggleTimeJacobian::new(
        Arc::clone(&zeros),
        Arc::clone(&zeros),
        Arc::clone(&zeros),
        Arc::new(Array2::<f64>::zeros((n, 0))), // p_m = 0
        Arc::new(Array2::<f64>::zeros((n, 0))), // p_g = 0
        Arc::clone(&offset_entry),
        Arc::clone(&offset_exit),
        Arc::clone(&offset_deriv),
        Arc::new(Array1::<f64>::zeros(n)), // marginal_offset = 0
        knots.clone(),
        degree,
        p_tw,
        0,
        0,
        1.0,
    );
    let empty: Vec<f64> = Vec::new();
    let state = crate::custom_family::FamilyLinearizationState {
        beta: &empty,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac = crate::custom_family::BlockEffectiveJacobian::effective_jacobian_at(&jac_cb, &state)
        .expect("timewiggle time jacobian at beta=0");
    assert_eq!(jac.dim(), (3 * n, p_tw));

    // q0 rows (0..n) must equal the wiggle basis at the entry pilot
    // coordinate (c_i = 1 at β_g = 0), not the broken all-zero placeholder.
    let basis_entry =
        monotone_wiggle_basis_with_derivative_order(offset_entry.view(), &knots, degree, 0)
            .expect("entry basis");
    for i in 0..n {
        for j in 0..p_tw {
            assert_close(
                jac[[i, j]],
                basis_entry[[i, j]],
                1e-12,
                &format!("q0 wiggle col ({i},{j})"),
            );
        }
    }
    // The time block must carry a positive structural eigenspace: its Gram
    // Jᵀ J is not the zero matrix.
    let gram = jac.t().dot(&jac);
    let trace: f64 = (0..p_tw).map(|j| gram[[j, j]]).sum();
    assert!(
        trace > 1e-6,
        "time block Gram is structurally zero (fully-aliased artifact): trace={trace}"
    );
}

#[test]
fn survival_marginal_slope_advertises_outer_hvp_at_large_psi_dim() {
    // `dummy_penalized_blockspec` materializes single-row designs, so the
    // family row count must be 1 to satisfy the HVP availability row guard
    // (`parameter_block_specs_match_rows`, added in 28a1c035f). The
    // "large psi dim" under test is the 32-column block below, not `n`.
    let n = 1usize;
    let family = make_block_psi_test_family(n);
    let specs = vec![
        dummy_penalized_blockspec(0, 0),
        dummy_penalized_blockspec(1, 31),
        dummy_penalized_blockspec(1, 1),
    ];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        use_outer_hessian: true,
        ..BlockwiseFitOptions::default()
    };

    let (gradient, hessian) = custom_family_outer_derivatives(&family, &specs, &options);

    assert!(family.inner_coefficient_hessian_hvp_available(&specs));
    assert!(family.outer_hyper_hessian_hvp_available(&specs));
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &options),
        ExactOuterDerivativeOrder::Second
    );
    assert!(
        crate::solver::estimate::reml::reml_outer_engine::prefer_outer_hessian_operator(
            50_001, 2, 32,
        )
    );
    assert_eq!(gradient, crate::solver::rho_optimizer::Derivative::Analytic);
    assert_eq!(
        hessian,
        crate::solver::rho_optimizer::DeclaredHessianForm::Either
    );
}

#[test]
fn survival_marginal_slope_coefficient_cost_uses_joint_coupled_formula() {
    // Rigid three-block shape: time p=12, marginal p=20, log-slope p=8.
    // The row kernel couples all three blocks, so the joint Hessian is
    // dense over (12+20+8)²=1600 entries per row. The override must
    // return n·(Σ p_b)², not the block-diagonal Σ n·p_b².
    let n = 200usize;
    let p_time = 12usize;
    let p_marg = 20usize;
    let p_log = 8usize;
    let family = SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(Array1::zeros(n)),
        weights: Arc::new(Array1::from_elem(n, 1.0)),
        z: Arc::new(Array1::zeros(n).insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, p_time))),
        design_exit: DesignMatrix::from(Array2::zeros((n, p_time))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((n, p_time))),
        offset_entry: Arc::new(Array1::zeros(n)),
        offset_exit: Arc::new(Array1::zeros(n)),
        derivative_offset_exit: Arc::new(Array1::ones(n)),
        marginal_design: DesignMatrix::from(Array2::zeros((n, p_marg))),
        logslope_design: DesignMatrix::from(Array2::zeros((n, p_log))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let specs = vec![
        dummy_penalized_blockspec(p_time, 1),
        dummy_penalized_blockspec(p_marg, 1),
        dummy_penalized_blockspec(p_log, 1),
    ];
    let p_total = (p_time + p_marg + p_log) as u64;
    let expected_joint = (n as u64) * p_total * p_total;
    let expected_block_diag_at_full_n =
        (n as u64) * ((p_time * p_time + p_marg * p_marg + p_log * p_log) as u64);
    assert_eq!(family.coefficient_hessian_cost(&specs), expected_joint);
    // Joint coupling exceeds block-diagonal by the cross-block fill
    // 2·n·(p_t·p_m + p_t·p_l + p_m·p_l).
    assert!(expected_joint > expected_block_diag_at_full_n);
}

#[test]
fn exact_outer_row_work_gate_keeps_large_timewiggle_link_models_under_linear_flex_budget() {
    let link_runtime = test_deviation_runtime();
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 80,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 12))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 12))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 12))),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::ones(1)),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 20))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 20))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let specs = vec![
        dummy_penalized_blockspec(12, 2),
        dummy_penalized_blockspec(20, 2),
        dummy_penalized_blockspec(link_runtime.basis_dim(), 2),
    ];
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        ExactOuterDerivativeOrder::Second
    );
}

#[test]
fn timewiggle_scorewarp_beta_hessian_directional_derivative_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_flat = Array1::zeros(slices.total);
    d_beta_flat[slices.time.start] = 0.07;
    d_beta_flat[slices.time.start + 1] = -0.03;
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_flat[h_range.start] = 0.02;
    }

    let directional = family
        .exact_newton_joint_hessian_directional_derivative(&block_states, &d_beta_flat)
        .expect("timewiggle flex beta-Hessian directional derivative should evaluate")
        .expect("directional derivative should exist");
    assert_eq!(directional.nrows(), slices.total);
    assert_eq!(directional.ncols(), slices.total);
    assert!(directional.iter().all(|value| value.is_finite()));
}

#[test]
fn timewiggle_scorewarp_beta_hessian_second_directional_derivative_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_u = Array1::zeros(slices.total);
    let mut d_beta_v = Array1::zeros(slices.total);
    d_beta_u[slices.time.start] = 0.07;
    d_beta_u[slices.time.start + 1] = -0.03;
    d_beta_u[slices.marginal.start] = 0.05;
    d_beta_u[slices.logslope.start] = -0.04;
    d_beta_v[slices.time.start + 2] = 0.06;
    d_beta_v[slices.marginal.start + 1] = -0.02;
    d_beta_v[slices.logslope.start] = 0.03;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_u[h_range.start] = 0.02;
        d_beta_v[h_range.start] = -0.01;
    }

    let second = family
        .exact_newton_joint_hessiansecond_directional_derivative(
            &block_states,
            &d_beta_u,
            &d_beta_v,
        )
        .expect("timewiggle flex beta-Hessian second directional derivative should evaluate")
        .expect("second directional derivative should exist");
    assert_eq!(second.nrows(), slices.total);
    assert_eq!(second.ncols(), slices.total);
    assert!(second.iter().all(|value| value.is_finite()));
}

#[test]
fn link_flex_bidirectional_timepoint_returns_finite_transport() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];

    let q_geom = family
        .row_dynamic_q_geometry(0, &block_states)
        .expect("row geometry");
    let primary = flex_primary_slices(&family);
    let g = block_states[2].eta[0];
    let beta_h = family.flex_score_beta(&block_states).expect("score beta");
    let beta_w = family.flex_link_beta(&block_states).expect("link beta");
    let (a1, _) = family
        .solve_row_survival_intercept_with_slot(q_geom.q1, g, beta_h, beta_w, None)
        .expect("solve intercept");

    let mut dir_u = Array1::zeros(primary.total);
    let mut dir_v = Array1::zeros(primary.total);
    dir_u[primary.q1] = 0.07;
    dir_u[primary.g] = -0.04;
    dir_v[primary.q0] = 0.03;
    if let Some(h_range) = primary.h.as_ref() {
        dir_u[h_range.start] = 0.02;
    }
    if let Some(w_range) = primary.w.as_ref() {
        dir_v[w_range.start] = -0.01;
    }

    let cached = family
        .build_cached_partition(&primary, a1, g, beta_h, beta_w)
        .expect("active bidirectional cached partition");
    let active = family
        .compute_survival_timepoint_bidirectional_exact_from_cached(
            0, &primary, q_geom.q1, primary.q1, a1, g, beta_h, beta_w, &cached, &dir_u, &dir_v,
        )
        .expect("active bidirectional transport");
    assert!(active.eta_uv_uv.iter().all(|value| value.is_finite()));
    assert!(active.chi_uv_uv.iter().all(|value| value.is_finite()));
    assert!(active.d_uv_uv.iter().all(|value| value.is_finite()));
    assert_eq!(active.eta_uv_uv.nrows(), primary.total);
    assert_eq!(active.eta_uv_uv.ncols(), primary.total);
    assert_eq!(active.chi_uv_uv.nrows(), primary.total);
    assert_eq!(active.chi_uv_uv.ncols(), primary.total);
    assert_eq!(active.d_uv_uv.nrows(), primary.total);
    assert_eq!(active.d_uv_uv.ncols(), primary.total);
    for u in 0..primary.total {
        for v in 0..primary.total {
            assert_eq!(active.eta_uv_uv[[u, v]], active.eta_uv_uv[[v, u]]);
            assert_eq!(active.chi_uv_uv[[u, v]], active.chi_uv_uv[[v, u]]);
            assert_eq!(active.d_uv_uv[[u, v]], active.d_uv_uv[[v, u]]);
        }
    }
}

#[test]
fn link_flex_blockwise_exact_newton_matches_joint_principal_blocks() {
    assert!(file!().ends_with(".rs"));
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let logslope_design = array![[1.0], [0.5]];
    let family = SurvivalMarginalSlopeFamily {
        n: 2,
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((2, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((2, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((2, 1))),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(logslope_design.clone()),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let marginal_beta = array![0.35, -0.1];
    let logslope_beta = array![0.2];
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_design.dot(&logslope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    assert_blockwise_matches_joint_principal_blocks(&family, &block_states);
}

#[test]
fn link_flex_marginal_psi_terms_return_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_beta = array![0.2];
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_beta.clone(),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(score_runtime.basis_dim()),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(link_runtime.basis_dim()),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_terms(&block_states, &derivative_blocks, 0)
        .expect("link flex psi terms should evaluate")
        .expect("psi terms should exist");
    assert!(terms.objective_psi.is_finite());
    assert_eq!(terms.score_psi.len(), slices.total);
    assert!(terms.score_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_operator.is_some());
}

#[test]
fn link_flex_marginal_psi_second_order_returns_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_design = array![[1.2, -0.3]];
    let logslope_beta = array![0.2, -0.05];
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(logslope_design.clone()),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_design.dot(&logslope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[0.3, 0.8]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_second_order_terms(&block_states, &derivative_blocks, 0, 1)
        .expect("link flex psi second-order path should evaluate")
        .expect("psi second-order terms should exist");
    assert!(terms.objective_psi_psi.is_finite());
    assert_eq!(terms.score_psi_psi.len(), slices.total);
    assert!(terms.score_psi_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_psi_operator.is_some());
}

#[test]
fn link_flex_marginal_psi_hessian_directional_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_flat = Array1::zeros(slices.total);
    d_beta_flat[slices.time.start] = 0.07;
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_flat[h_range.start] = 0.02;
    }
    if let Some(w_range) = slices.link_dev.as_ref() {
        d_beta_flat[w_range.start] = -0.03;
    }

    let hess_dir = family
        .psi_hessian_directional_derivative(&block_states, &derivative_blocks, 0, &d_beta_flat)
        .expect("link flex psi-Hessian directional path should evaluate")
        .expect("psi-Hessian directional derivative should exist");
    assert_eq!(hess_dir.nrows(), slices.total);
    assert_eq!(hess_dir.ncols(), slices.total);
    assert!(hess_dir.iter().all(|value| value.is_finite()));
}

#[test]
fn timewiggle_marginal_psi_terms_return_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_terms(&block_states, &derivative_blocks, 0)
        .expect("timewiggle psi terms should evaluate")
        .expect("psi terms should exist");
    assert!(terms.objective_psi.is_finite());
    assert_eq!(terms.score_psi.len(), slices.total);
    assert!(terms.score_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_operator.is_some());
}

#[test]
fn timewiggle_blockwise_exact_newton_matches_joint_principal_blocks() {
    assert!(file!().ends_with(".rs"));
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 2,
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0], [0.5]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: array![0.2, 0.1],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    assert_blockwise_matches_joint_principal_blocks(&family, &block_states);
}

#[test]
fn flex_timewiggle_fast_gradient_matches_dense_joint_gradient() {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_design = array![[1.0]];
    let logslope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(logslope_design.clone()),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_design.dot(&logslope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];

    let (fast_ll, fast_grad) = family
        .evaluate_exact_newton_joint_gradient_dynamic_q(&block_states)
        .expect("fast gradient should evaluate");
    let (dense_ll, dense_grad, _) = family
        .evaluate_exact_newton_joint_dynamic_q_dense(&block_states)
        .expect("dense joint derivatives should evaluate");

    assert_close(fast_ll, dense_ll, 1e-10, "log-likelihood");
    assert_eq!(fast_grad.len(), dense_grad.len());
    for idx in 0..fast_grad.len() {
        assert_close(
            fast_grad[idx],
            dense_grad[idx],
            1e-8,
            &format!("gradient[{idx}]"),
        );
    }
}

#[test]
fn row_dynamic_q_geometry_into_pooled_matches_fresh_allocation_bitwise() {
    // Regression: `row_dynamic_q_geometry_into` reuses a caller-owned
    // `SurvivalMarginalSlopeDynamicRow` workspace (resized + zero-filled
    // in place) instead of allocating nine fresh Array2/Array1 buffers
    // per row. Both code paths must return bit-for-bit identical
    // contents, with the workspace path additionally verified to leave
    // the same answer when re-entered against an already-populated
    // buffer (so the in-place `reset` correctly wipes stale state).
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_design = array![[1.0], [0.5]];
    let logslope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 2,
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(logslope_design.clone()),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_design.dot(&logslope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    // Compare each row, twice — first into a virgin workspace, then
    // into the same workspace re-used after a different row has
    // already populated it. Both must equal the fresh-allocation path
    // bit-for-bit.
    let mut workspace = SurvivalMarginalSlopeDynamicRow::empty_workspace();
    // Pre-load the workspace with row 1 so the row 0 comparison
    // exercises the buffer-reuse zeroing logic on a non-trivial state.
    family
        .row_dynamic_q_geometry_into(1, &block_states, &mut workspace)
        .expect("preload workspace with row 1");
    for row in [0usize, 1usize, 0usize] {
        let fresh = family
            .row_dynamic_q_geometry(row, &block_states)
            .expect("fresh-allocation row geometry");
        family
            .row_dynamic_q_geometry_into(row, &block_states, &mut workspace)
            .expect("pooled-workspace row geometry");
        assert_eq!(workspace.q0.to_bits(), fresh.q0.to_bits(), "row {row} q0");
        assert_eq!(workspace.q1.to_bits(), fresh.q1.to_bits(), "row {row} q1");
        assert_eq!(
            workspace.qd1.to_bits(),
            fresh.qd1.to_bits(),
            "row {row} qd1"
        );
        let array1_pairs: [(&Array1<f64>, &Array1<f64>, &str); 6] = [
            (&workspace.dq0_time, &fresh.dq0_time, "dq0_time"),
            (&workspace.dq1_time, &fresh.dq1_time, "dq1_time"),
            (&workspace.dqd1_time, &fresh.dqd1_time, "dqd1_time"),
            (&workspace.dq0_marginal, &fresh.dq0_marginal, "dq0_marginal"),
            (&workspace.dq1_marginal, &fresh.dq1_marginal, "dq1_marginal"),
            (
                &workspace.dqd1_marginal,
                &fresh.dqd1_marginal,
                "dqd1_marginal",
            ),
        ];
        for (lhs, rhs, label) in array1_pairs {
            assert_eq!(lhs.shape(), rhs.shape(), "row {row} {label} shape");
            for (i, (l, r)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert_eq!(
                    l.to_bits(),
                    r.to_bits(),
                    "row {row} {label}[{i}] lhs={l:.17e} rhs={r:.17e}",
                );
            }
        }
        let array2_pairs: [(&Array2<f64>, &Array2<f64>, &str); 9] = [
            (
                &workspace.d2q0_time_time,
                &fresh.d2q0_time_time,
                "d2q0_time_time",
            ),
            (
                &workspace.d2q1_time_time,
                &fresh.d2q1_time_time,
                "d2q1_time_time",
            ),
            (
                &workspace.d2qd1_time_time,
                &fresh.d2qd1_time_time,
                "d2qd1_time_time",
            ),
            (
                &workspace.d2q0_time_marginal,
                &fresh.d2q0_time_marginal,
                "d2q0_time_marginal",
            ),
            (
                &workspace.d2q1_time_marginal,
                &fresh.d2q1_time_marginal,
                "d2q1_time_marginal",
            ),
            (
                &workspace.d2qd1_time_marginal,
                &fresh.d2qd1_time_marginal,
                "d2qd1_time_marginal",
            ),
            (
                &workspace.d2q0_marginal_marginal,
                &fresh.d2q0_marginal_marginal,
                "d2q0_marginal_marginal",
            ),
            (
                &workspace.d2q1_marginal_marginal,
                &fresh.d2q1_marginal_marginal,
                "d2q1_marginal_marginal",
            ),
            (
                &workspace.d2qd1_marginal_marginal,
                &fresh.d2qd1_marginal_marginal,
                "d2qd1_marginal_marginal",
            ),
        ];
        for (lhs, rhs, label) in array2_pairs {
            assert_eq!(lhs.shape(), rhs.shape(), "row {row} {label} shape");
            for ((idx, l), r) in lhs.indexed_iter().zip(rhs.iter()) {
                assert_eq!(
                    l.to_bits(),
                    r.to_bits(),
                    "row {row} {label}[{idx:?}] lhs={l:.17e} rhs={r:.17e}",
                );
            }
        }
    }
}

#[test]
fn flex_timewiggle_operator_to_dense_matches_evaluate_dense_joint_hessian() {
    // Regression: SurvivalMarginalSlopeExactNewtonJointHessianWorkspace::
    // hessian_dense() now returns the already-built operator's
    // to_dense() instead of re-running
    // evaluate_exact_newton_joint_dynamic_q_dense (a second full n-row
    // sweep). Both code paths must agree on the joint p×p Hessian.
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2], [0.1, 0.6]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_design = array![[1.0], [0.5]];
    let logslope_beta = array![0.2];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 2,
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.15, -0.25].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_exit: DesignMatrix::from(array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        design_derivative_exit: DesignMatrix::from(array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ]),
        offset_entry: Arc::new(array![0.05, -0.02]),
        offset_exit: Arc::new(array![0.15, 0.08]),
        derivative_offset_exit: Arc::new(array![0.9, 1.1]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(logslope_design.clone()),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_design.dot(&logslope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
        ParameterBlockState {
            beta: Array1::zeros(link_runtime.basis_dim()),
            eta: Array1::zeros(2),
        },
    ];

    let (_, _, dense) = family
        .evaluate_exact_newton_joint_dynamic_q_dense(&block_states)
        .expect("dense joint Hessian");
    let (operator, _, _, _) = family
        .exact_newton_joint_hessian_operator(&block_states, &BlockwiseFitOptions::default())
        .expect("joint Hessian operator");
    let op_dense = operator.to_dense();

    assert_eq!(op_dense.shape(), dense.shape());
    let diff = max_abs_diff_mat(&op_dense, &dense);
    assert!(
        diff <= 1e-10,
        "operator.to_dense() differs from evaluate_dense by {diff:.3e}",
    );
}

#[test]
fn timewiggle_marginal_logslope_psi_second_order_returns_finite_joint_terms() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_design = array![[1.2, -0.3]];
    let logslope_beta = array![0.2, -0.05];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(logslope_design.clone()),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_design.dot(&logslope_beta),
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[0.3, 0.8]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];

    let slices = block_slices(&family, &block_states);
    let terms = family
        .psi_second_order_terms(&block_states, &derivative_blocks, 0, 1)
        .expect("timewiggle scorewarp psi second-order path should evaluate")
        .expect("psi second-order terms should exist");
    assert!(terms.objective_psi_psi.is_finite());
    assert_eq!(terms.score_psi_psi.len(), slices.total);
    assert!(terms.score_psi_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_psi_operator.is_some());
}

#[test]
fn timewiggle_marginal_psi_hessian_directional_returns_finite_matrix() {
    let score_runtime = test_deviation_runtime();
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let (time_wiggle_knots, time_wiggle_degree, time_wiggle_ncols) = standard_test_time_wiggle();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_exit: DesignMatrix::from(array![[0.0, 0.0, 0.0, 0.0, 0.0]]),
        design_derivative_exit: DesignMatrix::from(array![[1.0, 0.0, 0.0, 0.0, 0.0]]),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: Some(time_wiggle_knots),
        time_wiggle_degree: Some(time_wiggle_degree),
        time_wiggle_ncols,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.08, -0.03, 0.02, -0.01],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: array![0.2],
            eta: array![0.2],
        },
        ParameterBlockState {
            beta: Array1::zeros(score_runtime.basis_dim()),
            eta: Array1::zeros(1),
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ];
    let slices = block_slices(&family, &block_states);
    let mut d_beta_flat = Array1::zeros(slices.total);
    d_beta_flat[slices.time.start] = 0.07;
    d_beta_flat[slices.time.start + 1] = -0.03;
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;
    if let Some(h_range) = slices.score_warp.as_ref() {
        d_beta_flat[h_range.start] = 0.02;
    }

    let slices = block_slices(&family, &block_states);
    let hess_dir = family
        .psi_hessian_directional_derivative(&block_states, &derivative_blocks, 0, &d_beta_flat)
        .expect("timewiggle scorewarp psi-Hessian directional path should evaluate")
        .expect("psi-Hessian directional derivative should exist");
    assert_eq!(hess_dir.nrows(), slices.total);
    assert_eq!(hess_dir.ncols(), slices.total);
    assert!(hess_dir.iter().all(|value| value.is_finite()));
}

#[test]
fn sigma_exact_joint_psi_terms_returns_analytic_terms() {
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_beta = array![0.2];
    let sigma = 0.65;
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: Some(sigma),
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_beta.clone(),
        },
    ];
    let specs = vec![
        dummy_blockspec(1),
        dummy_blockspec(marginal_design.ncols()),
        dummy_blockspec(1),
    ];

    let terms = family
        .sigma_exact_joint_psi_terms(&block_states, &specs)
        .expect("sigma psi terms should evaluate analytically")
        .expect("sigma psi terms should be present");
    assert!(terms.objective_psi.is_finite());
    assert_eq!(
        terms.score_psi.len(),
        block_slices(&family, &block_states).total
    );
    assert!(terms.score_psi.iter().all(|value| value.is_finite()));
    assert!(terms.hessian_psi_operator.is_some());
}

#[test]
fn censored_rows_still_reject_invalid_time_derivative() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
            1, 1,
        )))),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
            1, 1,
        )))),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::ones((1, 1)),
        )),
        offset_entry: Arc::new(Array1::zeros(1)),
        offset_exit: Arc::new(Array1::zeros(1)),
        derivative_offset_exit: Arc::new(Array1::from_elem(1, 1e-6)),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
    ];

    let err = family
        .compute_row_primary_gradient_hessian_uncached(0, &block_states)
        .expect_err("censored rows must still enforce the time-derivative domain");
    assert!(
        err.contains("monotonicity violated at row 0"),
        "unexpected error: {err}"
    );
}

fn standard_test_time_wiggle() -> (Array1<f64>, usize, usize) {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let degree = 3usize;
    let ncols = time_wiggle_basis_ncols(&knots, degree).expect("timewiggle basis width");
    (knots, degree, ncols)
}

#[test]
fn exact_newton_evaluation_propagates_invalid_rows() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
    ];

    let err = family
        .evaluate(&block_states)
        .expect_err("invalid rows must abort exact-newton evaluation");
    assert!(
        err.contains("monotonicity violated"),
        "unexpected error: {err}"
    );
}

#[test]
fn time_constraints_use_exact_derivative_guard_rows() {
    let family = SurvivalMarginalSlopeFamily {
        n: 2,
        event: Arc::new(array![0.0, 1.0]),
        weights: Arc::new(array![1.0, 1.0]),
        z: Arc::new(array![0.0, 0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
            2, 2,
        )))),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::zeros((
            2, 2,
        )))),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0, 2.0], [3.0, 4.0]],
        )),
        offset_entry: Arc::new(Array1::zeros(2)),
        offset_exit: Arc::new(Array1::zeros(2)),
        derivative_offset_exit: Arc::new(array![0.25, 0.5]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((2, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((2, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        time_linear_constraints: time_derivative_guard_constraints(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0, 1.0],
                [1.0, -1.0]
            ])),
            &array![0.0, 0.25],
            1.0,
        )
        .expect("time derivative guard constraints"),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
            [0.0, 0.0],
            [0.0, 0.0]
        ])),
        offset: Array1::zeros(2),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let constraints = family
        .block_linear_constraints(&[], 0, &spec)
        .expect("constraint lookup")
        .expect("time constraints");
    let row_scale = 2.0_f64.sqrt();
    let expected_a = array![
        [1.0 / row_scale, 1.0 / row_scale],
        [1.0 / row_scale, -1.0 / row_scale]
    ];
    assert_eq!(constraints.a.dim(), expected_a.dim());
    for (got, want) in constraints.a.iter().zip(expected_a.iter()) {
        assert_relative_eq!(*got, *want, epsilon = 1e-12);
    }
    let expected_b = array![1.0 / row_scale, 0.75 / row_scale];
    assert_eq!(constraints.b.dim(), expected_b.dim());
    for (got, want) in constraints.b.iter().zip(expected_b.iter()) {
        assert_relative_eq!(*got, *want, epsilon = 1e-12);
    }
}

#[test]
fn time_block_constraints_synthesize_qd1_rows_when_stored_constraints_missing() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0, 0.0]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let constraints = family
        .block_linear_constraints(&[], 0, &spec)
        .expect("synthesized constraints")
        .expect("qd1 row");
    assert_eq!(constraints.a, array![[1.0, 0.0]]);
    assert_eq!(constraints.b, array![0.0]);
}

#[test]
fn time_block_max_feasible_step_uses_synthesized_qd1_rows() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let states = vec![ParameterBlockState {
        beta: array![0.4, 7.0],
        eta: array![0.0],
    }];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![-1.0, -10.0])
        .expect("synthesized qd1 step ceiling")
        .expect("binding synthesized qd1 row");

    assert_relative_eq!(alpha, 0.398, epsilon = 1e-12);
}

#[test]
fn coupled_qd1_guard_limits_time_step_before_post_update_projection() {
    let constraints = time_derivative_guard_constraints(
        &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0, 1.0]])),
        &array![0.0],
        1.0,
    )
    .expect("time derivative guard constraints")
    .expect("coupled row");
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1.0,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0, 1.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![0.0]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        time_linear_constraints: Some(constraints),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let states = vec![ParameterBlockState {
        beta: array![0.6, 0.6],
        eta: array![0.0],
    }];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![-1.0, 0.0])
        .expect("coupled qd1 step limit")
        .expect("binding coupled row");

    assert_relative_eq!(alpha, 0.199, epsilon = 1e-12);
}

#[test]
fn timewiggle_tail_constraints_are_part_of_time_block_feasibility() {
    let structural = time_derivative_guard_constraints(
        &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.0, 0.0
        ]])),
        &array![1e-6],
        1e-6,
    )
    .expect("time derivative guard constraints");
    let constraints = append_timewiggle_tail_nonnegative_constraints(structural, 3, 2)
        .expect("combined constraints")
        .expect("time constraints");

    assert_eq!(constraints.a, Array2::<f64>::eye(3));
    assert_eq!(constraints.b, Array1::<f64>::zeros(3));
}

#[test]
fn timewiggle_tail_step_is_clipped_before_it_can_flip_derivative() {
    assert!(file!().ends_with(".rs"));
    let constraints = append_timewiggle_tail_nonnegative_constraints(None, 2, 1)
        .expect("tail constraints")
        .expect("time constraints");
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[0.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        time_linear_constraints: Some(constraints),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 1,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let states = vec![ParameterBlockState {
        beta: array![0.0, 0.5],
        eta: array![0.0],
    }];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![0.0, -1.0])
        .expect("timewiggle tail step ceiling")
        .expect("negative tail step should be bounded");
    assert_relative_eq!(alpha, 0.4975, epsilon = 1e-12);
}

#[test]
fn time_block_post_update_rejects_infeasible_beta_instead_of_projecting() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            1.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        time_linear_constraints: append_timewiggle_tail_nonnegative_constraints(
            time_derivative_guard_constraints(
                &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0, 0.0]])),
                &array![1e-6],
                1e-6,
            )
            .expect("time derivative guard constraints"),
            2,
            1,
        )
        .expect("combined time constraints"),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 1,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0, 0.0]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![0.0, 0.0],
                eta: array![0.0],
            }],
            0,
            &spec,
            array![-0.3, -0.2],
        )
        .expect_err("post-update must not project an infeasible time beta");
    assert!(
        err.contains("violates monotonicity") && err.contains("proposed"),
        "unexpected error message: {err}"
    );
}

/// Regression guard for the phantom-multiplier failure mode: if a
/// hand-constructed family omits the derivative-guard rows from
/// `time_linear_constraints`, post-update must not repair the proposed
/// time beta by a hidden projection. The KKT system needs those rows; a
/// missing-row violation is an error, not a convergence mechanism.
#[test]
fn time_block_post_update_rejects_qd1_when_no_linear_constraints() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
        // qd1 = 1.0 · β[0] + 0.0 · β[1] + offset
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        // offset = derivative_guard exactly (the production setup).
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0, 0.0]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let current = array![0.4, 7.0];
    // qd1 at current = 1.0·0.4 + 0.0·7.0 + 1e-6 ≈ 0.4 (feasible)
    // qd1 at proposed = 1.0·-0.6 + 0.0·-3.0 + 1e-6 ≈ -0.6 (infeasible)
    let proposed = array![-0.6, -3.0];
    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: current.clone(),
                eta: array![0.0],
            }],
            0,
            &spec,
            proposed.clone(),
        )
        .expect_err("missing qd1 constraints must not be repaired by projection");
    assert!(
        err.contains("violates monotonicity")
            && err.contains("proposed")
            && err.contains("time_linear_constraints"),
        "unexpected error message: {err}"
    );
}

/// When `current` already violates the qd1 monotonicity, the projection
/// surfaces a structured error (with the row index and qd1 value)
/// instead of silently returning a still-infeasible β. This is the
/// invariant the score_warp / link_dev projection enforces; the time
/// block now matches.
#[test]
fn time_block_post_update_errors_when_current_violates_qd1() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![1e-6]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let spec = ParameterBlockSpec {
        name: "time_surface".to_string(),
        design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[0.0]])),
        offset: Array1::zeros(1),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    // current qd1 = -1.0 + 1e-6 < guard → infeasible.
    let err = family
        .post_update_block_beta(
            &[ParameterBlockState {
                beta: array![-1.0],
                eta: array![0.0],
            }],
            0,
            &spec,
            array![0.5],
        )
        .expect_err("infeasible current must surface an error");
    assert!(
        err.contains("violates monotonicity") && err.contains("row 0"),
        "unexpected error message: {err}"
    );
}

#[test]
fn time_block_feasible_step_stays_inside_derivative_guard() {
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![0.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-4,
        design_entry: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
            0.0, 0.0
        ]])),
        design_derivative_exit: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            array![[1.0, 0.0]],
        )),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![0.2]),
        marginal_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
            Array2::zeros((1, 0)),
        )),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: time_derivative_guard_constraints(
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[1.0, 0.0]])),
            &array![0.2],
            1e-4,
        )
        .expect("time derivative guard constraints"),
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let states = vec![
        ParameterBlockState {
            beta: array![0.0, 0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0],
        },
    ];
    let alpha = family
        .max_feasible_step_size(&states, 0, &array![-1.0, 0.0])
        .expect("time step ceiling")
        .expect("time step should be bounded");
    // Starting at beta=[0,0] with derivative q' = design·beta + offset = 0.2,
    // far above the 1e-4 guard. Stepping along [-1, 0] drives q' toward the
    // guard; the largest feasible α satisfies -α + 0.2 = 1e-4, i.e. α ≈ 0.1999,
    // and the shared `feasible_step_fraction` applies a 0.995 boundary backoff
    // so the post-step iterate stays *strictly* interior (slack > 0).
    assert!(
        alpha > 0.0 && alpha < 1.0,
        "expected an interior step, got {alpha}"
    );
    let feasible = &states[0].beta + &(array![-1.0, 0.0] * alpha);
    let slack = family
        .time_linear_constraints
        .as_ref()
        .expect("constraints")
        .a
        .row(0)
        .dot(&feasible)
        - family
            .time_linear_constraints
            .as_ref()
            .expect("constraints")
            .b[0];
    assert!(
        slack > 0.0,
        "boundary-backed-off step must stay strictly interior; slack={slack}"
    );
}

#[test]
fn mixed_blockwise_exact_newton_preserves_sparse_block_hessians() {
    let family = SurvivalMarginalSlopeFamily {
        n: 2,
        event: Arc::new(array![1.0, 0.0]),
        weights: Arc::new(array![1.0, 0.8]),
        z: Arc::new(array![0.1, -0.2].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.6]])),
        design_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![[0.9], [0.5]])),
        design_derivative_exit: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [1.0]])),
        offset_entry: Arc::new(array![0.0, 0.0]),
        offset_exit: Arc::new(array![0.0, 0.0]),
        derivative_offset_exit: Arc::new(array![0.05, 0.05]),
        marginal_design: sparse_design(&array![[1.0, 0.0], [0.0, 1.0]]),
        logslope_design: DesignMatrix::Dense(DenseDesignMatrix::from(array![[1.0], [0.5]])),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.4],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.2, -0.1],
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: array![0.3],
            eta: array![0.3, 0.3],
        },
    ];

    let eval = family
        .evaluate_blockwise_exact_newton(&block_states)
        .expect("mixed exact-newton evaluation");

    assert!(matches!(
        &eval.blockworking_sets[0],
        BlockWorkingSet::ExactNewton {
            hessian: SymmetricMatrix::Dense(_),
            ..
        }
    ));
    assert!(matches!(
        &eval.blockworking_sets[1],
        BlockWorkingSet::ExactNewton {
            hessian: SymmetricMatrix::Dense(_) | SymmetricMatrix::Sparse(_),
            ..
        }
    ));
    assert!(matches!(
        &eval.blockworking_sets[2],
        BlockWorkingSet::ExactNewton {
            hessian: SymmetricMatrix::Dense(_),
            ..
        }
    ));
}

/// Pairwise oracle for survival marginal-slope outer-HVP cross-checking.
///
/// Builds a small survival fixture (n=1, no time-wiggle, no link-dev,
/// no score-warp) with two ψ axes spanning the marginal and logslope
/// blocks. Calls `psi_second_order_terms(i, j)` for every (i, j) pair,
/// probes the operator returned via
/// `psi_hessian_directional_derivative_operator` at each axis i with
/// a fixed direction `d_beta_flat`, and writes the result to
/// `/tmp/survival_pairwise_oracle.json`.
///
/// Verification anchor for the operator-form survival outer-HVP that
/// pairs the existing ψψ likelihood-side directional pullback with
/// the directional trace-correction helpers being added under the
/// CTN HVP Phase 2 work. The oracle uses only existing public APIs.
#[test]
fn survival_marginal_slope_pairwise_oracle_dumps_json() {
    let marginal_design = array![[0.7, -0.2]];
    let marginal_beta = array![0.35, -0.1];
    let logslope_design = array![[1.0, 0.4]];
    let logslope_beta = array![0.2, -0.05];
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.15].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::ones((1, 1))),
        offset_entry: Arc::new(array![0.05]),
        offset_exit: Arc::new(array![0.15]),
        derivative_offset_exit: Arc::new(array![0.9]),
        marginal_design: DesignMatrix::from(marginal_design.clone()),
        logslope_design: DesignMatrix::from(logslope_design.clone()),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let block_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![0.0],
        },
        ParameterBlockState {
            beta: marginal_beta.clone(),
            eta: marginal_design.dot(&marginal_beta),
        },
        ParameterBlockState {
            beta: logslope_beta.clone(),
            eta: logslope_design.dot(&logslope_beta),
        },
    ];
    // Two ψ axes: one in marginal (block 1), one in logslope (block 2).
    let derivative_blocks = vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[1.0, -0.4]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            array![[0.6, 0.3]],
            Array2::zeros((2, 2)),
            None,
            None,
            None,
            None,
        )],
    ];

    let psi_dim = 2usize;

    let mut pair_records = Vec::new();
    for i in 0..psi_dim {
        for j in 0..psi_dim {
            let terms = family
                .psi_second_order_terms(&block_states, &derivative_blocks, i, j)
                .expect("survival pairwise call ok")
                .expect("pairwise returns Some for valid i,j");
            let g_inf = terms
                .score_psi_psi
                .iter()
                .fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(
                terms.objective_psi_psi.is_finite(),
                "objective_psi_psi non-finite at (i={i},j={j})"
            );
            assert!(
                terms.score_psi_psi.iter().all(|v| v.is_finite()),
                "score_psi_psi non-finite at (i={i},j={j})"
            );
            pair_records.push(serde_json::json!({
                "i": i,
                "j": j,
                "a": terms.objective_psi_psi,
                "g_inf": g_inf,
                "g": terms.score_psi_psi.to_vec(),
                "operator_present": terms.hessian_psi_psi_operator.is_some(),
            }));
        }
    }

    let slices = block_slices(&family, &block_states);
    let mut d_beta_flat = Array1::zeros(slices.total);
    d_beta_flat[slices.time.start] = 0.07;
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.marginal.start + 1] = -0.02;
    d_beta_flat[slices.logslope.start] = -0.04;
    d_beta_flat[slices.logslope.start + 1] = 0.03;

    let mut op_records = Vec::new();
    for i in 0..psi_dim {
        let op = family
            .psi_hessian_directional_derivative_operator_with_options(
                &block_states,
                &derivative_blocks,
                i,
                &d_beta_flat,
                &BlockwiseFitOptions::default(),
            )
            .expect("operator call ok")
            .expect("operator returns Some");
        let dim = op.dim();
        let mut probes: Vec<(&'static str, Array1<f64>)> = Vec::new();
        let mut e0 = Array1::<f64>::zeros(dim);
        if dim > 0 {
            e0[0] = 1.0;
        }
        probes.push(("e0", e0));
        let scale = 1.0 / (dim.max(1) as f64).sqrt();
        probes.push(("uniform", Array1::from_elem(dim, scale)));
        let alt: Array1<f64> = (0..dim)
            .map(|k| if k % 2 == 0 { 0.5 } else { -0.3 })
            .collect();
        probes.push(("alt", alt));

        let mut probe_outputs = Vec::new();
        for (label, v) in probes.iter() {
            let out = op.mul_vec(v);
            let v_inf = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let out_inf = out.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(
                out.iter().all(|x| x.is_finite()),
                "operator output non-finite at axis {i} probe {label}"
            );
            probe_outputs.push(serde_json::json!({
                "label": label,
                "v_inf": v_inf,
                "out_inf": out_inf,
            }));
        }
        op_records.push(serde_json::json!({
            "i": i,
            "dim": dim,
            "probes": probe_outputs,
        }));
    }

    let payload = serde_json::json!({
        "version": 1,
        "fixture": "survival_marginal_slope:n=1,no_wiggle,no_warp,psi_dim=2",
        "psi_dim": psi_dim,
        "p_total": slices.total,
        "pair_records": pair_records,
        "operator_records": op_records,
    });

    let path = std::path::Path::new("/tmp/survival_pairwise_oracle.json");
    std::fs::write(path, serde_json::to_string_pretty(&payload).unwrap())
        .expect("write oracle JSON");
    eprintln!(
        "[oracle] wrote {} pair records, {} operator records to {}",
        psi_dim * psi_dim,
        psi_dim,
        path.display()
    );
}

/// Closed-form test family with `gaussian_frailty_sd` set so the sigma-aware
/// joint psi paths fire. Same pseudo-random row layout as
/// `make_closed_form_test_family` so half-row subsamples remain
/// representative.
fn make_sigma_aware_closed_form_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    let mut family = make_closed_form_test_family(n);
    family.gaussian_frailty_sd = Some(0.6);
    family
}

fn rel_diff_array1_survival(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let mut max = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs() / b[i].abs().max(1.0);
        if d > max {
            max = d;
        }
    }
    max
}

fn rel_diff_array2_survival(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let mut max = 0.0f64;
    for ((i, j), &av) in a.indexed_iter() {
        let bv = b[[i, j]];
        let d = (av - bv).abs() / bv.abs().max(1.0);
        if d > max {
            max = d;
        }
    }
    max
}

#[test]
fn survival_sigma_psi_terms_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(0), dummy_blockspec(0)];

    let baseline = family
        .sigma_exact_joint_psi_terms(&states, &specs)
        .expect("baseline psi terms")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_full)
        .expect("with full mask")
        .expect("some");

    let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
        / baseline.objective_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi, &baseline.score_psi);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_sigma_psi_terms_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(0), dummy_blockspec(0)];

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_half)
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psi_terms_with_options(&states, &specs, &opts_raw)
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi;
    let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let exp_score = &raw.score_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi, &exp_score);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_sigma_psi_second_order_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let baseline = family
        .sigma_exact_joint_psisecond_order_terms(&states)
        .expect("baseline")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_full)
        .expect("with full mask")
        .expect("some");

    let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
        / baseline.objective_psi_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi_psi, &baseline.score_psi_psi);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_sigma_psi_second_order_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_half)
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psisecond_order_terms_with_options(&states, &opts_raw)
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi_psi;
    let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let exp_score = &raw.score_psi_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi_psi, &exp_score);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_sigma_psihessian_directional_derivative_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let slices = block_slices(&family, &states);
    let d_beta_flat = Array1::<f64>::zeros(slices.total);

    let baseline = family
        .sigma_exact_joint_psihessian_directional_derivative(&states, &d_beta_flat)
        .expect("baseline")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(
            &states,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full")
        .expect("some");

    let rel = rel_diff_array2_survival(&with_full, &baseline);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn survival_sigma_psihessian_directional_derivative_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_sigma_aware_closed_form_test_family(n);
    let states = closed_form_block_states(&family, 0.25);
    let slices = block_slices(&family, &states);
    let d_beta_flat = Array1::<f64>::zeros(slices.total);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(
            &states,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .sigma_exact_joint_psihessian_directional_derivative_with_options(
            &states,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp = &raw * factor;
    let rel = rel_diff_array2_survival(&scaled, &exp);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

/// Multi-row test family with non-empty marginal/logslope designs but no
/// score_warp / link_dev / time_wiggle. Drives the rigid block path of
/// `psi_terms_inner` so we can subsample-check Horvitz-Thompson scaling.
fn make_block_psi_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_entry: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_exit: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
    );
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
    // Single-column marginal/logslope designs with row-varying entries.
    let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
    });
    let logslope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
    });
    SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(marginal_design),
        logslope_design: DesignMatrix::from(logslope_design),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    }
}

fn block_psi_test_block_states(
    family: &SurvivalMarginalSlopeFamily,
    m_beta: f64,
    g_beta: f64,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family.logslope_design.to_dense().to_owned();
    let m_eta = m_design.dot(&array![m_beta]);
    let g_eta = g_design.dot(&array![g_beta]);
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: array![m_beta],
            eta: m_eta,
        },
        ParameterBlockState {
            beta: array![g_beta],
            eta: g_eta,
        },
    ]
}

/// Derivative blocks with a single ψ on the marginal block (block_idx=1).
/// `x_psi` has shape (n, 1) so the test family gets a per-row psi map.
fn block_psi_test_marginal_derivative_blocks(
    n: usize,
) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
    let x_psi = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
    });
    vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
        Vec::new(),
    ]
}

/// Derivative blocks with one ψ on marginal (block 1) and one on logslope
/// (block 2), so second-order terms can mix.
fn block_psi_test_dual_derivative_blocks(
    n: usize,
) -> Vec<Vec<crate::custom_family::CustomFamilyBlockPsiDerivative>> {
    let x_psi_m = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.4 + 0.3 * (((i * 41 + 13) % n) as f64) / (n as f64)
    });
    let x_psi_g = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 43 + 17) % n) as f64) / (n as f64)
    });
    vec![
        Vec::new(),
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi_m,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
        vec![crate::custom_family::CustomFamilyBlockPsiDerivative::new(
            None,
            x_psi_g,
            Array2::zeros((1, 1)),
            None,
            None,
            None,
            None,
        )],
    ]
}

#[test]
fn survival_psi_terms_inner_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);

    let baseline = family
        .psi_terms_inner(&states, &derivative_blocks, 0, None)
        .expect("baseline psi terms")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_full)
        .expect("with full mask")
        .expect("some");

    let obj_rel = ((with_full.objective_psi - baseline.objective_psi)
        / baseline.objective_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi, &baseline.score_psi);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_psi_terms_inner_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_half)
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts_raw)
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi;
    let obj_rel = ((scaled.objective_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective_psi rel {}", obj_rel);
    let exp_score = &raw.score_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi, &exp_score);
    assert!(score_rel < 1e-12, "score_psi rel {}", score_rel);
}

#[test]
fn survival_psi_terms_inner_batched_matches_per_axis() {
    // The batched first-order ψ row pass shares the per-row primary
    // gradient/Hessian across axes; this asserts it produces the same
    // ExactNewtonJointPsiTerms (objective, score, Hessian-operator action)
    // as K serial calls to `psi_terms_inner_with_options`.
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);
    let opts = BlockwiseFitOptions::default();

    let per_axis_0 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts)
        .expect("per-axis 0")
        .expect("some");
    let per_axis_1 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 1, None, &opts)
        .expect("per-axis 1")
        .expect("some");

    let batched = family
        .psi_terms_inner_batched_with_options(&states, &derivative_blocks, &[0, 1], None, &opts)
        .expect("batched")
        .expect("batched simple-spatial path returned None unexpectedly");
    assert_eq!(batched.len(), 2, "batched should yield one term per axis");

    let per_axis = [&per_axis_0, &per_axis_1];
    for (i, (lhs, rhs)) in per_axis.iter().zip(batched.iter()).enumerate() {
        let obj_rel =
            ((rhs.objective_psi - lhs.objective_psi) / lhs.objective_psi.abs().max(1.0)).abs();
        assert!(
            obj_rel < 1e-12,
            "axis {i} objective_psi rel {obj_rel} (per-axis={}, batched={})",
            lhs.objective_psi,
            rhs.objective_psi,
        );
        let score_rel = rel_diff_array1_survival(&rhs.score_psi, &lhs.score_psi);
        assert!(score_rel < 1e-12, "axis {i} score_psi rel {score_rel}");

        let op_a = lhs
            .hessian_psi_operator
            .as_ref()
            .expect("per-axis Hessian operator");
        let op_b = rhs
            .hessian_psi_operator
            .as_ref()
            .expect("batched Hessian operator");
        assert_eq!(op_a.dim(), op_b.dim(), "axis {i} operator dim mismatch");
        let dim = op_a.dim();
        let probe = Array1::from_shape_fn(dim, |j| {
            ((j as i64 * 37 + 11).rem_euclid(7)) as f64 * 0.1 - 0.3
        });
        let a = op_a.mul_vec(&probe);
        let b = op_b.mul_vec(&probe);
        let op_rel = rel_diff_array1_survival(&a, &b);
        assert!(op_rel < 1e-12, "axis {i} Hessian-action rel {op_rel}");
    }
}

#[test]
fn survival_psi_terms_inner_batched_subsample_matches_per_axis() {
    // Same equivalence under a half-row Horvitz-Thompson mask, exercising
    // the per-row weight branch of the batched fast path.
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let mut opts = BlockwiseFitOptions::default();
    opts.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::from_uniform_inclusion_mask(
        even_mask, n, 0xC0FFEE,
    )));

    let per_axis_0 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 0, None, &opts)
        .expect("per-axis 0")
        .expect("some");
    let per_axis_1 = family
        .psi_terms_inner_with_options(&states, &derivative_blocks, 1, None, &opts)
        .expect("per-axis 1")
        .expect("some");

    let batched = family
        .psi_terms_inner_batched_with_options(&states, &derivative_blocks, &[0, 1], None, &opts)
        .expect("batched")
        .expect("batched simple-spatial path returned None under subsample");
    assert_eq!(batched.len(), 2);

    let per_axis = [&per_axis_0, &per_axis_1];
    for (i, (lhs, rhs)) in per_axis.iter().zip(batched.iter()).enumerate() {
        let obj_rel =
            ((rhs.objective_psi - lhs.objective_psi) / lhs.objective_psi.abs().max(1.0)).abs();
        assert!(
            obj_rel < 1e-12,
            "axis {i} subsample objective_psi rel {obj_rel}"
        );
        let score_rel = rel_diff_array1_survival(&rhs.score_psi, &lhs.score_psi);
        assert!(
            score_rel < 1e-12,
            "axis {i} subsample score_psi rel {score_rel}"
        );

        let op_a = lhs.hessian_psi_operator.as_ref().unwrap();
        let op_b = rhs.hessian_psi_operator.as_ref().unwrap();
        let dim = op_a.dim();
        let probe = Array1::from_shape_fn(dim, |j| {
            ((j as i64 * 41 + 5).rem_euclid(11)) as f64 * 0.07 - 0.4
        });
        let a = op_a.mul_vec(&probe);
        let b = op_b.mul_vec(&probe);
        let op_rel = rel_diff_array1_survival(&a, &b);
        assert!(
            op_rel < 1e-12,
            "axis {i} subsample Hessian-action rel {op_rel}"
        );
    }
}

#[test]
fn survival_psi_second_order_terms_inner_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

    let baseline = family
        .psi_second_order_terms_inner(&states, &derivative_blocks, 0, 1, None)
        .expect("baseline psi second-order")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_second_order_terms_inner_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            None,
            &opts_full,
        )
        .expect("with full")
        .expect("some");

    let obj_rel = ((with_full.objective_psi_psi - baseline.objective_psi_psi)
        / baseline.objective_psi_psi.abs().max(1.0))
    .abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let score_rel = rel_diff_array1_survival(&with_full.score_psi_psi, &baseline.score_psi_psi);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_psi_second_order_terms_inner_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_dual_derivative_blocks(n);

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_second_order_terms_inner_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            None,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_second_order_terms_inner_with_options(
            &states,
            &derivative_blocks,
            0,
            1,
            None,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp_obj = factor * raw.objective_psi_psi;
    let obj_rel = ((scaled.objective_psi_psi - exp_obj) / exp_obj.abs().max(1.0)).abs();
    assert!(obj_rel < 1e-12, "objective rel {}", obj_rel);
    let exp_score = &raw.score_psi_psi * factor;
    let score_rel = rel_diff_array1_survival(&scaled.score_psi_psi, &exp_score);
    assert!(score_rel < 1e-12, "score rel {}", score_rel);
}

#[test]
fn survival_psi_hessian_directional_derivative_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let baseline = family
        .psi_hessian_directional_derivative(&states, &derivative_blocks, 0, &d_beta_flat)
        .expect("baseline psi-Hessian directional derivative")
        .expect("some");

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full mask")
        .expect("some");

    let rel = rel_diff_array2_survival(&with_full, &baseline);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn survival_psi_hessian_directional_derivative_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");

    let factor = n as f64 / m as f64;
    let exp = &raw * factor;
    let rel = rel_diff_array2_survival(&scaled, &exp);
    assert!(rel < 1e-12, "drift rel {}", rel);
}

#[test]
fn survival_psi_workspace_hessian_directional_derivative_is_operator_and_matches_dense() {
    let n = 40usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let specs = vec![dummy_blockspec(0), dummy_blockspec(1), dummy_blockspec(1)];
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let dense = family
        .psi_hessian_directional_derivative_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
        .expect("dense drift")
        .expect("dense drift available");
    let workspace = family
        .exact_newton_joint_psi_workspace_with_options(
            &states,
            &specs,
            &derivative_blocks,
            &BlockwiseFitOptions::default(),
        )
        .expect("workspace")
        .expect("workspace available");
    let result = workspace
        .hessian_directional_derivative(0, &d_beta_flat)
        .expect("workspace drift")
        .expect("workspace drift available");

    let crate::reml_contracts::DriftDerivResult::Operator(op) = result else {
        panic!("survival psi drift should use operator representation");
    };
    assert_eq!(op.dim(), dense.nrows());
    let operator_dense = op.to_dense();
    let rel = rel_diff_array2_survival(&operator_dense, &dense);
    assert!(rel < 1e-12, "operator/dense drift rel {rel}");
}

#[test]
fn survival_psi_hessian_directional_derivative_operator_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let baseline = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
        .expect("baseline operator")
        .expect("some");
    let baseline_dense = baseline.to_dense();

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full mask")
        .expect("some");
    let with_full_dense = with_full.to_dense();

    let rel = rel_diff_array2_survival(&with_full_dense, &baseline_dense);
    assert!(rel < 1e-12, "operator drift rel {}", rel);
}

#[test]
fn survival_psi_hessian_directional_derivative_operator_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 200usize;
    let family = make_block_psi_test_family(n);
    let states = block_psi_test_block_states(&family, 0.15, 0.25);
    let derivative_blocks = block_psi_test_marginal_derivative_blocks(n);
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled")
        .expect("some");
    let scaled_dense = scaled.to_dense();

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .psi_hessian_directional_derivative_operator_with_options(
            &states,
            &derivative_blocks,
            0,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw")
        .expect("some");
    let raw_dense = raw.to_dense();

    let factor = n as f64 / m as f64;
    let exp = &raw_dense * factor;
    let rel = rel_diff_array2_survival(&scaled_dense, &exp);
    assert!(rel < 1e-12, "operator drift rel {}", rel);
}

// ── Phase 7: joint-Hessian flex-no-wiggle directional-derivative
// operator subsample tests. The flex-no-wiggle helpers are the path
// taken by the joint-Hessian workspace's `directional_derivative_operator`
// when `effective_flex_active(states)` is true and timewiggle is off.
// We exercise the helpers directly through a flex-active fixture so the
// outer subsample threading is verified end-to-end.

fn make_flex_no_wiggle_test_family(n: usize) -> SurvivalMarginalSlopeFamily {
    let score_runtime = test_deviation_runtime();
    let event: Array1<f64> =
        Array1::from_iter((0..n).map(|i| if (i * 31 + 7) % 5 >= 3 { 1.0 } else { 0.0 }));
    let weights: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + ((i * 13 + 4) % 5) as f64 * 0.1));
    let z: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -1.0 + 2.0 * (((i * 17 + 5) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_entry: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| -0.4 + 0.7 * (((i * 11 + 3) % n) as f64 + 0.5) / (n as f64)),
    );
    let offset_exit: Array1<f64> = Array1::from_iter(
        (0..n).map(|i| 0.1 + 0.6 * (((i * 19 + 7) % n) as f64 + 0.5) / (n as f64)),
    );
    let derivative_offset_exit: Array1<f64> =
        Array1::from_iter((0..n).map(|i| 0.5 + 0.05 * ((i * 23 + 1) % 3) as f64));
    let marginal_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.3 + 0.4 * (((i * 29 + 11) % n) as f64) / (n as f64)
    });
    let logslope_design = Array2::from_shape_fn((n, 1), |(i, _)| {
        0.2 + 0.5 * (((i * 37 + 9) % n) as f64) / (n as f64)
    });
    SurvivalMarginalSlopeFamily {
        n,
        event: Arc::new(event),
        weights: Arc::new(weights),
        z: Arc::new(z.insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((n, 0))),
        design_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((n, 0))),
        offset_entry: Arc::new(offset_entry),
        offset_exit: Arc::new(offset_exit),
        derivative_offset_exit: Arc::new(derivative_offset_exit),
        marginal_design: DesignMatrix::from(marginal_design),
        logslope_design: DesignMatrix::from(logslope_design),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime),
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    }
}

fn flex_no_wiggle_test_block_states(
    family: &SurvivalMarginalSlopeFamily,
) -> Vec<ParameterBlockState> {
    let n = family.n;
    let m_design = family.marginal_design.to_dense().to_owned();
    let g_design = family.logslope_design.to_dense().to_owned();
    let m_beta = 0.15_f64;
    let g_beta = 0.25_f64;
    let m_eta = m_design.dot(&array![m_beta]);
    let g_eta = g_design.dot(&array![g_beta]);
    let score_dim = family
        .score_warp
        .as_ref()
        .map(|w| w.basis_dim())
        .unwrap_or(0);
    vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(n),
        },
        ParameterBlockState {
            beta: array![m_beta],
            eta: m_eta,
        },
        ParameterBlockState {
            beta: array![g_beta],
            eta: g_eta,
        },
        ParameterBlockState {
            beta: Array1::zeros(score_dim),
            eta: Array1::zeros(n),
        },
    ]
}

/// Joint-β assembly oracle (#1133): the Step-6 GPU joint-β pullback
/// `pullback_step6_joint_beta` (`Σ_rows Jᵀ g_p`, `Σ_rows Jᵀ H_p J`) must
/// reproduce the production CPU joint-β assembly
/// (`evaluate_exact_newton_joint_dynamic_q_dense`) bit-for-bit when fed the
/// real per-row primary jet `(g_p, H_p)` and the row's dense
/// primary→coefficient Jacobian `J`.
///
/// This is the CPU correctness anchor the on-device contraction is checked
/// against: it exercises the *exact* code path the device entry points
/// (`gpu::try_survival_flex_{gradient,hvp,dense_hessian}`) fold their
/// `SurvivalFlexStep6RowPullback` outputs through, so the joint-β assembly is
/// verified end-to-end against the family math (not just the synthetic
/// hand-built jets in `gpu::step6_tests`). The device-substrate jet builder
/// (Steps 2–5) still needs CUDA hardware; the host-side fold (Step 6) is
/// proven here.
///
/// Validity is confined to the GPU-eligible regime — `effective_flex_active`
/// && `!flex_timewiggle_active()` — where the q0/q1/qd1 primaries are *affine*
/// in β so the `∂²primary/∂β²` second-order design curvature (the `d2q*` terms
/// in `accumulate_dynamic_q_core_hessian`) vanishes and the pure `Jᵀ H_p J`
/// form is exact. This is the same gate the dispatcher applies before routing
/// to the GPU entry points (`joint_eval.rs`).
#[test]
fn step6_joint_beta_pullback_matches_cpu_dense_assembly_flex_no_wiggle() {
    use crate::families::survival::marginal_slope::gpu::{
        SurvivalFlexStep5RowOutputs, SurvivalFlexStep6RowPullback, pullback_step6_joint_beta,
    };

    let n = 24usize;
    let family = make_flex_no_wiggle_test_family(n);
    let states = flex_no_wiggle_test_block_states(&family);
    // Confirm we are in the GPU-eligible regime so the affine `Jᵀ H_p J`
    // identity holds (no `d2q` second-order design curvature).
    assert!(family.effective_flex_active(&states).unwrap());
    assert!(!family.flex_timewiggle_active());

    let slices = block_slices(&family, &states);
    let p = slices.total;
    let primary = flex_primary_slices(&family);
    let r = primary.total;
    let identity_blocks = flex_identity_block_pairs(&primary, &slices);

    // Production CPU joint-β assembly: the reference the Step-6 pullback must
    // reproduce.
    let (cpu_nll, cpu_grad, cpu_hess) = family
        .evaluate_exact_newton_joint_dynamic_q_dense(&states)
        .expect("cpu dense joint assembly");

    // Build the per-row Step-5 primary jet + dense primary→coefficient
    // Jacobian `J`, then fold through the Step-6 pullback.
    let mut q_geom = SurvivalMarginalSlopeDynamicRow::empty_workspace();
    let mut step5_rows: Vec<SurvivalFlexStep5RowOutputs> = Vec::with_capacity(n);
    let mut jacobians: Vec<Vec<f64>> = Vec::with_capacity(n);
    for row in 0..n {
        family
            .row_dynamic_q_geometry_into(row, &states, &mut q_geom)
            .expect("row geometry");
        let (row_nll, f_pi, f_pipi) = family
            .compute_row_flex_primary_gradient_hessian_exact(row, &states, &q_geom, &primary)
            .expect("row primary jet");

        // Step-5 primary outputs in the `pullback_step6_joint_beta` convention:
        //   * nll      : the CPU sweep accumulates `-= row_nll`, so the Step-6
        //                row nll carries the sign (`grad`/`H` fold the same J).
        //   * grad g_p : the CPU sweep accumulates `-= primary_gradient·dq`, so
        //                the raw-J convention requires `g_p = -f_pi`.
        //   * hess H_p : the CPU sweep accumulates `+= primary_hessian·dq·dq`
        //                (no sign flip), so `H_p = f_pipi` with the raw J.
        let g_p: Vec<f64> = f_pi.iter().map(|&v| -v).collect();
        let h_p: Vec<f64> = f_pipi.iter().copied().collect();
        assert_eq!(g_p.len(), r);
        assert_eq!(h_p.len(), r * r);

        // Dense row Jacobian `J[a*p + j] = ∂ primary_a / ∂ β_j` (raw, unsigned).
        let mut jac = vec![0.0_f64; r * p];

        // Core primaries q0/q1/qd1 (indices 0,1,2): time + marginal design rows.
        let dq_time = [&q_geom.dq0_time, &q_geom.dq1_time, &q_geom.dqd1_time];
        let dq_marginal = [
            &q_geom.dq0_marginal,
            &q_geom.dq1_marginal,
            &q_geom.dqd1_marginal,
        ];
        for q_idx in 0..3 {
            for (c, &v) in dq_time[q_idx].iter().enumerate() {
                jac[q_idx * p + slices.time.start + c] = v;
            }
            for (c, &v) in dq_marginal[q_idx].iter().enumerate() {
                jac[q_idx * p + slices.marginal.start + c] = v;
            }
        }
        // logslope primary g (index 3): the logslope design row.
        {
            let chunk = family
                .logslope_design
                .try_row_chunk(row..row + 1)
                .expect("logslope row");
            let g_row = chunk.row(0);
            for (c, &v) in g_row.iter().enumerate() {
                jac[3 * p + slices.logslope.start + c] = v;
            }
        }
        // Identity flex blocks (score_warp / link_dev): primary coordinate `a`
        // maps to joint coefficient `a` with unit Jacobian.
        for (primary_range, joint_range) in &identity_blocks {
            for local in 0..primary_range.len() {
                let a = primary_range.start + local;
                jac[a * p + joint_range.start + local] = 1.0;
            }
        }
        // Influence absorber primary `o_infl` (single scalar): maps to its γ
        // coefficients through the residualized design row `Z̃[row,:]`.
        if let (Some(infl_primary), Some(infl_joint)) = (primary.infl, slices.influence.as_ref()) {
            let z_tilde = family
                .influence_absorber
                .as_ref()
                .expect("influence Z̃ present when infl primary present");
            let z_row = z_tilde.row(row);
            for (i, &z) in z_row.iter().enumerate() {
                jac[infl_primary * p + infl_joint.start + i] = z;
            }
        }

        step5_rows.push(SurvivalFlexStep5RowOutputs {
            row_nll: -row_nll,
            grad: g_p,
            hess: h_p,
        });
        jacobians.push(jac);
    }

    let pullbacks: Vec<SurvivalFlexStep6RowPullback<'_>> = step5_rows
        .iter()
        .zip(jacobians.iter())
        .map(|(po, jac)| SurvivalFlexStep6RowPullback {
            primary: po,
            jacobian: jac.as_slice(),
        })
        .collect();

    let (step6_nll, step6_grad, step6_hess) =
        pullback_step6_joint_beta(&pullbacks, p).expect("step6 joint-β pullback");

    // nll + gradient + dense Hessian all match the production CPU assembly.
    assert_close(step6_nll, cpu_nll, 1e-9, "joint nll");
    assert_eq!(step6_grad.len(), cpu_grad.len());
    for j in 0..p {
        assert_close(step6_grad[j], cpu_grad[j], 1e-8, &format!("grad[{j}]"));
    }
    assert_eq!(step6_hess.shape(), &[p, p]);
    for a in 0..p {
        for b in 0..p {
            assert_close(
                step6_hess[[a, b]],
                cpu_hess[[a, b]],
                1e-8,
                &format!("hess[{a},{b}]"),
            );
        }
    }

    // The HVP entry point uses the same pullback: H·v from Step-6 must equal
    // the dense CPU Hessian applied to a probe direction.
    let mut v = Array1::<f64>::zeros(p);
    if p > 0 {
        v[0] = 0.37;
    }
    if p > 1 {
        v[p - 1] = -0.21;
    }
    let step6_hv = step6_hess.dot(&v);
    let cpu_hv = cpu_hess.dot(&v);
    for j in 0..p {
        assert_close(step6_hv[j], cpu_hv[j], 1e-8, &format!("Hv[{j}]"));
    }
}

#[test]
fn survival_jointhessian_flex_no_wiggle_operator_subsample_full_equals_unsampled() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 40usize;
    let family = make_flex_no_wiggle_test_family(n);
    let states = flex_no_wiggle_test_block_states(&family);
    assert!(family.effective_flex_active(&states).unwrap());
    assert!(!family.flex_timewiggle_active());
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let baseline = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &BlockwiseFitOptions::default(),
        )
        .expect("baseline operator");
    let baseline_dense = baseline.to_dense();

    let mut opts_full = BlockwiseFitOptions::default();
    opts_full.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask((0..n).collect(), n, 0xDEADBEEF),
    ));
    let with_full = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &opts_full,
        )
        .expect("with full mask");
    let with_full_dense = with_full.to_dense();

    let rel = rel_diff_array2_survival(&with_full_dense, &baseline_dense);
    assert!(
        rel < 1e-10,
        "joint Hessian flex-no-wiggle dH operator drift rel {}",
        rel
    );
}

#[test]
fn survival_jointhessian_flex_no_wiggle_operator_subsample_half_scales_correctly() {
    use crate::outer_subsample::OuterScoreSubsample;
    let n = 40usize;
    let family = make_flex_no_wiggle_test_family(n);
    let states = flex_no_wiggle_test_block_states(&family);
    assert!(family.effective_flex_active(&states).unwrap());
    assert!(!family.flex_timewiggle_active());
    let slices = block_slices(&family, &states);
    let mut d_beta_flat = Array1::<f64>::zeros(slices.total);
    d_beta_flat[slices.marginal.start] = 0.05;
    d_beta_flat[slices.logslope.start] = -0.04;

    let even_mask: Vec<usize> = (0..n).filter(|i| i % 2 == 0).collect();
    let m = even_mask.len();

    let mut opts_half = BlockwiseFitOptions::default();
    opts_half.outer_score_subsample = Some(Arc::new(
        OuterScoreSubsample::from_uniform_inclusion_mask(even_mask.clone(), n, 0xCAFE),
    ));
    let scaled = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &opts_half,
        )
        .expect("scaled");
    let scaled_dense = scaled.to_dense();

    let mut opts_raw = BlockwiseFitOptions::default();
    opts_raw.outer_score_subsample = Some(Arc::new(OuterScoreSubsample::with_uniform_weight(
        even_mask, m, 0, 1.0,
    )));
    let raw = family
        .exact_newton_joint_hessian_directional_derivative_operator_flex_no_wiggle_with_options(
            &states,
            &d_beta_flat,
            &opts_raw,
        )
        .expect("raw");
    let raw_dense = raw.to_dense();

    let factor = n as f64 / m as f64;
    let exp = &raw_dense * factor;
    let rel = rel_diff_array2_survival(&scaled_dense, &exp);
    assert!(
        rel < 1e-10,
        "joint Hessian flex-no-wiggle dH operator HT rel {}",
        rel
    );
}

#[test]
fn survival_auto_subsample_phase_counter_field_initializes_to_zero() {
    let family = make_closed_form_test_family(8);
    assert_eq!(
        family
            .auto_subsample_phase_counter
            .load(std::sync::atomic::Ordering::SeqCst),
        0,
        "fresh family must start at Phase-1 step 0"
    );
    assert!(
        family
            .auto_subsample_last_rho
            .lock()
            .expect("auto_subsample_last_rho mutex poisoned")
            .is_none(),
        "fresh family must have no recorded last-rho proxy"
    );
}

// ────────────────────────────────────────────────────────────────────
// Block 10 parity — third T_uv[r] and fourth Q_uv[r,s] contractions:
// crate::families::survival::marginal_slope::gpu::cpu_oracle_third_contraction /
// cpu_oracle_fourth_contraction must match the family CPU paths
// (row_flex_primary_third_contracted_exact / _fourth_contracted_exact)
// to within 5e-8 absolute / 5e-7 relative.
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct B10ParityFixture {
    label: &'static str,
    event: f64,
    weight: f64,
    z: f64,
    q0: f64,
    q1: f64,
    qd1: f64,
    score_eta: f64,
    h_scale: f64,
    w_scale: f64,
}

const B10_PARITY_FIXTURES: &[B10ParityFixture] = &[
    B10ParityFixture {
        label: "event_nonzero_warps",
        event: 1.0,
        weight: 0.75,
        z: -0.2,
        q0: -0.4,
        q1: 0.6,
        qd1: 0.85,
        score_eta: 0.32,
        h_scale: 0.05,
        w_scale: 0.04,
    },
    B10ParityFixture {
        label: "censored_left_tail",
        event: 0.0,
        weight: 1.35,
        z: -1.15,
        q0: -1.35,
        q1: -0.9,
        qd1: 0.42,
        score_eta: -0.55,
        h_scale: -0.035,
        w_scale: 0.025,
    },
    B10ParityFixture {
        label: "near_boundary_derivative",
        event: 1.0,
        weight: 0.2,
        z: 0.95,
        q0: 0.15,
        q1: 1.05,
        qd1: 0.08,
        score_eta: 0.72,
        h_scale: 0.015,
        w_scale: -0.02,
    },
    B10ParityFixture {
        label: "zero_warp_edge",
        event: 0.0,
        weight: 0.9,
        z: 0.0,
        q0: -0.05,
        q1: 0.25,
        qd1: 1.2,
        score_eta: 0.0,
        h_scale: 0.0,
        w_scale: 0.0,
    },
];

fn b10_flex_family_for_parity(
    fixture: B10ParityFixture,
) -> (SurvivalMarginalSlopeFamily, Vec<ParameterBlockState>) {
    let score_runtime = test_deviation_runtime();
    let link_runtime = test_deviation_runtime();
    let family = SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![fixture.event]),
        weights: Arc::new(array![fixture.weight]),
        z: Arc::new(array![fixture.z].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        offset_entry: Arc::new(array![fixture.q0]),
        offset_exit: Arc::new(array![fixture.q1]),
        derivative_offset_exit: Arc::new(array![fixture.qd1]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: Some(score_runtime.clone()),
        link_dev: Some(link_runtime.clone()),
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    };
    let h_dim = score_runtime.basis_dim();
    let w_dim = link_runtime.basis_dim();
    let h_beta: Array1<f64> = (0..h_dim)
        .map(|k| fixture.h_scale * ((k as f64 + 1.0).sin()))
        .collect::<Vec<_>>()
        .into();
    let w_beta: Array1<f64> = (0..w_dim)
        .map(|k| fixture.w_scale * ((k as f64 + 1.0).cos()))
        .collect::<Vec<_>>()
        .into();
    let block_states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![fixture.score_eta],
        },
        ParameterBlockState {
            beta: h_beta,
            eta: Array1::zeros(1),
        },
        ParameterBlockState {
            beta: w_beta,
            eta: Array1::zeros(1),
        },
    ];
    (family, block_states)
}

fn b10_direction_set(p: usize) -> Vec<(&'static str, Array1<f64>)> {
    let mixed: Array1<f64> = (0..p)
        .map(|k| 0.1 + 0.07 * ((k as f64 + 1.7).sin()))
        .collect::<Vec<_>>()
        .into();
    let alternating: Array1<f64> = (0..p)
        .map(|k| if k % 2 == 0 { 0.16 } else { -0.11 })
        .collect::<Vec<_>>()
        .into();
    let mut qd_axis = Array1::zeros(p);
    if p > 2 {
        qd_axis[2] = 1.0;
    }
    let zero = Array1::zeros(p);
    vec![
        ("mixed", mixed),
        ("alternating", alternating),
        ("qd_axis", qd_axis),
        ("zero", zero),
    ]
}

fn b10_pack_base(
    base: &SurvivalFlexTimepointExact,
) -> crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBase {
    crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBase {
        eta: base.eta,
        chi: base.chi,
        d: base.d,
        eta_u: base.eta_u.to_vec(),
        eta_uv: base.eta_uv.iter().copied().collect(),
        chi_u: base.chi_u.to_vec(),
        chi_uv: base.chi_uv.iter().copied().collect(),
        d_u: base.d_u.to_vec(),
        d_uv: base.d_uv.iter().copied().collect(),
    }
}

fn b10_pack_dir(
    ext: &SurvivalFlexTimepointDirectionalExact,
) -> crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointDirectional {
    crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointDirectional {
        eta_uv_dir: ext.eta_uv_dir.iter().copied().collect(),
        eta_u_dir: ext.eta_u_dir.to_vec(),
        chi_u_dir: ext.chi_u_dir.to_vec(),
        chi_uv_dir: ext.chi_uv_dir.iter().copied().collect(),
        d_u_dir: ext.d_u_dir.to_vec(),
        d_uv_dir: ext.d_uv_dir.iter().copied().collect(),
    }
}

fn b10_pack_bi(
    bi: &SurvivalFlexTimepointBiDirectionalExact,
) -> crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBiDirectional {
    crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10TimepointBiDirectional {
        eta_uv_uv: bi.eta_uv_uv.iter().copied().collect(),
        chi_uv_uv: bi.chi_uv_uv.iter().copied().collect(),
        d_uv_uv: bi.d_uv_uv.iter().copied().collect(),
    }
}

/// Test-only accessor that materialises the per-row timepoint jets
/// (entry/exit base + per-direction extensions + bidirectional) used
/// to assemble the third- and fourth-order contractions. Drives the
/// Block 10 CPU oracle parity tests below against
/// `row_flex_primary_third_contracted_exact` /
/// `row_flex_primary_fourth_contracted_exact`. Lives in `mod tests`
/// (rather than as a `#[cfg(test)]` method on the parent impl) so the
/// build.rs `#[cfg(test)] on src/ item` ban — which exists to prevent
/// `#[cfg(test)]` being used as a dead-code escape hatch — does not
/// fire on a legitimate test helper. Visibility into the family's
/// private methods is preserved by Rust's child-module visibility rule.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn flex_primary_timepoint_jets_for_test(
    family: &SurvivalMarginalSlopeFamily,
    row: usize,
    block_states: &[ParameterBlockState],
    dir1: &Array1<f64>,
    dir2: Option<&Array1<f64>>,
) -> Result<
    (
        SurvivalFlexTimepointExact,
        SurvivalFlexTimepointExact,
        SurvivalFlexTimepointDirectionalExact,
        SurvivalFlexTimepointDirectionalExact,
        Option<SurvivalFlexTimepointDirectionalExact>,
        Option<SurvivalFlexTimepointDirectionalExact>,
        Option<SurvivalFlexTimepointBiDirectionalExact>,
        Option<SurvivalFlexTimepointBiDirectionalExact>,
        f64,
        usize,
        usize,
    ),
    String,
> {
    family.ensure_scalar_flex_exact_score_geometry("flex_primary_timepoint_jets_for_test")?;
    let primary = flex_primary_slices(family);
    let q_geom = family.row_dynamic_q_geometry(row, block_states)?;
    let q0 = q_geom.q0;
    let q1 = q_geom.q1;
    let qd1 = q_geom.qd1;
    let g = block_states[2].eta[row];
    let beta_h = family.flex_score_beta(block_states)?;
    let beta_w = family.flex_link_beta(block_states)?;

    let (a0, d0) = family.solve_row_survival_intercept_with_slot(
        q0,
        g,
        beta_h,
        beta_w,
        Some((row, SurvivalInterceptSlotKind::Entry)),
    )?;
    let (a1, d1) = family.solve_row_survival_intercept_with_slot(
        q1,
        g,
        beta_h,
        beta_w,
        Some((row, SurvivalInterceptSlotKind::Exit)),
    )?;

    let entry_cached = family.build_cached_partition(&primary, a0, g, beta_h, beta_w)?;
    let exit_cached = family.build_cached_partition(&primary, a1, g, beta_h, beta_w)?;

    let entry_base = family.compute_survival_timepoint_exact_from_cached(
        row,
        &primary,
        q0,
        primary.q0,
        a0,
        g,
        d0,
        beta_h,
        beta_w,
        0.0,
        false,
        &entry_cached,
    )?;
    let exit_base = family.compute_survival_timepoint_exact_from_cached(
        row,
        &primary,
        q1,
        primary.q1,
        a1,
        g,
        d1,
        beta_h,
        beta_w,
        0.0,
        true,
        &exit_cached,
    )?;

    let entry_ext1 = family.compute_survival_timepoint_directional_exact_from_cached(
        row,
        &primary,
        q0,
        primary.q0,
        a0,
        g,
        beta_h,
        beta_w,
        &entry_cached,
        dir1,
        false,
    )?;
    let exit_ext1 = family.compute_survival_timepoint_directional_exact_from_cached(
        row,
        &primary,
        q1,
        primary.q1,
        a1,
        g,
        beta_h,
        beta_w,
        &exit_cached,
        dir1,
        true,
    )?;

    let (entry_ext2, exit_ext2, entry_bi, exit_bi) = if let Some(dir2_arr) = dir2 {
        let entry_ext2 = family.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            beta_h,
            beta_w,
            &entry_cached,
            dir2_arr,
            false,
        )?;
        let exit_ext2 = family.compute_survival_timepoint_directional_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            beta_h,
            beta_w,
            &exit_cached,
            dir2_arr,
            true,
        )?;
        let entry_bi = family.compute_survival_timepoint_bidirectional_exact_from_cached(
            row,
            &primary,
            q0,
            primary.q0,
            a0,
            g,
            beta_h,
            beta_w,
            &entry_cached,
            dir1,
            dir2_arr,
        )?;
        let exit_bi = family.compute_survival_timepoint_bidirectional_exact_from_cached(
            row,
            &primary,
            q1,
            primary.q1,
            a1,
            g,
            beta_h,
            beta_w,
            &exit_cached,
            dir1,
            dir2_arr,
        )?;
        (
            Some(entry_ext2),
            Some(exit_ext2),
            Some(entry_bi),
            Some(exit_bi),
        )
    } else {
        (None, None, None, None)
    };

    Ok((
        entry_base,
        exit_base,
        entry_ext1,
        exit_ext1,
        entry_ext2,
        exit_ext2,
        entry_bi,
        exit_bi,
        qd1,
        primary.qd1,
        primary.total,
    ))
}

fn b10_assert_parity(actual: &[f64], expected: &ndarray::Array2<f64>, label: &str) {
    let p = expected.nrows();
    assert_eq!(expected.ncols(), p, "{label}: expected non-square");
    assert_eq!(actual.len(), p * p, "{label}: flat length mismatch");
    for u in 0..p {
        for v in 0..p {
            let got = actual[u * p + v];
            let want = expected[[u, v]];
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1.0);
            assert!(
                abs <= 5e-8 || rel <= 5e-7,
                "{label}[{u},{v}]: got={got:.17e} want={want:.17e} abs={abs:.3e} rel={rel:.3e}",
            );
        }
    }
}

fn b10_third_oracle_from_family(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    dir: &Array1<f64>,
) -> Vec<f64> {
    let (entry_base, exit_base, entry_ext, exit_ext, _e2, _x2, _eb, _xb, qd1, qd1_idx, p_total) =
        flex_primary_timepoint_jets_for_test(family, 0, block_states, dir, None)
            .expect("third-contraction jets");

    let entry_b = b10_pack_base(&entry_base);
    let exit_b = b10_pack_base(&exit_base);
    let entry_d = b10_pack_dir(&entry_ext);
    let exit_d = b10_pack_dir(&exit_ext);
    let dir_vec: Vec<f64> = dir.to_vec();
    let inputs = crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10ThirdInputs {
        p: p_total,
        qd1_index: qd1_idx,
        qd1,
        w: family.weights[0],
        d: family.event[0],
        dir: &dir_vec,
        entry_base: &entry_b,
        exit_base: &exit_b,
        entry_ext: &entry_d,
        exit_ext: &exit_d,
    };
    crate::families::survival::marginal_slope::gpu::cpu_oracle_third_contraction(&inputs)
        .expect("oracle third")
}

fn b10_fourth_oracle_from_family(
    family: &SurvivalMarginalSlopeFamily,
    block_states: &[ParameterBlockState],
    dir_u: &Array1<f64>,
    dir_v: &Array1<f64>,
) -> Vec<f64> {
    let (
        entry_base,
        exit_base,
        entry_ext1,
        exit_ext1,
        entry_ext2,
        exit_ext2,
        entry_bi,
        exit_bi,
        qd1,
        qd1_idx,
        p_total,
    ) = flex_primary_timepoint_jets_for_test(family, 0, block_states, dir_u, Some(dir_v))
        .expect("fourth-contraction bi-jets");
    let entry_ext2 = entry_ext2.expect("entry ext2");
    let exit_ext2 = exit_ext2.expect("exit ext2");
    let entry_bi = entry_bi.expect("entry bi");
    let exit_bi = exit_bi.expect("exit bi");

    let entry_b = b10_pack_base(&entry_base);
    let exit_b = b10_pack_base(&exit_base);
    let entry_d1 = b10_pack_dir(&entry_ext1);
    let entry_d2 = b10_pack_dir(&entry_ext2);
    let exit_d1 = b10_pack_dir(&exit_ext1);
    let exit_d2 = b10_pack_dir(&exit_ext2);
    let entry_bi_p = b10_pack_bi(&entry_bi);
    let exit_bi_p = b10_pack_bi(&exit_bi);
    let dir_u_v: Vec<f64> = dir_u.to_vec();
    let dir_v_v: Vec<f64> = dir_v.to_vec();
    let inputs = crate::families::survival::marginal_slope::gpu::SurvivalFlexBlock10FourthInputs {
        p: p_total,
        qd1_index: qd1_idx,
        qd1,
        w: family.weights[0],
        d: family.event[0],
        dir_u: &dir_u_v,
        dir_v: &dir_v_v,
        entry_base: &entry_b,
        exit_base: &exit_b,
        entry_ext_u: &entry_d1,
        entry_ext_v: &entry_d2,
        exit_ext_u: &exit_d1,
        exit_ext_v: &exit_d2,
        entry_bi: &entry_bi_p,
        exit_bi: &exit_bi_p,
    };
    crate::families::survival::marginal_slope::gpu::cpu_oracle_fourth_contraction(&inputs)
        .expect("oracle fourth")
}

#[test]
fn block10_cpu_oracle_third_contraction_matches_family_shared_fixtures() {
    for &fixture in B10_PARITY_FIXTURES {
        let (family, block_states) = b10_flex_family_for_parity(fixture);
        let primary = flex_primary_slices(&family);
        for (dir_label, dir) in b10_direction_set(primary.total) {
            let expected = family
                .row_flex_primary_third_contracted_exact(0, &block_states, &dir)
                .unwrap_or_else(|err| {
                    panic!(
                        "{} / {dir_label}: cpu third contraction failed: {err}",
                        fixture.label
                    )
                });
            let actual = b10_third_oracle_from_family(&family, &block_states, &dir);
            assert_eq!(actual.len(), expected.nrows() * expected.ncols());
            b10_assert_parity(&actual, &expected, fixture.label);
        }
    }
}

#[test]
fn block10_cpu_oracle_fourth_contraction_matches_family_shared_fixtures() {
    for &fixture in B10_PARITY_FIXTURES {
        let (family, block_states) = b10_flex_family_for_parity(fixture);
        let primary = flex_primary_slices(&family);
        let dirs = b10_direction_set(primary.total);
        let pairs = [(0usize, 0usize), (0, 1), (1, 0), (1, 2), (2, 3), (3, 0)];
        for &(u_idx, v_idx) in &pairs {
            let (u_label, dir_u) = &dirs[u_idx];
            let (v_label, dir_v) = &dirs[v_idx];
            let expected = family
                .row_flex_primary_fourth_contracted_exact(0, &block_states, dir_u, dir_v)
                .unwrap_or_else(|err| {
                    panic!(
                        "{} / {u_label}->{v_label}: cpu fourth contraction failed: {err}",
                        fixture.label
                    )
                });
            let actual = b10_fourth_oracle_from_family(&family, &block_states, dir_u, dir_v);
            assert_eq!(actual.len(), expected.nrows() * expected.ncols());
            b10_assert_parity(&actual, &expected, fixture.label);
        }
    }
}

/// Build a one-row time-block family whose single derivative-design row has
/// the given coefficient and offset, so we can drive
/// `validate_time_qd1_feasible` with a controlled raw `qd1` and constraint
/// row scaling. Mirrors the field layout of the other in-module fixtures.
fn make_time_guard_family(deriv_coeff: f64, deriv_offset: f64) -> SurvivalMarginalSlopeFamily {
    SurvivalMarginalSlopeFamily {
        n: 1,
        event: Arc::new(array![1.0]),
        weights: Arc::new(array![1.0]),
        z: Arc::new(array![0.0].insert_axis(Axis(1))),
        score_covariance: unit_score_covariance(),
        gaussian_frailty_sd: None,
        derivative_guard: 1e-6,
        design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
        design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
        design_derivative_exit: DesignMatrix::from(array![[deriv_coeff]]),
        offset_entry: Arc::new(array![0.0]),
        offset_exit: Arc::new(array![0.0]),
        derivative_offset_exit: Arc::new(array![deriv_offset]),
        marginal_design: DesignMatrix::from(Array2::zeros((1, 1))),
        logslope_design: DesignMatrix::from(array![[1.0]]),
        logslope_surface_ranges: empty_logslope_surface_ranges(),
        score_warp: None,
        link_dev: None,
        influence_absorber: None,
        time_linear_constraints: None,
        time_wiggle_knots: None,
        time_wiggle_degree: None,
        time_wiggle_ncols: 0,
        intercept_warm_starts: None,
        auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
        auto_subsample_last_rho: Arc::new(Mutex::new(None)),
    }
}

/// Regression for #379: a full-rank PH marginal-slope fit aborted because
/// `validate_time_qd1_feasible` rejected a constrained time-block iterate
/// that the inequality-constrained active-set Newton solver considered
/// feasible. The solver only guarantees primal feasibility to
/// `ACTIVE_SET_PRIMAL_FEASIBILITY_TOL` in the *scaled* constraint-row units
/// produced by `time_derivative_guard_constraints` (row scale
/// `max(||design_row||, |guard - offset|, 1)`). On a design row with a
/// large norm, a scaled slack of ~1e-8 maps to a raw `qd1` shortfall of
/// ~1e-5 below the 1e-6 guard — exactly the overshoot in the issue. The
/// validator must accept that boundary iterate while still rejecting a
/// genuine monotonicity divergence.
#[test]
fn validate_time_qd1_accepts_scaled_boundary_overshoot_rejects_real_violation() {
    let guard = 1e-6_f64;

    // Boundary case: design row norm ~6000, so the solver's 1e-8 scaled
    // primal tolerance permits a raw qd1 shortfall of ~6e-5 below the guard.
    let deriv_coeff = 6000.0_f64;
    let family = make_time_guard_family(deriv_coeff, 0.0);
    // Choose beta so qd1 = guard - 6e-5 (raw overshoot ~6e-5, as observed).
    let target_qd1 = guard - 6e-5;
    let beta = array![target_qd1 / deriv_coeff];
    // Confirm the fixture actually reproduces the issue's regime: the raw
    // qd1 trips the historic 256·eps guard band (so the *old* validator
    // would have rejected this iterate and aborted the whole fit).
    assert!(
        super::survival_derivative_guard_violated(target_qd1, guard),
        "fixture must reproduce the sub-guard raw overshoot the old validator rejected",
    );
    // Scaled violation = 6e-5 / max(6000, |guard|, 1) = 1e-8, below the
    // solver-consistent feasibility band, so the iterate must be accepted.
    family
        .validate_time_qd1_feasible(&beta, "proposed")
        .expect("solver-feasible boundary iterate must be accepted");

    // Genuine violation: unit-scale row, qd1 driven far below the guard.
    let bad_family = make_time_guard_family(1.0, 0.0);
    let bad_beta = array![-0.5]; // qd1 = -0.5, scaled violation ~0.5 >> band
    let err = bad_family
        .validate_time_qd1_feasible(&bad_beta, "proposed")
        .expect_err("a true monotonicity divergence must still hard-error");
    assert!(
        err.contains("violates monotonicity"),
        "expected a monotonicity error, got: {err}",
    );
}

/// Fill every one of the 15 cross-block matrices of a
/// `BlockHessianAccumulator` with distinct, deterministic values keyed by
/// (block-pair, local-row, local-col) so that a transposed or mis-placed
/// block is impossible to miss in the dense/operator parity assertions.
fn fill_block_hessian_accumulator(
    p_t: usize,
    p_m: usize,
    p_g: usize,
    p_h: usize,
    p_w: usize,
) -> BlockHessianAccumulator {
    let mut acc = BlockHessianAccumulator::new(p_t, p_m, p_g, p_h, p_w, 0);
    // Per-pair base offsets keep the diagonal blocks symmetric (required
    // for a valid Hessian) while making the off-diagonal blocks distinct
    // and asymmetric, so `to_dense` must place each transpose correctly.
    let sym = |m: &mut Array2<f64>, base: f64| {
        let (r, c) = m.dim();
        for i in 0..r {
            for j in 0..c {
                m[[i, j]] = base + (i.min(j) as f64) * 0.5 + (i.max(j) as f64) * 0.25;
            }
        }
    };
    sym(&mut acc.h_tt, 1.0);
    sym(&mut acc.h_mm, 2.0);
    sym(&mut acc.h_gg, 3.0);
    sym(&mut acc.h_hh, 4.0);
    sym(&mut acc.h_ww, 5.0);
    let rect = |m: &mut Array2<f64>, base: f64| {
        let (r, c) = m.dim();
        for i in 0..r {
            for j in 0..c {
                m[[i, j]] = base + (i as f64) * 1.0 + (j as f64) * 0.1;
            }
        }
    };
    rect(&mut acc.h_tm, 10.0);
    rect(&mut acc.h_tg, 20.0);
    rect(&mut acc.h_th, 30.0);
    rect(&mut acc.h_tw, 40.0);
    rect(&mut acc.h_mg, 50.0);
    rect(&mut acc.h_mh, 60.0);
    rect(&mut acc.h_mw, 70.0);
    rect(&mut acc.h_gh, 80.0);
    rect(&mut acc.h_gw, 90.0);
    rect(&mut acc.h_hw, 100.0);
    acc
}

fn full_block_slices(p_t: usize, p_m: usize, p_g: usize, p_h: usize, p_w: usize) -> BlockSlices {
    let time = 0..p_t;
    let marginal = time.end..time.end + p_m;
    let logslope = marginal.end..marginal.end + p_g;
    let score_warp = logslope.end..logslope.end + p_h;
    let link_dev = score_warp.end..score_warp.end + p_w;
    let total = link_dev.end;
    BlockSlices {
        time,
        marginal,
        logslope,
        score_warp: Some(score_warp),
        link_dev: Some(link_dev),
        influence: None,
        total,
    }
}

/// Parity guard for issue #428: the dense and operator block-Hessian
/// assembly are now one storage abstraction with a single scatter, so
/// `BlockHessianOperator` (the operator wrapping the accumulator) must
/// agree *exactly* with the dense scatter — to_dense, matvec, and the
/// bilinear form — across the full five-block layout (time, marginal,
/// logslope, score_warp, link_dev) that the directional-derivative path
/// produces. Any future reintroduction of a second, divergent block
/// layout would break this test.
#[test]
fn block_hessian_dense_operator_parity_all_five_blocks() {
    let (p_t, p_m, p_g, p_h, p_w) = (3usize, 2, 2, 3, 2);
    let slices = full_block_slices(p_t, p_m, p_g, p_h, p_w);
    let acc = fill_block_hessian_accumulator(p_t, p_m, p_g, p_h, p_w);

    // The dense reference scatter (single source of truth).
    let dense = acc.to_dense(&slices);
    assert_eq!(dense.dim(), (slices.total, slices.total));

    // The full block Hessian must be symmetric: each off-diagonal block
    // appears with its transpose at the mirrored location.
    for i in 0..slices.total {
        for j in 0..slices.total {
            assert_relative_eq!(dense[[i, j]], dense[[j, i]], max_relative = 1e-12);
        }
    }

    // Operator densification must equal the dense scatter bit-for-bit:
    // both now route through `BlockHessianAccumulator::to_dense`.
    let op = acc.into_operator(slices.clone());
    let op_dense = op.to_dense();
    assert_eq!(op_dense.dim(), dense.dim());
    for i in 0..slices.total {
        for j in 0..slices.total {
            assert_eq!(
                op_dense[[i, j]],
                dense[[i, j]],
                "operator/dense mismatch at ({i}, {j})",
            );
        }
    }

    // Matvec parity: operator.mul_vec(v) == dense.dot(v) for arbitrary v,
    // including the directional-derivative use (the operator is exactly
    // what the operator-variant directional path returns).
    let v: Array1<f64> =
        Array1::from_iter((0..slices.total).map(|k| 0.7 + (k as f64) * 0.31 - 0.05 * k as f64));
    let u: Array1<f64> = Array1::from_iter((0..slices.total).map(|k| -1.3 + (k as f64) * 0.17));
    let mv = op.mul_vec(&v);
    let dense_mv = dense.dot(&v);
    for k in 0..slices.total {
        assert_relative_eq!(mv[k], dense_mv[k], max_relative = 1e-12);
    }
    // The view-based matvec must match the owned one exactly.
    let mv_view = op.mul_vec_view(v.view());
    for k in 0..slices.total {
        assert_eq!(mv_view[k], mv[k]);
    }

    // Bilinear parity: operator.bilinear(v, u) == vᵀ · dense · u.
    let bil = op.bilinear(&v, &u);
    let dense_bil = v.dot(&dense.dot(&u));
    assert_relative_eq!(bil, dense_bil, max_relative = 1e-12);
}

#[test]
fn zz_diag_failure1_flex_vs_rigid_vs_fdhess() {
    use crate::families::jet_tower::derived_third_contracted;
    // FAILURE 1 fixture row.
    let event = 1.0_f64;
    let weight = 0.75_f64;
    let zr = -0.2_f64;
    let q0 = -0.4_f64;
    let q1 = 0.6_f64;
    let qd1 = 0.85_f64;
    let gv = 0.32_f64;

    let make = |q0: f64, q1: f64, qd1: f64, g: f64| {
        let score_runtime = test_deviation_runtime();
        let link_runtime = test_deviation_runtime();
        let family = SurvivalMarginalSlopeFamily {
            n: 1,
            event: Arc::new(array![event]),
            weights: Arc::new(array![weight]),
            z: Arc::new(array![zr].insert_axis(Axis(1))),
            score_covariance: unit_score_covariance(),
            gaussian_frailty_sd: None,
            derivative_guard: 1e-6,
            design_entry: DesignMatrix::from(Array2::zeros((1, 1))),
            design_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            design_derivative_exit: DesignMatrix::from(Array2::zeros((1, 1))),
            offset_entry: Arc::new(array![q0]),
            offset_exit: Arc::new(array![q1]),
            derivative_offset_exit: Arc::new(array![qd1]),
            marginal_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_design: DesignMatrix::from(Array2::zeros((1, 0))),
            logslope_surface_ranges: empty_logslope_surface_ranges(),
            score_warp: Some(score_runtime.clone()),
            link_dev: Some(link_runtime.clone()),
            influence_absorber: None,
            time_linear_constraints: None,
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            intercept_warm_starts: None,
            auto_subsample_phase_counter: Arc::new(AtomicUsize::new(0)),
            auto_subsample_last_rho: Arc::new(Mutex::new(None)),
        };
        let sd = score_runtime.basis_dim();
        let ld = link_runtime.basis_dim();
        let bs = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![g],
            },
            ParameterBlockState {
                beta: Array1::zeros(sd),
                eta: Array1::zeros(1),
            },
            ParameterBlockState {
                beta: Array1::zeros(ld),
                eta: Array1::zeros(1),
            },
        ];
        (family, bs)
    };

    let (family, bs) = make(q0, q1, qd1, gv);
    let primary = flex_primary_slices(&family);
    let p = primary.total;
    let bidx = [primary.q0, primary.q1, primary.qd1, primary.g];

    // Flex Hessian at a primary point (q0,q1,qd1,g).
    let flex_hess = |q0: f64, q1: f64, qd1: f64, g: f64| -> Array2<f64> {
        let (fam, bs) = make(q0, q1, qd1, g);
        let qg = fam.row_dynamic_q_geometry(0, &bs).unwrap();
        let pr = flex_primary_slices(&fam);
        let (_, _, h) = fam
            .compute_row_flex_primary_gradient_hessian_exact(0, &bs, &qg, &pr)
            .unwrap();
        h
    };

    let dir4 = [0.7f64, -1.3, 0.5, 0.9];
    let dirvec = {
        let mut v = Array1::zeros(p);
        for (k, &s) in bidx.iter().enumerate() {
            v[s] = dir4[k];
        }
        v
    };

    // Production flex third-contracted.
    let flex_third = family
        .row_flex_primary_third_contracted_exact(0, &bs, &dirvec)
        .unwrap();

    // Independent FD of flex Hessian along dir4 (central, Richardson).
    let fd_dir = |h: f64| -> Array2<f64> {
        let hp = flex_hess(
            q0 + h * dir4[0],
            q1 + h * dir4[1],
            qd1 + h * dir4[2],
            gv + h * dir4[3],
        );
        let hm = flex_hess(
            q0 - h * dir4[0],
            q1 - h * dir4[1],
            qd1 - h * dir4[2],
            gv - h * dir4[3],
        );
        (&hp - &hm) / (2.0 * h)
    };
    let fd_coarse = fd_dir(1e-3);
    let fd_fine = fd_dir(5e-4);
    let fd_rich = (&fd_fine * 4.0 - &fd_coarse) / 3.0;

    // Rigid tower.
    let program = SurvivalMarginalSlopeRigidNllProgram {
        primaries: vec![[q0, q1, qd1, gv]],
        z: vec![zr],
        w: vec![weight],
        d: vec![event],
        probit_scale: family.probit_frailty_scale(),
    };
    let rigid = derived_third_contracted(&program, 0, &dir4).unwrap();

    const THIRD_CONTRACTED_TOL: f64 = 1e-5;
    for (u, &bu) in bidx.iter().enumerate() {
        for (v, &bv) in bidx.iter().enumerate() {
            let flex = flex_third[[bu, bv]];
            let fd = fd_rich[[bu, bv]];
            let rigid = rigid[u][v];

            assert_close(
                flex,
                fd,
                THIRD_CONTRACTED_TOL,
                &format!("third-contracted flex vs FD at primary block ({u}, {v})"),
            );
            assert_close(
                flex,
                rigid,
                THIRD_CONTRACTED_TOL,
                &format!("third-contracted flex vs rigid at primary block ({u}, {v})"),
            );
            assert_close(
                fd,
                rigid,
                THIRD_CONTRACTED_TOL,
                &format!("third-contracted FD vs rigid at primary block ({u}, {v})"),
            );
        }
    }
}
