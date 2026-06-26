//! Tests for the Bernoulli marginal-slope predictor's intrinsic evaluation
//! and the saved score-warp deviation runtime it carries.
//!
//! These exercise gam-crate types ([`BernoulliMarginalSlopePredictor`], the
//! [`crate::families::bms`] deviation runtimes, and [`SavedCompiledFlexBlock`])
//! through crate-internal (`pub(crate)`) seams —
//! [`BernoulliMarginalSlopePredictor::probit_frailty_scale`],
//! [`build_score_warp_deviation_block_from_seed`], and
//! [`empirical_intercept_from_marginal`]. They were swept into `gam-predict`'s
//! lib test module by the #1521 crate split, which is what left that crate's
//! test build red (the seams are not visible across the crate boundary). The
//! code under test is gam-owned, so they are homed back here next to it; the
//! genuinely gam-predict tests (its `point_state`/interval wrappers) stay in
//! `gam-predict`.

use crate::families::bms::{
    EmpiricalZGrid, LatentMeasureKind, empirical_intercept_from_marginal,
};
use crate::inference::model::{SavedCompiledFlexBlock, SavedLatentZNormalization};
use crate::inference::predict_io::{BernoulliMarginalSlopePredictor, PredictInput};
use crate::matrix::DesignMatrix;
use crate::probability::normal_cdf;
use crate::types::InverseLink;
use ndarray::{Array1, Array2, array};

fn saved_runtime_from_deviation_runtime(
    runtime: &crate::families::bms::DeviationRuntime,
) -> SavedCompiledFlexBlock {
    SavedCompiledFlexBlock {
        kernel: crate::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
        breakpoints: runtime.breakpoints().to_vec(),
        basis_dim: runtime.basis_dim(),
        span_c0: runtime
            .span_c0()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c1: runtime
            .span_c1()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c2: runtime
            .span_c2()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c3: runtime
            .span_c3()
            .outer_iter()
            .map(|row| row.to_vec())
            .collect(),
        anchor_correction: None,
        anchor_components: Vec::new(),
    }
}

#[test]
fn bernoulli_marginal_slope_predictor_rejects_structurally_invalid_or_unknown_runtime_kernel() {
    let seed = array![-1.5, -0.2, 0.6, 1.4];
    let prepared = crate::families::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::families::bms::DeviationBlockConfig {
            degree: 3,
            num_internal_knots: 3,
            ..Default::default()
        },
    )
    .expect("production score-warp runtime");
    let production_runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);
    let score_only = BernoulliMarginalSlopePredictor {
        beta_marginal: array![0.8],
        beta_logslope: array![1.6],
        beta_score_warp: Some(array![0.7, -0.4]),
        beta_link_dev: None,
        base_link: InverseLink::Standard(crate::types::StandardLink::Probit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::StandardNormal,
        baseline_marginal: 0.0,
        baseline_logslope: 0.0,
        covariance: None,
        score_warp_runtime: Some(SavedCompiledFlexBlock {
            kernel: "OldQuadrature".to_string(),
            ..production_runtime.clone()
        }),
        // existing field-init order (link_deviation_runtime is the next).
        link_deviation_runtime: None,
        gaussian_frailty_sd: None,
        latent_z_calibration: None,
        latent_z_conditional_calibration: None,
    };
    let err = score_only
        .score_warp_runtime
        .as_ref()
        .unwrap()
        .design(&array![0.0])
        .unwrap_err();
    assert!(err.to_string().contains("DenestedCubicTransport"));

    let err = crate::families::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::families::bms::DeviationBlockConfig {
            degree: 2,
            num_internal_knots: 3,
            ..Default::default()
        },
    )
    .expect_err("non-cubic deviation runtimes should be rejected");
    assert!(err.contains("degree must be 3"));

    let mut structurally_invalid = production_runtime.clone();
    structurally_invalid.span_c0[0].pop();
    let err = structurally_invalid.design(&array![0.0]).unwrap_err();
    assert!(err.to_string().contains("c0 row 0 has width"));

    let cubic = production_runtime;
    assert!(cubic.design(&array![0.0]).is_ok());
}

#[test]
fn saved_anchored_deviation_runtime_local_cubic_reconstructs_values() {
    let seed = array![-2.0, -0.75, 0.0, 1.0, 3.0];
    let prepared = crate::families::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::families::bms::DeviationBlockConfig {
            num_internal_knots: 4,
            ..Default::default()
        },
    )
    .expect("build saved anchored deviation runtime");
    let runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);
    let beta = Array1::from_iter(
        (0..runtime.basis_dim)
            .map(|idx| 0.02 * (idx as f64 + 1.0) * (-1.0_f64).powi(idx as i32)),
    );
    let n_spans = runtime.span_count().expect("span count");
    assert!(n_spans >= 2);
    for span_idx in 0..n_spans {
        let cubic = runtime
            .local_cubic_on_span(&beta, span_idx)
            .expect("local cubic");
        let x_eval = array![cubic.left, 0.5 * (cubic.left + cubic.right), cubic.right];
        let expected = runtime.design(&x_eval).expect("design").dot(&beta);
        let expected_d1 = runtime
            .first_derivative_design(&x_eval)
            .expect("d1 design")
            .dot(&beta);
        for i in 0..x_eval.len() {
            let x = x_eval[i];
            assert!((cubic.evaluate(x) - expected[i]).abs() < 1e-10);
            assert!((cubic.first_derivative(x) - expected_d1[i]).abs() < 1e-10);
            let selected = runtime.local_cubic_at(&beta, x).expect("local cubic at x");
            let expected_span_idx = if i == 0 && span_idx > 0 {
                span_idx - 1
            } else {
                span_idx
            };
            let expected_cubic = runtime
                .local_cubic_on_span(&beta, expected_span_idx)
                .expect("expected local cubic on span");
            assert_eq!(selected.left, expected_cubic.left);
            assert_eq!(selected.right, expected_cubic.right);
        }
    }
}

#[test]
fn saved_anchored_deviation_runtime_design_with_anchor_rows_applies_residual() {
    use crate::families::bms::deviation_runtime::ParametricAnchorBlock;
    use crate::inference::model::{SavedAnchorComponent, SavedAnchorKind};

    let seed = array![-2.0, -0.75, 0.0, 1.0, 3.0];
    let prepared = crate::families::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::families::bms::DeviationBlockConfig {
            num_internal_knots: 4,
            ..Default::default()
        },
    )
    .expect("build saved anchored deviation runtime");
    let mut runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);

    // Inject a non-trivial anchor residual: d = 3 anchor cols,
    // M = arbitrary 3 × basis_dim matrix, identity rotation.
    let d = 3usize;
    let m: Vec<Vec<f64>> = (0..d)
        .map(|i| {
            (0..runtime.basis_dim)
                .map(|j| 0.1 * (i as f64 + 1.0) - 0.05 * (j as f64 + 1.0))
                .collect()
        })
        .collect();
    runtime.anchor_correction = Some(m.clone());
    runtime.anchor_components = vec![SavedAnchorComponent {
        kind: SavedAnchorKind::Parametric {
            block: ParametricAnchorBlock::Marginal,
            ncols: d,
        },
    }];

    let values = array![-1.0, 0.0, 0.5, 2.0];
    let n = values.len();
    let anchor_rows = Array2::from_shape_fn((n, d), |(i, j)| {
        0.3 * (i as f64 + 1.0) - 0.1 * (j as f64 + 1.0)
    });

    let raw = runtime
        .design_uncorrected(&values)
        .expect("uncorrected design");
    let corrected = runtime
        .design_with_anchor_rows(&values, anchor_rows.view())
        .expect("design with anchor rows");

    // Manually compute expected: raw - anchor_rows · M
    let mut m_dense = Array2::<f64>::zeros((d, runtime.basis_dim));
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            m_dense[[i, j]] = v;
        }
    }
    let expected = &raw - &anchor_rows.dot(&m_dense);

    for i in 0..n {
        for j in 0..runtime.basis_dim {
            assert!(
                (corrected[[i, j]] - expected[[i, j]]).abs() < 1e-12,
                "residual-corrected design mismatch at ({i}, {j}): \
                 got {got}, expected {exp}",
                got = corrected[[i, j]],
                exp = expected[[i, j]],
            );
        }
    }

    // anchor_correction_matrix should produce N · M (n × basis_dim) so
    // that raw - correction == corrected, row by row.
    let correction = runtime
        .anchor_correction_matrix(anchor_rows.view())
        .expect("anchor correction matrix")
        .expect("Some correction when residual is present");
    for i in 0..n {
        for j in 0..runtime.basis_dim {
            assert!((raw[[i, j]] - correction[[i, j]] - corrected[[i, j]]).abs() < 1e-12,);
        }
    }
}

#[test]
fn bernoulli_marginal_slope_rigid_gaussian_frailty_uses_scaled_closed_form() {
    let predictor = BernoulliMarginalSlopePredictor {
        beta_marginal: array![0.7],
        beta_logslope: array![-0.4],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(crate::types::StandardLink::Probit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::StandardNormal,
        baseline_marginal: 0.1,
        baseline_logslope: -0.2,
        covariance: None,
        score_warp_runtime: None,
        link_deviation_runtime: None,
        gaussian_frailty_sd: Some(0.8),
        latent_z_calibration: None,
        latent_z_conditional_calibration: None,
    };
    let theta = predictor.theta();
    let input = PredictInput {
        design: DesignMatrix::from(array![[1.0], [1.0]]),
        offset: array![0.0, 0.05],
        design_noise: Some(DesignMatrix::from(array![[1.0], [1.0]])),
        offset_noise: Some(array![0.0, -0.1]),
        auxiliary_scalar: Some(array![-0.3, 1.2]),
        auxiliary_matrix: None,
    };

    let (eta, grad) = predictor
        .final_eta_and_gradient_from_theta(&input, &theta, true)
        .expect("rigid frailty path should evaluate");

    let scale = predictor.probit_frailty_scale();
    let marginal_eta = array![0.8, 0.85];
    let logslope_eta = array![-0.6, -0.7];
    let z = array![-0.3, 1.2];
    for i in 0..eta.len() {
        let sb = scale * logslope_eta[i];
        let c = (1.0 + sb * sb).sqrt();
        let expected_eta = marginal_eta[i] * c + sb * z[i];
        assert!((eta[i] - expected_eta).abs() <= 1e-12);
        let expected_d_marginal = c;
        let expected_d_logslope =
            marginal_eta[i] * scale * scale * logslope_eta[i] / c + scale * z[i];
        let grad = grad.as_ref().expect("gradient should be returned");
        assert!((grad[[i, 0]] - expected_d_marginal).abs() <= 1e-12);
        assert!((grad[[i, 1]] - expected_d_logslope).abs() <= 1e-12);
    }
}

#[test]
fn bernoulli_marginal_slope_predictor_uses_local_empirical_latent_law() {
    let grids = vec![
        EmpiricalZGrid {
            nodes: vec![-1.2, -0.2, 0.7],
            weights: vec![0.45, 0.35, 0.20],
        },
        EmpiricalZGrid {
            nodes: vec![-0.4, 0.6, 2.4],
            weights: vec![0.20, 0.35, 0.45],
        },
    ];
    let predictor = BernoulliMarginalSlopePredictor {
        beta_marginal: array![0.2],
        beta_logslope: array![0.9],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(crate::types::StandardLink::Probit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::LocalEmpirical {
            feature_cols: vec![0],
            input_scales: None,
            centers: vec![vec![-1.0], vec![1.0]],
            grids: grids.clone(),
            top_k: 1,
            bandwidth: 0.25,
            train_row_mixtures: std::sync::Arc::new(Vec::new()),
        },
        baseline_marginal: 0.0,
        baseline_logslope: 0.0,
        covariance: None,
        score_warp_runtime: None,
        link_deviation_runtime: None,
        gaussian_frailty_sd: None,
        latent_z_calibration: None,
        latent_z_conditional_calibration: None,
    };
    let input = PredictInput {
        design: DesignMatrix::from(array![[1.0], [1.0]]),
        offset: array![0.0, 0.0],
        design_noise: Some(DesignMatrix::from(array![[1.0], [1.0]])),
        offset_noise: Some(array![0.0, 0.0]),
        auxiliary_scalar: Some(array![0.0, 0.0]),
        auxiliary_matrix: Some(array![[-1.0], [1.0]]),
    };

    let (eta, _) = predictor
        .final_eta_and_gradient_from_theta(&input, &predictor.theta(), true)
        .expect("local empirical prediction");
    let (chain_eta, deta_dq) = predictor
        .predict_eta_and_q_chain(&input)
        .expect("local empirical q chain");

    for (row, grid) in grids.iter().enumerate() {
        let expected_intercept = empirical_intercept_from_marginal(
            normal_cdf(0.2),
            0.2,
            0.9,
            1.0,
            &grid.nodes,
            &grid.weights,
            None,
        )
        .expect("expected empirical intercept");
        assert!((eta[row] - expected_intercept).abs() <= 1e-10);
        assert!((chain_eta[row] - eta[row]).abs() <= 1e-12);
        assert!(deta_dq[row].is_finite() && deta_dq[row] > 0.0);
    }
}

#[test]
fn bernoulli_marginal_slope_predictor_rejects_nonprobit_base_link_scale() {
    let predictor = BernoulliMarginalSlopePredictor {
        beta_marginal: array![0.7],
        beta_logslope: array![-0.4],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(crate::types::StandardLink::Logit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::StandardNormal,
        baseline_marginal: 0.1,
        baseline_logslope: -0.2,
        covariance: None,
        score_warp_runtime: None,
        link_deviation_runtime: None,
        gaussian_frailty_sd: Some(0.8),
        latent_z_calibration: None,
        latent_z_conditional_calibration: None,
    };
    let theta = predictor.theta();
    let input = PredictInput {
        design: DesignMatrix::from(array![[1.0], [1.0]]),
        offset: array![0.0, 0.05],
        design_noise: Some(DesignMatrix::from(array![[1.0], [1.0]])),
        offset_noise: Some(array![0.0, -0.1]),
        auxiliary_scalar: Some(array![-0.3, 1.2]),
        auxiliary_matrix: None,
    };

    let err = predictor
        .final_eta_and_gradient_from_theta(&input, &theta, true)
        .expect_err("non-probit marginal-slope prediction should be rejected");
    assert!(err.to_string().contains("requires link(type=probit)"));
}

#[test]
fn saved_anchored_deviation_runtime_basis_cubic_matches_basis_column() {
    let seed = array![-2.0, -0.75, 0.0, 1.0, 3.0];
    let prepared = crate::families::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::families::bms::DeviationBlockConfig {
            num_internal_knots: 4,
            ..Default::default()
        },
    )
    .expect("build saved anchored deviation runtime");
    let runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);
    let cubic = runtime.basis_span_cubic(0, 1).expect("basis span cubic");
    let x_eval = array![cubic.left, 0.5 * (cubic.left + cubic.right), cubic.right];
    let design = runtime.design(&x_eval).expect("basis design");
    let d1 = runtime
        .first_derivative_design(&x_eval)
        .expect("basis d1 design");
    for i in 0..x_eval.len() {
        let x = x_eval[i];
        assert!((cubic.evaluate(x) - design[[i, 1]]).abs() < 1e-10);
        assert!((cubic.first_derivative(x) - d1[[i, 1]]).abs() < 1e-10);
        let selected = runtime.basis_cubic_at(1, x).expect("basis cubic at x");
        let expected_span_idx = 0;
        let expected_cubic = runtime
            .basis_span_cubic(expected_span_idx, 1)
            .expect("expected basis span cubic");
        assert_eq!(selected.left, expected_cubic.left);
        assert_eq!(selected.right, expected_cubic.right);
    }
}
