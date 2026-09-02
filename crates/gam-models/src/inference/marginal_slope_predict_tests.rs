//! Tests for the Bernoulli marginal-slope predictor's intrinsic evaluation
//! and the saved score-warp deviation runtime it carries.
//!
//! These exercise gam-crate types ([`BernoulliMarginalSlopePredictor`], the
//! [`crate::bms`] deviation runtimes, and [`SavedCompiledFlexBlock`])
//! through crate-internal (`pub(crate)`) seams —
//! [`BernoulliMarginalSlopePredictor::probit_frailty_scale`],
//! [`build_score_warp_deviation_block_from_seed`], and
//! [`empirical_intercept_from_marginal`]. The #1521 crate split swept them up
//! into the carved `gam-inference` crate's lib test module, which left that
//! crate's test build red: the seams are `pub(crate)` to `gam-models` and are
//! not visible across the crate boundary. The code under test lives here in
//! `gam-models` (`crate::inference::predict_io`, `crate::bms`), so the tests are
//! homed back next to it.

use crate::bms::{EmpiricalZGrid, LatentMeasureKind, empirical_intercept_from_marginal};
use crate::inference::model::{SavedCompiledFlexBlock, SavedLatentZNormalization};
use crate::inference::predict_io::{
    BernoulliMarginalSlopePredictor, LatentConditioningSpan, PredictInput,
};
use crate::probability::normal_cdf;
use gam_linalg::matrix::DesignMatrix;
use gam_problem::types::InverseLink;
use ndarray::{Array1, Array2, array};

fn saved_runtime_from_deviation_runtime(
    runtime: &crate::bms::DeviationRuntime,
) -> SavedCompiledFlexBlock {
    SavedCompiledFlexBlock {
        kernel: crate::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
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
    let prepared = crate::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::bms::DeviationBlockConfig {
            degree: 3,
            num_internal_knots: 3,
            ..Default::default()
        },
    )
    .expect("production score-warp runtime");
    let production_runtime = saved_runtime_from_deviation_runtime(&prepared.runtime);
    let score_only = BernoulliMarginalSlopePredictor {
        beta_marginal: array![0.8],
        beta_slope: array![1.6],
        beta_score_warp: Some(array![0.7, -0.4]),
        beta_link_dev: None,
        base_link: InverseLink::Standard(gam_problem::types::StandardLink::Probit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::StandardNormal,
        baseline_marginal: 0.0,
        baseline_slope: 0.0,
        covariance: None,
        score_warp_runtime: Some(SavedCompiledFlexBlock {
            kernel: "OldQuadrature".to_string(),
            ..production_runtime.clone()
        }),
        // existing field-init order (link_deviation_runtime is the next).
        link_deviation_runtime: None,
        gaussian_frailty_sd: None,
        latent_z_calibration: None,
        latent_conditioning_span: LatentConditioningSpan::PrimaryDesign,
        latent_z_conditional_calibration: None,
    };
    let err = score_only
        .score_warp_runtime
        .as_ref()
        .unwrap()
        .design(&array![0.0])
        .unwrap_err();
    assert!(err.to_string().contains("DenestedCubicTransport"));

    let err = crate::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::bms::DeviationBlockConfig {
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
fn saved_anchored_deviation_runtime_design_with_anchor_rows_applies_residual() {
    use crate::bms::deviation_runtime::ParametricAnchorBlock;
    use crate::inference::model::{SavedAnchorComponent, SavedAnchorKind};

    let seed = array![-2.0, -0.75, 0.0, 1.0, 3.0];
    let prepared = crate::bms::build_score_warp_deviation_block_from_seed(
        &seed,
        &crate::bms::DeviationBlockConfig {
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
        beta_slope: array![-0.4],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(gam_problem::types::StandardLink::Probit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::StandardNormal,
        baseline_marginal: 0.1,
        baseline_slope: -0.2,
        covariance: None,
        score_warp_runtime: None,
        link_deviation_runtime: None,
        gaussian_frailty_sd: Some(0.8),
        latent_z_calibration: None,
        latent_conditioning_span: LatentConditioningSpan::PrimaryDesign,
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
    let slope_eta = array![-0.6, -0.7];
    let z = array![-0.3, 1.2];
    for i in 0..eta.len() {
        let sb = scale * slope_eta[i];
        let c = (1.0 + sb * sb).sqrt();
        let expected_eta = marginal_eta[i] * c + sb * z[i];
        assert!((eta[i] - expected_eta).abs() <= 1e-12);
        let expected_d_marginal = c;
        let expected_d_slope =
            marginal_eta[i] * scale * scale * slope_eta[i] / c + scale * z[i];
        let grad = grad.as_ref().expect("gradient should be returned");
        assert!((grad[[i, 0]] - expected_d_marginal).abs() <= 1e-12);
        assert!((grad[[i, 1]] - expected_d_slope).abs() <= 1e-12);
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
        beta_slope: array![0.9],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(gam_problem::types::StandardLink::Probit),
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
        baseline_slope: 0.0,
        covariance: None,
        score_warp_runtime: None,
        link_deviation_runtime: None,
        gaussian_frailty_sd: None,
        latent_z_calibration: None,
        latent_conditioning_span: LatentConditioningSpan::PrimaryDesign,
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
    let (chain_eta, eta_q) = predictor
        .predict_eta_and_time_tangent(&input, &Array1::ones(2), &Array1::zeros(2))
        .expect("local empirical q tangent");

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
        assert!(eta_q[row].is_finite() && eta_q[row] > 0.0);
    }
}

#[test]
fn bernoulli_marginal_slope_predictor_rejects_nonprobit_base_link_scale() {
    let predictor = BernoulliMarginalSlopePredictor {
        beta_marginal: array![0.7],
        beta_slope: array![-0.4],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(gam_problem::types::StandardLink::Logit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::StandardNormal,
        baseline_marginal: 0.1,
        baseline_slope: -0.2,
        covariance: None,
        score_warp_runtime: None,
        link_deviation_runtime: None,
        gaussian_frailty_sd: Some(0.8),
        latent_z_calibration: None,
        latent_conditioning_span: LatentConditioningSpan::PrimaryDesign,
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

/// The predictor must rebuild the conditional calibration's conditioning span
/// `a(C)` from the block the FIT conditioned on, not from whatever the primary
/// design happens to be (gam#2768).
///
/// The two marginal-slope hosts package the primary design differently: the
/// Bernoulli predictor's IS the marginal design, while the survival predictor's
/// is the q-design `[time | timewiggle | marginal]`, because `q` is a function
/// of follow-up time as well as of the covariates. Conditioning on the time
/// columns would be a different map — i.e. a different model from the one the
/// coefficients were fitted under — so the span is named explicitly and this
/// gate is what holds the two hosts to their own choice.
#[test]
fn conditional_latent_calibration_conditions_on_the_named_design_block() {
    use crate::bms::LatentZConditionalCalibration;
    use ndarray::{Array2, array};

    // `m(C) = 0.25 + 0.5·x`, `v(C) ≡ 0.64` (mean-only correction, so `√v = 0.8`).
    let calibration = LatentZConditionalCalibration {
        mean_coeffs: vec![0.25, 0.5],
        var_coeffs: Vec::new(),
        basis_ncols: 1,
        var_floor: 1e-8,
        homoskedastic_var: 0.64,
        post_mean: 0.0,
        post_sd: 1.0,
        theta1_cov: Array2::<f64>::zeros((2, 2)),
    };
    let x = array![-1.0, 0.0, 2.0];
    let z = array![0.75, 0.25, 1.25];
    // Expected: ζ = (z − (0.25 + 0.5x)) / 0.8.
    let expected = array![
        (0.75 - (0.25 - 0.5)) / 0.8,
        (0.25 - 0.25) / 0.8,
        (1.25 - (0.25 + 1.0)) / 0.8,
    ];

    let predictor_with = |span: LatentConditioningSpan| BernoulliMarginalSlopePredictor {
        beta_marginal: array![0.0],
        beta_slope: array![0.0],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(gam_problem::types::StandardLink::Probit),
        z_column: "z".to_string(),
        latent_z_normalization: SavedLatentZNormalization { mean: 0.0, sd: 1.0 },
        latent_measure: LatentMeasureKind::StandardNormal,
        baseline_marginal: 0.0,
        baseline_slope: 0.0,
        covariance: None,
        score_warp_runtime: None,
        link_deviation_runtime: None,
        gaussian_frailty_sd: None,
        latent_z_calibration: None,
        latent_conditioning_span: span,
        latent_z_conditional_calibration: Some(calibration.clone()),
    };
    let input_from = |design: Array2<f64>| PredictInput {
        design: DesignMatrix::from(design),
        offset: Array1::<f64>::zeros(3),
        design_noise: None,
        offset_noise: None,
        auxiliary_scalar: None,
        auxiliary_matrix: None,
    };

    // Bernoulli host: the primary design IS `a(C)`.
    let mut marginal_only = Array2::<f64>::zeros((3, 1));
    marginal_only.column_mut(0).assign(&x);
    let bernoulli = predictor_with(LatentConditioningSpan::PrimaryDesign)
        .apply_latent_z_conditional_calibration(&z, &input_from(marginal_only.clone()))
        .expect("bernoulli conditioning span");

    // Survival host: the same `a(C)`, behind two leading time columns whose
    // values are deliberately large — conditioning on them instead would move
    // every ζ far from the expected value rather than subtly.
    let mut q_design = Array2::<f64>::zeros((3, 3));
    q_design.column_mut(0).assign(&array![7.0, -4.0, 11.0]);
    q_design.column_mut(1).assign(&array![-9.0, 6.0, 3.0]);
    q_design.column_mut(2).assign(&x);
    let survival = predictor_with(LatentConditioningSpan::PrimaryDesignTail { ncols: 1 })
        .apply_latent_z_conditional_calibration(&z, &input_from(q_design.clone()))
        .expect("survival conditioning span");

    for row in 0..3 {
        assert!(
            (bernoulli[row] - expected[row]).abs() < 1e-12,
            "bernoulli host row {row}: got {}, expected {}",
            bernoulli[row],
            expected[row]
        );
        assert!(
            (survival[row] - expected[row]).abs() < 1e-12,
            "survival host row {row}: got {}, expected {} — the tail block is the fitted \
             conditioning span, the leading time columns are not",
            survival[row],
            expected[row]
        );
    }

    // Naming the wrong block is refused, not silently mis-conditioned: the
    // calibration's own `basis_ncols` is the cross-check.
    let wrong = predictor_with(LatentConditioningSpan::PrimaryDesign)
        .apply_latent_z_conditional_calibration(&z, &input_from(q_design))
        .expect_err("a 3-column span against a 1-column calibration must be refused");
    assert!(
        wrong.to_string().contains("basis columns"),
        "the refusal must name the width mismatch; got {wrong}"
    );
}
