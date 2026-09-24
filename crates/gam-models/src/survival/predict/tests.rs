use super::*;
use crate::bms::LatentMeasureKind;
use crate::inference::model::SavedLatentZNormalization;
use crate::probability::{normal_cdf, normal_pdf};

#[test]
fn competing_risks_covariance_mode_selects_exact_requested_matrix() {
    let conditional = ndarray::array![[1.0, 0.2], [0.2, 2.0]];
    let corrected = ndarray::array![[1.5, 0.4], [0.4, 3.0]];

    let selected_conditional = select_survival_prediction_covariance(
        Some(&conditional),
        Some(&corrected),
        SurvivalPredictionCovarianceMode::Conditional,
    )
    .expect("conditional covariance");
    let selected_corrected = select_survival_prediction_covariance(
        Some(&conditional),
        Some(&corrected),
        SurvivalPredictionCovarianceMode::SmoothingCorrected,
    )
    .expect("smoothing-corrected covariance");

    assert_eq!(selected_conditional, &conditional);
    assert_eq!(selected_corrected, &corrected);
    assert_eq!(
        SurvivalPredictionCovarianceMode::Conditional.as_str(),
        "conditional"
    );
    assert_eq!(
        SurvivalPredictionCovarianceMode::SmoothingCorrected.as_str(),
        "smoothing-corrected"
    );
}

#[test]
fn competing_risks_smoothing_covariance_never_falls_back() {
    let conditional = ndarray::array![[1.0]];
    let error = select_survival_prediction_covariance(
        Some(&conditional),
        None,
        SurvivalPredictionCovarianceMode::SmoothingCorrected,
    )
    .expect_err("a corrected request must not substitute conditional covariance");
    assert_eq!(
        error.to_string(),
        "fit result does not contain smoothing-corrected covariance"
    );
}

#[test]
fn posterior_quadrature_second_moment_honors_cross_coordinate_covariance() {
    let posterior_mean = ndarray::array![0.4, -0.2];
    let covariance = ndarray::array![[0.9, 0.35], [0.35, 0.6]];
    let mut functional = Array2::from_elem((1, 1), PosteriorMoment::EMPTY);
    let mut recovered_cross_covariance = 0.0_f64;

    for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, weight| {
        functional[[0, 0]].merge(weight, PosteriorMoment::point(node[0] + 2.0 * node[1]));
        recovered_cross_covariance +=
            weight * (node[0] - posterior_mean[0]) * (node[1] - posterior_mean[1]);
        Ok(())
    })
    .expect("joint posterior quadrature");

    let expected_mean = posterior_mean[0] + 2.0 * posterior_mean[1];
    let expected_variance =
        covariance[[0, 0]] + 4.0 * covariance[[1, 1]] + 4.0 * covariance[[0, 1]];
    assert!((functional[[0, 0]].mean() - expected_mean).abs() <= 1e-12);
    assert!((recovered_cross_covariance - covariance[[0, 1]]).abs() <= 1e-12);

    let standard_error = posterior_standard_errors(&functional, "joint-covariance witness")
        .expect("posterior standard error");
    assert!((standard_error[[0, 0]].powi(2) - expected_variance).abs() <= 1e-11);
}

#[test]
fn posterior_quadrature_zero_covariance_has_zero_standard_error() {
    let posterior_mean = ndarray::array![0.25, -0.75];
    let covariance = Array2::<f64>::zeros((2, 2));
    let mut functional = Array2::from_elem((1, 1), PosteriorMoment::EMPTY);
    let mut node_count = 0usize;

    for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, weight| {
        functional[[0, 0]].merge(weight, PosteriorMoment::point(node[0].exp() + node[1].sin()));
        node_count += 1;
        Ok(())
    })
    .expect("rank-zero posterior quadrature");

    assert_eq!(node_count, 1, "rank-zero covariance has one exact node");
    let standard_error = posterior_standard_errors(&functional, "rank-zero witness")
        .expect("rank-zero posterior standard error");
    assert_eq!(standard_error[[0, 0]], 0.0);
}

/// gam#4086: a functional that is constant over the sigma points has zero
/// posterior variance at any rank. The raw `E f² − (E f)²` of the same
/// nodes cancels to a negative number beyond `128·eps` at rank 530, which
/// the removed round-off gate refused as a numerical failure; the centred
/// moments give exactly zero.
#[test]
fn posterior_quadrature_constant_functional_has_exactly_zero_standard_error_at_high_rank() {
    let rank = 530;
    let posterior_mean = Array1::<f64>::zeros(rank);
    let covariance = Array2::<f64>::eye(rank);
    let mut functional = Array2::from_elem((1, 1), PosteriorMoment::EMPTY);
    let mut raw_first = 0.0_f64;
    let mut raw_second = 0.0_f64;
    let mut node_count = 0usize;

    for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |_, weight| {
        let value = 1.0_f64;
        functional[[0, 0]].merge(weight, PosteriorMoment::point(value));
        raw_first += weight * value;
        raw_second += weight * value * value;
        node_count += 1;
        Ok(())
    })
    .expect("rank-530 posterior quadrature");

    assert_eq!(node_count, 2 * rank);
    assert!(
        raw_second - raw_first * raw_first < -128.0 * f64::EPSILON,
        "the raw moments no longer witness the cancellation: {:e}",
        raw_second - raw_first * raw_first
    );
    assert_eq!(functional[[0, 0]].mean(), 1.0);
    let standard_error = posterior_standard_errors(&functional, "constant witness")
        .expect("a constant functional's posterior standard error");
    assert_eq!(standard_error[[0, 0]], 0.0);
}

/// The pairwise merge is the exact weighted mean and centred second moment
/// of the merged nodes, whatever the order or weight scale.
#[test]
fn posterior_moment_merge_matches_two_pass_moments() {
    let nodes = [(0.5, 3.0), (0.125, -1.0), (0.25, 7.5), (0.125, 2.0)];
    let total: f64 = nodes.iter().map(|&(w, _)| w).sum();
    let mean = nodes.iter().map(|&(w, v)| w * v).sum::<f64>() / total;
    let variance =
        nodes.iter().map(|&(w, v)| w * (v - mean) * (v - mean)).sum::<f64>() / total;

    let mut forward = PosteriorMoment::EMPTY;
    for &(w, v) in &nodes {
        forward.merge(w, PosteriorMoment::point(v));
    }
    let mut halves = [PosteriorMoment::EMPTY; 2];
    for (index, &(w, v)) in nodes.iter().enumerate() {
        halves[index % 2].merge(w, PosteriorMoment::point(v));
    }
    let mut pooled = PosteriorMoment::EMPTY;
    pooled.merge(3.0, halves[1]);
    pooled.merge(3.0, halves[0]);
    for moment in [forward, pooled.scaled(0.25)] {
        assert!((moment.mean() - mean).abs() <= 4.0 * f64::EPSILON * mean.abs());
        assert!((moment.variance() - variance).abs() <= 8.0 * f64::EPSILON * variance);
    }
}

#[test]
fn posterior_quadrature_keeps_cone_coordinates_feasible_and_unbiased() {
    // Coordinate 0 is a structural monotone-I-spline baseline time
    // coefficient the fit constrained to β_0 ≥ 0; coordinate 1 is an
    // unconstrained covariate intercept. The covariance loads coordinate 0
    // with a spread wider than β̂_0, so the untruncated √rank·σ sigma point
    // steps β_0 below zero — the exact #2375 infeasible-node signature that
    // manufactures a non-monotone RP baseline the plugin evaluator refuses.
    let posterior_mean = ndarray::array![0.354, -8.30];
    let covariance = ndarray::array![[0.2304, 0.30], [0.30, 0.9604]];

    // Without the cone, the untruncated rule produces an infeasible node.
    let mut min_cone0_unconstrained = f64::INFINITY;
    for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, _| {
        min_cone0_unconstrained = min_cone0_unconstrained.min(node[0]);
        Ok(())
    })
    .expect("unconstrained quadrature");
    assert!(
        min_cone0_unconstrained < 0.0,
        "fixture must reproduce the infeasible-node bug (min β_0 = {min_cone0_unconstrained})"
    );

    // With the cone, every node stays feasible AND the rule stays unbiased
    // (symmetric ± steps, unchanged weights → the posterior mean is exact).
    let mut mean0 = 0.0_f64;
    let mut mean1 = 0.0_f64;
    let mut weight_sum = 0.0_f64;
    let mut min_cone0 = f64::INFINITY;
    for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
        assert!(
            node[0] >= -1e-12,
            "cone coordinate stepped below its β_0 ≥ 0 wall: {}",
            node[0]
        );
        min_cone0 = min_cone0.min(node[0]);
        mean0 += weight * node[0];
        mean1 += weight * node[1];
        weight_sum += weight;
        Ok(())
    })
    .expect("cone-truncated quadrature");
    assert!((weight_sum - 1.0).abs() <= 1e-12, "weights must sum to one");
    assert!(
        (mean0 - posterior_mean[0]).abs() <= 1e-12
            && (mean1 - posterior_mean[1]).abs() <= 1e-12,
        "cone truncation must leave the posterior mean unbiased (got [{mean0}, {mean1}])"
    );

    // Truncation shrinks — never inflates — the represented spread along the
    // constrained coordinate (a truncated Gaussian has smaller variance).
    let mut var0_unconstrained = 0.0_f64;
    for_each_survival_posterior_node(&posterior_mean, &covariance, &[], |node, weight| {
        var0_unconstrained += weight * (node[0] - posterior_mean[0]).powi(2);
        Ok(())
    })
    .expect("unconstrained spread");
    let mut var0_cone = 0.0_f64;
    for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
        var0_cone += weight * (node[0] - posterior_mean[0]).powi(2);
        Ok(())
    })
    .expect("cone spread");
    assert!(
        var0_cone <= var0_unconstrained + 1e-12 && var0_cone < var0_unconstrained,
        "cone spread {var0_cone} must be strictly smaller than the untruncated {var0_unconstrained}"
    );
}

#[test]
fn posterior_quadrature_cone_is_a_noop_far_from_the_wall() {
    // When β̂ sits comfortably inside the cone (every √rank·σ node stays
    // feasible), the fraction-to-boundary step never binds, so the cone rule
    // must reproduce the untruncated covariance to full precision — the fix
    // is inert on the healthy fits that dominate production.
    let posterior_mean = ndarray::array![40.0, -0.2];
    let covariance = ndarray::array![[0.9, 0.35], [0.35, 0.6]];
    let mut recovered_var0 = 0.0_f64;
    let mut recovered_cross = 0.0_f64;
    for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
        recovered_var0 += weight * (node[0] - posterior_mean[0]).powi(2);
        recovered_cross +=
            weight * (node[0] - posterior_mean[0]) * (node[1] - posterior_mean[1]);
        Ok(())
    })
    .expect("cone quadrature far from the wall");
    assert!((recovered_var0 - covariance[[0, 0]]).abs() <= 1e-11);
    assert!((recovered_cross - covariance[[0, 1]]).abs() <= 1e-11);
}

/// A cone coefficient sitting EXACTLY on its wall (`β̂_j = 0`) is the
/// ordinary state of an active box face, not an edge case: the fit's
/// `coefficient_lower_bounds` pins increments there routinely. The
/// fraction-to-boundary limit is then `0 / |f_j| = 0`, so every direction
/// that loads that coordinate collapses to the mean and contributes no
/// spread, while directions that do not load it keep the full `√rank` step.
///
/// This is the one place the rule cannot represent the posterior it is
/// approximating: a truncated Gaussian at an active bound is ONE-SIDED and
/// carries real mass, but no symmetric `±` pair can express that. Reporting
/// zero spread there is the conservative feasible answer rather than a
/// fabricated one, and pinning it here means a future switch to an
/// asymmetric rule has to change this test deliberately instead of silently.
#[test]
fn posterior_quadrature_radius_collapses_on_an_active_bound() {
    // Distinct eigenvalues so the PSD factor is axis-aligned and "the
    // direction that loads the pinned coordinate" is unambiguous.
    let posterior_mean = ndarray::array![0.0, 0.75];
    let covariance = ndarray::array![[0.5, 0.0], [0.0, 0.2]];

    let mut min_pinned = f64::INFINITY;
    let mut max_pinned = f64::NEG_INFINITY;
    let mut spread_unpinned = 0.0_f64;
    for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, weight| {
        min_pinned = min_pinned.min(node[0]);
        max_pinned = max_pinned.max(node[0]);
        spread_unpinned += weight * (node[1] - posterior_mean[1]).powi(2);
        Ok(())
    })
    .expect("active-bound quadrature");

    assert!(
        min_pinned >= 0.0,
        "an active bound must never be crossed, got {min_pinned}"
    );
    assert!(
        max_pinned.abs() <= 1e-12,
        "a direction loading an active-bound coordinate carries zero symmetric spread, \
         but the coordinate reached {max_pinned}"
    );
    // The unconstrained coordinate is untouched: collapsing one direction
    // must not collapse the whole rule.
    assert!(
        (spread_unpinned - covariance[[1, 1]]).abs() <= 1e-11,
        "a coordinate outside the cone keeps its full spread, got {spread_unpinned} want {}",
        covariance[[1, 1]]
    );
}

/// Round-off guard for the `β̂_j.max(0.0)` clamp in the fraction-to-boundary
/// limit. A converged active-set coefficient can land a few ulps BELOW its
/// wall, and the unclamped ratio `β̂_j / |f_j|` would then be NEGATIVE — a
/// negative step that silently inverts the `±` geometry of that direction
/// instead of shrinking it. The clamp sends the limit to zero, so the pair
/// collapses onto `β̂` and truncation never drives a coordinate further
/// outside the cone than the fit already left it.
#[test]
fn posterior_quadrature_clamps_a_roundoff_negative_cone_coordinate() {
    let roundoff_below_wall = -1e-15_f64;
    let posterior_mean = ndarray::array![roundoff_below_wall, 0.75];
    let covariance = ndarray::array![[0.5, 0.0], [0.0, 0.2]];

    let mut nodes = Vec::new();
    for_each_survival_posterior_node(&posterior_mean, &covariance, &[0], |node, _| {
        nodes.push(node[0]);
        Ok(())
    })
    .expect("round-off-negative cone quadrature");

    for value in &nodes {
        assert!(
            *value >= roundoff_below_wall,
            "truncation must never push a cone coordinate further below the wall than the \
             fit left it: node {value} < β̂ {roundoff_below_wall}"
        );
        assert!(
            (*value - roundoff_below_wall).abs() <= 1e-12,
            "a coordinate at the wall carries no spread, got {value}"
        );
    }
}

#[test]
fn probit_survival_hazard_uses_density_over_survival() {
    let eta = 2.0;
    let eta_t = 0.3;

    let (cum, hazard) =
        probit_survival_hazard_components(eta, eta_t).expect("valid components");

    let survival = normal_cdf(-eta);
    let expected_cum = -survival.ln();
    let expected_hazard = normal_pdf(eta) * eta_t / survival;
    assert!((cum - expected_cum).abs() <= 1e-14);
    assert!((hazard - expected_hazard).abs() <= 1e-14);
}

#[test]
fn probit_survival_hazard_stays_finite_in_right_tail() {
    let eta = 40.0;
    let eta_t = 9.694_340_360_912_401e-5;

    let event_density =
        (-0.5_f64 * eta * eta).exp() / (2.0 * std::f64::consts::PI).sqrt() * eta_t;
    assert_eq!(event_density, 0.0);

    let (cum, hazard) =
        probit_survival_hazard_components(eta, eta_t).expect("valid tail components");
    assert!(cum > 800.0, "right-tail cumulative hazard was {cum}");
    assert!(
        (3.87e-3..3.89e-3).contains(&hazard),
        "right-tail hazard was {hazard}"
    );
}

#[test]
fn probit_survival_hazard_accepts_zero_time_derivative_as_flat_hazard() {
    let (cum, hazard) =
        probit_survival_hazard_components(1.0, 0.0).expect("zero derivative is flat hazard");
    assert!(cum > 0.0);
    assert_eq!(hazard, 0.0);
}

#[test]
fn saved_marginal_slope_hazard_replay_differentiates_moving_slope_2767() {
    let predictor = BernoulliMarginalSlopePredictor {
        beta_marginal: ndarray::array![1.0],
        beta_slope: ndarray::array![1.0],
        beta_score_warp: None,
        beta_link_dev: None,
        base_link: InverseLink::Standard(StandardLink::Probit),
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
        latent_z_conditional_calibration: None,
        latent_conditioning_span: LatentConditioningSpan::PrimaryDesign,
        residual_repair: None,
        beta_residual: None,
    };
    let z = 1.1;
    let q = 0.8;
    let b = 0.65;
    let q_t = 0.31;
    let b_t = 0.22;
    let input_at = |q_value: f64, b_value: f64| PredictInput {
        design: DesignMatrix::from(ndarray::array![[q_value]]),
        offset: ndarray::array![0.0],
        design_noise: Some(DesignMatrix::from(ndarray::array![[b_value]])),
        offset_noise: Some(ndarray::array![0.0]),
        auxiliary_scalar: Some(ndarray::array![z]),
        auxiliary_matrix: None,
    };

    let (eta, eta_t) = predictor
        .predict_eta_and_time_tangent(
            &input_at(q, b),
            &ndarray::array![q_t],
            &ndarray::array![b_t],
        )
        .expect("saved value+tangent replay");
    let step = 1e-6;
    let eta_plus = predictor
        .predict_eta_and_time_tangent(
            &input_at(q + step * q_t, b + step * b_t),
            &ndarray::array![0.0],
            &ndarray::array![0.0],
        )
        .expect("positive saved replay")
        .0[0];
    let eta_minus = predictor
        .predict_eta_and_time_tangent(
            &input_at(q - step * q_t, b - step * b_t),
            &ndarray::array![0.0],
            &ndarray::array![0.0],
        )
        .expect("negative saved replay")
        .0[0];
    let eta_t_fd = (eta_plus - eta_minus) / (2.0 * step);
    assert!(
        (eta_t[0] - eta_t_fd).abs() <= 2e-9 * (1.0 + eta_t_fd.abs()),
        "complete saved eta tangent {} != centered FD {eta_t_fd}",
        eta_t[0]
    );

    let eta_q = (1.0_f64 + b * b).sqrt();
    let omitted_slope_chain = eta_q * q_t;
    assert!(
        (eta_t[0] - omitted_slope_chain).abs() > 0.1,
        "fixture must detect omission of eta_b*b_t"
    );
    let (_, analytic_hazard) = probit_survival_hazard_components(eta[0], eta_t[0])
        .expect("positive analytic hazard");
    let (_, fd_hazard) = probit_survival_hazard_components(eta[0], eta_t_fd)
        .expect("positive finite-difference hazard");
    assert!(
        (analytic_hazard - fd_hazard).abs() <= 2e-9 * (1.0 + fd_hazard.abs()),
        "analytic saved hazard {analytic_hazard} != FD hazard {fd_hazard}"
    );
}

/// gam#3026: the hazard is the derivative of the cumulative hazard reported
/// beside it, on both tails and through the bulk. `η(t) = a + c·log t` with
/// `c > 0`, `H(t) = −log Φ(−η(t))`, and `dH/dt` by a Richardson-extrapolated
/// central difference in `log t`, whose truncation error is `O(δ⁴)`.
#[test]
fn probit_survival_hazard_is_the_derivative_of_its_cumulative_hazard_3026() {
    let (a, c) = (-1.15, 0.95);
    let cumulative = |t: f64| {
        probit_survival_hazard_components(a + c * t.ln(), c / t)
            .expect("increasing index")
            .0
    };
    let delta = 1e-3;
    for t in [1e-4_f64, 1e-2, 0.3, 1.0, 5.0, 40.0, 1e3] {
        let (_, hazard) =
            probit_survival_hazard_components(a + c * t.ln(), c / t).expect("increasing index");
        let at = |m: f64| cumulative(t * (m * delta).exp());
        let d1 = (at(1.0) - at(-1.0)) / (2.0 * delta);
        let d2 = (at(2.0) - at(-2.0)) / (4.0 * delta);
        let derivative = (4.0 * d1 - d2) / 3.0 / t;
        assert!(
            (hazard - derivative).abs() <= 1e-8 * derivative.abs(),
            "at t={t}: hazard {hazard} is not dH/dt {derivative}"
        );
    }
}

/// gam#3026: where the index decreases the kernel returns the negative
/// hazard it is, still `dH/dt`, instead of a flat hazard:
/// `η(t) = a − c·log t` with `c > 0`, `dH/dt` by the same Richardson
/// difference in `log t`. The rate `−1.35e-3` is the one the deleted
/// clamp's test recorded (#1040), where the clamp reported `h = 0` beside
/// an `H` whose derivative is negative.
#[test]
fn probit_survival_hazard_is_signed_where_the_index_decreases_3026() {
    let (a, c) = (0.4, 0.3);
    let cumulative = |t: f64| {
        probit_survival_hazard_components(a - c * t.ln(), -c / t)
            .expect("finite index")
            .0
    };
    let delta = 1e-3;
    for t in [1e-2_f64, 0.3, 1.0, 5.0, 40.0] {
        let (_, hazard) =
            probit_survival_hazard_components(a - c * t.ln(), -c / t).expect("finite index");
        let at = |m: f64| cumulative(t * (m * delta).exp());
        let d1 = (at(1.0) - at(-1.0)) / (2.0 * delta);
        let d2 = (at(2.0) - at(-2.0)) / (4.0 * delta);
        let derivative = (4.0 * d1 - d2) / 3.0 / t;
        assert!(
            hazard < 0.0 && (hazard - derivative).abs() <= 1e-8 * derivative.abs(),
            "at t={t}: hazard {hazard} is not dH/dt {derivative}"
        );
    }
    let (cumulative, hazard) =
        probit_survival_hazard_components(-0.563, -1.35e-3).expect("finite index");
    let (log_survival, mills_ratio) = signed_probit_logcdf_and_mills_ratio(0.563);
    assert_eq!((cumulative, hazard), (-log_survival, mills_ratio * -1.35e-3));
}

/// gam#3026: a plug-in surface that reports a negative hazard at any cell
/// is refused by name, naming the cell, and one whose hazards are all
/// non-negative (a flat stretch's `0` included) publishes.
#[test]
fn a_surface_whose_survival_increases_is_refused_by_name_3026() {
    let surface = |hazard: Array2<f64>| SurvivalPredictResult {
        times: vec![1.0, 2.0],
        survival: Array2::from_elem((2, 2), 0.5),
        cumulative_hazard: Array2::from_elem((2, 2), 2.0_f64.ln()),
        hazard,
        linear_predictor: Array1::zeros(2),
        likelihood_mode: SurvivalLikelihoodMode::MarginalSlope,
        survival_se: None,
        eta_se: None,
        covariance_source: None,
        survival_plugin: None,
        survival_lower: None,
        survival_upper: None,
    };
    refuse_decreasing_survival(&surface(ndarray::array![[0.3, 0.0], [0.1, 0.2]]))
        .expect("non-negative hazards publish");
    match refuse_decreasing_survival(&surface(ndarray::array![[0.3, 0.2], [0.1, -4.0e-3]])) {
        Err(SurvivalPredictError::DecreasingSurvival { reason }) => assert!(
            reason.contains("row 1, time column 1"),
            "the refusal must name the cell: {reason}"
        ),
        other => panic!("a negative hazard must be refused by name, got {other:?}"),
    }
}

/// gam#3026: the exact anchored posterior's density is `−d/dt E[S]`, so the
/// published hazard `E f / E S` is `dH̄/dt`. The law makes the rate negative
/// with visible probability: `q(t) = u + v·log t` with `(u, v)` bivariate
/// normal and `P(v < 0) = Φ(−0.5/0.6) ≈ 0.20`. On the rigid frame with
/// `b = 0`, `η = q`, `η_q = 1` and `η_b = 0`. Given `q(t)`, `v` has mean
/// `m_v + β(q − m_q)`, and `E[S]` and `E[f]` are integrated by the same
/// trapezoid over `q`, which is spectrally accurate for a Gaussian weight.
/// The positive-part rule the node replaced is measured too, so the fixture
/// can tell the two apart.
#[test]
fn exact_anchor_node_density_is_the_derivative_of_the_posterior_survival_3026() {
    let (m_u, m_v) = (-0.4, 0.5);
    let (s_u, s_v, rho) = (0.7_f64, 0.6_f64, -0.3_f64);
    // Moments of (q(t), v) at log t = x.
    let law = |x: f64| {
        let mean_q = m_u + m_v * x;
        let var_q = s_u * s_u + 2.0 * rho * s_u * s_v * x + s_v * s_v * x * x;
        let cov_vq = rho * s_u * s_v + s_v * s_v * x;
        let var_v_given_q = s_v * s_v - cov_vq * cov_vq / var_q;
        (mean_q, var_q, cov_vq / var_q, var_v_given_q)
    };
    let nodes = 4001;
    let integrate = |t: f64, signed: bool| -> (f64, f64) {
        let x = t.ln();
        let (mean_q, var_q, beta, var_v_given_q) = law(x);
        let sd_q = var_q.sqrt();
        let (mut survival, mut density) = (0.0, 0.0);
        for k in 0..nodes {
            let w = -12.0 + 24.0 * k as f64 / (nodes - 1) as f64;
            let q = mean_q + sd_q * w;
            let weight = normal_pdf(w) * 24.0 / (nodes - 1) as f64;
            // d q / d t = v / t, so the tangent's conditional mean is E[v | q] / t.
            let v_mean = m_v + beta * (q - mean_q);
            let rate = if signed {
                v_mean / t
            } else {
                let sd = var_v_given_q.max(0.0).sqrt();
                (v_mean * normal_cdf(v_mean / sd) + sd * normal_pdf(v_mean / sd)) / t
            };
            let moments = exact_anchor_node_moments(q, 1.0, 0.0, [rate, 0.0]);
            survival += weight * moments.survival.mean();
            density += weight * moments.density;
        }
        (survival, density)
    };
    let delta = 1e-3;
    let mut positive_part_gap = 0.0_f64;
    for t in [0.2_f64, 1.0, 3.0, 12.0] {
        let (survival, density) = integrate(t, true);
        let at = |m: f64| -(integrate(t * (m * delta).exp(), true).0).ln();
        let d1 = (at(1.0) - at(-1.0)) / (2.0 * delta);
        let d2 = (at(2.0) - at(-2.0)) / (4.0 * delta);
        let derivative = (4.0 * d1 - d2) / 3.0 / t;
        let hazard = density / survival;
        assert!(
            (hazard - derivative).abs() <= 1e-7 * derivative.abs(),
            "at t={t}: posterior hazard E f / E S = {hazard} is not dH̄/dt = {derivative}"
        );
        let (_, positive_density) = integrate(t, false);
        positive_part_gap =
            positive_part_gap.max((positive_density / survival - derivative).abs() / derivative);
    }
    assert!(
        positive_part_gap > 1e-3,
        "the fixture must separate the signed density from the positive-part rule; the gap \
         was {positive_part_gap:.3e}"
    );
}

#[test]
fn probit_survival_hazard_rejects_infinite_time_derivative() {
    let err = probit_survival_hazard_components(1.0, f64::INFINITY)
        .expect_err("infinite derivative should be invalid");
    assert!(
        err.to_string()
            .contains("invalid survival index derivative")
    );
}

#[test]
fn probit_survival_hazard_rejects_nan_inputs() {
    // The upstream input gate is the only line that rejects NaN — the
    // output gate (`>= 0.0`) is dead-code for finite input because
    // `signed_probit_logcdf_and_mills_ratio` is provably NaN-free on the
    // finite domain (every internal branch clamps `erfcx`/`cdf` away from
    // zero). Pin both NaN slots so the input gate cannot regress.
    let err_eta =
        probit_survival_hazard_components(f64::NAN, 0.5).expect_err("NaN eta must be rejected");
    assert!(
        err_eta
            .to_string()
            .contains("invalid survival index derivative")
    );
    let err_dt = probit_survival_hazard_components(1.0, f64::NAN)
        .expect_err("NaN eta_derivative must be rejected");
    assert!(
        err_dt
            .to_string()
            .contains("invalid survival index derivative")
    );
}

#[test]
fn royston_parmar_hazard_is_cumulative_hazard_derivative() {
    let eta = 2.0_f64.ln();
    let eta_t = 0.25;

    let (cum, hazard) =
        royston_parmar_survival_hazard_components(eta, eta_t).expect("valid components");

    assert!((cum - 2.0).abs() <= 1e-14);
    assert!((hazard - 0.5).abs() <= 1e-14);
    assert_ne!(hazard, cum);
}

#[test]
fn royston_parmar_hazard_keeps_the_sign_of_a_negative_log_hazard_derivative() {
    // A negative time-derivative of log Λ(t) is a falling cumulative hazard: the
    // component returns the negative `dΛ/dt` it is, so a posterior rule's
    // `Σ S·h` stays `−d/dt Σ S` (gam#3575). Publication refuses it
    // (`refuse_decreasing_survival`); the component does not.
    let (cum, hazard) = royston_parmar_survival_hazard_components(0.0, -0.5)
        .expect("a finite negative derivative is a signed hazard");
    assert_eq!(cum, 1.0);
    assert_eq!(hazard, -0.5);
    let err = royston_parmar_survival_hazard_components(0.0, f64::NAN)
        .expect_err("a NaN derivative is not a hazard");
    assert!(
        err.to_string()
            .contains("invalid log-cumulative-hazard derivative")
    );
}

#[test]
fn royston_parmar_hazard_accepts_zero_derivative_as_flat_boundary() {
    // #1564: a monotone I-spline cumulative hazard is flat beyond its last
    // interior knot, so `d(log Λ)/dt == 0` exactly on any grid node past the
    // training support. That is a *valid* prediction (zero instantaneous
    // hazard, locally constant survival), not a numerical failure. The old
    // strict `> 0.0` gate rejected it and crashed saved-model RP predict.
    let eta = 1.9909019457445971_f64; // the exact η from the #1564 report
    let (cum, hazard) = royston_parmar_survival_hazard_components(eta, 0.0)
        .expect("zero derivative is a valid flat boundary, not an error");
    assert!((cum - eta.exp()).abs() <= 1e-12, "cum = Λ(t) = exp(η)");
    assert_eq!(
        hazard, 0.0,
        "flat cumulative hazard ⇒ zero instantaneous hazard"
    );
    // Survival is finite and well-defined at the boundary.
    let survival = (-cum).exp().clamp(0.0, 1.0);
    assert!(survival.is_finite() && (0.0..=1.0).contains(&survival));
}

#[test]
fn royston_parmar_hazard_zero_derivative_in_saturated_tail_is_zero_not_nan() {
    // The dangerous corner: a saturated tail (η large ⇒ Λ = exp(η) = +∞) that
    // also lands past the I-spline support (derivative == 0). The naive
    // product `+∞ * 0.0` is `NaN`, which would (a) trip the components guard
    // and (b) serialize to JSON `null` and break the Python parse (#1564,
    // bug 1). The hazard must resolve to the mathematically correct `0`.
    let eta = 1000.0_f64;
    assert!(
        eta.exp().is_infinite(),
        "test premise: exp(1000) overflows to +∞"
    );
    assert!(
        (f64::INFINITY * 0.0).is_nan(),
        "test premise: the naive product is NaN"
    );
    let (cum, hazard) = royston_parmar_survival_hazard_components(eta, 0.0)
        .expect("saturated + flat boundary must be valid");
    assert!(cum.is_infinite() && cum > 0.0, "cum saturates to +∞");
    assert_eq!(hazard, 0.0, "hazard at a flat boundary is 0, never NaN");
}

#[test]
fn royston_parmar_hazard_propagates_saturation_as_infinity() {
    // η = log Λ(t); a saturated RP fit can drive η well past the
    // exp(709.78)≈f64::MAX boundary in the right tail. The math is
    // S(t)→0, h(t)→∞; the helper must not reject this regime, because the
    // inner solver has already accepted the underlying fit.
    let eta = 1000.0_f64;
    let eta_t = 0.5_f64;
    assert!(eta.exp().is_infinite(), "test premise: exp(1000) overflows");

    let (cum, hazard) = royston_parmar_survival_hazard_components(eta, eta_t)
        .expect("saturated RP fit must yield a result, not an error");
    assert!(cum.is_infinite() && cum > 0.0, "expected +∞ cum, got {cum}");
    assert!(
        hazard.is_infinite() && hazard > 0.0,
        "expected +∞ hazard, got {hazard}"
    );

    // Consumer materializes survival via exp(-cum).clamp(0,1).
    let survival = (-cum).exp().clamp(0.0, 1.0);
    assert_eq!(survival, 0.0, "saturated cum_hazard must give survival 0");
}

#[test]
fn royston_parmar_hazard_rejects_nan_eta() {
    let err = royston_parmar_survival_hazard_components(f64::NAN, 0.5)
        .expect_err("NaN eta should be invalid");
    assert!(
        err.to_string()
            .contains("invalid log-cumulative-hazard derivative")
    );
}

#[test]
fn royston_parmar_hazard_left_tail_collapses_to_zero() {
    // η = log Λ(t); η → -∞ means Λ(t) → 0, so cum_hazard underflows to 0
    // and hazard rate underflows to 0. Survival → 1. No error.
    let eta = -1000.0_f64;
    let eta_t = 2.0_f64;
    assert_eq!(eta.exp(), 0.0, "test premise: exp(-1000) underflows to 0");

    let (cum, hazard) = royston_parmar_survival_hazard_components(eta, eta_t)
        .expect("RP left tail must remain valid");
    assert_eq!(
        cum, 0.0,
        "left-tail cum_hazard should underflow to 0, got {cum}"
    );
    assert_eq!(
        hazard, 0.0,
        "left-tail hazard should underflow to 0, got {hazard}"
    );

    // Consumer: survival = exp(-0) = 1.
    let survival = (-cum).exp().clamp(0.0, 1.0);
    assert_eq!(survival, 1.0);
}

#[test]
fn probit_survival_hazard_left_tail_collapses_to_zero() {
    // η→-∞ mirror of the right-tail test: survival → 1, hazard → 0.
    // Asymptote: Mills(η) = φ(η)/Φ(-η) → 0 as η → -∞ (φ underflows,
    // Φ(-η) → 1).  No error, no NaN, no spurious negativity.
    let eta = -40.0_f64;
    let eta_t = 1.5_f64;

    let (cum, hazard) =
        probit_survival_hazard_components(eta, eta_t).expect("left tail must remain valid");
    assert!(
        (0.0..1e-300).contains(&cum),
        "left-tail cum should be ~0, got {cum}"
    );
    assert_eq!(
        hazard, 0.0,
        "left-tail hazard should underflow to 0, got {hazard}"
    );
}

#[test]
fn location_scale_logit_hazard_is_failure_slope_over_survival() {
    let eta = 0.7;
    let eta_t = 0.4;

    let hazard = location_scale_hazard_component(
        eta,
        eta_t,
        &InverseLink::Standard(StandardLink::Logit),
    )
    .expect("valid logit hazard");

    let failure = 1.0 / (1.0 + (-eta).exp());
    assert!((hazard - failure * eta_t).abs() <= 1e-14);
}

#[test]
fn location_scale_cloglog_hazard_matches_log_cumulative_hazard_derivative() {
    let eta = 1.5;
    let eta_t = 0.2;

    let hazard = location_scale_hazard_component(
        eta,
        eta_t,
        &InverseLink::Standard(StandardLink::CLogLog),
    )
    .expect("valid cloglog hazard");

    assert!((hazard - eta.exp() * eta_t).abs() <= 1e-14);
}

// ---- Held-out survival scoring, one owner for every front door (#2899) --

#[test]
fn survival_score_grid_spans_zero_to_the_largest_time_strictly_increasing() {
    let times = [0.3, 0.1, 0.1, 0.1, 0.45, f64::NAN, -2.0, 0.0, 0.2];
    let grid = survival_score_grid(&times);
    assert_eq!(grid[0], 0.0);
    assert_eq!(grid[grid.len() - 1], 0.45);
    assert!(grid.windows(2).all(|pair| pair[1] > pair[0]), "grid={grid:?}");
    // Half the finite positive times tie at 0.1, so the low quantiles land
    // there exactly and collapse onto one knot.
    assert_eq!(grid.iter().filter(|knot| **knot == 0.1).count(), 1);
    assert_eq!(survival_score_grid(&[f64::NAN, -1.0]), vec![0.0, 1.0]);
}

#[test]
fn survival_prediction_scores_integrate_the_ipcw_brier_and_lift_it_over_the_null() {
    let time = [2.0, 8.0, 10.0, 3.0, 6.0, 1.5];
    let event = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let grid = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0];
    // Rows that are already survival paths, so the repair leaves them as they are.
    let model = Array2::from_shape_fn((6, 6), |(row, col)| {
        1.0 - col as f64 * (0.05 + 0.02 * row as f64)
    });
    let null = Array2::from_shape_fn((6, 6), |(_, col)| 1.0 - 0.1 * col as f64);
    let scores =
        survival_prediction_scores(&time, &event, &grid, model.view(), Some(null.view()));
    let censoring = KaplanMeier::fit_censoring(&time, &event);
    let model_ibs =
        integrated_ipcw_brier_score(model.view(), &time, &event, &grid, 10.0, &censoring)
            .expect("model integrated Brier");
    let null_ibs =
        integrated_ipcw_brier_score(null.view(), &time, &event, &grid, 10.0, &censoring)
            .expect("null integrated Brier");
    assert_eq!(scores.brier, Some(model_ibs));
    let lifted = scores.lifted_brier.expect("lifted Brier");
    assert!((lifted - (null_ibs - model_ibs) / null_ibs.abs()).abs() <= 1e-15);
    assert!(scores.hazard_quadratic_score.is_some() && scores.logloss.is_some());
    assert!(scores.lifted_hazard_quadratic_score.is_some());
    assert!(scores.lifted_logloss.is_some());
    let alone = survival_prediction_scores(&time, &event, &grid, model.view(), None);
    assert_eq!(alone.brier, scores.brier);
    assert_eq!(alone.logloss, scores.logloss);
    assert_eq!(alone.lifted_brier, None);
    assert_eq!(alone.nagelkerke_r2, None);
    let repeated_knot = [0.0, 1.0, 1.0, 3.0, 5.0, 7.0];
    assert_eq!(
        survival_prediction_scores(&time, &event, &repeated_knot, model.view(), None),
        SurvivalPredictionScores::default()
    );
}

#[test]
fn a_scoring_grid_that_does_not_start_at_the_origin_is_refused_3609() {
    // S = 1 holds only at t = 0. On a grid starting at 1.0 the repair would
    // pin the model's S(1.0) < 1 to 1, dropping −ln S(1.0) from every H(T_i),
    // and the event at T = 0.5 would sit before the first interval and read a
    // negative cumulative hazard.
    let time = [0.5, 2.0, 8.0, 3.0];
    let event = [1.0, 1.0, 0.0, 1.0];
    let origin = [0.0, 1.0, 2.0, 3.0, 5.0];
    let model = Array2::from_shape_fn((4, 5), |(row, col)| {
        1.0 - col as f64 * (0.05 + 0.02 * row as f64)
    });
    let scored = survival_prediction_scores(&time, &event, &origin, model.view(), None);
    assert!(scored.brier.is_some() && scored.logloss.is_some());
    for late in [[1.0, 2.0, 3.0, 5.0, 7.0], [-1.0, 1.0, 2.0, 3.0, 5.0]] {
        assert_eq!(
            survival_prediction_scores(&time, &event, &late, model.view(), None),
            SurvivalPredictionScores::default(),
            "grid {late:?}"
        );
    }
}

// ---- IPCW Brier score (Graf et al. 1999) -------------------------------

#[test]
fn kaplan_meier_censoring_is_right_continuous_step() {
    // Two censorings (events flipped) at t=4 and t=8; deaths at t=2,6.
    let time = [2.0, 4.0, 6.0, 8.0];
    let event = [1.0, 0.0, 1.0, 0.0];
    let g = KaplanMeier::fit_censoring(&time, &event);
    // Before the first censoring the censoring-survival is 1.
    assert!((g.at(0.0) - 1.0).abs() <= 1e-15);
    assert!((g.at(2.0) - 1.0).abs() <= 1e-15);
    assert!((g.at(3.999) - 1.0).abs() <= 1e-15);
    // At t=4 the at-risk set {4,6,8} loses one to censoring: G = 2/3.
    assert!((g.at(4.0) - 2.0 / 3.0).abs() <= 1e-12);
    assert!((g.at(5.0) - 2.0 / 3.0).abs() <= 1e-12);
    // A death at t=6 does not move the censoring KM.
    assert!((g.at(6.0) - 2.0 / 3.0).abs() <= 1e-12);
    // At t=8 the last (sole) at-risk subject is censored: G collapses to 0.
    assert!(g.at(8.0).abs() <= 1e-15);
    // `on_grid` is `at` mapped over the grid, NaN included (it precedes no step).
    let probe = [f64::NAN, -1.0, 0.0, 3.999, 4.0, 7.5, 8.0, 9.0, f64::INFINITY];
    let mapped: Vec<f64> = probe.iter().map(|&t| g.at(t)).collect();
    assert_eq!(g.on_grid(&probe), mapped);
    assert_eq!(g.at(f64::NAN), 1.0);
    assert_eq!(g.at(f64::INFINITY), 0.0);
}

/// The pair-loop definition of Harrell's C, written independently of the
/// Fenwick sweep: a pair is comparable when one subject had an event and the
/// other was observed strictly later, or at the same time but censored.
fn harrell_concordance_by_pairs(time: &[f64], event: &[f64], risk: &[f64]) -> Option<f64> {
    let mut comparable = 0.0_f64;
    let mut concordant = 0.0_f64;
    for i in 0..time.len() {
        for j in 0..time.len() {
            let i_fails_first = event[i] > 0.5
                && (time[j] > time[i] || (time[j] == time[i] && event[j] <= 0.5));
            if !i_fails_first {
                continue;
            }
            comparable += 1.0;
            if risk[i] > risk[j] {
                concordant += 1.0;
            } else if risk[i] == risk[j] {
                concordant += 0.5;
            }
        }
    }
    if comparable > 0.0 {
        Some(concordant / comparable)
    } else {
        None
    }
}

#[test]
fn harrell_concordance_tie_rules_match_hand_count() {
    // Tied block at t=2: two events (rows 1, 2) and one censoring (row 3).
    // Comparable pairs (early, late): row 0 with all four later rows
    // (risk 5 beats 4, 1, 3, 2: 4 concordant); rows 1 and 2 with the tied
    // censoring row 3 (4>3 concordant, 1<3 discordant) and with row 4
    // (4>2 concordant, 1<2 discordant). The tied event pair (1, 2) is not
    // comparable, and row 3 (censored) orders nothing after it.
    // C = (4 + 1 + 1) / (4 + 2 + 2) = 0.75.
    let time = [1.0, 2.0, 2.0, 2.0, 3.0];
    let event = [1.0, 1.0, 1.0, 0.0, 0.0];
    let risk = [5.0, 4.0, 1.0, 3.0, 2.0];
    assert_eq!(harrell_concordance(&time, &event, &risk), Some(0.75));

    // Two events at the same time are not comparable: nothing is orderable.
    assert_eq!(harrell_concordance(&[2.0, 2.0], &[1.0, 1.0], &[1.0, 0.0]), None);
    // An event tied with a censoring is comparable; the censored subject
    // outlived the death, so the larger risk on the event is concordant.
    assert_eq!(harrell_concordance(&[2.0, 2.0], &[0.0, 1.0], &[1.0, 3.0]), Some(1.0));
    // Equal risks score half credit.
    assert_eq!(harrell_concordance(&[1.0, 2.0], &[1.0, 1.0], &[7.0, 7.0]), Some(0.5));
    // All censored: no comparable pair.
    assert_eq!(harrell_concordance(&[1.0, 2.0], &[0.0, 0.0], &[1.0, 2.0]), None);
    // Non-finite inputs have no defined ordering.
    assert_eq!(harrell_concordance(&[1.0, f64::NAN], &[1.0, 1.0], &[1.0, 2.0]), None);
    assert_eq!(harrell_concordance(&[1.0, 2.0], &[1.0, 1.0], &[f64::NAN, 2.0]), None);
    // Length mismatch.
    assert_eq!(harrell_concordance(&[1.0, 2.0], &[1.0], &[1.0, 2.0]), None);
}

#[test]
fn harrell_concordance_sweep_equals_pair_definition_under_heavy_ties() {
    // Coarse integer times and risks force many tied time blocks and tied
    // risks; the O(n log n) sweep must reproduce the pair loop exactly.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |modulus: u64| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) % modulus
    };
    for trial in 0..200 {
        let n = (next(40) + 1) as usize;
        let time: Vec<f64> = (0..n).map(|_| (next(6) + 1) as f64).collect();
        let event: Vec<f64> = (0..n)
            .map(|_| if next(5) < 3 { 1.0 } else { 0.0 })
            .collect();
        let risk: Vec<f64> = (0..n).map(|_| next(5) as f64 - 2.0).collect();
        let sweep = harrell_concordance(&time, &event, &risk);
        let pairs = harrell_concordance_by_pairs(&time, &event, &risk);
        // Both are one correctly rounded division of the same exact rational
        // (the numerators are integer and half-integer counts), so they are
        // bitwise equal, not merely close.
        assert_eq!(sweep, pairs, "trial {trial}");
    }
}

#[test]
fn ipcw_brier_no_censoring_reduces_to_plain_brier() {
    // With no censoring G(t) ≡ 1, so the IPCW Brier is the ordinary Brier of
    // the predicted survival against the alive-indicator I(T_i > tau).
    let s_pred = [0.3, 0.7, 0.6, 0.2];
    let time = [2.0, 8.0, 10.0, 3.0];
    let event = [1.0, 1.0, 0.0, 1.0];
    let tau = 5.0;
    let g = KaplanMeier::fit_censoring(&time, &event);
    let bs = ipcw_brier_score(&s_pred, &time, &event, tau, &g).unwrap();
    // targets: dead→0 (subj1,4), alive→1 (subj2,3).
    let expected =
        (0.3f64.powi(2) + (1.0 - 0.7f64).powi(2) + (1.0 - 0.6f64).powi(2) + 0.2f64.powi(2))
            / 4.0;
    assert!(
        (bs - expected).abs() <= 1e-12,
        "bs={bs} expected={expected}"
    );
}

#[test]
fn ipcw_brier_reweights_by_inverse_censoring_probability() {
    // Hand-computed Graf estimator with real censoring weights.
    // times/events: death@2, cens@4, death@6, cens@8; tau=5.
    // Censoring KM: G(5)=2/3 (one censoring at t=4 among {4,6,8}); G(2)=1.
    let s_pred = [0.4, 0.5, 0.7, 0.8];
    let time = [2.0, 4.0, 6.0, 8.0];
    let event = [1.0, 0.0, 1.0, 0.0];
    let tau = 5.0;
    let g = KaplanMeier::fit_censoring(&time, &event);
    let bs = ipcw_brier_score(&s_pred, &time, &event, tau, &g).unwrap();
    // subj1 dead by 5: weight 1/G(2)=1, contrib 0.4²=0.16.
    // subj2 censored before 5: contributes 0.
    // subj3 alive: weight 1/G(5)=1.5, contrib 1.5·0.3²=0.135.
    // subj4 alive: weight 1/G(5)=1.5, contrib 1.5·0.2²=0.06.
    let expected = (0.16 + 0.0 + 0.135 + 0.06) / 4.0;
    assert!(
        (bs - expected).abs() <= 1e-12,
        "bs={bs} expected={expected}"
    );
}

#[test]
fn ipcw_brier_weights_an_event_tied_with_a_censoring_by_the_left_limit() {
    // Discretised follow-up: a death and a censoring tie at t = 2, and again
    // at t = 5. The censoring KM steps at each tie (G(2) = 4/5, G(5) = 8/15),
    // but a death at T was observed because C ≥ T, whose probability is the
    // left limit G(T−) (G(2−) = 1, G(5−) = 4/5). Weighting by the
    // right-continuous G(T) would count each tied death 1/(1 − c/n) times.
    let s_pred = [0.4, 0.6, 0.7, 0.8, 0.9];
    let time = [2.0, 2.0, 5.0, 5.0, 7.0];
    let event = [1.0, 0.0, 1.0, 0.0, 1.0];
    let g = KaplanMeier::fit_censoring(&time, &event);
    assert_eq!(g.before(2.0), 1.0);
    assert!((g.at(2.0) - 0.8).abs() <= 1e-15);
    assert!((g.before(5.0) - 0.8).abs() <= 1e-15);
    assert!((g.at(5.0) - 8.0 / 15.0).abs() <= 1e-15);
    // tau = 3: subj1 dead, weight 1/G(2−) = 1, contrib 0.4²; subj2 censored
    // at 2 contributes 0; subj3..5 alive, weight 1/G(3) = 5/4.
    let bs = ipcw_brier_score(&s_pred, &time, &event, 3.0, &g).unwrap();
    let expected = (0.16 + 1.25 * (0.09 + 0.04 + 0.01)) / 5.0;
    assert!((bs - expected).abs() <= 1e-12, "bs={bs} expected={expected}");
    // tau = 5: subj3 dead at the tie, weight 1/G(5−) = 5/4; subj5 alive,
    // weight 1/G(5) = 15/8; the two censored subjects contribute 0.
    let bs = ipcw_brier_score(&s_pred, &time, &event, 5.0, &g).unwrap();
    let expected = (0.16 + 1.25 * 0.49 + 1.875 * 0.01) / 5.0;
    assert!((bs - expected).abs() <= 1e-12, "bs={bs} expected={expected}");
}

#[test]
fn ipcw_brier_drops_invalid_rows_from_both_numerator_and_denominator() {
    // A NaN-time row and a non-positive-time row must not be counted at all.
    let s_pred = [0.3, 0.7, 0.5, 0.5];
    let time = [2.0, 8.0, f64::NAN, -1.0];
    let event = [1.0, 1.0, 1.0, 0.0];
    let g = KaplanMeier::fit_censoring(&time, &event);
    let bs = ipcw_brier_score(&s_pred, &time, &event, 5.0, &g).unwrap();
    // Only subj1 (dead, contrib 0.3²) and subj2 (alive, contrib 0.3²) count;
    // censoring KM has no censorings so G≡1.
    let expected = (0.3f64.powi(2) + (1.0 - 0.7f64).powi(2)) / 2.0;
    assert!(
        (bs - expected).abs() <= 1e-12,
        "bs={bs} expected={expected}"
    );
}

#[test]
fn integrated_ipcw_brier_of_constant_brier_is_that_constant() {
    // A survival matrix whose every column equals a perfect classifier yields
    // BS(t)=0 at every grid point, so the integral is 0.
    let time = [2.0, 8.0, 10.0, 3.0];
    let event = [1.0, 1.0, 0.0, 1.0];
    let grid = [0.0, 1.0, 2.5, 4.0, 6.0];
    // Perfect prediction at every grid time given the (no-censoring) data is
    // not generally achievable, so instead test the integral of a literally
    // constant-in-time Brier: replicate one column across the grid.
    let col = [0.3, 0.7, 0.6, 0.2];
    let mut surv = Array2::<f64>::zeros((4, grid.len()));
    for k in 0..grid.len() {
        for i in 0..4 {
            surv[[i, k]] = col[i];
        }
    }
    let g = KaplanMeier::fit_censoring(&time, &event);
    let per_time = ipcw_brier_score(&col, &time, &event, grid[2], &g).unwrap();
    // Because the predicted survival is identical at every grid time, BS(t)
    // is *not* constant (tau changes which subjects are "alive"), so use a
    // direct trapezoid as the oracle.
    let mut oracle_pts = Vec::new();
    for k in 0..grid.len() {
        oracle_pts.push((
            grid[k],
            ipcw_brier_score(&col, &time, &event, grid[k], &g).unwrap(),
        ));
    }
    let mut integral = 0.0;
    for w in oracle_pts.windows(2) {
        integral += 0.5 * (w[0].1 + w[1].1) * (w[1].0 - w[0].0);
    }
    let oracle = integral / (grid[grid.len() - 1] - grid[0]);
    let ibs =
        integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, f64::INFINITY, &g)
            .unwrap();
    assert!((ibs - oracle).abs() <= 1e-12, "ibs={ibs} oracle={oracle}");
    // Sanity: per-time value is in a sensible [0,1]-ish range.
    assert!(per_time >= 0.0);
}

#[test]
fn integrated_ipcw_brier_respects_the_horizon_cutoff() {
    let time = [2.0, 8.0, 10.0, 3.0];
    let event = [1.0, 1.0, 0.0, 1.0];
    let grid = [0.0, 2.0, 4.0, 100.0];
    let col = [0.3, 0.7, 0.6, 0.2];
    let mut surv = Array2::<f64>::zeros((4, grid.len()));
    for k in 0..grid.len() {
        for i in 0..4 {
            surv[[i, k]] = col[i];
        }
    }
    let g = KaplanMeier::fit_censoring(&time, &event);
    // Horizon 5 drops the extrapolation point at t=100: integral runs [0,4].
    let restricted =
        integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, 5.0, &g).unwrap();
    let full =
        integrated_ipcw_brier_score(surv.view(), &time, &event, &grid, f64::INFINITY, &g)
            .unwrap();
    // The huge [4,100] tail interval dominates the full integral, so the two
    // must differ substantially — the horizon guard is doing real work.
    assert!(
        (restricted - full).abs() > 1e-3,
        "horizon cutoff had no effect: restricted={restricted} full={full}"
    );
}

#[test]
fn integrated_ipcw_brier_rejects_malformed_grids() {
    let time = [2.0, 8.0];
    let event = [1.0, 0.0];
    let surv = Array2::<f64>::from_elem((2, 3), 0.5);
    let g = KaplanMeier::fit_censoring(&time, &event);
    // Non-increasing grid.
    let bad = [0.0, 2.0, 1.0];
    assert!(
        integrated_ipcw_brier_score(surv.view(), &time, &event, &bad, f64::INFINITY, &g)
            .is_none()
    );
    // Grid width mismatched to the survival matrix.
    let short = [0.0, 1.0];
    assert!(
        integrated_ipcw_brier_score(surv.view(), &time, &event, &short, f64::INFINITY, &g)
            .is_none()
    );
}

/// `S(t) = exp(-rate*t)` sampled on `times`, one identical row per subject.
fn exponential_survival_result(times: Vec<f64>, rate: f64, rows: usize) -> SurvivalPredictResult {
    let t = times.len();
    let mut survival = Array2::<f64>::zeros((rows, t));
    let mut hazard = Array2::<f64>::zeros((rows, t));
    let mut cumulative_hazard = Array2::<f64>::zeros((rows, t));
    for i in 0..rows {
        for (j, &time) in times.iter().enumerate() {
            survival[[i, j]] = (-rate * time).exp();
            hazard[[i, j]] = rate;
            cumulative_hazard[[i, j]] = rate * time;
        }
    }
    SurvivalPredictResult {
        times,
        hazard,
        survival,
        cumulative_hazard,
        linear_predictor: Array1::zeros(rows),
        likelihood_mode: SurvivalLikelihoodMode::MarginalSlope,
        survival_se: None,
        eta_se: None,
        covariance_source: None,
        survival_plugin: None,
        survival_lower: None,
        survival_upper: None,
    }
}

#[test]
fn rmst_over_prediction_horizon_matches_the_exponential_closed_form() {
    // For S(t) = exp(-rate*t) the restricted mean has a closed form,
    // RMST(tau) = (1 - exp(-rate*tau)) / rate, so the trapezoid sum is
    // checked against an exact value rather than against itself.
    let rate = 0.35_f64;
    let tau = 4.0_f64;
    let steps = 4000_usize;
    let times: Vec<f64> = (1..=steps)
        .map(|k| tau * (k as f64) / (steps as f64))
        .collect();
    let result = exponential_survival_result(times, rate, 3);

    let rmst = result
        .rmst_over_prediction_horizon()
        .expect("a positive finite grid yields an RMST column");
    let expected = (1.0 - (-rate * tau).exp()) / rate;

    assert!((rmst.tau - tau).abs() < 1e-12, "tau = {}", rmst.tau);
    assert_eq!(rmst.values.len(), 3);
    for value in rmst.values.iter() {
        // Trapezoid error on a convex curve is O(h^2); h = tau/steps here,
        // so 1e-6 is several orders above the discretization floor and
        // still far below any difference that would matter clinically.
        assert!(
            (value - expected).abs() < 1e-6,
            "rmst {value} vs closed form {expected}"
        );
    }
}

#[test]
fn rmst_over_prediction_horizon_integrates_to_the_last_grid_time() {
    let times = vec![0.5, 1.0, 2.5, 6.0];
    let result = exponential_survival_result(times.clone(), 0.2, 2);

    let horizon = result
        .rmst_over_prediction_horizon()
        .expect("non-empty grid");
    let explicit = result
        .restricted_mean_survival_time(6.0)
        .expect("explicit tau at the same horizon");

    assert_eq!(horizon.tau, 6.0, "tau is the last grid point");
    assert_eq!(horizon.values, explicit, "no second integration policy");
}

#[test]
fn rmst_over_prediction_horizon_declines_an_empty_or_degenerate_grid() {
    let empty = exponential_survival_result(Vec::new(), 0.2, 2);
    assert!(empty.rmst_over_prediction_horizon().is_none(), "empty grid");

    // A grid whose only point is the time origin encloses no area.
    let origin_only = exponential_survival_result(vec![0.0], 0.2, 2);
    assert!(
        origin_only.rmst_over_prediction_horizon().is_none(),
        "tau = 0 encloses no area"
    );
}

#[test]
fn rmst_over_prediction_horizon_declines_a_non_finite_curve() {
    let mut result = exponential_survival_result(vec![1.0, 2.0], 0.2, 2);
    result.survival[[1, 0]] = f64::NAN;
    assert!(
        result.rmst_over_prediction_horizon().is_none(),
        "a NaN anywhere on the integrated span refuses the whole column"
    );
}

#[test]
fn overall_rmst_over_prediction_horizon_reads_the_all_cause_surface() {
    // Two endpoints, each with constant hazard `rate`, so the all-cause
    // survival is exp(-2*rate*t) and the closed form applies to it.
    let rate = 0.25_f64;
    let tau = 3.0_f64;
    let steps = 3000_usize;
    let times: Vec<f64> = (1..=steps)
        .map(|k| tau * (k as f64) / (steps as f64))
        .collect();
    let rows = 2_usize;
    let t = times.len();
    let mut overall_survival = Array2::<f64>::zeros((rows, t));
    for i in 0..rows {
        for (j, &time) in times.iter().enumerate() {
            overall_survival[[i, j]] = (-2.0 * rate * time).exp();
        }
    }
    let per_cause = Array2::<f64>::zeros((rows, t));
    let result = CompetingRisksPredictResult {
        times,
        endpoint_names: vec!["a".to_string(), "b".to_string()],
        hazard: vec![per_cause.clone(), per_cause.clone()],
        survival: vec![per_cause.clone(), per_cause.clone()],
        cumulative_hazard: vec![per_cause.clone(), per_cause.clone()],
        cif: vec![per_cause.clone(), per_cause],
        overall_survival,
        linear_predictor: vec![Array1::zeros(rows), Array1::zeros(rows)],
        likelihood_mode: SurvivalLikelihoodMode::MarginalSlope,
        covariance_source: None,
        hazard_se: None,
        survival_se: None,
        cumulative_hazard_se: None,
        cif_se: None,
        overall_survival_se: None,
        eta_se: None,
        bands: None,
    };

    let rmst = result
        .overall_rmst_over_prediction_horizon()
        .expect("all-cause RMST over a positive grid");
    let expected = (1.0 - (-2.0 * rate * tau).exp()) / (2.0 * rate);

    assert!((rmst.tau - tau).abs() < 1e-12);
    for value in rmst.values.iter() {
        assert!(
            (value - expected).abs() < 1e-6,
            "all-cause rmst {value} vs closed form {expected}"
        );
    }
}

/// Deterministic uniforms and normals for the gam#3038 fixture.
struct Lcg3038(u64);

impl Lcg3038 {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.unit().ln()).sqrt();
        radius * (2.0 * std::f64::consts::PI * self.unit()).cos()
    }
}

/// gam#3038: a location-scale survival fit whose time channel sits against
/// its cone publishes its posterior-mean surfaces by integrating the
/// cone-truncated posterior `π = N(β_unc, Σ) | C` it reports, not the
/// moment-matched normal `N(E_π[β], Σ_π)`, whose sigma points leave `C`
/// where the model has no hazard.
///
/// The reference is independent of the rule: rejection draws from the
/// ambient `N(β_unc, Σ)` kept only inside `C`, every draw replayed through
/// the per-coefficient survival law. The published survival must match the
/// Monte Carlo `E_π[S]` and the published density `h·S` its `E_π[f]`, each
/// within four Monte Carlo standard errors plus the rule's certified
/// relative accuracy.
#[test]
fn location_scale_posterior_mean_integrates_the_cone_truncated_law_3038() {
    use crate::fit_orchestration::FitConfig;
    use crate::inference::model::FittedModel;
    use crate::inference::model_payload_builders::fit_formula_to_payload;
    use crate::survival::location_scale::factorize_psd_covariance;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // `log T = 0.4x + e^{−0.3 + 0.4x}·ε` with independent censoring: a
    // linear log-scale, as in the report.
    let n = 400;
    let mut rng = Lcg3038(0x3038);
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.normal();
        let log_t = 0.4 * x + (-0.3 + 0.4 * x).exp() * rng.normal();
        let censor = (-0.5 + 2.5 * rng.unit()).exp();
        let event_time = log_t.exp();
        let (time, event) = if event_time <= censor {
            (event_time, 1)
        } else {
            (censor, 0)
        };
        records.push(csv::StringRecord::from(vec![
            format!("{time:.17e}"),
            event.to_string(),
            format!("{x:.17e}"),
        ]));
    }
    let headers = ["time", "event", "x"].iter().map(|s| s.to_string()).collect();
    let data = gam_data::encode_recordswith_inferred_schema(headers, records)
        .expect("encode the #3038 fixture");
    let config = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some("x".to_string()),
        ..FitConfig::default()
    };
    let payload =
        fit_formula_to_payload("Surv(time, event) ~ x".to_string(), &data, &config)
            .expect("survival location-scale fit");
    let model = FittedModel::from_payload(payload);
    let mode = SurvivalPredictionCovarianceMode::Conditional;

    assert_eq!(
        SurvivalPosteriorIntegration::default_for(&model, mode).expect("default integration"),
        SurvivalPosteriorIntegration::TruncatedLaw,
        "the fixture's posterior must carry an active cone"
    );

    let frame = ndarray::array![[1.0, 0.0, -1.5], [1.0, 0.0, 0.0], [1.0, 0.0, 1.5]];
    let col_map = data.column_map();
    let zeros = Array1::<f64>::zeros(frame.nrows());
    let times = [0.25, 0.5, 1.0, 2.0, 4.0];
    let request = |estimand| SurvivalPredictRequest {
        model: &model,
        data: frame.view(),
        col_map: &col_map,
        training_headers: Some(&data.headers),
        primary_offset: &zeros,
        noise_offset: &zeros,
        time_grid: Some(&times),
        with_uncertainty: false,
        estimand,
    };
    let published = predict_survival(request(SurvivalPredictEstimand::PosteriorMean), mode)
        .expect("the posterior-mean surfaces integrate the truncated law");
    let sigma_point = predict_survival_posterior_mean_with(
        request(SurvivalPredictEstimand::PosteriorMean),
        mode,
        SurvivalPosteriorIntegration::SigmaPoint,
    );
    eprintln!(
        "[3038] moment-matched normal sigma points: {}",
        match &sigma_point {
            Ok(_) => "published".to_string(),
            Err(error) => format!("refused: {error}"),
        }
    );

    // The ambient law the cone truncates, in raw coefficients.
    let fit = fit_result_from_saved_model_for_prediction(&model).expect("saved fit");
    let geometry = fit.geometry.as_ref().expect("fit geometry");
    let constrained = geometry
        .constrained_posterior
        .as_ref()
        .expect("constrained posterior");
    let correction = constrained
        .correction()
        .expect("available moments")
        .expect("an active cone");
    let gauge = &geometry.coefficient_gauge;
    let unconstrained = constrained.unconstrained_center().expect("ambient centre");
    let lift = gauge.t_full.dot(&correction.lift);
    let ambient = fit.beta_covariance().expect("conditional covariance")
        + &lift
            .dot(&correction.removed_normal_variance)
            .dot(&lift.t());
    let factor = factorize_psd_covariance(&ambient, "#3038 ambient covariance")
        .expect("ambient factor")
        .factor;
    let center = gauge.t_full.dot(unconstrained) + &gauge.affine_shift;
    // Raw to active coordinates: every raw displacement `L·z` lies in the
    // range of the full-column-rank gauge `T`, so `T d_a = d_raw` is solved
    // exactly through its normal equations.
    let normal = gauge.t_full.t().dot(&gauge.t_full);
    let normal_inverse = {
        let eig = factorize_psd_covariance(&normal, "#3038 gauge normal equations")
            .expect("gauge factor");
        let scaled = &eig.eigenvectors * &eig.inv_sqrt_eigenvalues;
        scaled.dot(&scaled.t())
    };

    let chunks = 16usize;
    let per_chunk = 250usize;
    let (n_rows, n_times) = published.survival.dim();
    let sums = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut rng = Lcg3038(0x5eed_3038 ^ ((chunk as u64 + 1) << 32));
            let mut s1 = Array2::<f64>::zeros((n_rows, n_times));
            let mut s2 = Array2::<f64>::zeros((n_rows, n_times));
            let mut f1 = Array2::<f64>::zeros((n_rows, n_times));
            let mut f2 = Array2::<f64>::zeros((n_rows, n_times));
            let mut proposals = 0usize;
            let mut accepted = 0usize;
            while accepted < per_chunk {
                proposals += 1;
                let z = Array1::from_shape_fn(factor.ncols(), |_| rng.normal());
                let displacement = factor.dot(&z);
                let active = unconstrained
                    + &normal_inverse.dot(&gauge.t_full.t().dot(&displacement));
                let slack = constrained.constraints.a.dot(&active) - &constrained.constraints.b;
                if slack.iter().any(|&value| value < 0.0) {
                    continue;
                }
                accepted += 1;
                let draw_model =
                    saved_model_with_survival_coefficients(&model, &(&center + &displacement))
                        .expect("draw model");
                let draw = predict_survival_coefficient_law(
                    SurvivalPredictRequest {
                        model: &draw_model,
                        estimand: SurvivalPredictEstimand::Plugin,
                        ..request(SurvivalPredictEstimand::Plugin)
                    },
                    mode,
                )
                .expect("per-coefficient survival law at a feasible draw");
                let density = &draw.survival * &draw.hazard;
                s1 += &draw.survival;
                s2 += &draw.survival.mapv(|s| s * s);
                f1 += &density;
                f2 += &density.mapv(|f| f * f);
            }
            (s1, s2, f1, f2, proposals)
        })
        .reduce(
            || {
                (
                    Array2::<f64>::zeros((n_rows, n_times)),
                    Array2::<f64>::zeros((n_rows, n_times)),
                    Array2::<f64>::zeros((n_rows, n_times)),
                    Array2::<f64>::zeros((n_rows, n_times)),
                    0usize,
                )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
        );
    let draws = (chunks * per_chunk) as f64;
    eprintln!(
        "[3038] Monte Carlo: {} feasible draws of {} ambient proposals",
        chunks * per_chunk,
        sums.4
    );
    let mut worst_survival_gap = 0.0_f64;
    let mut largest_truncation_effect = 0.0_f64;
    let plugin = published
        .survival_plugin
        .as_ref()
        .expect("posterior-mean prediction carries the plug-in survival");
    for row in 0..n_rows {
        for time in 0..n_times {
            let cell = [row, time];
            let mc_survival = sums.0[cell] / draws;
            let mc_survival_se =
                ((sums.1[cell] / draws - mc_survival * mc_survival).max(0.0) / draws).sqrt();
            let mc_density = sums.2[cell] / draws;
            let mc_density_se =
                ((sums.3[cell] / draws - mc_density * mc_density).max(0.0) / draws).sqrt();
            let survival = published.survival[cell];
            let density = published.hazard[cell] * survival;
            let rule_accuracy = 2.0e-3 * (mc_survival * (1.0 - mc_survival)).sqrt();
            eprintln!(
                "[3038] row {row} t={:.2}: E[S] rule {survival:.6} MC {mc_survival:.6} \
                 (se {mc_survival_se:.1e}, plug-in {:.6}); E[f] rule {density:.6} MC \
                 {mc_density:.6} (se {mc_density_se:.1e})",
                times[time], plugin[cell]
            );
            assert!(
                (survival - mc_survival).abs() <= 4.0 * mc_survival_se + rule_accuracy,
                "row {row}, t={}: published E[S] {survival:.6} vs Monte Carlo \
                 {mc_survival:.6} (se {mc_survival_se:.2e})",
                times[time]
            );
            assert!(
                (density - mc_density).abs()
                    <= 4.0 * mc_density_se + 2.0e-3 * mc_density.abs(),
                "row {row}, t={}: published E[f] {density:.6} vs Monte Carlo \
                 {mc_density:.6} (se {mc_density_se:.2e})",
                times[time]
            );
            worst_survival_gap = worst_survival_gap.max((survival - mc_survival).abs());
            largest_truncation_effect =
                largest_truncation_effect.max((plugin[cell] - mc_survival).abs());
        }
    }
    eprintln!(
        "[3038] max |E[S] rule − MC| {worst_survival_gap:.2e}; max |plug-in − MC| \
         {largest_truncation_effect:.2e}"
    );
}

/// gam#3575: a Royston-Parmar fit whose monotone I-spline baseline has a
/// flat segment pins baseline coefficients on their bound `β_j = 0`. Its
/// posterior is `N(β̂ − H⁻¹g, H⁻¹)` restricted to the cone, and the
/// posterior-mean surfaces must integrate THAT law: a strictly positive
/// survival SE and moments matching a rejection Monte Carlo of the same
/// law. The fraction-to-boundary sigma-point rule this replaces collapsed
/// every node onto the mode, publishing the plug-in with SE ≈ 0.
///
/// The reference is independent of the rule: rejection draws from the
/// ambient `N(β_unc, Σ)` kept only inside the cone, every draw replayed
/// through the per-coefficient survival law. The published `E[S]` must
/// match the Monte Carlo mean within four Monte Carlo standard errors plus
/// the rule's certified relative accuracy, and the published variance
/// `SE²` the Monte Carlo variance within four standard errors of the
/// sample variance (from the fourth central moment) plus the accuracy the
/// rule certifies for `E[S²] − E[S]²`.
#[test]
fn royston_parmar_posterior_mean_integrates_the_cone_truncated_law_3575() {
    use crate::fit_orchestration::FitConfig;
    use crate::inference::model::FittedModel;
    use crate::inference::model_payload_builders::fit_formula_to_payload;
    use crate::survival::location_scale::factorize_psd_covariance;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    // Events in two clusters, early `t ∈ (0.1, 0.3)` and late
    // `t ∈ (8, 12)`, with subjects censored across the gap between them:
    // the gap carries exposure but no events, so the maximum-likelihood
    // baseline hazard is zero there and the monotone baseline presses
    // against its cone.
    let n = 300;
    let mut rng = Lcg3038(0x3575);
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.normal();
        let event_time = if rng.unit() < 0.5 {
            0.1 + 0.2 * rng.unit()
        } else {
            8.0 + 4.0 * rng.unit()
        };
        let censor = if rng.unit() < 0.3 {
            0.3 + 7.7 * rng.unit()
        } else {
            15.0
        };
        let (time, event) = if event_time <= censor {
            (event_time, 1)
        } else {
            (censor, 0)
        };
        records.push(csv::StringRecord::from(vec![
            format!("{time:.17e}"),
            event.to_string(),
            format!("{x:.17e}"),
        ]));
    }
    let headers = ["time", "event", "x"].iter().map(|s| s.to_string()).collect();
    let data = gam_data::encode_recordswith_inferred_schema(headers, records)
        .expect("encode the #3575 fixture");
    let config = FitConfig {
        survival_likelihood: Some("transformation".to_string()),
        ..FitConfig::default()
    };
    let payload =
        fit_formula_to_payload("Surv(time, event) ~ x".to_string(), &data, &config)
            .expect("Royston-Parmar survival fit");
    let model = FittedModel::from_payload(payload);
    let mode = SurvivalPredictionCovarianceMode::Conditional;

    let fit = fit_result_from_saved_model_for_prediction(&model).expect("saved fit");
    let geometry = fit.geometry.as_ref().expect("fit geometry");
    let constrained = geometry
        .constrained_posterior
        .as_ref()
        .expect("the Royston-Parmar fit publishes its cone-truncated posterior");
    eprintln!(
        "[3575] mode {:?}; published posterior mean {:?}",
        constrained.mode,
        fit.beta.to_vec()
    );
    assert_eq!(
        SurvivalPosteriorIntegration::default_for(&model, mode).expect("default integration"),
        SurvivalPosteriorIntegration::TruncatedLaw,
        "the fixture's posterior must carry an active cone"
    );

    let frame = ndarray::array![[1.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]];
    let col_map = data.column_map();
    let zeros = Array1::<f64>::zeros(frame.nrows());
    let times = [0.15, 0.25, 1.0, 4.0, 9.0, 11.0];
    let request = |estimand, with_uncertainty| SurvivalPredictRequest {
        model: &model,
        data: frame.view(),
        col_map: &col_map,
        training_headers: Some(&data.headers),
        primary_offset: &zeros,
        noise_offset: &zeros,
        time_grid: Some(&times),
        with_uncertainty,
        estimand,
    };
    let published = predict_survival(request(SurvivalPredictEstimand::PosteriorMean, true), mode)
        .expect("the posterior-mean surfaces integrate the truncated law");
    let published_se = published
        .survival_se
        .as_ref()
        .expect("an uncertainty request publishes the survival SE");
    let sigma_point = predict_survival_posterior_mean_with(
        request(SurvivalPredictEstimand::PosteriorMean, true),
        mode,
        SurvivalPosteriorIntegration::SigmaPoint,
    );
    match &sigma_point {
        Ok(result) => eprintln!(
            "[3575] fraction-to-boundary sigma points: survival SE {:?}",
            result.survival_se.as_ref().map(|se| se.iter().copied().collect::<Vec<f64>>())
        ),
        Err(error) => eprintln!("[3575] fraction-to-boundary sigma points refused: {error}"),
    }

    // The ambient law the cone truncates. The Royston-Parmar gauge is the
    // identity, so the active coordinates are the raw coefficients.
    let correction = constrained
        .correction()
        .expect("available moments")
        .expect("an active cone");
    let gauge = &geometry.coefficient_gauge;
    let unconstrained = constrained.unconstrained_center().expect("ambient centre");
    let lift = gauge.t_full.dot(&correction.lift);
    let ambient = fit.beta_covariance().expect("conditional covariance")
        + &lift
            .dot(&correction.removed_normal_variance)
            .dot(&lift.t());
    let factor = factorize_psd_covariance(&ambient, "#3575 ambient covariance")
        .expect("ambient factor")
        .factor;
    let center = gauge.t_full.dot(unconstrained) + &gauge.affine_shift;
    let normal = gauge.t_full.t().dot(&gauge.t_full);
    let normal_inverse = {
        let eig = factorize_psd_covariance(&normal, "#3575 gauge normal equations")
            .expect("gauge factor");
        let scaled = &eig.eigenvectors * &eig.inv_sqrt_eigenvalues;
        scaled.dot(&scaled.t())
    };

    let chunks = 16usize;
    let per_chunk = 250usize;
    let (n_rows, n_times) = published.survival.dim();
    let (samples, proposals) = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut rng = Lcg3038(0x5eed_3575 ^ ((chunk as u64 + 1) << 32));
            let mut samples = Vec::with_capacity(per_chunk);
            let mut proposals = 0usize;
            while samples.len() < per_chunk {
                proposals += 1;
                let z = Array1::from_shape_fn(factor.ncols(), |_| rng.normal());
                let displacement = factor.dot(&z);
                let active = unconstrained
                    + &normal_inverse.dot(&gauge.t_full.t().dot(&displacement));
                let slack = constrained.constraints.a.dot(&active) - &constrained.constraints.b;
                if slack.iter().any(|&value| value < 0.0) {
                    continue;
                }
                let draw_model =
                    saved_model_with_survival_coefficients(&model, &(&center + &displacement))
                        .expect("draw model");
                let draw = predict_survival_coefficient_law(
                    SurvivalPredictRequest {
                        model: &draw_model,
                        ..request(SurvivalPredictEstimand::Plugin, false)
                    },
                    mode,
                )
                .expect("per-coefficient survival law at a feasible draw");
                samples.push(draw.survival);
            }
            (samples, proposals)
        })
        .reduce(
            || (Vec::new(), 0usize),
            |mut a, b| {
                a.0.extend(b.0);
                (a.0, a.1 + b.1)
            },
        );
    let draws = samples.len() as f64;
    eprintln!(
        "[3575] Monte Carlo: {} feasible draws of {proposals} ambient proposals",
        samples.len()
    );
    let plugin = published
        .survival_plugin
        .as_ref()
        .expect("posterior-mean prediction carries the plug-in survival");
    let mut largest_jensen_gap = 0.0_f64;
    for row in 0..n_rows {
        for time in 0..n_times {
            let cell = [row, time];
            let mc_mean = samples.iter().map(|s| s[cell]).sum::<f64>() / draws;
            let central = |power: i32| {
                samples
                    .iter()
                    .map(|s| (s[cell] - mc_mean).powi(power))
                    .sum::<f64>()
                    / draws
            };
            let mc_variance = central(2);
            let mc_mean_se = (mc_variance / draws).sqrt();
            let mc_variance_se = ((central(4) - mc_variance * mc_variance).max(0.0) / draws).sqrt();
            let survival = published.survival[cell];
            let se = published_se[cell];
            let rule_accuracy = 2.0e-3 * (mc_mean * (1.0 - mc_mean)).sqrt();
            eprintln!(
                "[3575] row {row} t={:.2}: E[S] rule {survival:.6} MC {mc_mean:.6} \
                 (se {mc_mean_se:.1e}, plug-in {:.6}); SE rule {se:.3e} MC {:.3e}",
                times[time],
                plugin[cell],
                mc_variance.sqrt()
            );
            assert!(
                se > 0.0,
                "row {row}, t={}: the truncated posterior carries mass off the bound, yet the \
                 published survival SE is {se:e}",
                times[time]
            );
            assert!(
                (survival - mc_mean).abs() <= 4.0 * mc_mean_se + rule_accuracy,
                "row {row}, t={}: published E[S] {survival:.6} vs Monte Carlo {mc_mean:.6} \
                 (se {mc_mean_se:.2e})",
                times[time]
            );
            // `Var = E[S²] − E[S]²`: an error δ in each certified moment
            // moves it by at most `δ + 2·E[S]·δ ≤ 3δ`.
            assert!(
                (se * se - mc_variance).abs() <= 4.0 * mc_variance_se + 3.0 * rule_accuracy,
                "row {row}, t={}: published Var[S] {:.3e} vs Monte Carlo {mc_variance:.3e} \
                 (se {mc_variance_se:.2e})",
                times[time],
                se * se
            );
            largest_jensen_gap = largest_jensen_gap.max((survival - plugin[cell]).abs());
        }
    }
    eprintln!("[3575] max |E[S] − S(E[β])| {largest_jensen_gap:.2e}");
    assert!(
        largest_jensen_gap > 0.0,
        "the posterior-mean survival must integrate the law, not re-publish the plug-in"
    );
}

/// The sigma-point rule on a location-scale model replays its nodes on the
/// designs assembled once for the pass; the moments it publishes are the
/// sums the whole-re-prediction rule forms from 2·rank + 1 passes, which a
/// banded request still runs.
#[test]
fn location_scale_sigma_nodes_replay_the_whole_reprediction_rule() {
    use crate::fit_orchestration::FitConfig;
    use crate::inference::model::FittedModel;
    use crate::inference::model_payload_builders::fit_formula_to_payload;
    let n = 300;
    let mut rng = Lcg3038(0x5157);
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.normal();
        let log_t = 0.4 * x + (-0.3 + 0.4 * x).exp() * rng.normal();
        let censor = (-0.5 + 2.5 * rng.unit()).exp();
        let event_time = log_t.exp();
        let (time, event) = if event_time <= censor {
            (event_time, 1)
        } else {
            (censor, 0)
        };
        records.push(csv::StringRecord::from(vec![
            format!("{time:.17e}"),
            event.to_string(),
            format!("{x:.17e}"),
        ]));
    }
    let headers = ["time", "event", "x"].iter().map(|s| s.to_string()).collect();
    let data = gam_data::encode_recordswith_inferred_schema(headers, records)
        .expect("encode the fixture");
    let config = FitConfig {
        survival_likelihood: Some("location-scale".to_string()),
        survival_distribution: "gaussian".to_string(),
        noise_formula: Some("x".to_string()),
        ..FitConfig::default()
    };
    let payload = fit_formula_to_payload("Surv(time, event) ~ x".to_string(), &data, &config)
        .expect("survival location-scale fit");
    let model = FittedModel::from_payload(payload);
    let mode = SurvivalPredictionCovarianceMode::Conditional;
    let frame = ndarray::array![[1.0, 0.0, -1.5], [1.0, 0.0, 0.0], [1.0, 0.0, 1.5]];
    let col_map = data.column_map();
    let zeros = Array1::<f64>::zeros(frame.nrows());
    let times = [0.25, 0.5, 1.0, 2.0, 4.0];
    let request = || SurvivalPredictRequest {
        model: &model,
        data: frame.view(),
        col_map: &col_map,
        training_headers: Some(&data.headers),
        primary_offset: &zeros,
        noise_offset: &zeros,
        time_grid: Some(&times),
        with_uncertainty: false,
        estimand: SurvivalPredictEstimand::PosteriorMean,
    };
    let (fast_result, fast) = survival_sigma_point_posterior_moments(request(), mode, None)
        .expect("the sigma nodes replay on the assembled designs");
    // The reference: the rule's nodes each re-predicted whole, summed as the
    // published moments are.
    let (posterior_mean, active_covariance, cone_coords) =
        survival_prediction_posterior_factor(&model, mode).expect("posterior factor");
    let plugin = predict_survival(
        SurvivalPredictRequest {
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
            ..request()
        },
        mode,
    )
    .expect("plug-in surfaces");
    let (n_rows, n_times) = plugin.survival.dim();
    let mut survival_sum = Array2::<f64>::zeros((n_rows, n_times));
    let mut density_sum = Array2::<f64>::zeros((n_rows, n_times));
    let mut hazard_sum = Array2::<f64>::zeros((n_rows, n_times));
    let mut eta_sum = Array1::<f64>::zeros(n_rows);
    let mut weight_sum = 0.0;
    for_each_survival_posterior_node(&posterior_mean, &active_covariance, &cone_coords, |node, weight| {
        let draw_model = saved_model_with_survival_coefficients(&model, node)?;
        let draw = predict_survival_coefficient_law(
            SurvivalPredictRequest {
                model: &draw_model,
                with_uncertainty: false,
                estimand: SurvivalPredictEstimand::Plugin,
                ..request()
            },
            mode,
        )?;
        weight_sum += weight;
        for row in 0..n_rows {
            eta_sum[row] += weight * draw.linear_predictor[row];
            for time in 0..n_times {
                let survival = draw.survival[[row, time]];
                let hazard = draw.hazard[[row, time]];
                let density =
                    conditional_event_density(survival, draw.cumulative_hazard[[row, time]], hazard)?;
                survival_sum[[row, time]] += weight * survival;
                density_sum[[row, time]] += weight * density;
                hazard_sum[[row, time]] += weight * hazard;
            }
        }
        Ok(())
    })
    .expect("the whole-re-prediction rule");
    assert!((weight_sum - 1.0).abs() < 1e-12, "weights sum to one, got {weight_sum}");
    assert_eq!(fast_result.times, plugin.times);
    assert_eq!(fast.survival.dim(), (n_rows, n_times));
    for ((row, time), moment) in fast.survival.indexed_iter() {
        let reference = survival_sum[[row, time]];
        assert!(
            (moment.mean() - reference).abs() <= 1e-12 * reference.abs().max(1.0),
            "survival at ({row}, {time}): {} vs {reference}",
            moment.mean()
        );
        let (density, reference) = (fast.density_mean[[row, time]], density_sum[[row, time]]);
        assert!(
            (density - reference).abs() <= 1e-12 * reference.abs().max(1.0),
            "density at ({row}, {time}): {density} vs {reference}"
        );
        let (hazard, reference) = (fast.hazard_mean[[row, time]], hazard_sum[[row, time]]);
        assert!(
            (hazard - reference).abs() <= 1e-12 * reference.abs().max(1.0),
            "hazard at ({row}, {time}): {hazard} vs {reference}"
        );
    }
    for (row, moment) in fast.eta.iter().enumerate() {
        let reference = eta_sum[row];
        assert!(
            (moment.mean() - reference).abs() <= 1e-12 * reference.abs().max(1.0),
            "eta at row {row}: {} vs {reference}",
            moment.mean()
        );
    }
}
