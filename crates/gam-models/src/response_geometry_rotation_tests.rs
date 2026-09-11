use super::*;
use crate::fit_orchestration::{FitConfig, FitRequest, materialize};
use ndarray::array;

#[test]
fn formula_shared_tangent_fit_preserves_output_rotations_2627() {
    let n = 80;
    let mut rows = Vec::with_capacity(n);
    let mut response = Array2::zeros((n, 2));
    for row in 0..n {
        let t = (row as f64 - 39.5) / 20.0;
        let x = (0.7 * t).sin() + 0.05 * t;
        let z = (1.1 * t).cos() - 0.1 * t * t;
        let y0 = 0.4 * x - 0.3 * z + 0.6 * (1.8 * t).sin() + 0.02 * t;
        let y1 = -0.2 * x + 0.5 * z + 0.4 * (2.3 * t).cos() - 0.03 * t;
        rows.push(csv::StringRecord::from(vec![
            y0.to_string(),
            x.to_string(),
            z.to_string(),
            t.to_string(),
        ]));
        response[[row, 0]] = y0;
        response[[row, 1]] = y1;
    }
    let dataset = gam_data::encode_recordswith_inferred_schema(
        vec!["r".into(), "x".into(), "z".into(), "w".into()],
        rows,
    )
    .expect("finite numeric formula fixture");
    let config = FitConfig {
        family: Some("gaussian".into()),
        link: Some("identity".into()),
        ..FitConfig::default()
    };
    let materialized = materialize("r ~ x + z + s(w)", &dataset, &config)
        .expect("Gaussian formula materialization");
    let FitRequest::Standard(standard) = materialized.request else {
        panic!("Gaussian formula must materialize a standard request");
    };
    let design =
        gam_terms::smooth::build_term_collection_design(standard.data.view(), &standard.spec)
            .expect("formula predictor design");
    let penalties = design
        .penalties
        .iter()
        .map(|penalty| SharedTangentPenalty::new(penalty.col_range.start, penalty.local.clone()))
        .collect();
    let request = SharedTangentRemlRequest::new(
        design.design,
        response,
        (*standard.weights).clone(),
        None,
        penalties,
    );
    let (s, c) = 0.6_f64.sin_cos();
    let rotation = array![[c, -s], [s, c]];
    let mut rotated_request = request.clone();
    rotated_request.response = request.response.dot(&rotation.t());
    let prepared =
        PreparedSharedTangent::from_request(request.clone()).expect("base prepared request");
    let rotated_prepared = PreparedSharedTangent::from_request(rotated_request.clone())
        .expect("rotated prepared request");
    let rho = Array1::zeros(prepared.penalties.len());
    let fixed = prepared
        .evaluate(&rho)
        .expect("fixed smoothing base evaluation");
    let fixed_rotated = rotated_prepared
        .evaluate(&rho)
        .expect("fixed smoothing rotated evaluation");
    let fixed_error = max_error(
        &fixed_rotated.coefficients,
        &fixed.coefficients.dot(&rotation.t()),
    );
    eprintln!("fixed smoothing: coefficient rotation error={fixed_error:e}");
    assert!(
        fixed_error < 1.0e-7,
        "fixed smoothing rotation error={fixed_error:e}"
    );

    let base = fit_shared_tangent_reml(request).expect("base shared REML fit");
    let rotated = fit_shared_tangent_reml(rotated_request).expect("rotated shared REML fit");
    let coefficient_error = max_error(&rotated.coefficients, &base.coefficients.dot(&rotation.t()));
    let prediction_error = max_error(&rotated.fitted, &base.fitted.dot(&rotation.t()));
    eprintln!(
        "optimized: coefficient rotation error={coefficient_error:e}, prediction error={prediction_error:e}, \
         base lambdas={:?}, rotated lambdas={:?}, scores=({}, {}), iterations=({}, {})",
        base.lambdas,
        rotated.lambdas,
        base.reml_score,
        rotated.reml_score,
        base.outer_iterations,
        rotated.outer_iterations,
    );
    for (label, fit) in [("base", &base), ("rotated", &rotated)] {
        let rho = fit.lambdas.mapv(f64::ln);
        let evaluation = prepared.evaluate(&rho).expect("fit smoothing diagnostic");
        let rotated_evaluation = rotated_prepared.evaluate(&rho).expect("same-point rotated diagnostic");
        eprintln!(
            "{label}: gradient={:?}, rotated_gradient={:?}, Hessian={:?}, fixed_point_coefficient_error={:e}, certificate={:?}",
            evaluation.gradient, rotated_evaluation.gradient, evaluation.hessian,
            max_error(&rotated_evaluation.coefficients, &evaluation.coefficients.dot(&rotation.t())),
            fit.outer_certificate,
        );
    }
    if let SufficientStatistics::Isotropic { root, .. } = &prepared.statistics {
        let gram = root.t().dot(root);
        eprintln!(
            "design Gram eigenvalues={:?}",
            gram.eigh(Side::Lower).expect("Gram spectrum").0
        );
    }
    for (index, penalty) in prepared.penalties.iter().enumerate() {
        eprintln!(
            "penalty {index}: range={}..{}, rank={}, eigenvalues={:?}",
            penalty.column_start,
            penalty.column_start + penalty.local.nrows(),
            penalty.rank,
            penalty.local.eigh(Side::Lower).expect("penalty spectrum").0
        );
    }
    assert!(
        coefficient_error < 1.0e-7,
        "optimized coefficient rotation error={coefficient_error:e}"
    );
    assert!(
        prediction_error < 1.0e-7,
        "optimized prediction rotation error={prediction_error:e}"
    );
    // Rotating the response only relabels the outputs, so the REML criterion is
    // rotation-invariant and both fits certify the SAME optimum, stopping at two
    // points inside its certified ball. To first order
    // `ρ̂_base − ρ̂_rot = H⁻¹·(g_base − g_rot)`, so the log-λ displacement is bounded
    // by `(‖Pg_base‖ + ‖Pg_rot‖)/σ_min(H)` at the base optimum: the amplification
    // this criterion actually has, rather than an absolute constant on λ.
    let base_rho = base.lambdas.mapv(f64::ln);
    let rotated_rho = rotated.lambdas.mapv(f64::ln);
    let base_hessian = prepared
        .evaluate(&base_rho)
        .expect("same-point base diagnostic")
        .hessian;
    let sigma_min = base_hessian
        .eigh(Side::Lower)
        .expect("outer Hessian spectrum")
        .0
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let gradient_sum = base.outer_certificate.stationarity.projected_norm()
        + rotated.outer_certificate.stationarity.projected_norm();
    assert!(
        sigma_min > 0.0,
        "the certified optimum must be locally strict for a displacement ball to exist: \
         sigma_min={sigma_min:e}"
    );
    let rho_ball = gradient_sum / sigma_min;
    for (index, (base_value, rotated_value)) in base_rho.iter().zip(rotated_rho.iter()).enumerate()
    {
        let displacement = (base_value - rotated_value).abs();
        assert!(
            displacement <= rho_ball,
            "log-lambda[{index}] rotation displacement {displacement:e} exceeds the certified ball \
             {rho_ball:e} (projected-gradient sum {gradient_sum:e}, sigma_min {sigma_min:e})"
        );
    }
    assert!((base.sigma2 - rotated.sigma2).abs() <= 1.0e-9 * base.sigma2.max(1.0));
}

fn max_error(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    assert_eq!(left.dim(), right.dim());
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn streamed_qr_preserves_fit_across_chunks_and_zero_weights_2627() {
    let design = Array2::from_shape_fn((9, 3), |(row, col)| {
        let t = (row as f64 - 4.0) / 4.0;
        match col {
            0 => 1.0,
            1 => t,
            _ => t * t,
        }
    });
    let response = Array2::from_shape_fn((9, 2), |(row, output)| {
        ((row + 1) as f64 * (output + 2) as f64 * 0.3).sin()
    });
    let request = SharedTangentRemlRequest::new(
        gam_linalg::test_support::no_densify_design(design),
        response,
        array![0.0, 0.0, 1.0, 0.8, 1.2, 1.0, 0.0, 0.9, 1.1],
        None,
        vec![
            SharedTangentPenalty::new(1, array![[1.0, 0.0], [0.0, 0.0]]),
            SharedTangentPenalty::new(1, array![[0.0, 0.0], [0.0, 1.0]]),
        ],
    );
    let single =
        PreparedSharedTangent::from_request(request.clone()).expect("one-chunk preparation");
    let mut streamed = single.clone();
    streamed.statistics =
        assemble_isotropic_statistics(&request.design, &request.response, &request.weights, 2)
            .expect("streaming preparation with an initial empty weighted chunk");
    let rho = array![-0.3, 0.7];
    let left = single.evaluate(&rho).expect("single-chunk evaluation");
    let right = streamed.evaluate(&rho).expect("streamed evaluation");
    assert!(max_error(&left.coefficients, &right.coefficients) < 2.0e-12);
    assert!((left.cost - right.cost).abs() < 2.0e-12);
    for (a, b) in left.gradient.iter().zip(right.gradient.iter()) {
        assert!((a - b).abs() < 2.0e-12);
    }
    assert!(max_error(&left.hessian, &right.hessian) < 2.0e-12);
}

#[test]
fn shared_tangent_rejects_negative_penalty_without_positive_range_2627() {
    let request = SharedTangentRemlRequest::new(
        gam_linalg::test_support::no_densify_design(array![[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]),
        array![[0.2], [0.3], [0.7]],
        Array1::ones(3),
        None,
        vec![SharedTangentPenalty::new(1, array![[-1.0]])],
    );
    assert!(PreparedSharedTangent::from_request(request).is_err());
}

#[test]
fn shared_tangent_penalty_rank_is_independent_of_strength_2627() {
    let direction = array![[1.0, 1.0], [1.0, -1.0]] / 2.0_f64.sqrt();
    let penalties = (0..2)
        .map(|col| {
            let axis = direction.column(col).to_owned();
            SharedTangentPenalty::new(1, Array2::from_shape_fn((2, 2), |(a, b)| axis[a] * axis[b]))
        })
        .collect();
    let prepared = PreparedSharedTangent::from_request(SharedTangentRemlRequest::new(
        gam_linalg::test_support::no_densify_design(array![
            [1.0, -1.0, 0.5],
            [1.0, 0.0, -0.3],
            [1.0, 1.0, 0.2],
            [1.0, 0.5, 1.0],
        ]),
        array![[0.2, 0.5], [0.1, -0.2], [1.0, 0.8], [0.3, 0.4]],
        Array1::ones(4),
        None,
        penalties,
    ))
    .expect("orthogonal rank-one penalties");
    for rho in [array![0.0, 0.0], array![-20.0, 0.0], array![0.0, -20.0]] {
        let (penalty, lambdas) = prepared
            .combined_penalty(&rho)
            .expect("finite positive strengths");
        let spectrum = prepared
            .combined_penalty_spectrum(&penalty, &lambdas)
            .expect("positive curvature on the declared penalty range");
        assert_eq!(
            spectrum.rank, 2,
            "positive strengths cannot change the penalty nullity"
        );
        assert!((spectrum.log_pseudo_determinant - rho.sum()).abs() < 1.0e-6);
        for j in 0..2 {
            assert!((spectrum.traces[j] - 1.0).abs() < 1.0e-6);
            for k in 0..2 {
                let expected = if j == k { 1.0 } else { 0.0 };
                assert!((spectrum.cross_traces[[j, k]] - expected).abs() < 1.0e-6);
            }
        }
    }
}
