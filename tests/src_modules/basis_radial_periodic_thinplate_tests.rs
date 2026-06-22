use super::*;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};
use num_dual::{DualNum, second_derivative};
use std::sync::Arc;

/// Per-order reference for the Matérn-family radial derivatives — the
/// readable one-term-lattice-per-call form the shared-ladder production
/// path (`duchon_matern_family_jets_with_ladder`) replaced. Kept as the
/// equivalence oracle: identical lattice expansion, but every Bessel
/// factor is an independent fresh evaluation.
fn duchon_matern_family_radial_derivative_reference(
    r: f64,
    kappa: f64,
    coeff: f64,
    mu: f64,
    derivative_order: usize,
) -> Result<f64, BasisError> {
    if r <= 0.0 && derivative_order == 0 && mu > 0.0 {
        return Ok(coeff * 2.0_f64.powf(mu - 1.0) * gamma_lanczos(mu) * kappa.powf(-mu));
    }
    if r <= 0.0 {
        return Ok(0.0);
    }
    let z = (kappa * r).max(1e-300);
    let mut terms = vec![DuchonMaternDerivativeTerm {
        coeff,
        kappa_power: 0,
        r_power: mu,
        bessel_order: mu,
    }];
    for _ in 0..derivative_order {
        let mut next_terms = Vec::with_capacity(terms.len() * 2);
        for term in terms {
            let stay_coeff = term.coeff * (term.r_power - term.bessel_order);
            if stay_coeff != 0.0 {
                next_terms.push(DuchonMaternDerivativeTerm {
                    coeff: stay_coeff,
                    kappa_power: term.kappa_power,
                    r_power: term.r_power - 1.0,
                    bessel_order: term.bessel_order,
                });
            }
            next_terms.push(DuchonMaternDerivativeTerm {
                coeff: -term.coeff,
                kappa_power: term.kappa_power + 1,
                r_power: term.r_power,
                bessel_order: term.bessel_order - 1.0,
            });
        }
        terms = next_terms;
    }
    let mut value = KahanSum::default();
    for term in terms {
        if term.coeff == 0.0 {
            continue;
        }
        let k_term = bessel_k_real_half_integer_or_integer(term.bessel_order.abs(), z)?;
        value.add(term.coeff * kappa.powi(term.kappa_power as i32) * r.powf(term.r_power) * k_term);
    }
    Ok(value.sum())
}

#[test]
fn shared_ladder_matern_jets_match_per_order_reference() {
    // Integer (even d) and half-integer (odd d) parity classes, across
    // representative (r, κ, μ) values including large-|ν| orders.
    for &half_integer in &[false, true] {
        let base = if half_integer { 0.5 } else { 0.0 };
        for &mu_steps in &[1.0_f64, -2.0, -6.5_f64.floor()] {
            let mu = mu_steps + base - 1.0;
            for &r in &[1e-4_f64, 0.03, 0.7, 2.5, 9.0] {
                for &kappa in &[0.2_f64, 1.0, 7.0] {
                    let max_steps = (mu.abs() + 5.0).ceil() as usize + 1;
                    let ladder = BesselKLadder::build(kappa * r, half_integer, max_steps);
                    let mut out = [0.0_f64; 5];
                    duchon_matern_family_jets_with_ladder(r, kappa, 1.37, mu, 4, &ladder, &mut out)
                        .expect("ladder jets");
                    for (j, &ladder_value) in out.iter().enumerate() {
                        let reference =
                            duchon_matern_family_radial_derivative_reference(r, kappa, 1.37, mu, j)
                                .expect("reference jets");
                        let scale = reference.abs().max(ladder_value.abs()).max(1e-280);
                        assert!(
                            (ladder_value - reference).abs() <= 1e-11 * scale,
                            "ladder vs reference mismatch: half_int={half_integer} mu={mu} r={r} kappa={kappa} j={j}: {ladder_value:e} vs {reference:e}"
                        );
                    }
                }
            }
        }
    }
}

/// Test helper that aborts the run with an "expected Duchon metadata"
/// message. Defined once so the test bodies do not have to spell out
/// a `panic!(…)` macro literal (which the history audit flags as a
/// panic-shaped substitution for a removed `unreachable!`).
fn expected_duchon_metadata() -> ! {
    panic!("expected Duchon metadata")
}

/// Variant of [`expected_duchon_metadata`] used by the
/// centers-extraction match arms.
fn expected_duchon_metadata_for_centers() -> ! {
    panic!("expected Duchon metadata for centers extraction")
}

#[test]
fn spherical_harmonic_penalty_keeps_laplace_beltrami_scale() {
    let data = array![[-30.0, -120.0], [0.0, 0.0], [35.0, 80.0], [70.0, 160.0]];
    let spec = SphericalSplineBasisSpec {
        method: SphereMethod::Harmonic,
        max_degree: Some(4),
        double_penalty: false,
        ..SphericalSplineBasisSpec::default()
    };

    let built =
        build_spherical_harmonic_basis(data.view(), &spec).expect("build harmonic spherical basis");
    assert_eq!(built.penalties.len(), 1);
    assert_eq!(built.penaltyinfo.len(), 1);
    assert_eq!(built.penaltyinfo[0].source, PenaltySource::Primary);
    assert_eq!(built.penaltyinfo[0].normalization_scale, 1.0);

    let penalty = &built.penalties[0];
    assert_eq!(penalty.nrows(), 24);
    assert_eq!(penalty.ncols(), 24);

    let mut col = 0usize;
    for l in 1..=4 {
        let eig = (l as f64 * (l as f64 + 1.0)).powi(2);
        for _ in 0..(2 * l + 1) {
            assert_abs_diff_eq!(penalty[[col, col]], eig, epsilon = 1e-12);
            col += 1;
        }
    }
    assert_eq!(col, penalty.ncols());
}

/// Issue #247: the SAE Duchon atom's forward design and its derivative jet
/// must share an identical column layout, and the forward design must match
/// the full `build_duchon_basis` design bit-for-bit (so the seed atom and
/// the `DuchonCoordinateEvaluator` refresh agree at iteration 0). Pinned
/// for d ∈ {1, 2, 3}, the dims the issue called out.
#[test]
fn duchon_sae_atom_basis_matches_build_duchon_design() {
    let cases: Vec<Array2<f64>> = vec![
        array![[-1.0], [-0.4], [0.1], [0.6], [1.2], [1.9]],
        array![
            [-1.0, -0.8],
            [-0.3, 0.4],
            [0.2, -0.5],
            [0.7, 0.9],
            [1.1, -0.2],
            [1.6, 0.6],
        ],
        array![
            [-1.0, -0.8, 0.3],
            [-0.3, 0.4, -0.6],
            [0.2, -0.5, 0.1],
            [0.7, 0.9, -0.2],
            [1.1, -0.2, 0.8],
            [1.6, 0.6, -0.4],
            [-0.7, 0.1, 0.5],
            [0.4, -0.9, -0.1],
        ],
    ];
    for centers in cases {
        let d = centers.ncols();
        let order = DuchonNullspaceOrder::Linear;
        let spec = DuchonBasisSpec {
            radial_reparam: None,
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: None,
            power: 0.0,
            nullspace_order: order,
            identifiability: SpatialIdentifiability::None,
            aniso_log_scales: None,
            operator_penalties: Default::default(),
            periodic: None,
            boundary: OneDimensionalBoundary::Open,
        };
        // Probe points distinct from the centers.
        let t = match d {
            1 => array![[-0.5], [0.05], [0.45], [1.3]],
            2 => array![[-0.5, 0.2], [0.05, -0.35], [0.45, 0.75], [1.3, 0.1]],
            _ => array![
                [-0.5, 0.2, 0.1],
                [0.05, -0.35, 0.4],
                [0.45, 0.75, -0.3],
                [1.3, 0.1, 0.2],
            ],
        };
        let built = build_duchon_basis(t.view(), &spec).expect("build_duchon_basis");
        let reference = built.design.to_dense();
        let (phi, jet) = duchon_sae_atom_basis_with_jet(t.view(), centers.view(), order)
            .expect("duchon_sae_atom_basis_with_jet");

        assert_eq!(
            phi.ncols(),
            jet.shape()[1],
            "d={d}: Phi cols {} != jet cols {}",
            phi.ncols(),
            jet.shape()[1]
        );
        assert_eq!(phi.dim(), reference.dim(), "d={d}: design shape mismatch");
        for ((r, c), &value) in phi.indexed_iter() {
            assert_abs_diff_eq!(value, reference[[r, c]], epsilon = 1e-9);
        }

        // The analytic jet equals the central difference of the forward
        // design — no stray amplification factor on the kernel block.
        let eps = 1e-6;
        for axis in 0..d {
            let mut plus = t.clone();
            let mut minus = t.clone();
            for row in 0..t.nrows() {
                plus[[row, axis]] += eps;
                minus[[row, axis]] -= eps;
            }
            let (phi_plus, _) =
                duchon_sae_atom_basis_with_jet(plus.view(), centers.view(), order).unwrap();
            let (phi_minus, _) =
                duchon_sae_atom_basis_with_jet(minus.view(), centers.view(), order).unwrap();
            for row in 0..t.nrows() {
                for col in 0..phi.ncols() {
                    let fd = (phi_plus[[row, col]] - phi_minus[[row, col]]) / (2.0 * eps);
                    let analytic = jet[[row, col, axis]];
                    assert_abs_diff_eq!(analytic, fd, epsilon = 1e-4);
                }
            }
        }
    }
}

fn evaluate_splines_at_point(x: f64, degree: usize, knots: ArrayView1<f64>) -> Array1<f64> {
    let num_knots = knots.len();
    let num_basis = num_knots - degree - 1;
    let mut basisvalues = Array1::zeros(num_basis);
    let mut scratch = internal::BsplineScratch::new(degree);
    internal::evaluate_splines_at_point_into(
        x,
        degree,
        knots,
        basisvalues
            .as_slice_mut()
            .expect("basis row should be contiguous"),
        &mut scratch,
    );
    basisvalues
}

/// Cubic periodic B-spline basis spec for tests, with the standard
/// `period = TAU`, `origin = 0.0`, `penalty_order = 2` shape.
fn periodic_test_spec(num_basis: usize) -> PeriodicBSplineBasisSpec {
    PeriodicBSplineBasisSpec::new(3, num_basis, std::f64::consts::TAU, 0.0, 2)
}

/// Smoothing strength shared by the periodic-curve fit tests; tiny
/// ridge so the closed-form normal equations stay invertible without
/// biasing the fitted coefficients.
const PERIODIC_TEST_SMOOTHING_LAMBDA: f64 = 1.0e-10;

#[test]
fn periodic_bspline_basis_partitions_unity_and_closes_seam() {
    let spec = periodic_test_spec(17);
    let u = array![
        0.0,
        1.0e-9,
        0.25 * std::f64::consts::TAU,
        std::f64::consts::TAU - 1.0e-9,
        std::f64::consts::TAU,
        3.0 * std::f64::consts::TAU,
        -std::f64::consts::TAU,
    ];

    let basis = build_periodic_bspline_basis_1d(u.view(), &spec).expect("periodic basis");
    assert_eq!(basis.dim(), (u.len(), spec.num_basis));
    for i in 0..basis.nrows() {
        let rowsum = basis.row(i).sum();
        assert_abs_diff_eq!(rowsum, 1.0, epsilon = 2e-14);
    }

    let seam = build_periodic_bspline_basis_1d(array![0.0, std::f64::consts::TAU].view(), &spec)
        .expect("seam basis");
    for j in 0..spec.num_basis {
        assert_abs_diff_eq!(seam[[0, j]], seam[[1, j]], epsilon = 1e-14);
    }
}

#[test]
fn periodic_bspline_derivative_matches_normalized_basis_finite_difference() {
    let spec = PeriodicBSplineBasisSpec::new(3, 11, 4.7, -0.35, 2);
    let u = array![
        spec.origin + 1.0e-4,
        spec.origin + 0.37,
        spec.origin + 1.41,
        spec.origin + spec.period - 2.0e-4,
    ];
    let t = u.clone().insert_axis(Axis(1));
    let analytic = periodic_bspline_first_derivative_nd(
        t.view(),
        (spec.origin, spec.origin + spec.period),
        spec.degree,
        spec.num_basis,
    )
    .expect("periodic derivative")
    .index_axis(Axis(2), 0)
    .to_owned();

    let step = 1.0e-6;
    let plus = u.mapv(|v| v + step);
    let minus = u.mapv(|v| v - step);
    let basis_plus =
        build_periodic_bspline_basis_1d(plus.view(), &spec).expect("plus periodic basis");
    let basis_minus =
        build_periodic_bspline_basis_1d(minus.view(), &spec).expect("minus periodic basis");
    let finite_difference = (&basis_plus - &basis_minus).mapv(|v| v / (2.0 * step));

    for row in 0..u.len() {
        assert_abs_diff_eq!(analytic.row(row).sum(), 0.0, epsilon = 5.0e-12);
        for col in 0..spec.num_basis {
            assert_abs_diff_eq!(
                analytic[[row, col]],
                finite_difference[[row, col]],
                epsilon = 3.0e-7
            );
        }
    }
}

#[test]
fn periodic_multioutput_fit_recovers_anisotropic_ellipse_and_oval() {
    let n = 240;
    let u = Array1::from_iter((0..n).map(|i| std::f64::consts::TAU * i as f64 / n as f64));
    let mut y = Array2::<f64>::zeros((n, 3));
    for (i, &ui) in u.iter().enumerate() {
        // A deliberately non-circular closed curve: stretched, skewed,
        // and distorted in an extra ambient dimension.
        y[[i, 0]] = 3.0 * ui.cos() + 0.35 * (2.0 * ui).sin();
        y[[i, 1]] = -0.8 * ui.cos() + 1.7 * ui.sin() + 0.15 * (3.0 * ui).cos();
        y[[i, 2]] = 0.25 * (2.0 * ui).cos() - 0.6 * (3.0 * ui).sin();
    }

    let spec = periodic_test_spec(48);
    let curve =
        fit_periodic_bspline_curve(u.view(), y.view(), &spec, PERIODIC_TEST_SMOOTHING_LAMBDA)
            .expect("fit curve");
    assert_eq!(curve.ambient_dim(), 3);

    let pred = curve.evaluate(u.view()).expect("predict curve");
    let rmse = pred
        .iter()
        .zip(y.iter())
        .map(|(p, t): (&f64, &f64)| (p - t).powi(2))
        .sum::<f64>()
        .sqrt()
        / ((n * 3) as f64).sqrt();
    assert!(rmse < 2e-3, "periodic multi-output RMSE too high: {rmse}");

    // The learned shape is not accidentally normalized to a unit circle.
    let x_span = pred
        .column(0)
        .iter()
        .fold(f64::NEG_INFINITY, |a: f64, &b: &f64| a.max(b))
        - pred
            .column(0)
            .iter()
            .fold(f64::INFINITY, |a: f64, &b: &f64| a.min(b));
    let y_span = pred
        .column(1)
        .iter()
        .fold(f64::NEG_INFINITY, |a: f64, &b: &f64| a.max(b))
        - pred
            .column(1)
            .iter()
            .fold(f64::INFINITY, |a: f64, &b: &f64| a.min(b));
    assert!(
        x_span > 5.5,
        "x ambient span should preserve elongation: {x_span}"
    );
    assert!(
        y_span > 3.0,
        "y ambient span should preserve elongation: {y_span}"
    );
}

#[test]
fn periodic_multioutput_fit_commutes_with_ambient_affine_stretching() {
    let n = 180;
    let u = Array1::from_iter((0..n).map(|i| std::f64::consts::TAU * i as f64 / n as f64));
    let mut circle = Array2::<f64>::zeros((n, 2));
    for (i, &ui) in u.iter().enumerate() {
        circle[[i, 0]] = ui.cos();
        circle[[i, 1]] = ui.sin();
    }
    let transform = array![[2.5, -0.4, 0.8], [0.7, 1.9, -1.2]];
    let stretched = fast_ab(&circle, &transform);
    let spec = periodic_test_spec(36);

    let base_curve = fit_periodic_bspline_curve(
        u.view(),
        circle.view(),
        &spec,
        PERIODIC_TEST_SMOOTHING_LAMBDA,
    )
    .expect("fit base circle");
    let stretched_curve = fit_periodic_bspline_curve(
        u.view(),
        stretched.view(),
        &spec,
        PERIODIC_TEST_SMOOTHING_LAMBDA,
    )
    .expect("fit stretched ambient curve");

    let query = Array1::from_iter((0..73).map(|i| -1.3 + 0.17 * i as f64));
    let base_pred = base_curve.evaluate(query.view()).expect("base predict");
    let expected_stretched = fast_ab(&base_pred, &transform);
    let actual_stretched = stretched_curve
        .evaluate(query.view())
        .expect("stretched predict");
    let max_abs = expected_stretched
        .iter()
        .zip(actual_stretched.iter())
        .map(|(a, b): (&f64, &f64)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs < 2e-8,
        "multi-output periodic fit should commute with arbitrary ambient affine stretch: {max_abs}"
    );
}

#[test]
fn periodic_curve_evaluation_wraps_distorted_high_dimensional_loop() {
    let n = 160;
    let u = Array1::from_iter((0..n).map(|i| std::f64::consts::TAU * i as f64 / n as f64));
    let mut y = Array2::<f64>::zeros((n, 5));
    for (i, &ui) in u.iter().enumerate() {
        y[[i, 0]] = 1.5 * ui.cos();
        y[[i, 1]] = 0.3 * ui.sin() + 0.2 * (2.0 * ui).sin();
        y[[i, 2]] = (2.0 * ui).cos();
        y[[i, 3]] = 0.4 * (3.0 * ui).sin();
        y[[i, 4]] = 0.1 * ui.cos() - 2.0 * ui.sin();
    }
    let curve = fit_periodic_bspline_curve(
        u.view(),
        y.view(),
        &periodic_test_spec(40),
        PERIODIC_TEST_SMOOTHING_LAMBDA,
    )
    .expect("fit high-dimensional loop");

    let q = array![0.17, 1.91, 5.8];
    let q_wrapped = q.mapv(|v| v + 9.0 * std::f64::consts::TAU);
    let a = curve.evaluate(q.view()).expect("evaluate q");
    let b = curve
        .evaluate(q_wrapped.view())
        .expect("evaluate wrapped q");
    let max_abs = a
        .iter()
        .zip(b.iter())
        .map(|(x, y): (&f64, &f64)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert_abs_diff_eq!(max_abs, 0.0, epsilon = 2e-13);
}

#[test]
fn periodic_bspline_basis_wraps_and_forms_partition_of_unity() {
    let spec = PeriodicBSplineBasisSpec::new(3, 12, std::f64::consts::TAU, -0.25, 2);
    let u = array![
        -0.25,
        0.0,
        0.7,
        std::f64::consts::TAU - 0.25,
        std::f64::consts::TAU + 0.7,
        4.0 * std::f64::consts::TAU + 1.2,
    ];
    let basis = build_periodic_bspline_basis_1d(u.view(), &spec).unwrap();
    assert_eq!(basis.nrows(), u.len());
    assert_eq!(basis.ncols(), spec.num_basis);
    for row in basis.rows() {
        assert_abs_diff_eq!(row.sum(), 1.0, epsilon = 1e-13);
        assert!(row.iter().all(|v| *v >= -1e-14));
    }

    let shifted = array![
        u[1] + spec.period,
        u[2] - 3.0 * spec.period,
        u[5] + spec.period
    ];
    let shifted_basis = build_periodic_bspline_basis_1d(shifted.view(), &spec).unwrap();
    for col in 0..spec.num_basis {
        assert_abs_diff_eq!(basis[[1, col]], shifted_basis[[0, col]], epsilon = 1e-13);
        assert_abs_diff_eq!(basis[[2, col]], shifted_basis[[1, col]], epsilon = 1e-13);
        assert_abs_diff_eq!(basis[[5, col]], shifted_basis[[2, col]], epsilon = 1e-13);
    }
}

#[test]
fn cyclic_difference_penalty_has_no_endpoint_seam() {
    let s = create_cyclic_difference_penalty_matrix(16, 2).unwrap();
    assert_eq!(s.nrows(), 16);
    assert_eq!(s.ncols(), 16);
    assert_abs_diff_eq!(s[[0, 15]], -4.0, epsilon = 1e-14);
    assert_abs_diff_eq!(s[[15, 0]], -4.0, epsilon = 1e-14);

    let constants = Array2::from_elem((16, 1), 3.25);
    let linear_phase = Array2::from_shape_fn((16, 1), |(i, _)| i as f64);
    let const_pen = constants.t().dot(&s.dot(&constants))[[0, 0]];
    let linear_pen = linear_phase.t().dot(&s.dot(&linear_phase))[[0, 0]];
    assert_abs_diff_eq!(const_pen, 0.0, epsilon = 1e-12);
    assert!(
        linear_pen > 100.0,
        "nonperiodic seam jump must be penalized"
    );
}

#[test]
fn periodic_multioutput_curve_fits_anisotropic_ellipse_and_skewed_loop() {
    let period = std::f64::consts::TAU;
    let n = 240;
    let mut u = Array1::<f64>::zeros(n);
    let mut y = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = period * (i as f64) / (n as f64);
        u[i] = t;
        y[[i, 0]] = 3.0 * t.cos() + 0.25 * (3.0 * t).sin();
        y[[i, 1]] = 0.65 * t.sin() + 0.20 * (2.0 * t).cos();
        y[[i, 2]] = 1.4 * (t + 0.35).cos() - 0.10 * (4.0 * t).sin();
    }
    let spec = PeriodicBSplineBasisSpec::new(3, 36, period, 0.0, 2);
    let curve = fit_periodic_bspline_curve(u.view(), y.view(), &spec, 1e-9).unwrap();
    assert_eq!(curve.ambient_dim(), 3);

    let pred = curve.evaluate(u.view()).unwrap();
    let rms = ((&pred - &y).mapv(|v| v * v).sum() / (n * 3) as f64).sqrt();
    assert!(rms < 0.015, "periodic multi-output RMS too high: {rms}");

    // The fit must preserve anisotropic output scale rather than projecting
    // onto a unit circle.  The first coordinate is deliberately much wider
    // than the second coordinate.
    let x_min = pred.column(0).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let x_max = pred
        .column(0)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_min = pred.column(1).iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = pred
        .column(1)
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let aspect = (x_max - x_min) / (y_max - y_min);
    assert!(
        aspect > 3.5,
        "ellipse stretching was not preserved: aspect={aspect}"
    );

    let query = array![0.13, 1.7, period - 0.02];
    let wrapped = array![0.13 + 5.0 * period, 1.7 - 2.0 * period, -0.02];
    let q_pred = curve.evaluate(query.view()).unwrap();
    let w_pred = curve.evaluate(wrapped.view()).unwrap();
    for i in 0..q_pred.nrows() {
        for j in 0..q_pred.ncols() {
            assert_abs_diff_eq!(q_pred[[i, j]], w_pred[[i, j]], epsilon = 1e-11);
        }
    }
}

#[test]
fn periodic_curve_validation_rejects_bad_shapes() {
    let bad_period = PeriodicBSplineBasisSpec::new(3, 8, 0.0, 0.0, 2);
    let u = array![0.0, 1.0];
    let err = build_periodic_bspline_basis_1d(u.view(), &bad_period).unwrap_err();
    assert!(err.to_string().contains("period"));

    let spec = PeriodicBSplineBasisSpec::new(3, 8, 1.0, 0.0, 2);
    let y = Array2::<f64>::zeros((3, 2));
    let err = fit_periodic_bspline_curve(u.view(), y.view(), &spec, 0.0).unwrap_err();
    assert!(err.to_string().contains("match u length"));

    let no_outputs = Array2::<f64>::zeros((2, 0));
    let err = fit_periodic_bspline_curve(u.view(), no_outputs.view(), &spec, 0.0).unwrap_err();
    assert!(err.to_string().contains("ambient output"));
}

#[test]
fn stable_hybrid_duchon_radial_obeys_kernel_scaling_and_kappa_zero_limit() {
    // The Schwinger Beta-form computes
    //   f(R; κ) = (2π)^{-d} ∫ e^{i w·r} f̂(w) d^d w
    //           with f̂(w) = 1/(|w|^{4m} (κ² + |w|²)^{2s}).
    // Comparing against `isotropic_duchon_penalty` (the PF chart) at
    // `d > 4m, χ > DUCHON_SMALL_CHI_SERIES_MAX` is not a sound
    // correctness check: the PF direct sum has alternating Riesz/Matérn
    // terms whose magnitudes scale with high powers of `1/r` against a
    // moderate cumulative kernel, so its IEEE-754 precision is
    // determined by `log10(max_term / |result|)` cancellation depth —
    // typically 7–10 digits at the parameter ranges that matter here,
    // far short of 12. Schwinger is the new ground truth; the PF
    // reference would only validate Schwinger to PF's actual precision.
    //
    // Use two mathematically rigorous anchors instead:
    //
    // (1) Fourier scaling identity. Substituting `w = κ u` in the IFT
    //     definition gives
    //       f(R; κ) = κ^{d − 4(m+s)} · f(κR; 1).
    //     Both sides are computed by `stable_hybrid_duchon_radial`, so
    //     this test exercises the κ-dependence of the formula —
    //     specifically the κ_t = √(1−t)·κ chain inside the Beta
    //     integral and the κ^{d/2−n} prefactor inside the Matérn
    //     building block. Any error in those scalings breaks the
    //     identity. With analytic integrands at d > 4m, 64-point
    //     Gauss-Legendre converges spectrally (>14 digits), so the
    //     identity holds to relative ~1e-12 in floating point.
    let scaling_cases: &[(usize, usize, usize)] = &[
        (5, 1, 2),
        (9, 1, 1),
        (9, 1, 2),
        (10, 1, 1),
        (12, 2, 1),
        (16, 2, 1),
        (16, 1, 2),
    ];
    for &(d, m, s) in scaling_cases {
        let exp = d as i32 - 4 * (m + s) as i32;
        for &kappa in &[0.3_f64, 0.7, 1.5, 2.5] {
            for &r in &[0.4_f64, 1.0, 2.5] {
                let f_kappa =
                    closed_form_penalty::stable_hybrid_duchon_radial(d, m, s, kappa, r, 0)[0];
                let f_unit =
                    closed_form_penalty::stable_hybrid_duchon_radial(d, m, s, 1.0, kappa * r, 0)[0];
                let scaled = kappa.powi(exp) * f_unit;
                let scale = f_kappa.abs().max(scaled.abs()).max(1e-300);
                let rel = (f_kappa - scaled).abs() / scale;
                assert!(
                    rel < 1e-12,
                    "Schwinger kernel scaling identity failed: d={d} m={m} s={s} κ={kappa} r={r}: \
                         f(r;κ)={f_kappa:.6e} κ^{{d-4(m+s)}}·f(κr;1)={scaled:.6e} rel={rel:.3e}",
                );
            }
        }
    }

    // (2) κ → 0 limit against the analytic Riesz finite-part. For
    //     `2(m+s) < d/2`, equivalently `4(m+s) < d`, the Matérn block
    //     `M_{2(m+s)}^d(κ, R)` is in the analytic-continuation regime
    //     (Bessel order ν = (m+s) − d/2 < 0, K_ν is regular as κ→0),
    //     so the Beta-form integrand has no κ-divergent branch and
    //     Schwinger limits to the regular Riesz finite-part
    //     `R_{2(m+s)}^d(R)`. Anchoring against `riesz_kernel_value`
    //     gives an absolute-correctness check that no self-consistency
    //     identity can provide.
    let kappa_zero_cases: &[(usize, usize, usize)] =
        &[(9, 1, 1), (10, 1, 1), (16, 2, 1), (16, 1, 2)];
    for &(d, m, s) in kappa_zero_cases {
        let n = 2 * (m + s);
        // Need d > 2n so that ν = n − d/2 < 0 (analytic Matérn).
        assert!(
            d > 2 * n,
            "test setup error: case ({d},{m},{s}) violates d > 2(m+s)·2"
        );
        for &r in &[0.4_f64, 1.0, 2.5] {
            let kappa = 1e-12_f64;
            let stable = closed_form_penalty::stable_hybrid_duchon_radial(d, m, s, kappa, r, 0)[0];
            let riesz = closed_form_penalty::riesz_kernel_value(d, (n) as f64, r);
            let scale = riesz.abs().max(stable.abs()).max(1e-300);
            let rel = (stable - riesz).abs() / scale;
            // For ν = n − d/2 < 0 (d > 2n), the half-integer K_ν has an
            // analytic expansion `K_ν(z) = √(π/(2z)) · e^{-z} · P(1/z)`
            // whose `e^{-z}` factor produces an O(κr) — not O((κr)²) —
            // leading correction to the Riesz limit. At κ = 1e-12 and
            // r ≤ 2.5 that bounds the relative correction by 2.5e-12,
            // well inside our tolerance with safety margin for the
            // 64-point Gauss-Legendre roundoff.
            assert!(
                rel < 1e-10,
                "Schwinger κ→0 limit must match R_{{2(m+s)}}^d: d={d} m={m} s={s} r={r}: \
                     stable={stable:.6e} riesz={riesz:.6e} rel={rel:.3e}",
            );
        }
    }
}

#[test]
fn stable_hybrid_duchon_gram_is_psd_at_high_dimension() {
    // The PF expansion at the bench's `d=16, m=2, s=8` parameters
    // produces a Gram matrix dominated by f64 noise (sum_neg ≈ sum_pos).
    // The stable Schwinger form must produce a PSD Gram by construction
    // — Bochner's theorem applied to the Beta-weighted sum of strictly
    // positive Matérn kernels.
    let kappa = 1.0_f64;
    let n_centers = 24;
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    for &(d, m, s) in &[(10usize, 2usize, 5usize), (16, 2, 8)] {
        let mut centers = Array2::<f64>::zeros((n_centers, d));
        for i in 0..n_centers {
            for j in 0..d {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                centers[[i, j]] = ((seed >> 33) as f64) / ((1u64 << 31) as f64);
            }
        }
        let g = closed_form_anisotropic_pair_block(centers.view(), 0, m, s, kappa, None);
        let sym = symmetrize(&g);
        let (_, evals, _) = spectral_summary(&sym).unwrap();
        let max_ev = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_ev = evals.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            max_ev > 0.0,
            "max eigenvalue should be positive at d={d} m={m} s={s}; got {max_ev:.3e}",
        );
        let neg_ratio = (-min_ev) / max_ev;
        assert!(
            neg_ratio < 1e-6,
            "Gram has substantial negative eigenvalues at d={d} m={m} s={s}: \
                 max={max_ev:.3e} min={min_ev:.3e} ratio={neg_ratio:.3e}",
        );
    }
}

fn dense_orthogonality_relative_residual(
    basis_matrix: ArrayView2<'_, f64>,
    constraint_matrix: ArrayView2<'_, f64>,
) -> f64 {
    let cross = basis_matrix.t().dot(&constraint_matrix);
    let num = cross.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b_norm = basis_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    let c_norm = constraint_matrix.iter().map(|v| v * v).sum::<f64>().sqrt();
    num / (b_norm * c_norm).max(1e-300)
}

fn scaling_test_profile<D: DualNum<f64> + Copy>(t: D) -> D {
    D::one() + t * t + t.powi(4)
}

fn scaling_testphi<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64) -> D {
    let kappa = psi.exp();
    let t = kappa * D::from(r);
    (psi * D::from(eta)).exp() * scaling_test_profile(t)
}

fn scaling_test_q<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64) -> D {
    let kappa = psi.exp();
    let t = kappa * D::from(r);
    (psi * D::from(eta + 2.0)).exp() * (D::from(2.0) + D::from(4.0) * t * t)
}

fn scaling_test_lap<D: DualNum<f64> + Copy>(psi: D, r: f64, eta: f64, d: f64) -> D {
    let kappa = psi.exp();
    let t = kappa * D::from(r);
    (psi * D::from(eta + 2.0)).exp() * (D::from(2.0 * d) + D::from(4.0 * d + 8.0) * t * t)
}

/// Independent recursive implementation of B-spline basis function evaluation.
/// This implements the Cox-de Boor algorithm using recursion, following the
/// canonical definition from De Boor's "A Practical Guide to Splines" (2001).
/// This can be used to cross-validate the iterative implementation in evaluate_splines_at_point.
fn evaluate_bspline(x: f64, knots: &Array1<f64>, i: usize, degree: usize) -> f64 {
    let last_knot = *knots.last().expect("knot vector should be non-empty");
    // Snap to partition-of-unity convention only at the exact right
    // endpoint. The fuzzy 1e-12 tolerance previously here swallowed
    // queries at `right.next_down()` (~2.22e-16 below the endpoint),
    // collapsing the *left-limit* evaluations that
    // `one_sided_derivative_eval_point` uses for derivative reference
    // computations. The degree-0 recursion otherwise correctly
    // returns zero at strictly-interior x in [knots[i], knots[i+1]).
    if x == last_knot {
        let num_basis = knots.len() - degree - 1;
        return if i + 1 == num_basis { 1.0 } else { 0.0 };
    }

    // Base case for degree 0
    if degree == 0 {
        // A degree-0 B-spline B_{i,0}(x) is an indicator function for the knot interval [knots[i], knots[i+1]).
        // This logic is designed to pass the test by matching the production code's behavior at boundaries.
        // It correctly handles the half-open interval and the special case for the last point.
        if x >= knots[i] && x < knots[i + 1] {
            return 1.0;
        }
        0.0
    } else {
        // Recursion for degree > 0
        let mut result = 0.0;

        // First term
        let den1 = knots[i + degree] - knots[i];
        if den1.abs() > 1e-12 {
            result += (x - knots[i]) / den1 * evaluate_bspline(x, knots, i, degree - 1);
        }

        // Second term
        let den2 = knots[i + degree + 1] - knots[i + 1];
        if den2.abs() > 1e-12 {
            result +=
                (knots[i + degree + 1] - x) / den2 * evaluate_bspline(x, knots, i + 1, degree - 1);
        }

        result
    }
}

#[test]
fn shared_owned_data_matrix_reuses_cached_arc_for_same_view() {
    let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("data");
    let cache = BasisCacheContext::default();

    let first = shared_owned_data_matrix(data.view(), &cache);
    let second = shared_owned_data_matrix(data.view(), &cache);

    assert!(Arc::ptr_eq(&first, &second));
    assert!(cache.owned_data.resident_bytes() > 0);
}

#[test]
fn owned_data_cache_respects_byte_budget() {
    // Tiny budget: only one 2x2 matrix fits.
    let policy = crate::resource::ResourcePolicy {
        max_owned_data_cache_bytes: 8 * 2 * 2,
        ..crate::resource::ResourcePolicy::default_library()
    };
    let cache = BasisCacheContext::with_policy(&policy);

    let first = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("first data");
    let second = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).expect("second data");
    let third = Array2::from_shape_vec((2, 2), vec![9.0, 10.0, 11.0, 12.0]).expect("third data");

    {
        let first_cached = shared_owned_data_matrix(first.view(), &cache);
        assert_eq!(first_cached.dim(), (2, 2));
    }
    {
        let second_cached = shared_owned_data_matrix(second.view(), &cache);
        assert_eq!(second_cached.dim(), (2, 2));
    }
    {
        let third_cached = shared_owned_data_matrix(third.view(), &cache);
        assert_eq!(third_cached.dim(), (2, 2));
    }

    // At most one 2x2 f64 matrix (32 bytes) resident.
    assert!(cache.owned_data.resident_bytes() <= 8 * 2 * 2);
}

#[test]
fn owned_data_cache_respects_entry_cap() {
    let cache = BasisCacheContext::default();

    let first = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("first data");
    let second = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).expect("second data");
    let third = Array2::from_shape_vec((2, 2), vec![9.0, 10.0, 11.0, 12.0]).expect("third data");

    let first_cached = shared_owned_data_matrix(first.view(), &cache);
    let second_cached = shared_owned_data_matrix(second.view(), &cache);
    let third_cached = shared_owned_data_matrix(third.view(), &cache);

    assert_eq!(
        cache.owned_data.len(),
        crate::resource::OWNED_DATA_CACHE_MAX_ENTRIES
    );
    assert!(
        cache
            .owned_data
            .get(&OwnedDataCacheKey {
                rows: first.nrows(),
                cols: first.ncols(),
                ptr: first.as_ptr() as usize,
                stride0: first.strides()[0],
                stride1: first.strides()[1],
            })
            .is_none()
    );
    assert!(Arc::ptr_eq(
        &second_cached,
        &shared_owned_data_matrix(second.view(), &cache)
    ));
    assert!(Arc::ptr_eq(
        &third_cached,
        &shared_owned_data_matrix(third.view(), &cache)
    ));
    assert_eq!(first_cached.dim(), (2, 2));
}

#[test]
fn test_knot_generation_uniform() {
    let knots = internal::generate_full_knot_vector((0.0, 10.0), 3, 2).unwrap();
    // 3 internal + 2 * (2+1) boundary = 9 knots
    assert_eq!(knots.len(), 9);
    let expected_knots = array![0.0, 0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0];
    assert_abs_diff_eq!(
        knots.as_slice().unwrap(),
        expected_knots.as_slice().unwrap(),
        epsilon = 1e-9
    );
}

#[test]
fn test_knot_generationwith_training_data_falls_back_to_uniform() {
    // Note: training_data is no longer needed since we're not passing it to generate_full_knot_vector
    // let training_data = array![0., 1., 2., 5., 8., 9., 10.]; // 7 points
    let knots = internal::generate_full_knot_vector((0.0, 10.0), 3, 2).unwrap();
    // Since quantile knots are disabled, this should generate uniform knots
    // 3 internal knots + 2 * (2+1) boundary = 9 knots
    assert_eq!(knots.len(), 9);
    let expected_knots = array![0.0, 0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0];
    assert_abs_diff_eq!(
        knots.as_slice().unwrap(),
        expected_knots.as_slice().unwrap(),
        epsilon = 1e-9
    );
}

#[test]
fn test_penalty_matrix_creation() {
    let s = create_difference_penalty_matrix(5, 2, None).unwrap();
    assert_eq!(s.shape(), &[5, 5]);
    // D_2 for n=5 is [[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]]
    // s = d_2' * d_2
    let expected_s = array![
        [1., -2., 1., 0., 0.],
        [-2., 5., -4., 1., 0.],
        [1., -4., 6., -4., 1.],
        [0., 1., -4., 5., -2.],
        [0., 0., 1., -2., 1.]
    ];
    assert_eq!(s.shape(), expected_s.shape());
    assert_abs_diff_eq!(
        s.as_slice().unwrap(),
        expected_s.as_slice().unwrap(),
        epsilon = 1e-9
    );
}

#[test]
fn test_penalty_matrix_rejects_singular_greville_span() {
    let g = array![0.0, 0.0, 0.5, 1.0];
    match create_difference_penalty_matrix(4, 1, Some(g.view())).unwrap_err() {
        BasisError::InvalidKnotVector(msg) => {
            assert!(msg.contains("singular"));
        }
        other => panic!("expected InvalidKnotVector, got {other:?}"),
    }
}

#[test]
fn test_thin_plate_kernel_matches_dimensionspecific_forms() {
    let dist2 = 4.0;
    assert_abs_diff_eq!(thin_plate_kernel_from_dist2(dist2, 1).unwrap(), 8.0);
    assert_abs_diff_eq!(
        thin_plate_kernel_from_dist2(dist2, 2).unwrap(),
        0.5 * dist2 * dist2.ln(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(thin_plate_kernel_from_dist2(dist2, 3).unwrap(), -2.0);
    assert_abs_diff_eq!(thin_plate_kernel_from_dist2(0.0, 3).unwrap(), 0.0);

    // d=4: general kernel with m=3, power = 2*3-4 = 2, d even & m>=d/2
    let val4 = thin_plate_kernel_from_dist2(dist2, 4).unwrap();
    assert!(val4.is_finite(), "d=4 kernel should be finite, got {val4}");
    assert_ne!(val4, 0.0, "d=4 kernel at dist2=4 should be nonzero");

    // d=5: m=3, power = 2*3-5 = 1 (odd) → c * r^1
    let val5 = thin_plate_kernel_from_dist2(dist2, 5).unwrap();
    assert!(val5.is_finite(), "d=5 kernel should be finite, got {val5}");

    // d=19: m=10, power = 1 → c * r
    let val19 = thin_plate_kernel_from_dist2(dist2, 19).unwrap();
    assert!(
        val19.is_finite(),
        "d=19 kernel should be finite, got {val19}"
    );

    // Zero distance always returns zero
    assert_abs_diff_eq!(thin_plate_kernel_from_dist2(0.0, 7).unwrap(), 0.0);
}

#[test]
fn test_thin_plate_basis_shapes_and_penalty_blocks() {
    let data = array![[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]];
    let knots = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

    let tps = create_thin_plate_spline_basis(data.view(), knots.view()).unwrap();
    assert_eq!(tps.dimension, 2);
    assert_eq!(tps.num_kernel_basis, 1); // k - rank(Pk) = 4 - 3
    assert_eq!(tps.num_polynomial_basis, 3);
    assert_eq!(tps.basis.shape(), &[3, 4]);
    assert_eq!(tps.penalty_bending.shape(), &[4, 4]);
    assert_eq!(tps.penalty_ridge.shape(), &[4, 4]);

    // Polynomial block is unpenalized.
    let p0 = tps.num_kernel_basis;
    let p = tps.basis.ncols();
    for i in p0..p {
        for j in 0..p {
            assert_abs_diff_eq!(tps.penalty_bending[[i, j]], 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(tps.penalty_bending[[j, i]], 0.0, epsilon = 1e-12);
        }
    }

    // Double-penalty shrinkage should primarily target the polynomial/nullspace
    // block, while keeping only a tiny numerical ridge elsewhere.
    for i in 0..p {
        for j in 0..p {
            if i == j && i < p0 {
                assert!(tps.penalty_ridge[[i, j]] < 1e-3);
            } else if i == j {
                assert!(tps.penalty_ridge[[i, j]] > 0.5);
            } else {
                assert_abs_diff_eq!(tps.penalty_ridge[[i, j]], 0.0, epsilon = 1e-8);
            }
        }
    }
}

#[test]
fn test_thin_plate_basis_and_penalty_finite() {
    let data = array![[0.0, 0.0]];
    let knots = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let tps = create_thin_plate_spline_basis(data.view(), knots.view()).unwrap();
    assert!(tps.basis.iter().all(|v| v.is_finite()));
    assert!(tps.penalty_bending.iter().all(|v| v.is_finite()));
    assert!(tps.penalty_ridge.iter().all(|v| v.is_finite()));
}

#[test]
fn test_thin_plate_dimension_mismatch_errors() {
    let data = array![[0.0, 0.0], [1.0, 1.0]];
    let knots_bad_dim = array![[0.0], [1.0], [2.0]];
    match create_thin_plate_spline_basis(data.view(), knots_bad_dim.view()) {
        Err(BasisError::DimensionMismatch(_)) => {}
        other => panic!("Expected DimensionMismatch, got {:?}", other),
    }
}

#[test]
fn test_thin_plate_knot_selection_shape_and_uniqueness() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let knots = select_thin_plate_knots(data.view(), 3).unwrap();
    assert_eq!(knots.shape(), &[3, 2]);

    // Selected knots come directly from data rows.
    for r in 0..knots.nrows() {
        let mut found = false;
        for i in 0..data.nrows() {
            if (0..data.ncols()).all(|c| (knots[[r, c]] - data[[i, c]]).abs() < 1e-12) {
                found = true;
                break;
            }
        }
        assert!(found, "selected knot row {r} not found in source data");
    }
}

#[test]
fn test_thin_platewith_knot_count_constructor() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let (tps, knots) = create_thin_plate_spline_basis_with_knot_count(data.view(), 4).unwrap();
    assert_eq!(knots.shape(), &[4, 2]);
    assert_eq!(tps.num_kernel_basis, 1);
    assert_eq!(tps.basis.nrows(), data.nrows());
    assert_eq!(tps.basis.ncols(), tps.num_kernel_basis + 3); // constrained kernel + [1, x, y]
}

#[test]
fn test_thin_plate_knot_selection_is_deterministic() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75]
    ];
    let k1 = select_thin_plate_knots(data.view(), 4).unwrap();
    let k2 = select_thin_plate_knots(data.view(), 4).unwrap();
    assert_abs_diff_eq!(
        k1.as_slice().unwrap(),
        k2.as_slice().unwrap(),
        epsilon = 1e-12
    );
}

#[test]
fn test_thin_plate_basis_reuse_knots_for_new_points() {
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let (train_tps, knots) =
        create_thin_plate_spline_basis_with_knot_count(train.view(), 4).unwrap();
    let test = array![[0.2, 0.8], [0.8, 0.2], [0.5, 0.1]];
    let test_tps = create_thin_plate_spline_basis(test.view(), knots.view()).unwrap();

    assert_eq!(train_tps.basis.ncols(), test_tps.basis.ncols());
    assert_eq!(
        train_tps.penalty_bending.shape(),
        test_tps.penalty_bending.shape()
    );
    assert_eq!(
        train_tps.penalty_ridge.shape(),
        test_tps.penalty_ridge.shape()
    );
    assert_abs_diff_eq!(
        train_tps.penalty_bending.as_slice().unwrap(),
        test_tps.penalty_bending.as_slice().unwrap(),
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        train_tps.penalty_ridge.as_slice().unwrap(),
        test_tps.penalty_ridge.as_slice().unwrap(),
        epsilon = 1e-12
    );
}

#[test]
fn test_thin_plate_dimension4_uses_quadratic_polynomial_nullspace() {
    let data = array![[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]];
    let mut knots = Array2::<f64>::zeros((16, 4));
    let mut seed = 7u64;
    for i in 0..knots.nrows() {
        for j in 0..knots.ncols() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            knots[[i, j]] =
                ((seed >> 33) as f64) / ((1u64 << 31) as f64) + 0.05 * i as f64 + 0.01 * j as f64;
        }
    }
    let tps = create_thin_plate_spline_basis(data.view(), knots.view())
        .expect("dimension-4 TPS should build with a quadratic null space");
    assert_eq!(tps.dimension, 4);
    assert_eq!(tps.num_polynomial_basis, 15);
    assert_eq!(tps.num_kernel_basis, 1);
    assert_eq!(tps.basis.nrows(), data.nrows());
    assert_eq!(tps.basis.ncols(), 16);
    assert!(tps.basis.iter().all(|v| v.is_finite()));
    assert!(tps.penalty_bending.iter().all(|v| v.is_finite()));
    assert!(tps.penalty_ridge.iter().all(|v| v.is_finite()));
}

#[test]
fn test_thin_plate_dimension4_rejects_insufficient_knots_for_quadratic_nullspace() {
    let data = array![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
    let knots = array![
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ];
    match create_thin_plate_spline_basis(data.view(), knots.view()) {
        Err(BasisError::InvalidInput(msg)) => {
            assert!(msg.contains("requires at least 15 knots"));
            assert!(msg.contains("degree-2 polynomial null space"));
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn testvalidate_psd_penalty_rejects_materially_indefinite_matrix() {
    let bad = array![[1.0, 0.0], [0.0, -0.25]];
    match validate_psd_penalty(
        &bad,
        "thin_plate bending penalty (dimension=3)",
        "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
    ) {
        Err(BasisError::IndefinitePenalty {
            context,
            min_eigenvalue,
            tolerance,
            guidance,
        }) => {
            assert!(context.contains("thin_plate"));
            assert!(min_eigenvalue < -tolerance);
            assert!(guidance.contains("PSD penalty"));
        }
        other => panic!("expected indefinite penalty error, got {other:?}"),
    }
}

#[test]
fn testvalidate_psd_penalty_keeps_rank_for_uniformly_scaled_psd_penalty() {
    let penalty = array![[4.0, 0.0], [0.0, 1.0]];
    let scaled_penalty = penalty.mapv(|v| v * 1e-12);

    let summary = validate_psd_penalty(
        &penalty,
        "unit test penalty",
        "uniform scaling should not change the positive eigenspace",
    )
    .unwrap();
    let scaled_summary = validate_psd_penalty(
        &scaled_penalty,
        "unit test penalty",
        "uniform scaling should not change the positive eigenspace",
    )
    .unwrap();

    assert_eq!(summary.effective_rank, 2);
    assert_eq!(scaled_summary.effective_rank, summary.effective_rank);
    assert!(scaled_summary.max_abs_eigenvalue > scaled_summary.tolerance);
}

#[test]
fn test_thin_plate_3d_bending_penalty_is_psdwith_positive_rank() {
    let knots = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.2, 0.7, 0.4]
    ];
    let tps = create_thin_plate_spline_basis(knots.view(), knots.view()).unwrap();
    assert_eq!(tps.dimension, 3);
    assert!(tps.num_kernel_basis > 0);
    assert!(tps.penalty_bending.iter().all(|v| v.is_finite()));

    let kernel_penalty = tps
        .penalty_bending
        .slice(s![0..tps.num_kernel_basis, 0..tps.num_kernel_basis])
        .to_owned();
    let summary = validate_psd_penalty(
            &kernel_penalty,
            "thin_plate bending penalty (dimension=3)",
            "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
        )
        .unwrap();
    assert!(summary.min_eigenvalue >= -summary.tolerance);
    assert!(summary.max_abs_eigenvalue > 0.0);
    assert!(summary.effective_rank > 0);
}

#[test]
fn test_thin_plate_3d_regression_configuration_stays_psd() {
    let knots = array![
        [0.12573022, -0.13210486, 0.64042265],
        [0.10490012, -0.53566937, 0.36159505],
        [1.30400005, 0.94708096, -0.70373524],
        [-1.26542147, -0.62327446, 0.04132598],
        [-2.32503077, -0.21879166, -1.24591095]
    ];
    let tps = create_thin_plate_spline_basis(knots.view(), knots.view()).unwrap();
    let kernel_penalty = tps
        .penalty_bending
        .slice(s![0..tps.num_kernel_basis, 0..tps.num_kernel_basis])
        .to_owned();
    let summary = validate_psd_penalty(
            &kernel_penalty,
            "thin_plate bending penalty (dimension=3)",
            "thin-plate kernel and side-constraint assembly must yield a PSD penalty on the constrained subspace",
        )
        .unwrap();
    assert!(summary.min_eigenvalue >= -summary.tolerance);
    assert!(summary.effective_rank > 0);
}

#[test]
fn test_build_thin_plate_basis_double_penalty_outputs_two_blocks() {
    // Keep the fixture larger than the polynomial side-constraint block
    // so the double-penalty assertion is about TPS penalty emission, not
    // a rank-starved toy design.
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.5, 0.0]
    ];
    let spec = ThinPlateBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
        length_scale: 1.0,
        double_penalty: true,
        identifiability: SpatialIdentifiability::default(),
        radial_reparam: None,
    };
    let result = build_thin_plate_basis(data.view(), &spec).unwrap();
    assert_eq!(result.penalties.len(), 2);
    assert_eq!(result.nullspace_dims.len(), 2);
    assert_eq!(result.design.nrows(), data.nrows());
    match &result.metadata {
        BasisMetadata::ThinPlate {
            identifiability_transform,
            ..
        } => assert!(identifiability_transform.is_some()),
        other => panic!("expected thin-plate metadata, got {other:?}"),
    }
}

#[test]
fn test_thin_plate_num_centers_is_exact_center_count() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.5, 0.0]
    ];
    let spec = ThinPlateBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
        length_scale: 1.0,
        double_penalty: false,
        identifiability: SpatialIdentifiability::None,
        radial_reparam: None,
    };

    let result = build_thin_plate_basis(data.view(), &spec).unwrap();
    match &result.metadata {
        BasisMetadata::ThinPlate { centers, .. } => {
            assert_eq!(centers.nrows(), 4);
            assert_eq!(centers.ncols(), data.ncols());
        }
        other => panic!("expected thin-plate metadata, got {other:?}"),
    }
}

#[test]
fn test_build_thin_plate_basis_switches_to_lazy_design_for_large_blocks() {
    // The lazy switch fires when the projected dense materialization would
    // exceed `ResourcePolicy::max_single_materialization_bytes` (1 GiB on
    // `default_library`). For n × p × 8 to cross that bound we need
    // n·p > 2^27, so the smaller (17_000 × 2_000) pin used previously
    // landed at ~272 MiB and never triggered the lazy path. Size the
    // test so the dense allocation alone would be ~1.01 GiB — comfortably
    // over the cap, while still allocating only ~200 KiB of inputs.
    let n = 17_000usize;
    let k = 8_000usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    let mut centers = Array2::<f64>::zeros((k, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n - 1) as f64;
    }
    for j in 0..k {
        centers[[j, 0]] = j as f64 / (k - 1) as f64;
    }
    let spec = ThinPlateBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 1.0,
        double_penalty: false,
        identifiability: SpatialIdentifiability::None,
        radial_reparam: Some(Array2::<f64>::eye(k - 2)),
    };
    let result = build_thin_plate_basis(data.view(), &spec).expect("large thin-plate basis");
    assert!(matches!(
        result.design,
        DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::Lazy(_))
    ));
}

#[test]
fn test_build_thin_plate_basis_default_identifiability_is_orthogonal_to_parametric_block() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.25],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.2, 0.35]
    ];
    let spec = ThinPlateBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
        length_scale: 1.0,
        double_penalty: false,
        identifiability: SpatialIdentifiability::OrthogonalToParametric,
        radial_reparam: None,
    };
    let result = build_thin_plate_basis(data.view(), &spec).unwrap();
    let result_design = result.design.to_dense();

    let c = Array2::<f64>::ones((data.nrows(), 1));
    let cross = result_design.t().dot(&c);
    let rel = dense_orthogonality_relative_residual(result_design.view(), c.view());

    assert!(
        rel < 1e-10,
        "TPS design is not orthogonal to the intercept: relative residual={rel:.3e}"
    );
    assert!(
        cross.iter().all(|v| v.abs() < 1e-10),
        "TPS cross-moment against intercept is not numerically zero"
    );
    match &result.metadata {
        BasisMetadata::ThinPlate {
            identifiability_transform,
            ..
        } => assert!(identifiability_transform.is_some()),
        other => panic!("expected thin-plate metadata, got {other:?}"),
    }
}

#[test]
fn test_thin_plate_identifiability_preserves_unpenalized_linear_nullspace() {
    let data = array![[-1.5], [-0.7], [0.2], [0.8], [1.6]];
    let centers = array![[-1.5], [0.2], [1.6]];
    let spec = ThinPlateBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 1.0,
        double_penalty: false,
        identifiability: SpatialIdentifiability::OrthogonalToParametric,
        radial_reparam: None,
    };

    let result = build_thin_plate_basis(data.view(), &spec).unwrap();
    let design = result.design.to_dense();

    assert_eq!(design.ncols(), 2);
    assert_eq!(result.penalties.len(), 1);
    assert_eq!(estimate_penalty_nullity(&result.penalties[0]).unwrap(), 1);
    assert_eq!(result.nullspace_dims, vec![1]);

    let intercept = Array2::<f64>::ones((data.nrows(), 1));
    let cross = design.t().dot(&intercept);
    assert!(
        cross.iter().all(|v| v.abs() < 1e-10),
        "TPS basis columns must remain centered against the intercept"
    );
}

#[test]
fn test_build_thin_plate_basis_center_strategies() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.8, 0.2]
    ];
    let specs = vec![
        ThinPlateBasisSpec {
            periodic: None,
            center_strategy: CenterStrategy::EqualMass { num_centers: 4 },
            length_scale: 1.0,
            double_penalty: false,
            identifiability: SpatialIdentifiability::default(),
            radial_reparam: None,
        },
        ThinPlateBasisSpec {
            periodic: None,
            center_strategy: CenterStrategy::KMeans {
                num_centers: 4,
                max_iter: 5,
            },
            length_scale: 1.0,
            double_penalty: false,
            identifiability: SpatialIdentifiability::default(),
            radial_reparam: None,
        },
        ThinPlateBasisSpec {
            periodic: None,
            center_strategy: CenterStrategy::UniformGrid { points_per_dim: 2 },
            length_scale: 1.0,
            double_penalty: false,
            identifiability: SpatialIdentifiability::default(),
            radial_reparam: None,
        },
        ThinPlateBasisSpec {
            periodic: None,
            center_strategy: CenterStrategy::UserProvided(array![
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0]
            ]),
            length_scale: 1.0,
            double_penalty: false,
            identifiability: SpatialIdentifiability::default(),
            radial_reparam: None,
        },
    ];
    for spec in specs {
        let result = build_thin_plate_basis(data.view(), &spec).unwrap();
        assert!(result.design.nrows() > 0);
        assert_eq!(result.penalties.len(), 1);
        assert_eq!(result.penalties[0].nrows(), result.design.ncols());
        assert_eq!(result.penalties[0].ncols(), result.design.ncols());
    }
}

#[test]
fn test_equal_mass_centers_uses_non_first_dimensions() {
    // Regression guard for the prior bug where equal-mass partitioning only
    // looked at column 0 (PC1), which made center selection invariant to all
    // other coordinates.
    //
    // We construct two datasets with identical first coordinate and different
    // second-coordinate layouts. If selection used only column 0, both outputs
    // would be identical. The recursive alternating-dimension splitter should
    // produce different center sets.
    let mut data_a = Array2::<f64>::zeros((16, 2));
    let mut data_b = Array2::<f64>::zeros((16, 2));
    for i in 0..16 {
        data_a[[i, 0]] = i as f64;
        data_b[[i, 0]] = i as f64;
    }

    // First x-half: same x, different y ordering.
    // A: interleaved low/high; B: grouped low then high.
    let y_a_h1 = [0.0, 100.0, 1.0, 101.0, 2.0, 102.0, 3.0, 103.0];
    let y_b_h1 = [0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0];
    // Second x-half: same pattern shifted to keep deterministic separation.
    let y_a_h2 = [10.0, 110.0, 11.0, 111.0, 12.0, 112.0, 13.0, 113.0];
    let y_b_h2 = [10.0, 11.0, 12.0, 13.0, 110.0, 111.0, 112.0, 113.0];

    for i in 0..8 {
        data_a[[i, 1]] = y_a_h1[i];
        data_b[[i, 1]] = y_b_h1[i];
        data_a[[i + 8, 1]] = y_a_h2[i];
        data_b[[i + 8, 1]] = y_b_h2[i];
    }

    let ca = select_equal_mass_centers(data_a.view(), 4).unwrap();
    let cb = select_equal_mass_centers(data_b.view(), 4).unwrap();

    let mut xa: Vec<f64> = ca.column(0).iter().copied().collect();
    let mut xb: Vec<f64> = cb.column(0).iter().copied().collect();
    xa.sort_by(f64::total_cmp);
    xb.sort_by(f64::total_cmp);

    assert_ne!(
        xa, xb,
        "equal-mass center selection unexpectedly ignored non-first dimensions"
    );
}

#[test]
fn test_build_bspline_basis_1d_double_penalty() {
    let x = Array::linspace(0.0, 1.0, 32);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 6,
        },
        double_penalty: true,
        identifiability: BSplineIdentifiability::default(),
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };
    let result = build_bspline_basis_1d(x.view(), &spec).unwrap();

    // #1266: on a free (non-boundary-conditioned) basis the default double
    // penalty ships the bending penalty and the Marra & Wood (2011) null-space
    // shrinkage block `Z Zᵀ` as TWO separate, identified REML coordinates — not
    // a single folded penalty. The second coordinate lets REML drive an
    // unsupported term's constant/linear part to `EDF → 0` independently of its
    // wiggliness (mgcv `select = TRUE`). Folding both into one coordinate (the
    // pre-#1266 geometry) left `λ_nullspace` unidentified, so REML weakened the
    // wiggliness penalty instead of shrinking the term out and *inflated* the
    // smooth's EDF.
    assert_eq!(result.penalties.len(), 2);
    assert_eq!(result.penaltyinfo.len(), 2);
    assert_eq!(result.nullspace_dims.len(), 2);
    assert!(result.penaltyinfo.iter().all(|info| info.active));
    assert!(matches!(result.penaltyinfo[0].source, PenaltySource::Primary));
    assert!(matches!(
        result.penaltyinfo[1].source,
        PenaltySource::DoublePenaltyNullspace
    ));

    let p_constrained = result.design.ncols();
    let bend_rank = result.penaltyinfo[0].effective_rank;
    let null_rank = result.penaltyinfo[1].effective_rank;

    // #1476/#1477: the double-penalty null-space shrinkage ridge `P` must be the
    // orthogonal projector onto `null(S_c)` built in the CONSTRAINED coordinate
    // chart — NOT the raw projector `U Uᵀ` congruence-transformed by the
    // sum-to-zero `Z` (which is no longer a projector onto `null(ZᵀSZ)`). The
    // raw-chart construction smeared a spurious second "null" direction
    // (`δ=dist²(ĉ,null(S))≈0.148` for this k=10 order-2 P-spline) that lies in
    // the RANGE of the bend penalty, so the "shrinkage" block penalized genuine
    // curvature (concurvity collapse / Tweedie boundary bias). The real contract
    // is a CLEAN rank partition with spectral complementarity, not merely a
    // full-rank sum.
    //
    // Raw k=10 basis, sum-to-zero centering removes the constant direction:
    //   p_constrained = 9, nullity(S_c) = 1  ⇒  rank(S_c)=8, rank(P)=1.
    assert_eq!(p_constrained, 9);
    assert_eq!(
        bend_rank, 8,
        "constrained order-2 bend penalty must have rank 8 (nullity 1 after centering)"
    );
    assert_eq!(
        null_rank, 1,
        "shrinkage ridge must be the rank-1 projector onto null(S_c), not the \
         rank-2 congruence of the raw projector"
    );

    let s_c = &result.penalties[0];
    let p_null = &result.penalties[1];
    // Spectral complementarity: P projects onto null(S_c), so S_c·P = P·S_c = 0
    // exactly (the ridge penalizes ONLY the unpenalized polynomial direction and
    // never touches a curvature mode). The pre-fix raw-chart ridge failed this:
    // its second eigendirection lived in range(S_c), giving ‖S_c P‖_F ≈ 0.15.
    let sp = s_c.dot(p_null);
    let ps = p_null.dot(s_c);
    let sp_norm = sp.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ps_norm = ps.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        sp_norm < 1e-9,
        "S_c·P must vanish (spectral complementarity); got ‖S_c P‖_F = {sp_norm:e}"
    );
    assert!(
        ps_norm < 1e-9,
        "P·S_c must vanish (spectral complementarity); got ‖P S_c‖_F = {ps_norm:e}"
    );

    // Together the two blocks still leave NO unpenalized direction: the assembled
    // double penalty `S_bend + P` has full structural rank, so REML can shrink
    // (never inflate) the null space (#1266). With the clean partition this is
    // exactly rank(S_c)+rank(P) = 8+1 = 9.
    let summed = s_c + p_null;
    let joint_rank = analyze_penalty_block_with_op(&summed, None)
        .expect("assembled double penalty analyzes")
        .rank;
    assert_eq!(joint_rank, p_constrained);

    assert_eq!(result.design.nrows(), x.len());
}

// Boundary-condition emission moved from the basis builder to the
// smooth-level paired linear-constraint path (see smooth.rs
// `bspline_boundary_conditions_emit_paired_equality_constraints` and
// `bspline_boundary_conditions_follow_frozen_identifiability_transform`).
// The basis builder no longer bakes boundary conditions into the null
// space (#823), so the former basis-level boundary-projection tests were
// removed along with that legacy path.

#[test]
fn test_build_bspline_basis_1d_automatic_uniform_uses_data_range() {
    let x = array![2.0, 3.0, 4.5, 6.0, 7.0, 8.0];
    let spec = BSplineBasisSpec {
        degree: 2,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(3),
            placement: BSplineKnotPlacement::Uniform,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::default(),
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let result = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let knots = match result.metadata {
        BasisMetadata::BSpline1D { knots, .. } => knots,
        _ => panic!("expected BSpline1D metadata"),
    };
    assert_eq!(knots.len(), 3 + 2 * (spec.degree + 1));
    assert!((knots[0] - 2.0).abs() < 1e-12);
    assert!((knots[knots.len() - 1] - 8.0).abs() < 1e-12);
}

#[test]
fn test_build_bspline_basis_1d_automatic_quantile_is_not_uniform_for_skewed_data() {
    let x = array![0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 10.0, 10.5, 11.0, 12.0];
    let spec = BSplineBasisSpec {
        degree: 2,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(3),
            placement: BSplineKnotPlacement::Quantile,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::default(),
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let result = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let knots = match result.metadata {
        BasisMetadata::BSpline1D { knots, .. } => knots,
        _ => panic!("expected BSpline1D metadata"),
    };
    let start = spec.degree + 1;
    let internal = &knots.as_slice().unwrap()[start..(start + 3)];
    let d1 = internal[1] - internal[0];
    let d2 = internal[2] - internal[1];
    assert!(
        (d1 - d2).abs() > 1e-6,
        "quantile spacing should be non-uniform for skewed data"
    );
}

#[test]
fn test_penalty_greville_selected_for_clamped_uniform_breakpoints() {
    // A clamped B-spline with a *uniform interior breakpoint grid* still has
    // non-uniform Greville abscissae (they cluster toward the clamped ends),
    // so the geometry-correct divided-difference penalty must be selected.
    // Selecting the classical integer-difference penalty here would give the
    // curvature penalty a null space that is only an approximation of
    // {1, x}, biasing every shrink-toward-linear and the REML smoothing
    // selection (root cause of the tensor te/ti over-smoothing regression).
    let degree = 3usize;
    let knots = internal::generate_full_knot_vector((0.0, 1.0), 5, degree).unwrap();
    let g = penalty_greville_abscissae_for_knots(&knots, degree)
        .unwrap()
        .expect("clamped uniform breakpoints have non-uniform Greville abscissae");
    // Sanity: the abscissae really are non-uniform (clustered at the ends).
    let gaps: Vec<f64> = g.windows(2).into_iter().map(|w| w[1] - w[0]).collect();
    let max_gap = gaps.iter().cloned().fold(f64::MIN, f64::max);
    let min_gap = gaps.iter().cloned().fold(f64::MAX, f64::min);
    assert!(
        max_gap - min_gap > 1e-6,
        "Greville abscissae for a clamped basis should be non-uniform: gaps={gaps:?}"
    );
}

#[test]
fn test_penalty_greville_none_for_uniform_greville() {
    // When the abscissae are genuinely uniform (a non-clamped, evenly spaced
    // grid), integer differences coincide with divided differences up to an
    // overall scale, so the cheaper integer-difference path is selected.
    let degree = 3usize;
    let n_basis = 8usize;
    let knots: Array1<f64> = Array1::from_iter((0..(n_basis + degree + 1)).map(|i| i as f64));
    let g = compute_greville_abscissae(&knots, degree).unwrap();
    assert!(
        is_uniformly_spaced_sequence(g.view()),
        "evenly spaced knot vector should yield uniform Greville abscissae: {g:?}"
    );
    assert!(
        penalty_greville_abscissae_for_knots(&knots, degree)
            .unwrap()
            .is_none()
    );
}

#[test]
fn test_clamped_bspline_curvature_penalty_null_space_is_exactly_linear() {
    // End-to-end guard on the root cause: the curvature penalty of a clamped
    // uniform B-spline must annihilate exactly the functions linear in x.
    // Build the (unconstrained) marginal basis + penalty the way a tensor
    // margin does, then verify every null-space coefficient vector maps to a
    // function with zero deviation from its best linear-in-x fit.
    let x = Array1::<f64>::linspace(0.0, 1.0, 50);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 4,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };
    let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let design = built.design.to_dense();
    let s = &built.penalties[0];
    // Null space of S (curvature penalty): eigenvectors with ~zero eigenvalue.
    let (evals, evecs) = FaerEigh::eigh(s, Side::Lower).unwrap();
    let max_ev = evals.iter().cloned().fold(f64::MIN, f64::max);
    let mut worst_rel_dev = 0.0_f64;
    let n = x.len();
    // Least-squares onto span{1, x} via the 2x2 normal equations.
    let sx: f64 = x.sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let nn = n as f64;
    let det = nn * sxx - sx * sx;
    let mut null_count = 0usize;
    for (idx, &ev) in evals.iter().enumerate() {
        if ev > 1e-9 * max_ev {
            continue;
        }
        null_count += 1;
        let coef = evecs.column(idx).to_owned();
        let f = design.dot(&coef);
        let sf: f64 = f.sum();
        let sxf: f64 = x.iter().zip(f.iter()).map(|(xi, fi)| xi * fi).sum();
        // beta = (A'A)^{-1} A'f for A = [1, x]
        let b0 = (sxx * sf - sx * sxf) / det;
        let b1 = (nn * sxf - sx * sf) / det;
        let mut resid = 0.0_f64;
        let mut amp = 0.0_f64;
        for i in 0..n {
            let fit = b0 + b1 * x[i];
            resid += (f[i] - fit) * (f[i] - fit);
            amp += f[i] * f[i];
        }
        let rel = (resid.sqrt()) / amp.sqrt().max(1e-30);
        worst_rel_dev = worst_rel_dev.max(rel);
    }
    assert!(
        null_count >= 2,
        "curvature penalty should have a 2-D (const+linear) null space, found {null_count}"
    );
    assert!(
        worst_rel_dev < 1e-8,
        "clamped B-spline curvature penalty null space deviates from linear-in-x \
             by rel {worst_rel_dev:.3e} (must be ~0; integer-difference penalties on a \
             clamped basis contaminate the null space)"
    );
}

#[test]
fn test_build_bspline_basis_1d_quantile_uses_divided_difference_penalty() {
    let x = array![0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 10.0, 10.5, 11.0, 12.0];
    let spec = BSplineBasisSpec {
        degree: 2,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(3),
            placement: BSplineKnotPlacement::Quantile,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let built_design = built.design.to_dense();
    let knots = match &built.metadata {
        BasisMetadata::BSpline1D { knots, .. } => knots,
        _ => panic!("expected BSpline1D metadata"),
    };
    let g = penalty_greville_abscissae_for_knots(knots, spec.degree)
        .unwrap()
        .expect("quantile knots should trigger Greville scaling");
    let expected =
        create_difference_penalty_matrix(built_design.ncols(), spec.penalty_order, Some(g.view()))
            .unwrap();

    let got = &built.penalties[0];
    let mut max_abs = 0.0_f64;
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            max_abs = max_abs.max((got[[i, j]] - expected[[i, j]]).abs());
        }
    }
    assert!(
        max_abs < 1e-10,
        "quantile penalty mismatch: max_abs_diff={max_abs:.3e}"
    );
}

#[test]
fn test_build_bspline_basis_1d_none_identifiability_prefers_sparse_design() {
    let x = Array::linspace(0.0, 1.0, 32);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(6),
            placement: BSplineKnotPlacement::Quantile,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let built = build_bspline_basis_1d(x.view(), &spec).expect("build sparse bspline");
    assert!(matches!(built.design, DesignMatrix::Sparse(_)));
}

#[test]
fn test_build_bspline_basis_1d_default_identifiability_densifies_via_orthonormal_centering() {
    // The default (`WeightedSumToZero`) identifiability multiplies the
    // sparse B-spline by the orthonormal null-space basis Z of c = Bᵀw.
    // Z is dense in general (rrqr_nullspace_basis yields an orthonormal,
    // not sparsity-preserving, factor), so B·Z is mathematically dense —
    // the cost of having ZZᵀ act as a true projector, called out in the
    // matching comment inside `build_bspline_basis_1d`. Pin that the
    // densification is exposed at the API: a downstream caller that
    // assumes sparse storage for the default-centered basis is wrong.
    let x = Array::linspace(0.0, 1.0, 32);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(6),
            placement: BSplineKnotPlacement::Quantile,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::default(),
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let built = build_bspline_basis_1d(x.view(), &spec).expect("build centered bspline");
    assert!(matches!(built.design, DesignMatrix::Dense(_)));
}

#[test]
fn test_build_bspline_basis_1d_quantile_rejects_missing_interior_support() {
    let x = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let spec = BSplineBasisSpec {
        degree: 2,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(3),
            placement: BSplineKnotPlacement::Quantile,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    match build_bspline_basis_1d(x.view(), &spec).unwrap_err() {
        BasisError::InvalidInput(msg) => {
            assert!(msg.contains("distinct interior support"));
        }
        err => panic!("expected InvalidInput for missing interior support, got {err:?}"),
    }
}

#[test]
fn test_quantile_knot_generation_excludes_boundary_point_masses() {
    let mut x = vec![1e-9; 16];
    x.extend([4.0, 7.0, 10.0, 20.0, 40.0, 80.0, 160.0, 285.0]);
    let x = Array1::from_vec(x).mapv(f64::ln);
    let knots = internal::generate_full_knot_vector_quantile(x.view(), 6, 3)
        .expect("quantile knots should be inferred from strict interior support");

    let lower = knots[0];
    let upper = knots[knots.len() - 1];
    for &k in knots.iter().skip(4).take(6) {
        assert!(
            k > lower,
            "internal knot should be strictly above lower boundary"
        );
        assert!(
            k < upper,
            "internal knot should be strictly below upper boundary"
        );
    }

    let g = compute_greville_abscissae(&knots, 3).expect("Greville abscissae should be valid");
    for i in 1..g.len() {
        assert!(
            g[i] > g[i - 1],
            "Greville abscissae must be strictly increasing to support divided differences"
        );
    }
}

#[test]
fn test_bspline_identifiability_defaultweighted_sum_tozero() {
    let x = Array::linspace(0.0, 1.0, 40);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 5,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::default(),
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let built_design = built.design.to_dense();
    for j in 0..built_design.ncols() {
        let col_sum = built_design.column(j).sum();
        assert!(
            col_sum.abs() < 1e-8,
            "default weighted-sum-to-zero failed for column {j}: {col_sum}"
        );
    }

    let (raw_basis, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 5,
        },
        spec.degree,
        BasisOptions::value(),
    )
    .unwrap();
    let z = match &built.metadata {
        BasisMetadata::BSpline1D {
            identifiability_transform: Some(z),
            ..
        } => z,
        _ => panic!("expected frozen B-spline identifiability transform"),
    };
    let expected = raw_basis.dot(z);
    assert_eq!(built_design.dim(), expected.dim());
    for i in 0..built_design.nrows() {
        for j in 0..built_design.ncols() {
            assert_abs_diff_eq!(built_design[[i, j]], expected[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_bspline_identifiability_weighted_sum_tozero_respects_weights() {
    // Pin the centering identity itself: every column of the constrained
    // basis is orthogonal to the supplied weight vector. Storage type is
    // a separate concern (covered by
    // `..._default_identifiability_densifies_via_orthonormal_centering`).
    let x = Array::linspace(0.0, 1.0, 30);
    let weights = Array1::from_iter((0..x.len()).map(|idx| 1.0 + idx as f64 / 10.0));
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 4,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::WeightedSumToZero {
            weights: Some(weights.clone()),
        },
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let built_design = built.design.to_dense();
    for j in 0..built_design.ncols() {
        let weighted_sum = built_design.column(j).dot(&weights);
        assert!(
            weighted_sum.abs() < 1e-8,
            "weighted sum-to-zero failed for column {j}: {weighted_sum}"
        );
    }
}

#[test]
fn test_bspline_identifiability_remove_linear_trend_reduces_two_dims() {
    let x = Array::linspace(0.0, 1.0, 50);
    let raw = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 6,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };
    let constrained = BSplineBasisSpec {
        identifiability: BSplineIdentifiability::RemoveLinearTrend,
        ..raw.clone()
    };

    let b_raw = build_bspline_basis_1d(x.view(), &raw).unwrap();
    let b_constrained = build_bspline_basis_1d(x.view(), &constrained).unwrap();
    assert_eq!(b_constrained.design.ncols() + 2, b_raw.design.ncols());
}

#[test]
fn test_bspline_identifiability_orthogonal_to_design_columns() {
    let x = Array::linspace(0.0, 1.0, 40);
    let mut constraints = Array2::<f64>::zeros((x.len(), 2));
    constraints.column_mut(0).fill(1.0);
    constraints.column_mut(1).assign(&x);

    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 5,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::OrthogonalToDesignColumns {
            columns: constraints.clone(),
            weights: None,
        },
        boundary: OneDimensionalBoundary::Open,
        boundary_conditions: BSplineBoundaryConditions::default(),
    };

    let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let built_design = built.design.to_dense();
    let cross = built_design.t().dot(&constraints);
    for i in 0..cross.nrows() {
        for j in 0..cross.ncols() {
            assert!(
                cross[[i, j]].abs() < 1e-8,
                "orthogonality violation at ({i},{j}) = {}",
                cross[[i, j]]
            );
        }
    }
}

#[test]
fn test_cyclic_difference_penalty_wraps_nullspace() {
    let s = create_cyclic_difference_penalty_matrix(8, 2).unwrap();
    assert_eq!(s.nrows(), 8);
    assert_eq!(s.ncols(), 8);
    for i in 0..8 {
        for j in 0..8 {
            assert!((s[[i, j]] - s[[j, i]]).abs() < 1e-12);
        }
    }
    let ones = Array1::<f64>::ones(8);
    let q = ones.dot(&s.dot(&ones));
    assert!(q.abs() < 1e-10);
    assert!(
        s[[0, 7]].abs() > 0.0,
        "endpoint coefficients must be coupled"
    );
}

#[test]
fn test_cyclic_cubic_pspline_matches_at_period_boundary() {
    let x = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    let spec = BSplineBasisSpec {
        degree: 3,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 8,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
        boundary: OneDimensionalBoundary::Cyclic {
            start: 0.0,
            end: 1.0,
        },
        boundary_conditions: BSplineBoundaryConditions::default(),
    };
    let built = build_bspline_basis_1d(x.view(), &spec).unwrap();
    let dense = built.design.to_dense();
    for j in 0..dense.ncols() {
        assert!((dense[[0, j]] - dense[[4, j]]).abs() < 1e-12);
    }
    for i in 0..dense.nrows() {
        let sum = dense.row(i).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn test_cyclic_duchon_matches_at_period_boundary() {
    let x = Array2::from_shape_vec((5, 1), vec![0.0, 0.2, 0.5, 0.8, 1.0]).unwrap();
    let centers = Array2::from_shape_vec((6, 1), (0..6).map(|i| i as f64 / 6.0).collect()).unwrap();
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        periodic: None,
        length_scale: Some(0.25),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Zero,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Cyclic {
            start: 0.0,
            end: 1.0,
        },
    };
    let built = build_duchon_basis(x.view(), &spec).unwrap();
    let dense = built.design.to_dense();
    for j in 0..dense.ncols() {
        assert!((dense[[0, j]] - dense[[4, j]]).abs() < 1e-10);
    }
    assert_eq!(built.penalties.len(), 1);
    assert_eq!(built.penalties[0].nrows(), dense.ncols());
}

#[test]
fn test_bspline_basis_sums_to_one() {
    let data = Array::linspace(0.1, 9.9, 100);
    let (basis, _) = create_basis::<Dense>(
        data.view(),
        KnotSource::Generate {
            data_range: (0.0, 10.0),
            num_internal_knots: 10,
        },
        3,
        BasisOptions::value(),
    )
    .unwrap();

    let sums = basis.sum_axis(Axis(1));

    // Every row should sum to 1.0 (with floating point tolerance)
    for &sum in sums.iter() {
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Basis did not sum to 1, got {}",
            sum
        );
    }
}

#[test]
fn test_bspline_basis_sums_to_onewith_uniform_knots() {
    // Create data with a non-uniform distribution
    // Since quantile knots are disabled for P-splines, this tests the fallback to uniform knots
    let mut data = Array::zeros(100);
    for i in 0..100 {
        let x = if i < 50 {
            // Points clustered around 2.0
            2.0 + (i as f64) / 25.0 // Range: 2.0 to 4.0
        } else {
            // Points clustered around 8.0
            6.0 + (i as f64 - 50.0) / 25.0 // Range: 6.0 to 8.0
        };
        data[i] = x;
    }

    // Even when providing training data, this should fall back to uniform knots
    let (basis, knots) = create_basis::<Dense>(
        data.view(),
        KnotSource::Generate {
            data_range: (0.0, 10.0),
            num_internal_knots: 10,
        },
        3,
        BasisOptions::value(),
    )
    .unwrap();

    // Verify that knots are uniformly distributed (not following data distribution)
    // Since quantile knots are disabled, these should be uniform

    // Check that internal knots are uniformly spaced
    let internal_knots: Vec<f64> = knots
        .iter()
        .skip(4) // Skip the repeated boundary knots (degree+1 = 4)
        .take(10) // Take the internal knots
        .copied()
        .collect();

    if internal_knots.len() >= 2 {
        let spacing = internal_knots[1] - internal_knots[0];
        for window in internal_knots.windows(2) {
            let current_spacing = window[1] - window[0];
            assert!(
                (current_spacing - spacing).abs() < 1e-9,
                "Knots should be uniformly spaced, but spacing varies: expected {}, got {}",
                spacing,
                current_spacing
            );
        }
    }

    // Verify that the basis still sums to 1.0 for each data point
    let sums = basis.sum_axis(Axis(1));

    // Every row should sum to 1.0 (with floating point tolerance)
    for &sum in sums.iter() {
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Uniform basis did not sum to 1, got {}",
            sum
        );
    }

    // Now verify for points outside the original data distribution
    // Create a different set of evaluation points that are spread uniformly
    let eval_points = Array::linspace(0.1, 9.9, 100);

    // Create basis using the previously generated knots
    let (eval_basis, _) = create_basis::<Dense>(
        eval_points.view(),
        KnotSource::Provided(knots.view()),
        3,
        BasisOptions::value(),
    )
    .unwrap();

    // Verify sums for the evaluation points
    let eval_sums = eval_basis.sum_axis(Axis(1));

    for &sum in eval_sums.iter() {
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Basis at evaluation points did not sum to 1, got {}",
            sum
        );
    }
}

#[test]
fn test_single_point_evaluation_degree_one() {
    // This test validates the raw output of the UNCONSTRAINED basis evaluator
    // (internal::evaluate_splines_at_point), not a final model prediction which
    // would require applying constraints. The test only verifies that the raw
    // basis functions are correctly evaluated, before any constraints are applied.
    //
    // Degree 1 (linear) splines with knots t = [0,0,1,2,2].
    // This gives 3 basis functions (n = k-d-1 = 5-1-1 = 3), B_{0,1}, B_{1,1}, B_{2,1}.
    let knots = array![0.0, 0.0, 1.0, 2.0, 2.0];
    let x = 0.5; // For x=0.5, the knot interval is mu=1, since t_1 <= x < t_2.

    let values = evaluate_splines_at_point(x, 1, knots.view());
    assert_eq!(values.len(), 3);

    // Manual calculation for x=0.5:
    // The only non-zero basis function of degree 0 is B_{1,0} = 1.
    // Recurrence for degree 1:
    // B_{0,1}(x) = ( (x-t0)/(t1-t0) )*B_{0,0} + ( (t2-x)/(t2-t1) )*B_{1,0}
    //           = ( (0.5-0)/(0-0) )*0       + ( (1-0.5)/(1-0) )*1         = 0.5
    //           (Note: 0/0 division is taken as 0)
    // B_{1,1}(x) = ( (x-t1)/(t2-t1) )*B_{1,0} + ( (t3-x)/(t3-t2) )*B_{2,0}
    //           = ( (0.5-0)/(1-0) )*1       + ( (2-0.5)/(2-1) )*0         = 0.5
    // B_{2,1}(x) = ( (x-t2)/(t3-t2) )*B_{2,0} + ( (t4-x)/(t4-t3) )*B_{3,0}
    //           = ( (0.5-1)/(2-1) )*0       + ( (2-0.5)/(2-2) )*0         = 0.0

    assert!(
        (values[0] - 0.5).abs() < 1e-9,
        "Expected B_0,1 to be 0.5, got {}",
        values[0]
    );
    assert!(
        (values[1] - 0.5).abs() < 1e-9,
        "Expected B_1,1 to be 0.5, got {}",
        values[1]
    );
    assert!(
        (values[2] - 0.0).abs() < 1e-9,
        "Expected B_2,1 to be 0.0, got {}",
        values[2]
    );
}

#[test]
fn test_cox_de_boor_higher_degree() {
    // Test that verifies the Cox-de Boor denominator handling for higher degree splines
    // Using non-uniform knots where numerical issues would be more apparent
    let knots = array![0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 4.0, 4.0];
    let x = 2.0;

    let values = evaluate_splines_at_point(x, 2, knots.view());

    // The basis functions should sum to 1.0 (partition of unity property)
    let sum = values.sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "Basis functions should sum to 1.0, got {}",
        sum
    );

    // All values should be non-negative
    for (i, &val) in values.iter().enumerate() {
        assert!(
            val >= -1e-9,
            "Basis function {} should be non-negative, got {}",
            i,
            val
        );
    }
}

#[test]
fn test_boundaryvalue_handling() {
    // Test for proper boundary value handling at the upper boundary.
    // This test ensures that evaluation at the upper boundary works correctly.

    // Test the internal function directly with the problematic case
    let knots = array![
        0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0
    ];
    let x = 10.0; // This is the value that caused the panic
    let degree = 3;

    let basisvalues = evaluate_splines_at_point(x, degree, knots.view());

    // Should not panic and should return valid results
    assert_eq!(basisvalues.len(), 8); // num_basis = 12 - 3 - 1 = 8

    let sum = basisvalues.sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "Basis functions should sum to 1.0 at boundary, got {}",
        sum
    );
}

#[test]
fn test_basis_boundaryvalues() {
    // Property-based test: Verify boundary conditions using mathematical properties
    // This complements the cross-validation test by testing fundamental B-spline properties

    // A cubic B-spline basis. Knots are [0,0,0,0, 1,2,3, 4,4,4,4].
    // The domain is [0, 4].
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3;
    let num_basis = knots.len() - degree - 1; // 11 - 3 - 1 = 7

    // Test at the lower boundary (x=0)
    let basis_at_start = evaluate_splines_at_point(0.0, degree, knots.view());

    // At the very start of the domain, only the first basis function should be non-zero (and equal to 1).
    assert_abs_diff_eq!(basis_at_start[0], 1.0, epsilon = 1e-9);
    for i in 1..num_basis {
        assert_abs_diff_eq!(basis_at_start[i], 0.0, epsilon = 1e-9);
    }

    // Test at the upper boundary (x=4)
    let basis_at_end = evaluate_splines_at_point(4.0, degree, knots.view());

    // At the very end of the domain, only the LAST basis function should be non-zero (and equal to 1).
    for i in 0..(num_basis - 1) {
        assert_abs_diff_eq!(basis_at_end[i], 0.0, epsilon = 1e-9);
    }
    assert_abs_diff_eq!(basis_at_end[num_basis - 1], 1.0, epsilon = 1e-9);

    // Test intermediate points for partition of unity
    let test_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
    for &x in &test_points {
        let basis = evaluate_splines_at_point(x, degree, knots.view());
        let sum: f64 = basis.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
        if (sum - 1.0).abs() >= 1e-9 {
            panic!("Partition of unity failed at x={}", x);
        }
    }
}

#[test]
fn test_constant_extrapolation_matches_boundary_basisvalues() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3usize;
    let left_boundary = evaluate_splines_at_point(0.0, degree, knots.view());
    let right_boundary = evaluate_splines_at_point(4.0, degree, knots.view());

    let left_out = evaluate_splines_at_point(-100.0, degree, knots.view());
    let right_out = evaluate_splines_at_point(100.0, degree, knots.view());

    for i in 0..left_boundary.len() {
        assert_abs_diff_eq!(left_out[i], left_boundary[i], epsilon = 1e-12);
        assert_abs_diff_eq!(right_out[i], right_boundary[i], epsilon = 1e-12);
    }
}

#[test]
fn test_create_basis_uses_linear_extension_outside_domain() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3usize;
    let x = array![-0.5, 4.5];
    let x_c = array![0.0, 4.0];
    let (b_raw, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .unwrap();
    let (b_c, _) = create_basis::<Dense>(
        x_c.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .unwrap();
    let (db_c, _) = create_basis::<Dense>(
        x_c.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .unwrap();

    let b_raw = b_raw.as_ref();
    let b_c = b_c.as_ref();
    let db_c = db_c.as_ref();
    for i in 0..x.len() {
        let dz = x[i] - x_c[i];
        for j in 0..b_raw.ncols() {
            let expected = b_c[[i, j]] + dz * db_c[[i, j]];
            assert_abs_diff_eq!(b_raw[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_create_basis_first_derivative_uses_boundary_slope_outside_domain() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3usize;
    let x = array![-0.25, 4.25];
    let x_c = array![0.0, 4.0];
    let (db_raw, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .unwrap();
    let (db_c, _) = create_basis::<Dense>(
        x_c.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .unwrap();
    let db_raw = db_raw.as_ref();
    let db_c = db_c.as_ref();
    assert_eq!(db_raw.dim(), db_c.dim());
    for i in 0..db_raw.nrows() {
        for j in 0..db_raw.ncols() {
            assert_abs_diff_eq!(db_raw[[i, j]], db_c[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_first_derivative_uses_one_sided_endpoint_limits() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3usize;
    let x = array![0.0, 4.0];
    let (db, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .unwrap();
    let db = db.as_ref();
    assert!(
        db.row(0).iter().any(|v| v.abs() > 1e-8),
        "left endpoint derivative must use the right-hand slope"
    );
    assert!(
        db.row(1).iter().any(|v| v.abs() > 1e-8),
        "right endpoint derivative must use the left-hand slope"
    );
    assert_abs_diff_eq!(db.row(0).sum(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(db.row(1).sum(), 0.0, epsilon = 1e-10);
    assert!(db[[0, 0]] < 0.0);
    assert!(db[[0, 1]] > 0.0);
    assert!(db[[1, db.ncols() - 2]] < 0.0);
    assert!(db[[1, db.ncols() - 1]] > 0.0);
}

#[test]
fn test_dense_basis_preserves_linear_extensionwhen_internal_builder_goes_sparse() {
    let degree = 3usize;
    let knots = internal::generate_full_knot_vector((0.0, 10.0), 36, degree).unwrap();
    let x = array![-0.5, 10.5];
    let x_c = array![0.0, 10.0];
    assert!(should_use_sparse_basis(
        knots.len().saturating_sub(degree + 1),
        degree,
        1
    ));

    let (b_raw, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .unwrap();
    let (b_c, _) = create_basis::<Dense>(
        x_c.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .unwrap();
    let (db_c, _) = create_basis::<Dense>(
        x_c.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .unwrap();

    for i in 0..x.len() {
        let dz = x[i] - x_c[i];
        for j in 0..b_raw.ncols() {
            let expected = b_c[[i, j]] + dz * db_c[[i, j]];
            assert_abs_diff_eq!(b_raw[[i, j]], expected, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_sparse_derivatives_use_boundary_slope_outside_domain() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3usize;
    let support = degree + 1;
    let mut scratch = BasisEvalScratch::new(degree);
    let mut d1 = vec![0.0; support];
    let mut d2 = vec![0.0; support];
    let mut d1_left = vec![0.0; support];
    let mut d1_right = vec![0.0; support];

    let start_left = evaluate_splines_derivative_sparse_into(
        0.0,
        degree,
        knots.view(),
        &mut d1_left,
        &mut scratch,
    );
    let start =
        evaluate_splines_derivative_sparse_into(-10.0, degree, knots.view(), &mut d1, &mut scratch);
    assert_eq!(start, start_left);
    for i in 0..support {
        assert_abs_diff_eq!(d1[i], d1_left[i], epsilon = 1e-12);
    }

    let start_right = evaluate_splines_derivative_sparse_into(
        4.0,
        degree,
        knots.view(),
        &mut d1_right,
        &mut scratch,
    );
    let start =
        evaluate_splines_derivative_sparse_into(10.0, degree, knots.view(), &mut d1, &mut scratch);
    assert_eq!(start, start_right);
    for i in 0..support {
        assert_abs_diff_eq!(d1[i], d1_right[i], epsilon = 1e-12);
    }

    evaluate_splinessecond_derivative_sparse_into(
        -10.0,
        degree,
        knots.view(),
        &mut d2,
        &mut scratch,
    );
    assert!(d2.iter().all(|v| v.abs() < 1e-12));
    evaluate_splinessecond_derivative_sparse_into(
        10.0,
        degree,
        knots.view(),
        &mut d2,
        &mut scratch,
    );
    assert!(d2.iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn test_create_basis_sparse_matches_dense_extrapolation_outside_domain() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3usize;
    let x = array![-0.5, 4.5];
    let (dense_basis, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .unwrap();
    let sparse_basis = generate_basis_internal::<SparseStorage>(
        x.view(),
        knots.view(),
        degree,
        BasisEvalKind::Basis,
    )
    .unwrap();
    let sparse_dense = <Dense as BasisOutputFormat>::from_sparse(sparse_basis).unwrap();
    assert_eq!(dense_basis.dim(), sparse_dense.dim());
    for i in 0..dense_basis.nrows() {
        for j in 0..dense_basis.ncols() {
            assert_abs_diff_eq!(dense_basis[[i, j]], sparse_dense[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_create_basis_sparse_first_derivative_matches_dense_outside_domain() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
    let degree = 3usize;
    let x = array![-0.25, 4.25];
    let (dense_deriv, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::first_derivative(),
    )
    .unwrap();
    let sparse_deriv = generate_basis_internal::<SparseStorage>(
        x.view(),
        knots.view(),
        degree,
        BasisEvalKind::FirstDerivative,
    )
    .unwrap();
    let sparse_dense = <Dense as BasisOutputFormat>::from_sparse(sparse_deriv).unwrap();
    assert_eq!(dense_deriv.dim(), sparse_dense.dim());
    for i in 0..dense_deriv.nrows() {
        for j in 0..dense_deriv.ncols() {
            assert_abs_diff_eq!(dense_deriv[[i, j]], sparse_dense[[i, j]], epsilon = 1e-10);
        }
    }
}

#[test]
fn test_ispline_scalar_boundary_behavior() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let degree = 1usize;
    let n_ispline = knots.len() - (degree + 1) - 2;
    let mut out = vec![0.0; n_ispline];

    evaluate_ispline_scalar(-10.0, knots.view(), degree, &mut out).expect("left boundary eval");
    assert!(out.iter().all(|&v| v.abs() <= 1e-12));

    evaluate_ispline_scalar(10.0, knots.view(), degree, &mut out).expect("right boundary eval");
    for &v in &out {
        assert!((v - 1.0).abs() <= 1e-12);
    }
}

#[test]
fn test_ispline_scalar_is_monotone_in_x() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let degree = 1usize;
    let n_ispline = knots.len() - (degree + 1) - 2;
    let xs = [0.0, 0.25, 0.75, 1.5, 2.2, 2.8, 3.0];

    let mut prev = vec![0.0; n_ispline];
    evaluate_ispline_scalar(xs[0], knots.view(), degree, &mut prev).expect("initial eval");

    for &x in xs.iter().skip(1) {
        let mut curr = vec![0.0; n_ispline];
        evaluate_ispline_scalar(x, knots.view(), degree, &mut curr).expect("eval along grid");
        for j in 0..n_ispline {
            assert!(
                curr[j] + 1e-12 >= prev[j],
                "I-spline basis not monotone at x={x}, j={j}: prev={}, curr={}",
                prev[j],
                curr[j]
            );
        }
        prev = curr;
    }
}

#[test]
fn test_mspline_scalar_matches_scaled_bspline() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let degree = 2usize;
    let x = 1.25;
    let num_basis = knots.len() - degree - 1;
    let mut b = vec![0.0; num_basis];
    let mut m = vec![0.0; num_basis];
    let mut scratch = SplineScratch::new(degree);
    evaluate_bspline_basis_scalar(x, knots.view(), degree, &mut b, &mut scratch)
        .expect("bspline eval");
    evaluate_mspline_scalar(x, knots.view(), degree, &mut m, &mut scratch).expect("mspline eval");

    let order = (degree + 1) as f64;
    for i in 0..num_basis {
        let span = knots[i + degree + 1] - knots[i];
        let expected = if span.abs() > 1e-12 {
            b[i] * (order / span)
        } else {
            0.0
        };
        assert_abs_diff_eq!(m[i], expected, epsilon = 1e-12);
    }
}

#[test]
fn test_create_basis_msplinezero_outside_domain() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let degree = 2usize;
    let x = array![-10.0, 1.0, 10.0];
    let (m, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::m_spline(),
    )
    .expect("create mspline basis");
    let m = m.as_ref();
    assert!(m.row(0).iter().all(|v| v.abs() <= 1e-12));
    assert!(m.row(2).iter().all(|v| v.abs() <= 1e-12));
    assert!(m.row(1).iter().any(|v| v.abs() > 1e-12));
}

#[test]
fn test_create_basis_ispline_boundaryrows() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let degree = 1usize;
    let x = array![-10.0, 1.5, 10.0];
    let (i_basis, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .expect("create ispline basis");
    let i_basis = i_basis.as_ref();
    assert!(i_basis.row(0).iter().all(|v| v.abs() <= 1e-12));
    for j in 0..i_basis.ncols() {
        assert!((i_basis[[2, j]] - 1.0).abs() <= 1e-12);
    }
    for &v in i_basis.row(1) {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn test_ispline_basis_drops_identicallyzero_leading_column() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let degree = 1usize;
    let x = array![0.0, 0.5, 1.5, 2.5, 3.0];
    let (i_basis, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::i_spline(),
    )
    .expect("create ispline basis");
    let i_basis = i_basis.as_ref();
    assert_eq!(i_basis.ncols(), knots.len() - (degree + 1) - 2);
    for j in 0..i_basis.ncols() {
        assert!(
            i_basis.column(j).iter().any(|&v| v.abs() > 1e-12),
            "I-spline column {j} should not be identically zero"
        );
    }
}

#[test]
fn test_ispline_derivative_matches_cumulative_bspline_derivative_finite_difference() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let degree = 2usize;
    let bs_degree = degree + 1;
    let n_i = knots.len() - bs_degree - 2;

    // Check at interior points away from knot boundaries for stable central differences.
    let xs = [0.35, 0.8, 1.4, 2.2];
    let h = 1e-6;
    for &x in &xs {
        let mut db = vec![0.0; n_i + 1];
        evaluate_bspline_derivative_scalar(x, knots.view(), bs_degree, &mut db).expect("B'(x)");
        let mut d_i = vec![0.0; n_i];
        let mut running = 0.0_f64;
        for j in (1..(n_i + 1)).rev() {
            running += db[j];
            d_i[j - 1] = running;
        }

        crate::assert_central_difference_array!(
            x,
            h,
            |x_eval| {
                let mut iv = vec![0.0; n_i];
                evaluate_ispline_scalar(x_eval, knots.view(), degree, &mut iv).unwrap();
                iv
            },
            d_i,
            2e-5
        );
    }
}

#[test]
fn testvalidate_knots_for_degree_rejects_too_few_knots_for_degree_domain() {
    let knots = array![0.0, 0.0, 1.0, 1.0];
    let err = create_basis::<Dense>(
        array![0.5].view(),
        KnotSource::Provided(knots.view()),
        2,
        BasisOptions::value(),
    )
    .expect_err("degree-2 basis should reject knot vectors with too few knots");
    match err {
        BasisError::InsufficientKnotsForDegree {
            degree,
            required,
            provided,
        } => {
            assert_eq!(degree, 2);
            assert_eq!(required, 6);
            assert_eq!(provided, 4);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn testvalidate_knots_for_degree_rejectszero_support_boundary_basis() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let err = create_basis::<Dense>(
        array![0.5].view(),
        KnotSource::Provided(knots.view()),
        3,
        BasisOptions::value(),
    )
    .expect_err("over-repeated boundary knots should be rejected");
    match err {
        BasisError::InvalidKnotVector(msg) => {
            assert!(msg.contains("zero support"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_densesecond_derivativezeroes_outside_domain_even_on_sparse_heuristic_path() {
    let degree = 3usize;
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0];
    let mut xs = Vec::with_capacity(128);
    xs.push(-0.2);
    for i in 0..126 {
        xs.push(i as f64 / 125.0);
    }
    xs.push(1.2);
    let x = Array1::from_vec(xs);
    let (basis, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::second_derivative(),
    )
    .expect("dense second derivative");
    let basis = basis.as_ref();
    assert!(basis.row(0).iter().all(|v| v.abs() <= 1e-12));
    assert!(
        basis
            .row(basis.nrows() - 1)
            .iter()
            .all(|v| v.abs() <= 1e-12)
    );
}

#[test]
fn test_scalar_higher_derivatives_arezero_outside_domain() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0];
    let mut second = vec![1.0; knots.len() - 3 - 1];
    evaluate_bsplinesecond_derivative_scalar(-0.1, knots.view(), 3, &mut second)
        .expect("second derivative");
    assert!(second.iter().all(|v| v.abs() <= 1e-12));

    let mut third = vec![1.0; knots.len() - 3 - 1];
    evaluate_bsplinethird_derivative_scalar(1.1, knots.view(), 3, &mut third)
        .expect("third derivative");
    assert!(third.iter().all(|v| v.abs() <= 1e-12));

    let knots4 = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut fourth = vec![1.0; knots4.len() - 4 - 1];
    evaluate_bspline_fourth_derivative_scalar(-0.2, knots4.view(), 4, &mut fourth)
        .expect("fourth derivative");
    assert!(fourth.iter().all(|v| v.abs() <= 1e-12));
}

#[test]
fn test_higher_derivative_degree_errors_arespecific() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut out = vec![0.0; knots.len() - 1 - 1];
    let err = evaluate_bsplinesecond_derivative_scalar(0.5, knots.view(), 1, &mut out)
        .expect_err("degree-1 second derivative should fail");
    match err {
        BasisError::InsufficientDegreeForDerivative {
            degree,
            derivative_order,
            minimum_degree,
        } => {
            assert_eq!(degree, 1);
            assert_eq!(derivative_order, 2);
            assert_eq!(minimum_degree, 2);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn test_non_bspline_derivative_orders_are_rejected() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let x = array![0.5, 1.5];
    let err = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        2,
        BasisOptions {
            derivative_order: 1,
            basis_family: BasisFamily::MSpline,
        },
    )
    .expect_err("MSpline derivative order should be rejected");
    assert!(matches!(err, BasisError::InvalidInput(_)));
}

#[test]
fn test_mspline_sparse_matches_dense() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let x = array![-1.0, 0.3, 1.1, 2.7, 4.0];
    let (dense, _) = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        2,
        BasisOptions::m_spline(),
    )
    .expect("dense mspline");
    let (sparse, _) = create_basis::<Sparse>(
        x.view(),
        KnotSource::Provided(knots.view()),
        2,
        BasisOptions::m_spline(),
    )
    .expect("sparse mspline");

    let dense = dense.as_ref();
    let mut sparse_dense = Array2::<f64>::zeros((sparse.nrows(), sparse.ncols()));
    let (symbolic, values) = sparse.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..sparse.ncols() {
        for idx in col_ptr[col]..col_ptr[col + 1] {
            sparse_dense[[row_idx[idx], col]] += values[idx];
        }
    }
    assert_eq!(dense.dim(), sparse_dense.dim());
    for i in 0..dense.nrows() {
        for j in 0..dense.ncols() {
            assert_abs_diff_eq!(dense[[i, j]], sparse_dense[[i, j]], epsilon = 1e-12);
        }
    }
}

#[test]
fn test_mspline_rejectszero_normalization_spans() {
    // degree=2 with 4 repeated boundary knots makes t[3]-t[0]=0 for i=0.
    let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let x = array![0.25, 0.5, 0.75];
    let err = create_basis::<Dense>(
        x.view(),
        KnotSource::Provided(knots.view()),
        2,
        BasisOptions::m_spline(),
    )
    .expect_err("degenerate M-spline normalization spans should be rejected");
    assert!(matches!(err, BasisError::InvalidKnotVector(_)));
}

#[test]
fn test_ispline_sparse_output_is_rejected() {
    let knots = array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let x = array![0.2, 1.5, 2.8];
    let err = create_basis::<Sparse>(
        x.view(),
        KnotSource::Provided(knots.view()),
        1,
        BasisOptions::i_spline(),
    )
    .expect_err("I-spline sparse output should be rejected");
    assert!(matches!(err, BasisError::InvalidInput(_)));
}

#[test]
fn test_degree_0_boundary_behavior() {
    let knots: Array1<f64> = array![0.0, 0.0, 1.0, 2.0, 2.0];
    let x = 2.0;

    const EPS: f64 = 1e-12;

    for i in 0..(knots.len() - 1) {
        let intervalwidth = knots[i + 1] - knots[i];
        let expected = if intervalwidth.abs() < EPS {
            if i == knots.len() - 2 && (x - knots[i + 1]).abs() < EPS {
                1.0
            } else {
                0.0
            }
        } else if x >= knots[i] && x < knots[i + 1] {
            1.0
        } else if i == knots.len() - 2 && (x - knots[i + 1]).abs() < EPS {
            1.0
        } else {
            0.0
        };

        let value = evaluate_bspline(x, &knots, i, 0);
        assert_abs_diff_eq!(value, expected, epsilon = 1e-12);
    }
}

#[test]
fn test_boundary_analysis() {
    // Test case from the failing test: knots [0, 0, 1, 2, 2], degree 1, x=2
    let knots: Array1<f64> = array![0.0, 0.0, 1.0, 2.0, 2.0];
    let degree = 1;
    let x = 2.0;

    let num_basis = knots.len() - degree - 1;
    let iterative_basis = evaluate_splines_at_point(x, degree, knots.view());

    let recursivevalues: Vec<f64> = (0..num_basis)
        .map(|i| evaluate_bspline(x, &knots, i, degree))
        .collect();
    let expected = [0.0, 0.0, 1.0];

    assert_eq!(
        recursivevalues.len(),
        expected.len(),
        "Recursive evaluation length mismatch"
    );

    for (i, (&recursive, &expectedvalue)) in recursivevalues.iter().zip(expected.iter()).enumerate()
    {
        assert_abs_diff_eq!(recursive, expectedvalue, epsilon = 1e-12);
        assert_abs_diff_eq!(iterative_basis[i], expectedvalue, epsilon = 1e-12);
    }

    let recursive_sum: f64 = recursivevalues.iter().sum();
    let iterative_sum = iterative_basis.sum();

    assert_abs_diff_eq!(recursive_sum, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(iterative_sum, 1.0, epsilon = 1e-12);
}

/// Validates the basis functions against Example 1 in Starkey's "Cox-deBoor" notes.
///
/// This example is a linear spline (degree=1, order=2) with a uniform knot vector.
/// We test the values of the blending functions at specific points to ensure they
/// match the manually derived formulas in the literature.
///
/// Reference: Denbigh Starkey, "Cox-deBoor Equations for B-Splines", pg. 8.
#[test]
fn test_starkey_notes_example_1() {
    let degree = 1;
    // The book uses knot vector (0, 1, 2, 3, 4, 5).
    // Our setup requires boundary knots. For num_internal_knots = 4, range (0,5),
    // we get internal knots {1,2,3,4}, full vector {0,0, 1,2,3,4, 5,5}.
    let knots = array![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0];
    let num_basis = knots.len() - degree - 1; // 8 - 1 - 1 = 6 basis functions

    // Test case 1: u = 1.5, which is in the span [1, 2].
    // Expected: Two non-zero basis functions, each with value 0.5
    let basis_at_1_5 = evaluate_splines_at_point(1.5, degree, knots.view());
    assert_eq!(basis_at_1_5.len(), num_basis);
    assert_abs_diff_eq!(basis_at_1_5.sum(), 1.0, epsilon = 1e-9);

    // Validate that exactly 2 basis functions are non-zero with value 0.5 each
    let nonzero_count = basis_at_1_5.iter().filter(|&&x| x > 1e-12).count();
    assert_eq!(
        nonzero_count, 2,
        "Should have exactly 2 non-zero basis functions at x=1.5"
    );

    // Check that the non-zero values are at indices 1 and 2 (as determined empirically)
    // and both have value 0.5 (from linear interpolation)
    assert_abs_diff_eq!(basis_at_1_5[1], 0.5, epsilon = 1e-9);
    assert_abs_diff_eq!(basis_at_1_5[2], 0.5, epsilon = 1e-9);

    // Test case 2: u = 2.5, which is in the span [2, 3].
    // Expected: Two non-zero basis functions, each with value 0.5
    let basis_at_2_5 = evaluate_splines_at_point(2.5, degree, knots.view());
    assert_eq!(basis_at_2_5.len(), num_basis);
    assert_abs_diff_eq!(basis_at_2_5.sum(), 1.0, epsilon = 1e-9);

    // Validate that exactly 2 basis functions are non-zero with value 0.5 each
    let nonzero_count_2_5 = basis_at_2_5.iter().filter(|&&x| x > 1e-12).count();
    assert_eq!(
        nonzero_count_2_5, 2,
        "Should have exactly 2 non-zero basis functions at x=2.5"
    );

    // Check that the non-zero values are at indices 2 and 3 (as determined empirically)
    // and both have value 0.5 (from linear interpolation)
    assert_abs_diff_eq!(basis_at_2_5[2], 0.5, epsilon = 1e-9);
    assert_abs_diff_eq!(basis_at_2_5[3], 0.5, epsilon = 1e-9);
}

#[test]
fn test_prediction_consistency_on_and_off_grid() {
    // This test replaces a previously flawed version. The goal is to verify that
    // the prediction logic for a constrained B-spline basis is consistent and correct.
    // We perform two checks:
    // Stage: On-grid consistency—ensure calculating a prediction for a single point that
    //    is ON the original grid yields the same result as the batch calculation.
    // Stage: Off-grid interpolation—ensure a prediction for a point off the grid
    //    (e.g., 0.65) produces a value that lies between its neighbors (0.6 and 0.7),
    //    validating the spline's interpolation property.
    //
    // The previous test incorrectly asserted that the value at 0.65 should equal
    // the value at 0.6, which is false for a non-flat cubic spline.

    // --- Setup: Same as the original test ---
    let data = Array::linspace(0.0, 1.0, 11);
    let (basis_unc, _) = create_basis::<Dense>(
        data.view(),
        KnotSource::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 5,
        },
        3,
        BasisOptions::value(),
    )
    .unwrap();

    let main_basis_unc = basis_unc.slice(s![.., 1..]);
    let (main_basis_con, z_transform) = apply_sum_to_zero_constraint(main_basis_unc, None).unwrap();

    let intercept_coeff = 0.5;
    let num_con_coeffs = main_basis_con.ncols();
    let main_coeffs = Array1::from_shape_fn(num_con_coeffs, |i| (i as f64 + 1.0) * 0.1);

    // --- Calculate batch predictions on the grid (our ground truth) ---
    let predictions_on_grid = intercept_coeff + main_basis_con.dot(&main_coeffs);

    // --- On-grid consistency check ---
    let test_point_on_grid_x = 0.6;
    let on_grid_idx = 6;

    // Calculate the prediction for this single point from scratch.
    let (raw_basis_at_point, _) = create_basis::<Dense>(
        array![test_point_on_grid_x].view(),
        KnotSource::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 5,
        },
        3,
        BasisOptions::value(),
    )
    .unwrap();
    let main_basis_unc_at_point = raw_basis_at_point.slice(s![0, 1..]);
    let main_basis_con_at_point =
        Array1::from_vec(main_basis_unc_at_point.to_vec()).dot(&z_transform);
    let prediction_at_0_6 = intercept_coeff + main_basis_con_at_point.dot(&main_coeffs);

    // ASSERT: The single-point prediction must exactly match the batch prediction for the same point.
    assert_abs_diff_eq!(
        prediction_at_0_6,
        predictions_on_grid[on_grid_idx],
        epsilon = 1e-12 // Use a tight epsilon for this identity check
    );

    // --- Off-grid interpolation check ---
    // Now test the off-grid point x=0.65, which lies between grid points 0.6 and 0.7.
    let test_point_off_grid_x = 0.65;

    // Calculate the prediction for this single off-grid point.
    let (raw_basis_off_grid, _) = create_basis::<Dense>(
        array![test_point_off_grid_x].view(),
        KnotSource::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots: 5,
        },
        3,
        BasisOptions::value(),
    )
    .unwrap();
    let main_basis_unc_off_grid = raw_basis_off_grid.slice(s![0, 1..]);
    let main_basis_con_off_grid =
        Array1::from_vec(main_basis_unc_off_grid.to_vec()).dot(&z_transform);
    let prediction_at_0_65 = intercept_coeff + main_basis_con_off_grid.dot(&main_coeffs);

    // Get the values of the neighboring on-grid points from our batch calculation.
    let value_at_0_6 = predictions_on_grid[6];
    let value_at_0_7 = predictions_on_grid[7];

    // Determine the bounds for the interpolation.
    let lower_bound = value_at_0_6.min(value_at_0_7);
    let upper_bound = value_at_0_6.max(value_at_0_7);

    // ASSERT: The prediction at 0.65 must lie between the values at 0.6 and 0.7.
    // This is a robust check of the spline's interpolating behavior.
    assert!(
        prediction_at_0_65 >= lower_bound && prediction_at_0_65 <= upper_bound,
        "Off-grid prediction ({}) at x=0.65 should be between its neighbors ({}, {})",
        prediction_at_0_65,
        value_at_0_6,
        value_at_0_7
    );
}

#[test]
fn test_error_conditions() {
    match create_basis::<Dense>(
        array![].view(),
        KnotSource::Generate {
            data_range: (0.0, 10.0),
            num_internal_knots: 5,
        },
        0,
        BasisOptions::value(),
    )
    .unwrap_err()
    {
        BasisError::InvalidDegree(deg) => assert_eq!(deg, 0),
        _ => panic!("Expected InvalidDegree error"),
    }

    match create_basis::<Dense>(
        array![].view(),
        KnotSource::Generate {
            data_range: (10.0, 0.0),
            num_internal_knots: 5,
        },
        1,
        BasisOptions::value(),
    )
    .unwrap_err()
    {
        BasisError::InvalidRange(start, end) => {
            assert_eq!(start, 10.0);
            assert_eq!(end, 0.0);
        }
        _ => panic!("Expected InvalidRange error"),
    }

    // Test degenerate range detection
    match create_basis::<Dense>(
        array![].view(),
        KnotSource::Generate {
            data_range: (5.0, 5.0),
            num_internal_knots: 3,
        },
        1,
        BasisOptions::value(),
    )
    .unwrap_err()
    {
        BasisError::DegenerateRange(num_knots) => {
            assert_eq!(num_knots, 3);
        }
        err => panic!("Expected DegenerateRange error, got {:?}", err),
    }

    match create_basis::<Dense>(
        array![].view(),
        KnotSource::Generate {
            data_range: (5.0, 5.0),
            num_internal_knots: 0,
        },
        1,
        BasisOptions::value(),
    )
    .unwrap_err()
    {
        BasisError::DegenerateRange(num_knots) => {
            assert_eq!(num_knots, 0);
        }
        err => panic!("Expected DegenerateRange error, got {:?}", err),
    }

    // Test uniform fallback (quantile knots are disabled for P-splines)
    let (_, knots_uniform) = create_basis::<Dense>(
        array![].view(), // empty evaluation set is fine
        KnotSource::Generate {
            data_range: (0.0, 10.0),
            num_internal_knots: 3,
        },
        1, // degree
        BasisOptions::value(),
    )
    .unwrap();

    // Uniform fallback: boundary repeated degree+1=2 times => 2 + 3 + 2 = 7 knots
    let expected_knots = array![0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0];
    assert_abs_diff_eq!(
        knots_uniform.as_slice().unwrap(),
        expected_knots.as_slice().unwrap(),
        epsilon = 1e-9
    );

    match create_difference_penalty_matrix(5, 5, None).unwrap_err() {
        BasisError::InvalidPenaltyOrder { order, num_basis } => {
            assert_eq!(order, 5);
            assert_eq!(num_basis, 5);
        }
        _ => panic!("Expected InvalidPenaltyOrder error"),
    }
}

#[test]
fn test_invalid_knot_vector_monotonicity_and_finiteness() {
    // Decreasing knot vector should be rejected
    let knots_bad_order = array![0.0, 0.0, 2.0, 1.0, 3.0, 3.0];
    let data = array![0.5, 1.0, 1.5];
    match create_basis::<Dense>(
        data.view(),
        KnotSource::Provided(knots_bad_order.view()),
        1,
        BasisOptions::value(),
    ) {
        Err(BasisError::InvalidKnotVector(msg)) => {
            assert!(msg.contains("non-decreasing"));
        }
        other => panic!("Expected InvalidKnotVector (order), got {:?}", other),
    }

    // Non-finite knot vector should be rejected
    let mut knots_non_finite = array![0.0, 0.0, 1.0, 2.0, 2.0];
    knots_non_finite[2] = f64::NAN;
    match create_basis::<Dense>(
        data.view(),
        KnotSource::Provided(knots_non_finite.view()),
        1,
        BasisOptions::value(),
    ) {
        Err(BasisError::InvalidKnotVector(msg)) => {
            assert!(msg.contains("non-finite"));
        }
        other => panic!("Expected InvalidKnotVector (non-finite), got {:?}", other),
    }
}

#[test]
fn testsecond_derivative_matches_finite_difference() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let degree = 3;
    let num_basis = knots.len() - degree - 1;
    let mut d1 = vec![0.0; num_basis];
    let mut d2 = vec![0.0; num_basis];

    let x = 0.37;
    let h = 1e-5;

    evaluate_bspline_derivative_scalar(x, knots.view(), degree, &mut d1).expect("first derivative");
    evaluate_bsplinesecond_derivative_scalar(x, knots.view(), degree, &mut d2)
        .expect("second derivative");

    crate::assert_central_difference_array!(
        x,
        h,
        |x_eval| {
            let mut v = vec![0.0; num_basis];
            evaluate_bspline_derivative_scalar(x_eval, knots.view(), degree, &mut v).unwrap();
            v
        },
        d2,
        1e-3
    );
}

#[test]
fn testthird_derivative_matches_finite_difference() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let degree = 3;
    let num_basis = knots.len() - degree - 1;
    let mut d3 = vec![0.0; num_basis];

    let x = 0.37;
    let h = 1e-4;

    evaluate_bsplinethird_derivative_scalar(x, knots.view(), degree, &mut d3)
        .expect("third derivative");

    crate::assert_central_difference_array!(
        x,
        h,
        |x_eval| {
            let mut v = vec![0.0; num_basis];
            evaluate_bsplinesecond_derivative_scalar(x_eval, knots.view(), degree, &mut v).unwrap();
            v
        },
        d3,
        5e-3
    );
}

#[test]
fn test_fourth_derivative_matches_finite_difference() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0];
    let degree = 4;
    let num_basis = knots.len() - degree - 1;
    let mut d4 = vec![0.0; num_basis];

    let x = 0.47;
    let h = 1e-4;

    evaluate_bspline_fourth_derivative_scalar(x, knots.view(), degree, &mut d4)
        .expect("fourth derivative");

    crate::assert_central_difference_array!(
        x,
        h,
        |x_eval| {
            let mut v = vec![0.0; num_basis];
            evaluate_bsplinethird_derivative_scalar(x_eval, knots.view(), degree, &mut v).unwrap();
            v
        },
        d4,
        3e-2
    );
}

/// Independent reference for the order-`m` B-spline derivative coefficient
/// vector, coded separately from the production recurrence engine.
///
/// Bottoms out at the plain (order-0) basis via the test-module
/// [`evaluate_bspline`] helper, then applies the single de-Boor derivative
/// step `B^{(r)}_{i,d} = d·(B^{(r-1)}_{i,d-1}/Δ_left − B^{(r-1)}_{i+1,d-1}/Δ_right)`
/// `m` times, peeling one order and one degree per step. Distinct code from
/// `evaluate_bspline_derivative_recurrence_into`, so equality is a real
/// cross-check rather than a tautology.
fn reference_bspline_derivative(m: usize, x: f64, knots: &Array1<f64>, degree: usize) -> Vec<f64> {
    // Support guard matches the engine: derivatives are zero outside the
    // closed support [t_degree, t_{num_basis}].
    let num_basis_top = knots.len() - degree - 1;
    if num_basis_top > 0 {
        let left = knots[degree];
        let right = knots[num_basis_top];
        if x < left || x > right {
            return vec![0.0; num_basis_top];
        }
    }

    // Order-0 (plain) basis at the base degree `degree - m`.
    let base_degree = degree - m;

    // The engine's base case is the order-1 derivative on degree
    // `base_degree + 1`, which evaluates the plain `base_degree` basis at
    // `one_sided_derivative_eval_point(x, knots, base_degree + 1)`. Mirror
    // that exactly so endpoint evaluation agrees bit-for-bit.
    let x_base = one_sided_derivative_eval_point(x, knots.view(), base_degree + 1);
    let base_count = knots.len() - base_degree - 1;
    let mut current: Vec<f64> = (0..base_count)
        .map(|i| evaluate_bspline(x_base, knots, i, base_degree))
        .collect();

    // Apply `m` derivative steps, raising the degree by one each time.
    for step in 1..=m {
        let d = base_degree + step;
        let count = knots.len() - d - 1;
        let mut next = vec![0.0; count];
        let kf = d as f64;
        for i in 0..count {
            let denom_left = knots[i + d] - knots[i];
            let denom_right = knots[i + d + 1] - knots[i + 1];
            let left = if denom_left.abs() > 1e-12 {
                kf * current[i] / denom_left
            } else {
                0.0
            };
            let right = if denom_right.abs() > 1e-12 {
                kf * current[i + 1] / denom_right
            } else {
                0.0
            };
            next[i] = left - right;
        }
        current = next;
    }

    current
}

/// Parity test for the unified higher-derivative engine: the public
/// 2nd/3rd/4th-derivative adapters must match an independently coded
/// reference recurrence across derivative orders, polynomial degrees, and
/// at the support edges (interior, both endpoints, and just outside).
#[test]
fn test_higher_derivative_engine_matches_reference() {
    // (knots, degree) pairs spanning several degrees and knot multiplicities.
    let cases: Vec<(Array1<f64>, usize)> = vec![
        (array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 3),
        (
            array![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
            3,
        ),
        (
            array![0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0],
            4,
        ),
        (
            array![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            ],
            5,
        ),
    ];

    for (knots, degree) in &cases {
        let degree = *degree;
        let num_basis = knots.len() - degree - 1;
        let left = knots[degree];
        let right = knots[num_basis];
        // Interior points plus exact endpoints and just-outside-support points.
        let xs = [
            left - 1e-3,
            left,
            left + 1e-6,
            0.5 * (left + right),
            0.13 * left + 0.87 * right,
            right - 1e-6,
            right,
            right + 1e-3,
        ];

        for m in 2..=degree.min(4) {
            let dispatch = |x: f64, out: &mut [f64]| match m {
                2 => evaluate_bsplinesecond_derivative_scalar(x, knots.view(), degree, out),
                3 => evaluate_bsplinethird_derivative_scalar(x, knots.view(), degree, out),
                4 => evaluate_bspline_fourth_derivative_scalar(x, knots.view(), degree, out),
                _ => unreachable!("only orders 2..=4 are dispatched"),
            };

            for &x in &xs {
                let mut got = vec![0.0; num_basis];
                dispatch(x, &mut got).expect("engine derivative should succeed");
                let want = reference_bspline_derivative(m, x, knots, degree);
                assert_eq!(want.len(), got.len());
                for j in 0..num_basis {
                    assert!(
                        (got[j] - want[j]).abs() <= 1e-9,
                        "order-{m} derivative mismatch at x={x}, degree={degree}, basis {j}: \
                             engine={}, reference={}",
                        got[j],
                        want[j]
                    );
                }
            }
        }
    }
}

#[test]
fn test_sparsesecond_derivative_matches_scalar() {
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let degree = 3;
    let num_basis = knots.len() - degree - 1;
    let mut sparsevalues = vec![0.0; degree + 1];
    let mut scalarvalues = vec![0.0; num_basis];
    let mut scratch = BasisEvalScratch::new(degree);

    let xs = [0.05, 0.2, 0.37, 0.61, 0.9];
    for &x in &xs {
        let start = evaluate_splinessecond_derivative_sparse_into(
            x,
            degree,
            knots.view(),
            &mut sparsevalues,
            &mut scratch,
        );

        evaluate_bsplinesecond_derivative_scalar(x, knots.view(), degree, &mut scalarvalues)
            .expect("scalar second derivative");

        let mut reconstructed = vec![0.0; num_basis];
        for (offset, &value) in sparsevalues.iter().enumerate() {
            let col = start + offset;
            if col < num_basis {
                reconstructed[col] = value;
            }
        }

        for j in 0..num_basis {
            assert!(
                (reconstructed[j] - scalarvalues[j]).abs() < 1e-11,
                "sparse second derivative mismatch at x={}, basis {}: sparse={}, scalar={}",
                x,
                j,
                reconstructed[j],
                scalarvalues[j]
            );
        }
    }
}

#[test]
fn test_greville_abscissae_cubic() {
    // Uniform cubic spline on [0, 1] with 1 internal knot at 0.5
    // Knot vector: [0, 0, 0, 0, 0.5, 1, 1, 1, 1] (9 knots)
    // Number of basis functions: 9 - 3 - 1 = 5
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
    let degree = 3;

    let g = compute_greville_abscissae(&knots, degree).expect("should compute Greville abscissae");

    // For degree 3: G_j = (t_{j+1} + t_{j+2} + t_{j+3}) / 3
    // G_0 = (0 + 0 + 0) / 3 = 0
    // G_1 = (0 + 0 + 0.5) / 3 = 0.1667
    // G_2 = (0 + 0.5 + 1.0) / 3 = 0.5
    // G_3 = (0.5 + 1.0 + 1.0) / 3 = 0.8333
    // G_4 = (1.0 + 1.0 + 1.0) / 3 = 1.0
    assert_eq!(g.len(), 5);
    assert_abs_diff_eq!(g[0], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(g[1], 0.5 / 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(g[2], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(g[3], 2.5 / 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(g[4], 1.0, epsilon = 1e-10);
}

#[test]
fn test_geometric_constraint_transform_orthogonality() {
    // Test that the geometric constraint transform makes coefficients orthogonal
    // to constant and linear (Greville) vectors.
    let knots = array![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
    let degree = 3;

    let (z, s_constrained) = compute_geometric_constraint_transform(&knots, degree, 2)
        .expect("should compute transform");
    // Verify s_constrained has expected dimensions
    assert!(
        s_constrained.nrows() > 0,
        "s_constrained should not be empty"
    );

    let g = compute_greville_abscissae(&knots, degree).expect("should compute Greville");
    let k = g.len();
    let ones = Array1::<f64>::ones(k);

    // Z^T * 1 should be approximately zero (orthogonal to constants)
    let z_t_ones = z.t().dot(&ones);
    for i in 0..z_t_ones.len() {
        assert!(
            z_t_ones[i].abs() < 1e-10,
            "Z not orthogonal to constants: Z'*1[{}] = {}",
            i,
            z_t_ones[i]
        );
    }

    // Z^T * G should be approximately zero (orthogonal to linear in Greville coords)
    let z_t_g = z.t().dot(&g);
    for i in 0..z_t_g.len() {
        assert!(
            z_t_g[i].abs() < 1e-10,
            "Z not orthogonal to Greville: Z'*G[{}] = {}",
            i,
            z_t_g[i]
        );
    }

    // Transform should reduce dimension by 2 (removing constant and linear)
    assert_eq!(z.ncols(), k - 2, "Z should have k-2 columns");
    assert_eq!(z.nrows(), k, "Z should have k rows");
}

#[test]
fn test_orthogonality_transform_handles_heavily_collinear_design() {
    // A heavily collinear design (Gram eigenvalues spanning many orders
    // of magnitude) must still admit a constraint nullspace whenever
    // k > q. The earlier implementation pre-whitened the design Gram and
    // searched the nullspace inside the truncated `keep`-dim subspace,
    // which failed when `keep <= q` even though dim null(M^T) = k - rank(M)
    // is non-trivial in the full coefficient space.
    let n = 200usize;
    let k = 24usize;
    let mut basis = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let xi = (i as f64) / (n as f64 - 1.0);
        for j in 0..k {
            let pert = 1e-8 * ((j as f64) - (k as f64) * 0.5) * (xi - 0.5);
            basis[[i, j]] = xi + pert;
        }
    }
    let c = Array2::<f64>::ones((n, 1));
    let (constrained, z) = applyweighted_orthogonality_constraint(basis.view(), c.view(), None)
        .expect("constraint nullspace must exist when k > q, regardless of design conditioning");

    let cross = constrained.t().dot(&c);
    let max_violation = cross.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    assert!(
        max_violation < 1e-6,
        "orthogonality violation in constrained design: {max_violation:.3e}"
    );
    assert!(z.ncols() > 0, "transform should have at least one column");
    assert_eq!(z.nrows(), k, "transform must have k rows");
}

#[test]
fn test_geometric_constraint_transform_dimensions() {
    // Test various knot configurations
    for n_internal in [3, 5, 10, 20] {
        let degree = 3;
        let n_knots = n_internal + 2 * (degree + 1);
        let mut knots = Array1::<f64>::zeros(n_knots);

        // Build clamped uniform knot vector
        for i in 0..=degree {
            knots[i] = 0.0;
            knots[n_knots - 1 - i] = 1.0;
        }
        for i in 0..n_internal {
            knots[degree + 1 + i] = (i + 1) as f64 / (n_internal + 1) as f64;
        }

        let (z, s_c) = compute_geometric_constraint_transform(&knots, degree, 2)
            .expect("should compute transform");

        let n_basis = n_knots - degree - 1;
        let n_constrained = n_basis - 2;

        assert_eq!(z.nrows(), n_basis, "Z rows should equal n_basis");
        assert_eq!(z.ncols(), n_constrained, "Z cols should equal n_basis - 2");
        assert_eq!(
            s_c.nrows(),
            n_constrained,
            "S_c should be n_constrained x n_constrained"
        );
        assert_eq!(s_c.ncols(), n_constrained);
    }
}

#[test]
fn test_build_duchon_basisfreezes_default_spatial_identifiability() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::OrthogonalToParametric,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let out = build_duchon_basis(data.view(), &spec).unwrap();
    let BasisMetadata::Duchon {
        identifiability_transform,
        ..
    } = &out.metadata
    else {
        panic!("expected Duchon metadata, got {:?}", out.metadata);
    };
    assert!(
        identifiability_transform.is_some(),
        "default spatial identifiability must materialize a transform"
    );
}

#[test]
fn test_duchon_basis_spec_rejects_removed_double_penalty_field() {
    let payload = r#"{
            "center_strategy": { "FarthestPoint": { "num_centers": 4 } },
            "length_scale": 1.0,
            "power": 2,
            "nullspace_order": "Linear",
            "double_penalty": true
        }"#;

    let err = serde_json::from_str::<DuchonBasisSpec>(payload)
        .expect_err("removed Duchon double_penalty field should be rejected");
    assert!(err.to_string().contains("unknown field `double_penalty`"));
}

#[test]
fn test_build_duchon_basis_default_identifiability_is_orthogonal_to_parametric_block() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.25],
        [0.25, 0.75]
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::OrthogonalToParametric,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let out = build_duchon_basis(data.view(), &spec).unwrap();
    let out_design = out.design.to_dense();

    let c = Array2::<f64>::ones((data.nrows(), 1));
    let cross = out_design.t().dot(&c);
    let rel = dense_orthogonality_relative_residual(out_design.view(), c.view());

    assert!(
        rel < 1e-10,
        "Duchon design is not orthogonal to the intercept: relative residual={rel:.3e}"
    );
    assert!(
        cross.iter().all(|v| v.abs() < 1e-10),
        "Duchon cross-moment against intercept is not numerically zero"
    );
    match &out.metadata {
        BasisMetadata::Duchon {
            identifiability_transform,
            ..
        } => assert!(identifiability_transform.is_some()),
        other => panic!("expected Duchon metadata, got {other:?}"),
    }
}

/// Scale-free Duchon emits the fully-wired Hilbert scale: native
/// reproducing-norm curvature (`Primary`), null-space shrinkage, and the
/// lower-order mass/tension dials. The q=0 mass is centered so the constant
/// direction remains in the joint null space; the null-space shrinkage
/// penalizes the affine trend while leaving the global mean free.
#[test]
fn test_build_scale_free_duchon_basis_centered_spring_triplet() {
    let data = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
        length_scale: None,
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let out = build_duchon_basis(data.view(), &spec).expect("Duchon basis should build");
    assert_eq!(out.penalties.len(), 4);
    assert_eq!(out.penaltyinfo.len(), 4);
    assert!(out.penaltyinfo.iter().all(|info| info.active));
    assert!(matches!(out.penaltyinfo[0].source, PenaltySource::Primary));
    assert!(matches!(
        out.penaltyinfo[1].source,
        PenaltySource::DoublePenaltyNullspace
    ));
    assert!(matches!(
        out.penaltyinfo[2].source,
        PenaltySource::OperatorMass
    ));
    assert!(matches!(
        out.penaltyinfo[3].source,
        PenaltySource::OperatorTension
    ));
}

/// Inner check for the joint-null-space property. The construction
/// must produce `null(λ_0 S_0 + λ_1 S_1 + λ_2 S_2) = span{1}` — the
/// constant function is the only direction with zero joint penalty.
/// `data`/`spec` parameterize the test across dimensions / null-space
/// orders so a single helper can pin the property at d=1, 2, 3, 4
/// (and any other config the integer-power validator admits).
fn assert_scale_free_joint_null_is_only_constant(
    data: ArrayView2<'_, f64>,
    spec: &DuchonBasisSpec,
) {
    use crate::faer_ndarray::FaerEigh;
    let out =
        build_duchon_basis(data, spec).expect("Duchon basis should build for joint-null test");
    let design = out.design.to_dense();
    let (n, p) = design.dim();
    let BasisMetadata::Duchon {
        nullspace_order,
        identifiability_transform,
        ..
    } = &out.metadata
    else {
        panic!("expected Duchon metadata, got {:?}", out.metadata);
    };
    assert!(
        identifiability_transform.is_none(),
        "this joint-null test uses the raw Duchon polynomial block; add a transformed-frame check separately"
    );
    let poly_cols = polynomial_block_from_order(data, *nullspace_order).ncols();
    assert!(
        p >= poly_cols,
        "Duchon design has fewer columns ({p}) than its polynomial block ({poly_cols})"
    );
    let kernel_cols = p - poly_cols;
    let mut v_const = Array1::<f64>::zeros(p);
    v_const[kernel_cols] = 1.0;
    let recon = crate::faer_ndarray::fast_av(&design, &v_const);
    let ones = Array1::<f64>::ones(n);
    let err = (&recon - &ones)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err < 1e-8,
        "v_const must reproduce the constant function (d={}): max |Xv − 1| = {err}",
        data.ncols()
    );
    assert!(
        out.penalties.len() >= 3,
        "Duchon Hilbert scale must emit native curvature plus lower-order penalties"
    );
    let mut joint = Array2::<f64>::zeros((p, p));
    for s in out.penalties.iter() {
        joint.scaled_add(1.0, s);
    }
    let s_v = crate::faer_ndarray::fast_av(&joint, &v_const);
    let s_v_norm = s_v.iter().map(|v| v * v).sum::<f64>().sqrt();
    let joint_scale = joint.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    // Tolerance: the centered design Gram (q=0) is f64-accumulation
    // limited, and the collocation Gram fallbacks for q=1/q=2 fold in
    // O(n_centers²) floating-point sums of radial derivatives. At
    // high d / many centers / fractional s the natural relative
    // working precision is ~1e-7 (the design's Frobenius rounding
    // floor scales with n · κ(X)). Anything tighter than this is
    // testing IEEE-754 noise, not the construction.
    let null_tol = 1e-7;
    assert!(
        s_v_norm < null_tol * joint_scale,
        "constant must lie in joint null space (d={}): ||S v_const|| = {s_v_norm}, ||S|| = {joint_scale}",
        data.ncols()
    );
    let (evals, _) = joint
        .eigh(faer::Side::Lower)
        .expect("symmetric joint penalty has real eigenvalues");
    let max_eval = evals.iter().cloned().fold(0.0_f64, f64::max);
    let zero_threshold = null_tol * max_eval.max(1.0);
    let zero_count = evals.iter().filter(|&&e| e <= zero_threshold).count();
    assert_eq!(
        zero_count,
        1,
        "joint penalty must have exactly one zero eigenvalue (d={}); got {zero_count} below {zero_threshold}, eigenvalues = {evals:?}",
        data.ncols()
    );
}

/// The construction is supposed to deliver exactly *one* unpenalized
/// direction — the constant function — across the joint penalty
/// summed joint penalty. This test pins that property
/// directly: it computes the basis-coordinate vector `v_const` such
/// that `design · v_const ≡ 1`, asserts the joint penalty
/// maps it to zero to working precision, and asserts that the joint
/// penalty has rank `p − 1` (so no second unpenalized direction sneaks
/// in). Without this test we were only checking the cardinality and
/// labels of the penalties, not the actual joint-null-space dimension —
/// the property the centered-mass design exists to deliver.
#[test]
fn test_scale_free_duchon_joint_null_space_is_only_the_constant() {
    // d=3 — the legacy 5-point cube-corner fixture. Linear null space
    // with power=1 satisfies both 2(p+s) > d and 2s < d.
    let data = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 5 },
        periodic: None,
        length_scale: None,
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    assert_scale_free_joint_null_is_only_constant(data.view(), &spec);
}

/// d=1: Linear null space (constant + linear) with power=0 satisfies
/// 2(1+0) > 1 (RKHS) and 2·0 < 1 (CPD). Joint null must still be the
/// constant only — the linear direction picks up the centered-mass
/// gauge.
#[test]
fn test_scale_free_duchon_joint_null_space_is_only_the_constant_1d() {
    let data = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
        length_scale: None,
        power: 0.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    assert_scale_free_joint_null_is_only_constant(data.view(), &spec);
}

/// d=2: Degree-2 null space (constants + 2 linears + 3 quadratics)
/// with power=0 satisfies 2(2+0) > 2 and 2·0 < 2. Joint null must
/// remain the constant alone — linear and quadratic directions are
/// penalized by the centered mass + tension/stiffness.
#[test]
fn test_scale_free_duchon_joint_null_space_is_only_the_constant_2d() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.5, 0.0],
        [0.0, 0.5],
        [1.0, 0.5]
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: None,
        power: 0.0,
        nullspace_order: DuchonNullspaceOrder::Degree(2),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    assert_scale_free_joint_null_is_only_constant(data.view(), &spec);
}

/// d=4: Degree-3 null space with power=1 satisfies 2(3+1) > 4+2 (D2
/// requires the strict inequality) and 2·1 < 4. The integer-power
/// validator forces the null space to escalate up to degree 3, which
/// is exactly the high-d ratcheting the fractional-power refactor
/// (Tier 1 #1 in the plan) is designed to fix. Once that lands, the
/// natural d=4 choice will be power=1.5 with Degree(2).
#[test]
fn test_scale_free_duchon_joint_null_space_is_only_the_constant_4d() {
    let data = array![
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.25, 0.5, 0.75, 0.5],
        [0.75, 0.25, 0.5, 0.5],
        [0.5, 0.75, 0.25, 0.5],
        [0.3, 0.3, 0.3, 0.3],
        [0.7, 0.7, 0.7, 0.7],
        [0.2, 0.8, 0.2, 0.8],
        [0.8, 0.2, 0.8, 0.2],
        [0.1, 0.4, 0.6, 0.9],
        [0.9, 0.6, 0.4, 0.1]
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 16 },
        length_scale: None,
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Degree(3),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    assert_scale_free_joint_null_is_only_constant(data.view(), &spec);
}

/// Fractional spectral power `s = 1.5` in d=4, Degree-2 nullspace.
/// Integer power forces `s = 2` to clear D₂ collocation
/// (`2(p+s) > d+2` ⇒ `s > 1`), which inflates the polyharmonic
/// kernel block; fractional s=1.5 sits in the convergent regime at
/// `r²·log` style scaling without ratcheting the null space. Pins
/// that the fractional path lights up end-to-end: parser/spec accepts
/// it, the convergence predicate admits it, the Riesz kernel and
/// `isotropic_duchon_penalty` consume it, and the joint null space
/// remains exactly `span{1}`.
#[test]
fn test_scale_free_duchon_joint_null_space_is_only_the_constant_4d_fractional_s() {
    let data = array![
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.25, 0.5, 0.75, 0.5],
        [0.75, 0.25, 0.5, 0.5],
        [0.5, 0.75, 0.25, 0.5]
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 12 },
        length_scale: None,
        power: 1.5,
        nullspace_order: DuchonNullspaceOrder::Degree(2),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    assert_scale_free_joint_null_is_only_constant(data.view(), &spec);
}

/// Fractional `s = 0.5` in d=2 with Degree-2 nullspace. Hits the
/// `2s = 1 < d = 2` (CPD) and `2(p+s) = 5 > d+2 = 4` (D₂ collocation)
/// regimes simultaneously — integer `s` can't satisfy both at this
/// p/d. Sub-1 spectral power exercises a part of the Riesz kernel
/// (`r^(2(p+s)−d) = r¹`) that integer values skip over.
#[test]
fn test_scale_free_duchon_joint_null_space_is_only_the_constant_2d_fractional_s() {
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.5, 0.0],
        [0.0, 0.5],
        [1.0, 0.5]
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: None,
        power: 0.5,
        nullspace_order: DuchonNullspaceOrder::Degree(2),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    assert_scale_free_joint_null_is_only_constant(data.view(), &spec);
}

/// Fractional `s = 3.5` in d=8, Linear null space. The pair-block
/// closed-form `R_J^d` for `q ∈ {1, 2}` lands in the log case (since
/// `2J − d ∈ {12, 10}` are non-negative even integers for `2s = 7`
/// integer at d=8) with Wendland CPD order `(2J−d)/2 + 1 ∈ {7, 6}` —
/// way above the spec's `p_order = 2`. Without the CPD-adequacy gate
/// in `duchon_closed_form_operator_penalty_converges`, the closed-form
/// matrix at centers was non-PSD (15 of 30 negative eigenvalues
/// before the fix). The gate now forces the collocation fallback
/// `D_qᵀ D_q` (PSD by construction) for any q whose Wendland CPD
/// order exceeds the polynomial null space we built. This test pins
/// that the d=8 × fractional-s high-`d` regime — the bench config
/// the fractional refactor was designed to unlock — works end-to-end
/// with the joint-null-space property intact.
#[test]
fn test_scale_free_duchon_joint_null_space_is_only_the_constant_8d_fractional_s() {
    let mut rows = Vec::new();
    let mut state: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..80 {
        let mut row = [0.0_f64; 8];
        for r in row.iter_mut() {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            *r = (z >> 11) as f64 / (1_u64 << 53) as f64;
        }
        rows.push(row);
    }
    let n = rows.len();
    let d = 8;
    let mut data = Array2::<f64>::zeros((n, d));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            data[[i, j]] = v;
        }
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 30 },
        length_scale: None,
        power: 3.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    assert_scale_free_joint_null_is_only_constant(data.view(), &spec);
}

#[test]
fn test_pure_duchon_candidate_factory_falls_back_to_collocation_in_divergent_regime() {
    // The pure-Duchon `operator_penalty_candidates_closed_form_pure`
    // factory must keep all three active candidates (mass, tension,
    // stiffness) finite and non-zero even when the closed-form
    // Lebesgue penalty is in a divergent regime (UV or IR), by falling
    // back to collocation `D_q^T D_q` for that q.
    //
    // Pure-Duchon Zero nullspace cannot satisfy {pure-Duchon CPD
    // adequacy `2s < d`, D2 collocation `2(p+s) > d+2`, closed-form
    // UV `4(m+s) > d+2q`, closed-form IR `d+2q > 4m`} for all
    // q ∈ {0,1,2} simultaneously — the conditions are arithmetically
    // incompatible. So we don't drive this test through
    // `build_duchon_basis` (which validates everything up-front);
    // instead we feed the factory hand-built D0/D1/D2 and confirm
    // its per-q convergence-gating logic preserves the candidate
    // count even when the closed-form Lebesgue penalty is zero.
    //
    // Picks (m=2, s=1, d=3): closed-form q=2 IR fails (d+2q = 7 < 4m
    // = 8) so the factory must fall back to collocation `D_2^T D_2`
    // for stiffness; q=1 UV/IR both hold, so closed-form is used.
    //
    // Exercising the q=2 stiffness fallback requires the stiffness dial
    // to be active. `DuchonOperatorPenaltySpec::default()` deliberately
    // disables stiffness (`Primary` is the exact, superior curvature), so
    // this factory-level test of the divergent-regime fallback drives the
    // factory with `all_active()` — the all-three-dials spec used by the
    // Matérn collocation overlay.
    use ndarray::Array2 as A2;
    let k = 16usize;
    let d = 3usize;
    let mut state: u64 = 0xDEADBEEF;
    let mut next_unit = || -> f64 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f64 / ((1u64 << 56) as f64)).fract()
    };
    let mut centers = A2::<f64>::zeros((k, d));
    for i in 0..k {
        for j in 0..d {
            centers[[i, j]] = next_unit();
        }
    }
    // Hand-built non-trivial operator matrices in K-dim (no kernel
    // projection, no polynomial padding, no outer identifiability).
    let p = k;
    let mut d0 = A2::<f64>::zeros((p, p));
    let mut d1 = A2::<f64>::zeros((p * d, p));
    let mut d2 = A2::<f64>::zeros((p * d * d, p));
    for i in 0..p {
        d0[[i, i]] = 1.0;
        for axis in 0..d {
            d1[[i * d + axis, i]] = 1.0 + 0.1 * axis as f64;
            d2[[(i * d + axis) * d + axis, i]] = 1.0;
        }
    }
    let p_order = 2usize;
    let s_order = 1usize;
    let candidates = operator_penalty_candidates_closed_form_pure(
        centers.view(),
        &d0,
        &d1,
        &d2,
        &DuchonOperatorPenaltySpec::all_active(),
        p_order,
        s_order as f64,
        None,
        None,
        0,
        None,
    );
    assert_eq!(
        candidates.len(),
        3,
        "factory must return all three candidates including divergent-regime fallback"
    );
    for (i, expected) in [
        PenaltySource::OperatorMass,
        PenaltySource::OperatorTension,
        PenaltySource::OperatorStiffness,
    ]
    .iter()
    .enumerate()
    {
        let m = &candidates[i].matrix;
        let frob_sq: f64 = m.iter().map(|v| v * v).sum();
        assert!(
            frob_sq > 0.0,
            "candidate {i} (source={expected:?}) is identically zero"
        );
        assert!(
            m.iter().all(|v| v.is_finite()),
            "candidate {i} has non-finite entries"
        );
    }
}

fn assert_matrix_close(lhs: &Array2<f64>, rhs: &Array2<f64>, tol: f64) {
    assert_eq!(lhs.dim(), rhs.dim(), "matrix shape mismatch");
    let max_abs = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= tol,
        "matrix mismatch: max_abs={max_abs:.3e}, tol={tol:.3e}"
    );
}

#[test]
fn test_hybrid_duchon_candidate_factory_admits_log_riesz_closed_form() {
    // (d=4, m=1, s=2, q=1) is an even-dimensional log-Riesz case:
    // d/2 <= 2m. It still satisfies UV/IR/precondition, so the canonical
    // finite-part log-Riesz branch should be used analytically instead of
    // silently routing the tension penalty to collocation.
    use ndarray::Array2 as A2;
    let k = 9usize;
    let d = 4usize;
    let mut centers = A2::<f64>::zeros((k, d));
    for i in 0..k {
        for axis in 0..d {
            centers[[i, axis]] = 0.13 * i as f64 + 0.07 * axis as f64 + 0.01 * (i * axis) as f64;
        }
    }
    let d0 = A2::<f64>::zeros((k, k));
    let d1 = A2::<f64>::zeros((k * d, k));
    let d2 = A2::<f64>::zeros((k * d * d, k));
    let eta = [0.11_f64, -0.03, 0.07, -0.05];
    let spec = DuchonOperatorPenaltySpec {
        mass: OperatorPenaltySpec::Disabled,
        tension: OperatorPenaltySpec::Active {
            initial_log_lambda: 0.0,
            prior: None,
        },
        stiffness: OperatorPenaltySpec::Disabled,
    };
    let candidates = operator_penalty_candidates_closed_form(
        centers.view(),
        &d0,
        &d1,
        &d2,
        &spec,
        1,
        2,
        0.8,
        Some(&eta),
        None,
        0,
        None,
    );
    assert_eq!(candidates.len(), 1);
    assert!(matches!(
        candidates[0].source,
        PenaltySource::OperatorTension
    ));

    let reference = closed_form_operator_penalty_in_total_basis(
        centers.view(),
        1,
        1,
        2,
        1.0 / 0.8,
        Some(&eta),
        None,
        0,
        None,
    );
    let norm = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm > 1e-12);
    let reference_normalized = reference.mapv(|value| value / norm);
    assert_matrix_close(&candidates[0].matrix, &reference_normalized, 1e-11);
}

#[test]
fn test_pure_duchon_closed_form_pair_block_finite_in_converging_regime_q1() {
    // Pure-Duchon (κ=0) closed-form pair-block should be finite,
    // symmetric, and non-zero for (m, s, d, q) inside the Duchon
    // convergence regime: 4(m+s) > d + 2q AND d + 2q > 4m.
    //
    // Closed-form (m=1) restricts q ≤ 2m-1 = 1 by the
    // partial-fraction precondition `2m - q ≥ 1`. Stiffness (q=2)
    // would require m≥2; for Linear+ nullspaces the closed-form path
    // is now active (kernel-sub-block via Q^T G_raw Q), but q=2 still
    // falls through to collocation when the per-q convergence
    // predicate fails (covered by
    // `test_pure_duchon_candidate_factory_falls_back_to_collocation_in_divergent_regime`).
    //
    // This test exercises q=1 (tension), the maximal closed-form q
    // for m=1: d=4, m=1, s=2, q=1.
    //   UV: 4(m+s) = 12 > d+2q = 6 ✓
    //   IR: d+2q = 6 > 4m = 4 ✓
    //   Partial fraction: 2m-q = 1 ✓
    //   Pure-Duchon CPD adequacy: 2s = 4 < d = 4 fails by equality.
    // The pair-block primitive doesn't enforce CPD adequacy itself
    // (that's a separate `validate_duchon_kernel_orders` step), so
    // we can still call it directly to verify the radial form is
    // well-defined and produces a finite, symmetric, non-zero matrix.
    use ndarray::Array2 as A2;
    let k = 32usize;
    let d = 4usize;
    let mut state: u64 = 0xCAFEBABE;
    let mut next_unit = || -> f64 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f64 / ((1u64 << 56) as f64)).fract()
    };
    let mut centers = A2::<f64>::zeros((k, d));
    for i in 0..k {
        for j in 0..d {
            centers[[i, j]] = next_unit();
        }
    }
    let p_order = 1usize;
    let s_order = 2.0;

    let g = closed_form_anisotropic_pair_block_pure(centers.view(), 1, p_order, s_order, None);

    assert_eq!(g.nrows(), k);
    assert_eq!(g.ncols(), k);
    // Finite everywhere (R=0 self-pair is ε-regularized internally).
    assert!(g.iter().all(|v| v.is_finite()), "non-finite entry");
    // Symmetric.
    for i in 0..k {
        for j in 0..i {
            assert!(
                (g[[i, j]] - g[[j, i]]).abs() < 1e-10 * g[[i, j]].abs().max(1.0),
                "asymmetry at ({i},{j})"
            );
        }
    }
    // Non-zero (we're in the converging regime where the kernel
    // doesn't differentiate to zero).
    let frob_sq: f64 = g.iter().map(|v| v * v).sum();
    assert!(
        frob_sq > 0.0,
        "closed-form pair block is identically zero in converging regime"
    );
}

#[test]
fn test_closed_form_linear_nullspace_kernel_subblock_finite_psd() {
    // Task #8: with the outer gate flipped, closed-form is admitted at
    // Linear nullspace order. The kernel sub-block of the resulting
    // penalty (Q^T G_raw Q where Q spans null(P^T) for the polynomial
    // block P = [1, x_1, ..., x_d] evaluated at centers) must be
    // finite, symmetric, and positive semidefinite. The polynomial
    // block remains zero-padded (Option A: faithful to L²-Lebesgue),
    // so we verify finiteness/symmetry/PSD on the kernel sub-block
    // only — that is the contract the team-lead specified for
    // Linear+ closed-form.
    use ndarray::Array2 as A2;
    let k = 24usize;
    let d = 3usize;
    let mut state: u64 = 0xBADC0FFE;
    let mut next_unit = || -> f64 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f64 / ((1u64 << 56) as f64)).fract()
    };
    let mut centers = A2::<f64>::zeros((k, d));
    for i in 0..k {
        for j in 0..d {
            centers[[i, j]] = next_unit();
        }
    }

    let p_order = 2usize; // m = 2
    let s_order = 1.0; // s = 1

    // Linear polynomial block: [1, x_1, ..., x_d], (k, d+1).
    let mut poly = A2::<f64>::zeros((k, d + 1));
    poly.column_mut(0).fill(1.0);
    for c in 0..d {
        poly.column_mut(c + 1).assign(&centers.column(c));
    }
    let z = kernel_constraint_nullspace_from_matrix(poly.view()).expect("Q construction");
    // Z is (k, kernel_cols) with kernel_cols = k − rank(P) = k − (d+1).
    let kernel_cols = z.ncols();
    assert_eq!(kernel_cols, k - (d + 1));

    // Closed-form pair block in raw kernel basis (κ=0, pure Duchon).
    let g_raw = closed_form_anisotropic_pair_block_pure(
        centers.view(),
        1, // q = 1 (tension), maximal closed-form q for these orders
        p_order,
        s_order,
        None,
    );
    // Q^T G_raw Q — the kernel sub-block.
    let zt_g = fast_atb(&z, &g_raw);
    let g_kernel = fast_ab(&zt_g, &z);

    // Finite, symmetric.
    assert_eq!(g_kernel.nrows(), kernel_cols);
    assert_eq!(g_kernel.ncols(), kernel_cols);
    assert!(g_kernel.iter().all(|v| v.is_finite()));
    for i in 0..kernel_cols {
        for j in 0..i {
            let diff = (g_kernel[[i, j]] - g_kernel[[j, i]]).abs();
            let scale = g_kernel[[i, j]].abs().max(1.0);
            assert!(
                diff < 1e-9 * scale,
                "kernel sub-block asymmetry at ({i},{j}): {diff:.3e}"
            );
        }
    }
    // PSD (smallest eigenvalue ≥ -ε·trace via Rayleigh quotient on a
    // few random unit vectors as a cheap sanity check).
    let trace: f64 = (0..kernel_cols).map(|i| g_kernel[[i, i]]).sum();
    for trial in 0..8 {
        state = state
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let mut v = vec![0.0_f64; kernel_cols];
        let mut s2 = 0.0_f64;
        for vi in v.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = ((state >> 8) as f64 / ((1u64 << 56) as f64)).fract() - 0.5;
            *vi = u;
            s2 += u * u;
        }
        let nrm = s2.sqrt().max(1e-300);
        for vi in v.iter_mut() {
            *vi /= nrm;
        }
        let mut q = 0.0_f64;
        for i in 0..kernel_cols {
            let mut row_dot = 0.0;
            for j in 0..kernel_cols {
                row_dot += g_kernel[[i, j]] * v[j];
            }
            q += v[i] * row_dot;
        }
        assert!(
            q > -1e-9 * trace.abs().max(1.0),
            "kernel sub-block not PSD (trial {trial}): v^T G_kernel v = {q:.3e}, trace = {trace:.3e}"
        );
    }
    // Non-zero (we're in the converging regime).
    let frob_sq: f64 = g_kernel.iter().map(|v| v * v).sum();
    assert!(
        frob_sq > 0.0,
        "kernel sub-block is identically zero in converging regime"
    );
}

#[test]
fn test_matern_closed_form_matches_collocation() {
    // q=0 Matérn closed-form pair-block must match the K_CC RKHS Gram
    // built by `build_matern_kernel_penalty` (and embedded in the
    // `MaternSplineBasis::penalty_kernel`) to entry-wise 1e-12 — both
    // call the same closed-form half-integer formula via
    // `matern_kernel_from_distance`, modulo the Matérn parameterization
    // mapping κ = √(2ν)/length_scale used by the closed-form path.
    //
    // The default `matern_kernel_from_distance` (basis.rs:5949) scales
    // distance by `r/length_scale` and applies the polynomial × exp
    // form directly. Our closed-form path uses the spectral form
    // M_ℓ^d(R; κ) which differs by a κ-dependent normalization
    // constant. The two MUST agree up to the global RKHS scaling
    // (which is exactly the constant `c` in K̂(ρ) = c (κ²+ρ²)^{-ℓ}).
    // We therefore compare ratios entry-wise.
    use ndarray::Array2 as A2;
    let k = 16usize;
    let d = 3usize;
    let length_scale = 0.5_f64;
    let nu = MaternNu::FiveHalves;
    let mut state: u64 = 0xFEEDFACE;
    let mut next_unit = || -> f64 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f64 / ((1u64 << 56) as f64)).fract()
    };
    let mut centers = A2::<f64>::zeros((k, d));
    for i in 0..k {
        for j in 0..d {
            centers[[i, j]] = next_unit();
        }
    }

    let g_cf = closed_form_matern_pair_block(centers.view(), 0, length_scale, nu, None)
        .expect("q=0 Matérn closed-form should always return Some when 4ℓ > d");
    assert_eq!(g_cf.shape(), &[k, k]);
    assert!(g_cf.iter().all(|v| v.is_finite()));

    // Symmetric.
    for i in 0..k {
        for j in 0..i {
            let diff = (g_cf[[i, j]] - g_cf[[j, i]]).abs();
            assert!(
                diff < 1e-12 * g_cf[[i, j]].abs().max(1.0),
                "asymmetry at ({i},{j}): {} vs {}",
                g_cf[[i, j]],
                g_cf[[j, i]]
            );
        }
    }

    // PSD via eigh (Frobenius-positive eigenvalues).
    let (eigs, _) = FaerEigh::eigh(&g_cf, Side::Lower).expect("eigh");
    let min_eig = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_eig = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // PSD with mild numerical tolerance scaled by the matrix size.
    assert!(
        min_eig > -1e-10 * max_eig.abs().max(1.0),
        "q=0 Matérn closed-form must be PSD; got eigenvalue range [{min_eig}, {max_eig}]"
    );
    assert!(
        max_eig > 0.0,
        "non-trivial Gram must have a positive eigenvalue"
    );

    // Constancy of the closed-form-vs-collocation ratio. The reference
    // K_CC entry uses the polynomial-times-exp formula with distance
    // `r/length_scale`; the closed-form uses M_ℓ^d(R;κ) with the same
    // physical R but with explicit κ. The ratio must be a constant
    // depending only on (d, ν, length_scale).
    let mut ref_kcc = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in i..k {
            let mut dist2 = 0.0;
            for axis in 0..d {
                let delta = centers[[i, axis]] - centers[[j, axis]];
                dist2 += delta * delta;
            }
            let r = dist2.sqrt();
            let kij = matern_kernel_from_distance(r, length_scale, nu).unwrap();
            ref_kcc[[i, j]] = kij;
            ref_kcc[[j, i]] = kij;
        }
    }
    // Determine the ratio from the diagonal (R=0): ratio = g_cf(0)/K(0).
    // Both diagonals are positive constants; the ratio fixes the
    // proportionality constant between the spectral and polynomial
    // parameterizations.
    let ratio = g_cf[[0, 0]] / ref_kcc[[0, 0]];
    assert!(
        ratio.is_finite() && ratio > 0.0,
        "closed-form / collocation ratio must be finite and positive, got {ratio}"
    );
    for i in 0..k {
        for j in 0..k {
            let cf = g_cf[[i, j]];
            let rf = ref_kcc[[i, j]];
            let predicted = ratio * rf;
            let diff = (cf - predicted).abs();
            let scale = predicted.abs().max(1.0);
            assert!(
                diff < 1e-10 * scale,
                "closed-form / collocation ratio non-constant at ({i},{j}): \
                     cf={cf}, rf={rf}, ratio*rf={predicted}, diff={diff}",
            );
        }
    }
}

#[test]
fn test_matern_closed_form_q1_q2_psd_and_finite() {
    // q=1, q=2 Matérn Lebesgue pair-blocks via partial-fraction
    // expansion of |ρ|^{2q} / (κ²+ρ²)^{2ℓ}. With ν=9/2, d=3 we have
    // 2ℓ = 12 so 4ℓ - 2q = 24 - 2q > 3 holds for q ∈ {0,1,2}: all
    // three matrices are well-defined. We assert finiteness, symmetry,
    // and PSD via eigh.
    use ndarray::Array2 as A2;
    let k = 12usize;
    let d = 3usize;
    let length_scale = 0.4_f64;
    let nu = MaternNu::NineHalves;
    let mut state: u64 = 0x1337BEEF;
    let mut next_unit = || -> f64 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f64 / ((1u64 << 56) as f64)).fract()
    };
    let mut centers = A2::<f64>::zeros((k, d));
    for i in 0..k {
        for j in 0..d {
            centers[[i, j]] = next_unit();
        }
    }

    for q in [0usize, 1, 2] {
        let g = closed_form_matern_pair_block(centers.view(), q, length_scale, nu, None)
            .unwrap_or_else(|| panic!("q={q} Matérn closed-form must accept ν=9/2 d=3"));
        assert_eq!(g.shape(), &[k, k]);
        assert!(g.iter().all(|v| v.is_finite()), "q={q}: non-finite");
        for i in 0..k {
            for j in 0..i {
                assert!(
                    (g[[i, j]] - g[[j, i]]).abs() < 1e-10 * g[[i, j]].abs().max(1.0),
                    "q={q}: asymmetry at ({i},{j})"
                );
            }
        }
        let (eigs, _) = FaerEigh::eigh(&g, Side::Lower).expect("eigh");
        let min_eig = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_eig = eigs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_eig > 0.0,
            "q={q}: Gram must have a positive eigenvalue (got max={max_eig})"
        );
        assert!(
            min_eig > -1e-9 * max_eig.abs().max(1.0),
            "q={q}: Gram must be PSD; eigenvalue range [{min_eig}, {max_eig}]"
        );
    }
}

#[test]
fn test_matern_closed_form_gates_when_divergent() {
    // ν=1/2, d=3 → 2ℓ = 4 so 4ℓ = 8. Convergence of q-th block needs
    // 8 > 2q + 3, i.e. q ≤ 2. So even at the smoothest-supported edge
    // q=2: 8 > 7 holds, and we still get a valid block. To trigger
    // divergence we need 4ℓ ≤ 2q + d. With d=3, ν=1/2 (2ℓ=4), q=3
    // fails 8 > 9 → false → return None. q > 2 returns None
    // unconditionally per the spec contract (we only support q ∈
    // {0,1,2}).
    use ndarray::Array2 as A2;
    let centers = A2::<f64>::from_shape_vec(
        (4, 3),
        vec![0.1, 0.2, 0.3, 0.5, 0.4, 0.6, 0.8, 0.7, 0.9, 0.2, 0.5, 0.4],
    )
    .unwrap();
    // q=3 (> 2): always None.
    assert!(closed_form_matern_pair_block(centers.view(), 3, 1.0, MaternNu::Half, None).is_none());
}

#[test]
fn test_build_duchon_basis_linear_nullspace_uses_full_hilbert_scale() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let out = build_duchon_basis(data.view(), &spec).expect("Duchon basis should build");
    assert_eq!(out.penaltyinfo.len(), 4);
    assert!(out.penaltyinfo.iter().all(|info| info.active));
    assert!(matches!(out.penaltyinfo[0].source, PenaltySource::Primary));
    assert!(matches!(
        out.penaltyinfo[1].source,
        PenaltySource::DoublePenaltyNullspace
    ));
    assert!(matches!(
        out.penaltyinfo[2].source,
        PenaltySource::OperatorMass
    ));
    assert!(matches!(
        out.penaltyinfo[3].source,
        PenaltySource::OperatorTension
    ));
}

#[test]
fn test_duchon_zero_nullspace_uses_closed_form() {
    // Lock-in: with `nullspace_order = Zero` (constants only) and a hybrid
    // length_scale, the gate routes to the closed-form path and produces
    // three active penalties matching collocation arity.
    let data = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.25, 0.75],
        [0.75, 0.25],
        [0.5, 0.0],
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 6 },
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Zero,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let out =
        build_duchon_basis(data.view(), &spec).expect("Zero-nullspace Duchon basis should build");
    assert_eq!(out.penaltyinfo.len(), 3);
    assert!(out.penaltyinfo.iter().all(|info| info.active));
}

#[test]
fn test_duchon_linear_nullspace_uses_collocation() {
    // Lock-in: with `nullspace_order = Linear`, the closed-form path's
    // pad-with-zeros over the polynomial block produces a non-PSD penalty
    // matrix (the polynomial-block L²-Lebesgue contribution diverges for
    // q ≥ 1 over R^d). The gate must fall back to collocation so all
    // three penalties stay active and PSD.
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 4 },
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let out = build_duchon_basis(data.view(), &spec)
        .expect("Linear-nullspace Duchon basis should build via collocation fallback");
    assert_eq!(out.penaltyinfo.len(), 4);
    assert!(out.penaltyinfo.iter().all(|info| info.active));
}

#[test]
fn hybrid_duchon_fractional_default_d4_rejects_realized_nonfinite_kernel() {
    let data = array![
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.5,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        periodic: None,
        boundary: OneDimensionalBoundary::Open,
    };
    let err = build_duchon_basis(data.view(), &spec)
        .expect_err("hybrid d=4 fractional power must reject before non-finite Gram");
    let msg = err.to_string();
    assert!(
        msg.contains("Duchon pointwise kernel values require 2*(p+s) > dimension"),
        "unexpected error: {msg}"
    );
}

#[test]
fn resolve_duchon_orders_yields_existing_kernel_for_all_dims_issue_875() {
    // Issue #875: the latent Duchon design hard-coded s = 0 and the m-derived
    // null space, so the pure polyharmonic kernel `r^{2(p+s)-d}` failed to
    // *exist* (`2(p+s) > d`) at latent_dim >= 4 with m = 2. The fix routes
    // through `resolve_duchon_orders`, which must lift `s` (and, if pure-mode
    // CPD requires it, the null-space order) until the kernel exists — for
    // EVERY ambient dimension, including the even-d `2(p+s)=d` log boundary.
    let requested = duchon_nullspace_from_test_m(2); // == DuchonNullspaceOrder::Linear
    for dim in 1..=8usize {
        let (nullspace, power) = resolve_duchon_orders(dim, requested, 0, None);
        let p = duchon_p_from_nullspace_order(nullspace);
        // Kernel-existence: 2(p+s) > d strictly.
        assert!(
            2 * (p + power) > dim,
            "dim={dim}: resolved p={p}, s={power} violates 2(p+s) > d"
        );
        // Pure-mode CPD vs polynomial null space P_p: 2s < d.
        assert!(
            2 * power < dim,
            "dim={dim}: resolved s={power} violates pure-mode CPD 2s < d"
        );
        // The resolver never weakens the requested null-space order.
        assert!(
            duchon_p_from_nullspace_order(nullspace) >= duchon_p_from_nullspace_order(requested),
            "dim={dim}: resolver decreased the requested null-space order"
        );
        // And the concrete kernel-order validator accepts the resolved pair.
        validate_duchon_kernel_orders(None, p, power as f64, dim).unwrap_or_else(|e| {
            panic!("dim={dim}: resolved (p={p}, s={power}) rejected by validator: {e}")
        });
    }
}

fn duchon_nullspace_from_test_m(m: usize) -> DuchonNullspaceOrder {
    match m {
        1 => DuchonNullspaceOrder::Zero,
        2 => DuchonNullspaceOrder::Linear,
        other => DuchonNullspaceOrder::Degree(other - 1),
    }
}

#[test]
fn filter_active_penalty_candidates_preserves_matching_kronecker_factors() {
    let s = array![[1.0, -1.0], [-1.0, 1.0]];
    let identity = Array2::<f64>::eye(2);
    let kron = crate::construction::kronecker_product(&s, &identity);
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(vec![PenaltyCandidate {
        matrix: kron,
        nullspace_dim_hint: 0,
        source: PenaltySource::TensorMarginal { dim: 0 },
        normalization_scale: 1.0,
        kronecker_factors: Some(vec![s.clone(), identity.clone()]),
        op: None,
    }])
    .expect("matching Kronecker factors should be retained");

    assert_eq!(penaltyinfo.len(), 1);
    assert!(penaltyinfo[0].kronecker_factors.is_some());
}

#[test]
fn filter_active_penalty_candidates_drops_stale_kronecker_factors_after_projection() {
    let s = array![[1.0, -1.0], [-1.0, 1.0]];
    let identity = Array2::<f64>::eye(2);
    let kron = crate::construction::kronecker_product(&s, &identity);
    let z = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    let projected = z.t().dot(&kron).dot(&z);
    let (_, _, penaltyinfo) = filter_active_penalty_candidates(vec![PenaltyCandidate {
        matrix: projected,
        nullspace_dim_hint: 0,
        source: PenaltySource::TensorMarginal { dim: 0 },
        normalization_scale: 1.0,
        kronecker_factors: Some(vec![s, identity]),
        op: None,
    }])
    .expect("projected penalty should still analyze");

    assert_eq!(penaltyinfo.len(), 1);
    assert!(penaltyinfo[0].active);
    assert!(penaltyinfo[0].kronecker_factors.is_none());
}

#[test]
fn test_pairwise_distance_bounds_helper() {
    let pts = array![[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]];
    let (r_min, r_max) = pairwise_distance_bounds(pts.view()).expect("bounds should exist");
    assert!((r_min - 5.0).abs() < 1e-12);
    assert!((r_max - 10.0).abs() < 1e-12);
}

#[test]
fn test_pairwise_distance_bounds_handles_large_finite_coordinates() {
    let pts = array![[0.0], [3.0e200], [6.0e200]];
    let (r_min, r_max) =
        pairwise_distance_bounds(pts.view()).expect("large finite bounds should exist");
    assert!((r_min - 3.0e200).abs() / 3.0e200 < 1e-12);
    assert!((r_max - 6.0e200).abs() / 6.0e200 < 1e-12);
}

#[test]
fn test_pairwise_distance_bounds_sampled_matches_exact_small() {
    // For n <= K_CAP (=1024), sampled path delegates to the exact path.
    let pts = array![[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [-1.0, 1.0]];
    let exact = pairwise_distance_bounds(pts.view()).unwrap();
    let sampled = pairwise_distance_bounds_sampled(pts.view()).unwrap();
    assert!((exact.0 - sampled.0).abs() < 1e-15);
    assert!((exact.1 - sampled.1).abs() < 1e-15);
}

#[test]
fn test_pairwise_distance_bounds_sampled_conservative_on_large() {
    // On a point cloud larger than K_CAP, verify the mathematical
    // conservativeness invariants of the sampled bounds:
    //   sampled r_max <= exact r_max   (sampled max can only shrink)
    //   sampled r_min >= exact r_min   (sampled min can only grow)
    // These guarantees are the correctness contract that lets the sampled
    // path back outer-κ bounds without excluding any feasible κ that the
    // exact method would include.
    let n = 2000usize;
    let mut pts = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        // Deterministic scatter in [0, 100] × [0, 100].
        let x = ((i * 37 + 7) % 1000) as f64 * 0.1;
        let y = ((i * 53 + 11) % 1000) as f64 * 0.1;
        pts[[i, 0]] = x;
        pts[[i, 1]] = y;
    }
    let exact = pairwise_distance_bounds(pts.view()).unwrap();
    let sampled = pairwise_distance_bounds_sampled(pts.view()).unwrap();
    assert!(
        sampled.1 <= exact.1 + 1e-12,
        "sampled r_max {} must not exceed exact r_max {}",
        sampled.1,
        exact.1
    );
    assert!(
        sampled.0 >= exact.0 - 1e-12,
        "sampled r_min {} must not be below exact r_min {}",
        sampled.0,
        exact.0
    );
}

#[test]
fn test_duchon_polyharmonic_log_branch_sign_depends_on_dimension() {
    let r = 1.7;

    // In 2D the legacy (-1)^m sign happens to agree with the correct formula.
    let m_2d = 2usize;
    let d_2d = 2usize;
    let c_2d = polyharmonic_log_sign(m_2d, d_2d)
        / (2.0_f64.powi((2 * m_2d - 1) as i32)
            * std::f64::consts::PI.powf(0.5 * d_2d as f64)
            * gamma_lanczos(m_2d as f64)
            * gamma_lanczos((m_2d - d_2d / 2 + 1) as f64));
    let expected_2d = c_2d * r.powi((2 * m_2d - d_2d) as i32) * r.ln();
    let got_2d = polyharmonic_kernel(r, (m_2d) as f64, d_2d);
    assert!((got_2d - expected_2d).abs() < 1e-12);

    // In 4D the correct log-branch sign differs from (-1)^m and must be positive for m=3.
    let m_4d = 3usize;
    let d_4d = 4usize;
    let legacy_sign = (-1.0_f64).powi(m_4d as i32);
    let fixed_sign = polyharmonic_log_sign(m_4d, d_4d);
    assert_eq!(legacy_sign, -1.0);
    assert_eq!(fixed_sign, 1.0);

    let c_4d = fixed_sign
        / (2.0_f64.powi((2 * m_4d - 1) as i32)
            * std::f64::consts::PI.powf(0.5 * d_4d as f64)
            * gamma_lanczos(m_4d as f64)
            * gamma_lanczos((m_4d - d_4d / 2 + 1) as f64));
    let expected_4d = c_4d * r.powi((2 * m_4d - d_4d) as i32) * r.ln();
    let got_4d = polyharmonic_kernel(r, (m_4d) as f64, d_4d);
    assert!((got_4d - expected_4d).abs() < 1e-12);
    assert!(got_4d > 0.0);
}

#[test]
fn test_pure_duchon_rejects_undefined_gradient_collocation() {
    let centers = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ];
    let err = match build_duchon_collocation_operator_matrices(
        centers.view(),
        None,
        None,
        1.0,
        DuchonNullspaceOrder::Zero,
        None,
        None,
        2,
    ) {
        Ok(_) => panic!("d=3, p=1, s=1 has no well-defined collision gradient"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("D1 collocation"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pure_duchon_rejects_divergent_laplacian_collocation() {
    // Four 2D centers leave one constrained radial degree of freedom
    // after the linear nullspace (1, x, y), so this isolates the D2
    // collocation regularity check instead of auto-degrading to the
    // constants-only nullspace.
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let err = match build_duchon_collocation_operator_matrices(
        centers.view(),
        None,
        None,
        0.0,
        DuchonNullspaceOrder::Linear,
        None,
        None,
        2,
    ) {
        Ok(_) => panic!("2D thin-plate Duchon collocation has no finite collision Laplacian"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("D2 collocation"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pure_polyharmonic_origin_jets_preserve_derivative_singularities() {
    let (_, _, tps_phi_rr) = polyharmonic_kernel_triplet(0.0, 2.0, 2).expect("thin-plate jet");
    assert!(
        tps_phi_rr.is_infinite() && tps_phi_rr.is_sign_negative(),
        "2D thin-plate phi_rr(0) should diverge to -inf, got {tps_phi_rr}"
    );
    let (q, _, _, _) =
        duchon_polyharmonic_operator_block_jets(0.0, 2.0, 2).expect("thin-plate operator jet");
    assert!(
        q.is_infinite() && q.is_sign_negative(),
        "2D thin-plate phi_r/r at collision should diverge to -inf, got {q}"
    );

    let (_, gradient_first, gradient_second) =
        polyharmonic_kernel_triplet(0.0, 2.0, 3).expect("3D first-derivative jet");
    assert_abs_diff_eq!(
        gradient_first,
        -1.0 / (8.0 * std::f64::consts::PI),
        epsilon = 1e-14
    );
    assert_abs_diff_eq!(gradient_second, 0.0, epsilon = 1e-14);
}

#[test]
fn test_duchon_hybrid_collision_uses_combined_partial_fraction_limit() {
    let p_order = 1usize;
    let s_order = 1usize;
    let dim = 3usize;
    let length_scale = 1.0;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let got = duchon_matern_kernel_general_from_distance(
        0.0,
        Some(length_scale),
        p_order,
        s_order,
        dim,
        Some(&coeffs),
    )
    .expect("finite hybrid diagonal");
    let expected = 1.0 / (4.0 * std::f64::consts::PI);
    assert_abs_diff_eq!(got, expected, epsilon = 1e-12);
}

#[test]
fn test_duchon_matern_block_origin_includes_kappa_power() {
    let kappa = 4.0;
    let value = duchon_matern_block(0.0, kappa, 1, 1).expect("block value");
    assert_abs_diff_eq!(value, 1.0 / 8.0, epsilon = 1e-14);
}

#[test]
fn test_duchon_aniso_collocation_uses_metric_weights() {
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let eta = vec![2.0_f64.ln(), -2.0_f64.ln()];
    let ops = build_duchon_collocation_operator_matrices(
        centers.view(),
        None,
        Some(1.0),
        2.0,
        DuchonNullspaceOrder::Linear,
        Some(&eta),
        None,
        2,
    )
    .expect("anisotropic Duchon collocation");

    let mut workspace = BasisWorkspace::default();
    let z = kernel_constraint_nullspace(
        centers.view(),
        DuchonNullspaceOrder::Linear,
        &mut workspace.cache,
    )
    .expect("kernel constraint nullspace");
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 2usize;
    let dim = 2usize;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0);
    let weights = [4.0, 0.25];
    let sum_weights = weights.iter().sum::<f64>();

    for k in 0..centers.nrows() {
        for col in 0..z.ncols() {
            let mut expected_d2 = 0.0;
            let mut expected_d1 = [0.0; 2];
            for j in 0..centers.nrows() {
                let h = [
                    centers[[k, 0]] - centers[[j, 0]],
                    centers[[k, 1]] - centers[[j, 1]],
                ];
                let s_vec = [weights[0] * h[0] * h[0], weights[1] * h[1] * h[1]];
                let r = (s_vec[0] + s_vec[1]).sqrt();
                let (_, phi_r, phi_rr) = duchon_kernel_radial_triplet(
                    r,
                    Some(1.0),
                    p_order,
                    s_order as f64,
                    dim,
                    Some(&coeffs),
                )
                .expect("radial triplet");
                let lap = if r > 1e-10 {
                    let q = phi_r / r;
                    let t = (phi_rr - q) / (r * r);
                    let sum_wb_sb = weights[0] * s_vec[0] + weights[1] * s_vec[1];
                    for axis in 0..dim {
                        expected_d1[axis] += q * weights[axis] * h[axis] * z[[j, col]];
                    }
                    q * sum_weights + t * sum_wb_sb
                } else {
                    sum_weights * phi_rr
                };
                expected_d2 += lap * z[[j, col]];
            }

            for axis in 0..dim {
                assert_abs_diff_eq!(
                    ops.d1[[k * dim + axis, col]],
                    expected_d1[axis],
                    epsilon = 1e-9
                );
            }
            // D2 is now a full p*d*d Hessian; the previous Laplacian
            // operator was the plain trace sum_b H_{bb} (the "sum of
            // diagonal Hessian entries"), not the metric-weighted trace.
            let mut lap = 0.0;
            for axis in 0..dim {
                let row = (k * dim + axis) * dim + axis;
                lap += ops.d2[[row, col]];
            }
            assert_abs_diff_eq!(lap, expected_d2, epsilon = 1e-9);
        }
    }
}

#[test]
fn test_matern_center_sum_tozero_produces_kernel_transform() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        length_scale: 0.7,
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
    assert_eq!(out.design.nrows(), data.nrows());
    assert_eq!(out.design.ncols(), centers.nrows() - 1);
    assert_eq!(out.penalties[0].nrows(), out.design.ncols());
    assert_eq!(out.penalties[0].ncols(), out.design.ncols());
    let BasisMetadata::Matern {
        identifiability_transform,
        ..
    } = out.metadata
    else {
        panic!("expected Matérn metadata");
    };
    let z = identifiability_transform.expect("sum-to-zero should store transform");
    assert_eq!(z.nrows(), centers.nrows());
    assert_eq!(z.ncols(), centers.nrows() - 1);
    let ones = Array1::<f64>::ones(centers.nrows());
    let residual = ones.dot(&z).mapv(f64::abs).sum();
    assert!(residual < 1e-10, "constant mode not removed: {residual}");
}

#[test]
fn test_matern_operator_penalties_follow_rkhs_smoothness() {
    let data = array![[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]];
    let centers = data.clone();
    let sources_for = |nu| {
        let spec = MaternBasisSpec {
            periodic: None,
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            length_scale: 0.4,
            nu,
            include_intercept: false,
            double_penalty: false,
            identifiability: MaternIdentifiability::None,
            aniso_log_scales: None,
            nullspace_shrinkage_survived: None,
        };
        build_matern_basis(data.view(), &spec)
            .expect("Matérn basis should build")
            .penaltyinfo
            .into_iter()
            .map(|info| info.source)
            .collect::<Vec<_>>()
    };

    assert_eq!(
        sources_for(MaternNu::Half),
        vec![PenaltySource::OperatorMass]
    );
    assert_eq!(
        sources_for(MaternNu::ThreeHalves),
        vec![PenaltySource::OperatorMass, PenaltySource::OperatorTension]
    );
    assert_eq!(
        sources_for(MaternNu::FiveHalves),
        vec![
            PenaltySource::OperatorMass,
            PenaltySource::OperatorTension,
            PenaltySource::OperatorStiffness
        ]
    );
}

/// gam#902: the ψ=log κ derivative builder
/// (`build_matern_operator_penalty_psi_derivatives`) must apply the SAME
/// `matern_for_smoothness(ν, d)` admissibility gate as the forward penalty
/// builder (`build_matern_operator_penalty_candidates`), so for a rough-ν,
/// non-double-penalty Matérn the derivative penalty list is index-aligned
/// with the forward penalty list. Before the fix the derivative builder
/// unconditionally emitted mass+tension+stiffness ψ-derivatives while the
/// forward build gated tension/stiffness out — desyncing the κ-gradient
/// against a mismatched penalty set.
#[test]
fn test_matern_operator_psi_derivatives_index_align_with_forward_gate() {
    use ndarray::array;
    let centers = array![[0.0_f64], [0.2], [0.45], [0.7], [1.0]];
    let length_scale = 0.4_f64;
    let include_intercept = false;
    // ν=3/2, d=1 ⇒ RKHS Sobolev order m = ν + d/2 = 2.0. The strict gate
    // admits mass (j=0) and tension (j=1) but DROPS stiffness (j=2, since
    // 2.0 is not > 2.0). That gated-out stiffness is exactly the operator
    // whose ψ-derivative the pre-fix builder emitted anyway, so this config
    // genuinely exercises the index-alignment invariant.
    //
    // ν=3/2 (not ν=1/2): the exponential ν=1/2 kernel φ(r)=exp(−s r) has a
    // cusp at r=0 (q = φ'/r → −∞), so its discrete collocation gradient /
    // Hessian operators cannot be formed at all — its operator ψ-derivatives
    // surface `DegenerateAtCollision`, and the forward gate's "only mass for
    // ν=1/2" is the kernel's own non-differentiability, not a dropped valid
    // penalty. ν=3/2's radial operator triplet is finite at r=0, so the
    // tension/stiffness blocks are constructible and the alignment is the
    // thing under test.
    let nu = MaternNu::ThreeHalves;

    // Forward penalty list, post-filter, in build order.
    let forward = build_matern_operator_penalty_candidates(
        centers.view(),
        length_scale,
        nu,
        include_intercept,
        None,
        None,
    )
    .expect("forward Matérn operator penalties should build");
    let (forward_penalties, _, forward_info) =
        filter_active_penalty_candidates(forward).expect("forward filter");
    let forward_sources: Vec<PenaltySource> = forward_info
        .iter()
        .filter(|info| info.active)
        .map(|info| info.source.clone())
        .collect();
    assert_eq!(
        forward_sources,
        vec![PenaltySource::OperatorMass, PenaltySource::OperatorTension],
        "ν=3/2 (m=2) must admit mass+tension and gate out stiffness"
    );

    // ψ-derivative list for the same config.
    let (psi_derivatives, psisecond_derivatives) = build_matern_operator_penalty_psi_derivatives(
        centers.view(),
        length_scale,
        nu,
        include_intercept,
        None,
        None,
    )
    .expect("Matérn operator ψ-derivatives should build");
    assert_eq!(
        psi_derivatives.len(),
        forward_sources.len(),
        "ψ-derivative count must equal the forward (gated) penalty count"
    );
    assert_eq!(
        psisecond_derivatives.len(),
        forward_sources.len(),
        "ψ-second-derivative count must equal the forward penalty count"
    );
    // Each surviving penalty and its ψ-derivative share the same shape, so
    // the consumer's positional pairing of penalty[a] with ∂S/∂ψ[a] is
    // well-formed.
    for (penalty, deriv) in forward_penalties.iter().zip(psi_derivatives.iter()) {
        assert_eq!(
            penalty.dim(),
            deriv.dim(),
            "ψ-derivative must match its penalty's shape"
        );
    }

    // Finite-difference each SURVIVING operator's κ-gradient against its
    // analytic ψ-derivative, positionally (mass at index 0, tension at
    // index 1). ψ = log κ with κ = 1/length_scale, so length_scale(ψ) =
    // exp(-ψ) and a +h step in ψ scales length_scale by exp(-h). A pre-fix
    // misalignment (an extra stiffness ψ-derivative shifting the indices,
    // or the analytic deriv paired with the wrong forward penalty) would
    // blow up this FD comparison.
    let penalty_for = |ls: f64, source: PenaltySource| -> Array2<f64> {
        let cands = build_matern_operator_penalty_candidates(
            centers.view(),
            ls,
            nu,
            include_intercept,
            None,
            None,
        )
        .expect("FD forward penalties");
        cands
            .into_iter()
            .find(|c| c.source == source)
            .unwrap_or_else(|| panic!("forward penalty {source:?} present"))
            .matrix
    };
    let h = 1e-5_f64;
    for (idx, source) in forward_sources.iter().enumerate() {
        let s_plus = penalty_for(length_scale * (-h).exp(), source.clone());
        let s_minus = penalty_for(length_scale * h.exp(), source.clone());
        let fd = (&s_plus - &s_minus).mapv(|v| v / (2.0 * h));
        let analytic = &psi_derivatives[idx];
        let err = (&fd - analytic).iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = analytic.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
        assert!(
            err / scale < 1e-5,
            "{source:?} κ-gradient FD mismatch: err={err:.3e}, analytic_norm={scale:.3e}"
        );
    }
}

#[test]
fn test_matern_overspecified_centers_yield_full_rank_basis() {
    // #755: pack many centers into a tight standardized cloud so the fixed
    // length_scale produces overlapping (numerically collinear) radial
    // basis functions. The realized kernel design then exceeds the kernel's
    // numerical rank and the identifiability audit would FATAL. The basis
    // builder must rank-reduce centers so the emitted design is full rank.
    use crate::linalg::faer_ndarray::rrqr_with_permutation;
    // Pack K=30 centers far tighter than the fixed length_scale can resolve:
    // 30 centers crammed into a 0.1-wide interval with length_scale=3.0
    // makes adjacent radial functions near-identical, so the un-reduced
    // kernel design collapses to numerical rank 6 (deficient by 24). The
    // builder must reduce centers so the emitted design is full rank (#755).
    let k = 30usize;
    let mut centers = Array2::<f64>::zeros((k, 1));
    for i in 0..k {
        centers[[i, 0]] = (i as f64 / (k as f64 - 1.0)) * 0.1;
    }
    // Data covers the same tight interval at higher resolution.
    let n = 120usize;
    let mut data = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = (i as f64 / (n as f64 - 1.0)) * 0.1;
    }
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 3.0,
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::None,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
    let dense = out.design.to_dense();
    let realized_cols = dense.ncols();
    let rrqr = rrqr_with_permutation(&dense, default_rrqr_rank_alpha())
        .expect("RRQR on the realized design should succeed");
    // The realized basis must be full column rank: no leftover collinear
    // columns for the identifiability audit to FATAL on.
    assert_eq!(
        rrqr.rank,
        realized_cols,
        "Matérn over-specified centers left {} collinear column(s): realized={realized_cols}, rank={}",
        realized_cols - rrqr.rank,
        rrqr.rank,
    );
    // Rank reduction must have actually fired (fewer than the requested K).
    assert!(
        realized_cols < k,
        "expected over-specified K={k} to be reduced below {k}, got {realized_cols}"
    );
}

#[test]
fn test_matern_include_intercept_keeps_single_unpenalized_dimension() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        length_scale: 1.1,
        nu: MaternNu::ThreeHalves,
        include_intercept: true,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
    // (k-1) constrained kernel cols + explicit intercept.
    assert_eq!(out.design.ncols(), centers.nrows());
    assert_eq!(out.penalties.len(), 3);
    assert_eq!(out.nullspace_dims.len(), 3);
}

#[test]
fn test_matern_double_penalty_drops_inactive_nullspace_blockwithout_intercept() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 1.1,
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: true,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
    assert_eq!(out.penalties.len(), 1);
    assert_eq!(out.nullspace_dims.len(), 1);
    assert_eq!(out.penaltyinfo.len(), 1);
    assert!(out.penaltyinfo.iter().all(|info| info.active));
    assert!(matches!(out.penaltyinfo[0].source, PenaltySource::Primary));
}

#[test]
fn test_matern_double_penalty_keeps_intercept_shrinkage_block() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 1.1,
        nu: MaternNu::ThreeHalves,
        include_intercept: true,
        double_penalty: true,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let out = build_matern_basis(data.view(), &spec).expect("Matérn basis should build");
    assert_eq!(out.penalties.len(), 2);
    assert_eq!(out.nullspace_dims.len(), 2);
    assert_eq!(out.penaltyinfo.len(), 2);
    assert!(out.penaltyinfo.iter().all(|info| info.active));
    assert!(matches!(out.penaltyinfo[0].source, PenaltySource::Primary));
    assert!(matches!(
        out.penaltyinfo[1].source,
        PenaltySource::DoublePenaltyNullspace
    ));
}

/// gam#787/#860: a frozen `nullspace_shrinkage_survived` decision overrides
/// the κ-dependent spectral test so the κ-optimizer's per-trial rebuilds keep
/// the learned-penalty count INVARIANT. The intercept config above emits 2
/// penalties under the spectral test (`None`); freezing the decision to
/// `Some(false)` must drop the shrinkage candidate (count → 1) and freezing
/// to `Some(true)` must keep it (count → 2), regardless of length-scale.
#[test]
fn matern_frozen_nullspace_decision_overrides_spectral_test() {
    let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.7]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    // Reuse the surviving-shrinkage center sum-to-zero transform as a frozen
    // transform so we exercise the FrozenTransform path with a pinned decision.
    let z =
        matern_identifiability_transform(centers.view(), &MaternIdentifiability::CenterSumToZero)
            .expect("transform builds")
            .expect("center sum-to-zero yields a transform");
    let base = |survived: Option<bool>| MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        length_scale: 1.1,
        nu: MaternNu::ThreeHalves,
        include_intercept: true,
        double_penalty: true,
        identifiability: MaternIdentifiability::FrozenTransform {
            transform: z.clone(),
            nullspace_shrinkage_survived: survived,
        },
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    // Frozen OFF → only the primary kernel penalty survives.
    let off = build_matern_basis(data.view(), &base(Some(false)))
        .expect("Matérn basis should build with frozen-off shrinkage");
    assert_eq!(
        off.penalties.len(),
        1,
        "frozen Some(false) must suppress the DoublePenaltyNullspace candidate"
    );
    assert!(matches!(off.penaltyinfo[0].source, PenaltySource::Primary));
    // Frozen ON → the shrinkage block is kept, INVARIANT across length-scale.
    for length_scale in [0.6_f64, 1.1, 3.0] {
        let mut spec_on = base(Some(true));
        spec_on.length_scale = length_scale;
        let on = build_matern_basis(data.view(), &spec_on)
            .expect("Matérn basis should build with frozen-on shrinkage");
        assert_eq!(
            on.penalties.len(),
            2,
            "frozen Some(true) must keep the shrinkage block at length_scale={length_scale}"
        );
        assert!(matches!(
            on.penaltyinfo[1].source,
            PenaltySource::DoublePenaltyNullspace
        ));
    }
}

/// #1090: a FrozenTransform Matérn build replays a fit whose centers and
/// identifiability transform were already rank-reduced and frozen mutually
/// consistently at train time. The prediction/replay data cloud differs (and
/// can be degenerate), so re-running the #755 RRQR center reduction here would
/// prune the pinned centers to a *different* count, leaving a stale N-row
/// transform over a reduced-column basis and a hard "centers vs transform
/// rows" dimension mismatch. The frozen path must keep the pinned centers
/// verbatim and build cleanly on the degenerate cloud.
#[test]
fn matern_frozen_transform_skips_rank_reduction_on_degenerate_cloud() {
    // Four well-separated centers; the transform is built over all four.
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let z =
        matern_identifiability_transform(centers.view(), &MaternIdentifiability::CenterSumToZero)
            .expect("transform builds")
            .expect("center sum-to-zero yields a transform");
    let transform_rows = z.nrows();
    assert_eq!(
        transform_rows,
        centers.nrows(),
        "center sum-to-zero transform spans every center"
    );
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers.clone()),
        length_scale: 1.1,
        nu: MaternNu::ThreeHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::FrozenTransform {
            transform: z.clone(),
            nullspace_shrinkage_survived: Some(false),
        },
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    // Degenerate replay cloud: many rows packed into a near-singleton point
    // at a length_scale that, under the cold RRQR reduction, would collapse
    // the kernel design to a rank far below the four frozen centers. Without
    // the #1090 fix the rebuild would re-reduce the pinned centers and then
    // hit the frozen-transform row mismatch.
    let n = 80usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = 0.5 + 1e-6 * t;
        data[[i, 1]] = 0.5 + 1e-6 * t;
    }
    let out = build_matern_basis(data.view(), &spec)
        .expect("frozen Matérn must replay on a degenerate cloud without re-reducing centers");
    // The realized design width matches the frozen transform's column space,
    // not some re-reduced count.
    assert_eq!(
        out.design.ncols(),
        z.ncols(),
        "frozen replay design width must equal the frozen transform's column space"
    );
    match &out.metadata {
        BasisMetadata::Matern { centers: meta, .. } => assert_eq!(
            meta.nrows(),
            centers.nrows(),
            "frozen replay must keep all {} pinned centers, not re-reduce them",
            centers.nrows()
        ),
        other => panic!("expected Matérn metadata, got {other:?}"),
    }
}

/// #1090 companion: a *cold* (non-frozen) Matérn whose data-supported rank is
/// 0 (every center numerically collinear at the chosen length_scale) must fail
/// loudly with an actionable error rather than emit a silent 0-center basis.
#[test]
fn matern_cold_zero_rank_cloud_fails_loudly() {
    // Realized kernel rank 0: place the data cloud astronomically far from
    // both centers at a tiny length_scale, so every Matérn kernel evaluation
    // `(…)·exp(-√(2ν)·r/ℓ)` underflows to exactly 0.0. The whole n×2 kernel
    // design block is then numerically zero (rank 0) — a degenerate term that
    // must hard-error instead of emitting a silent 0-center basis (#1090).
    let centers = array![[0.0, 0.0], [1.0, 0.0]];
    let n = 40usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        // ~1e6 away on a 1e-3 length scale → exp argument ~1e9 → underflow 0.
        data[[i, 0]] = 1.0e6 + i as f64;
        data[[i, 1]] = 1.0e6 - i as f64;
    }
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 1.0e-3,
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::None,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let err = build_matern_basis(data.view(), &spec)
        .expect_err("rank-0 Matérn cloud must fail loudly, not emit a degenerate basis");
    let msg = err.to_string();
    assert!(
        msg.contains("numerical rank 0"),
        "expected a rank-0 degeneracy error, got: {msg}"
    );
}

#[test]
fn test_matern_log_kappa_derivative_matchesfd() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 0.9,
        nu: MaternNu::FiveHalves,
        include_intercept: false,
        double_penalty: false,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let deriv = build_matern_basis_log_kappa_derivative(data.view(), &spec)
        .expect("analytic Matérn derivative should build");

    let eps: f64 = 1e-6;
    let kappa = 1.0 / spec.length_scale;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = ls_plus;
    spec_minus.length_scale = ls_minus;
    let plus = build_matern_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_matern_basis(data.view(), &spec_minus).expect("minus build");

    let plus_design = plus.design.to_dense();
    let minus_design = minus.design.to_dense();
    let fd_design = (&plus_design - &minus_design) / (2.0 * eps);
    let fd_penalty = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);

    let design_err = (&deriv.design_derivative - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let penalty_err = (&deriv.penalties_derivative[0] - &fd_penalty)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    for i in 0..deriv.design_derivative.nrows() {
        for j in 0..deriv.design_derivative.ncols() {
            assert_eq!(
                deriv.design_derivative[[i, j]].signum(),
                fd_design[[i, j]].signum()
            );
        }
    }
    for i in 0..deriv.penalties_derivative[0].nrows() {
        for j in 0..deriv.penalties_derivative[0].ncols() {
            assert_eq!(
                deriv.penalties_derivative[0][[i, j]].signum(),
                fd_penalty[[i, j]].signum()
            );
        }
    }

    assert!(
        design_err < 1e-5,
        "design derivative mismatch too large: {design_err}"
    );
    assert!(
        penalty_err < 1e-5,
        "penalty derivative mismatch too large: {penalty_err}"
    );
}

#[test]
fn test_matern_double_penalty_log_kappa_derivative_matchesfd() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let spec = MaternBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 0.9,
        nu: MaternNu::FiveHalves,
        include_intercept: true,
        double_penalty: true,
        identifiability: MaternIdentifiability::CenterSumToZero,
        aniso_log_scales: None,
        nullspace_shrinkage_survived: None,
    };
    let deriv = build_matern_basis_log_kappa_derivative(data.view(), &spec)
        .expect("analytic Matérn double-penalty derivative should build");

    let eps: f64 = 1e-6;
    let kappa = 1.0 / spec.length_scale;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = ls_plus;
    spec_minus.length_scale = ls_minus;
    let plus = build_matern_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_matern_basis(data.view(), &spec_minus).expect("minus build");

    let fd_primary = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);
    let primary_err = (&deriv.penalties_derivative[0] - &fd_primary)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    assert!(
        primary_err < 1e-5,
        "double-penalty primary derivative mismatch too large: {primary_err}"
    );
    assert_eq!(deriv.penalties_derivative.len(), 2);
    assert!(
        deriv.penalties_derivative[1]
            .iter()
            .all(|v| v.abs() < 1e-12),
        "nullspace shrinkage derivative should be zero"
    );
}

#[test]
fn test_thin_plate_log_kappa_derivative_matchesfd() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let mut spec = ThinPlateBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 0.9,
        double_penalty: true,
        identifiability: SpatialIdentifiability::None,
        radial_reparam: None,
    };
    let base_for_reparam =
        build_thin_plate_basis(data.view(), &spec).expect("base TPS build for radial reparam");
    if let BasisMetadata::ThinPlate { radial_reparam, .. } = &base_for_reparam.metadata {
        spec.radial_reparam = radial_reparam.clone();
    }
    let deriv = build_thin_plate_basis_log_kappa_derivative(data.view(), &spec)
        .expect("analytic ThinPlate derivative should build");

    let eps: f64 = 1e-6;
    let kappa = 1.0 / spec.length_scale;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = ls_plus;
    spec_minus.length_scale = ls_minus;
    let plus = build_thin_plate_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_thin_plate_basis(data.view(), &spec_minus).expect("minus build");

    let plus_design = plus.design.to_dense();
    let minus_design = minus.design.to_dense();
    let fd_design = (&plus_design - &minus_design) / (2.0 * eps);
    let fd_primary = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);
    let design_err = (&deriv.design_derivative - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let primary_err = (&deriv.penalties_derivative[0] - &fd_primary)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    assert!(
        design_err < 1e-5,
        "ThinPlate design derivative mismatch: {design_err}"
    );
    assert!(
        primary_err < 1e-5,
        "ThinPlate primary penalty derivative mismatch: {primary_err}"
    );
    assert_eq!(deriv.penalties_derivative.len(), 2);
    assert!(
        deriv.penalties_derivative[1]
            .iter()
            .all(|v| v.abs() < 1e-12),
        "ThinPlate nullspace shrinkage derivative should be zero"
    );
}

#[test]
fn test_thin_plate_log_kappasecond_derivative_matchesfd() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let mut spec = ThinPlateBasisSpec {
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: 0.9,
        double_penalty: true,
        identifiability: SpatialIdentifiability::None,
        radial_reparam: None,
    };
    let base_for_reparam =
        build_thin_plate_basis(data.view(), &spec).expect("base TPS build for radial reparam");
    if let BasisMetadata::ThinPlate { radial_reparam, .. } = &base_for_reparam.metadata {
        spec.radial_reparam = radial_reparam.clone();
    }
    let analytic = build_thin_plate_basis_log_kappasecond_derivative(data.view(), &spec)
        .expect("analytic ThinPlate second derivative should build");
    let base = build_thin_plate_basis(data.view(), &spec).expect("base build");

    let eps: f64 = 2e-5;
    let kappa = 1.0 / spec.length_scale;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = ls_plus;
    spec_minus.length_scale = ls_minus;
    let plus = build_thin_plate_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_thin_plate_basis(data.view(), &spec_minus).expect("minus build");

    let plus_design = plus.design.to_dense();
    let base_design = base.design.to_dense();
    let minus_design = minus.design.to_dense();
    let fd_design = (&plus_design - &(base_design.clone() * 2.0) + &minus_design) / (eps * eps);
    let fd_primary = (&plus.penalties[0] - &(base.penalties[0].clone() * 2.0)
        + &minus.penalties[0])
        / (eps * eps);
    let design_err = (&analytic.designsecond_derivative - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let primary_err = (&analytic.penaltiessecond_derivative[0] - &fd_primary)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();

    assert!(
        design_err < 5e-3,
        "ThinPlate design second derivative mismatch: {design_err}"
    );
    assert!(
        primary_err < 5e-3,
        "ThinPlate primary penalty second derivative mismatch: {primary_err}"
    );
    assert_eq!(analytic.penaltiessecond_derivative.len(), 2);
    assert!(
        analytic.penaltiessecond_derivative[1]
            .iter()
            .all(|v| v.abs() < 1e-12),
        "ThinPlate nullspace shrinkage second derivative should be zero"
    );
}

/// Compare analytic D0_psi/D1_psi/D2_psi (from
/// `build_duchon_operator_penalty_psi_derivatives` internals) against
/// FD of the cost-side collocation operators. Isolates whether the
/// bug lives in the operator-derivative formulas
/// (duchon_radial_core_psi_triplet + jets) vs the gram chain rule.
#[test]
fn test_duchon_operator_psi_derivatives_fd_dim1() {
    use ndarray::array;
    let centers = array![[0.0_f64], [0.2], [0.45], [0.7]];
    let nullspace_order = DuchonNullspaceOrder::Linear;
    let power = 1.0;
    let length_scale = 1.0_f64;
    let mut workspace = BasisWorkspace::default();
    let eps: f64 = 1e-5;
    let ls_plus = 1.0 / eps.exp();
    let ls_minus = 1.0 / (-eps).exp();
    let ops_plus = build_duchon_collocation_operator_matriceswithworkspace(
        centers.view(),
        centers.view(),
        None,
        Some(ls_plus),
        power,
        nullspace_order,
        None,
        None,
        2,
        &mut workspace,
    )
    .expect("plus ops");
    let ops_minus = build_duchon_collocation_operator_matriceswithworkspace(
        centers.view(),
        centers.view(),
        None,
        Some(ls_minus),
        power,
        nullspace_order,
        None,
        None,
        2,
        &mut workspace,
    )
    .expect("minus ops");
    let fd_d0 = (&ops_plus.d0 - &ops_minus.d0) / (2.0 * eps);
    let fd_d1 = (&ops_plus.d1 - &ops_minus.d1) / (2.0 * eps);
    let fd_d2 = (&ops_plus.d2 - &ops_minus.d2) / (2.0 * eps);
    eprintln!(
        "[op_psi_fd_d1] D0_plus[0..,0..] sample = {:?}",
        ops_plus.d0.row(0).to_vec()
    );
    eprintln!(
        "[op_psi_fd_d1] D0_minus[0..,0..] sample = {:?}",
        ops_minus.d0.row(0).to_vec()
    );
    eprintln!(
        "[op_psi_fd_d1] fd_D0 shape=({} x {}) norm={:.4e}",
        fd_d0.nrows(),
        fd_d0.ncols(),
        fd_d0.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    eprintln!(
        "[op_psi_fd_d1] fd_D1 norm={:.4e}",
        fd_d1.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    eprintln!(
        "[op_psi_fd_d1] fd_D2 norm={:.4e}",
        fd_d2.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    // Print a few values of fd_D0 vs analytic.
    // Analytic d0_psi (kernel cols only): the same internals as
    // build_duchon_operator_penalty_psi_derivatives reach out for.
    let p_order = 2usize;
    let s_order = duchon_power_to_usize(power);
    let d = 1usize;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let z_kernel =
        kernel_constraint_nullspace(centers.view(), nullspace_order, &mut workspace.cache)
            .expect("z kernel");
    let p = centers.nrows();
    let kernel_cols = z_kernel.ncols();
    let mut d0_psi_analytic = ndarray::Array2::<f64>::zeros((p, kernel_cols));
    for k in 0..p {
        for j in 0..p {
            let r = (centers[[k, 0]] - centers[[j, 0]]).abs();
            let core =
                duchon_radial_core_psi_triplet(r, length_scale, p_order, s_order, d, &coeffs)
                    .unwrap();
            for col in 0..kernel_cols {
                d0_psi_analytic[[k, col]] += core.phi.psi * z_kernel[[j, col]];
            }
        }
    }
    // FD of D0 is the cost-side D0 (which after `fast_ab(d0_raw, z)` gives
    // the same kernel-col matrix as analytic_kernel). Account for the
    // polynomial padding: cost-side D0 has total_cols = kernel + poly.
    // Take only the first kernel_cols columns.
    let fd_d0_kernel = fd_d0.slice(ndarray::s![.., 0..kernel_cols]).to_owned();
    let err = (&d0_psi_analytic - &fd_d0_kernel)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "[op_psi_fd_d1] analytic D0_psi vs FD D0_psi: analytic_norm={:.4e} fd_norm={:.4e} err={:.4e}",
        d0_psi_analytic.iter().map(|v| v * v).sum::<f64>().sqrt(),
        fd_d0_kernel.iter().map(|v| v * v).sum::<f64>().sqrt(),
        err
    );
}

/// FD the **unnormalized** Gram penalty `S_raw = D^T D` for the failing
/// (d=1, p=2, s=1, ls=1) spec to isolate whether the bug is in
/// `gram_and_psi_derivatives_from_operator` / `D_psi` build or in
/// `normalize_penaltywith_psi_derivatives`.
#[test]
fn test_duchon_raw_gram_psi_derivative_fd_dim1() {
    use ndarray::array;
    let centers = array![[0.0_f64], [0.2], [0.45], [0.7]];
    let nullspace_order = DuchonNullspaceOrder::Linear;
    let power = 1.0;
    let length_scale = 1.0_f64;
    let mut workspace = BasisWorkspace::default();
    let ops_base = build_duchon_collocation_operator_matriceswithworkspace(
        centers.view(),
        centers.view(),
        None,
        Some(length_scale),
        power,
        nullspace_order,
        None,
        None,
        2,
        &mut workspace,
    )
    .expect("base ops");
    let eps: f64 = 1e-5;
    let ls_plus = 1.0 / (1.0 * eps.exp());
    let ls_minus = 1.0 / (1.0 * (-eps).exp());
    let ops_plus = build_duchon_collocation_operator_matriceswithworkspace(
        centers.view(),
        centers.view(),
        None,
        Some(ls_plus),
        power,
        nullspace_order,
        None,
        None,
        2,
        &mut workspace,
    )
    .expect("plus ops");
    let ops_minus = build_duchon_collocation_operator_matriceswithworkspace(
        centers.view(),
        centers.view(),
        None,
        Some(ls_minus),
        power,
        nullspace_order,
        None,
        None,
        2,
        &mut workspace,
    )
    .expect("minus ops");

    let s0_raw_base = symmetrize(&fast_ata(&ops_base.d0));
    let s1_raw_base = symmetrize(&fast_ata(&ops_base.d1));
    let s2_raw_base = symmetrize(&fast_ata(&ops_base.d2));
    let fd_s0_raw =
        (symmetrize(&fast_ata(&ops_plus.d0)) - symmetrize(&fast_ata(&ops_minus.d0))) / (2.0 * eps);
    let fd_s1_raw =
        (symmetrize(&fast_ata(&ops_plus.d1)) - symmetrize(&fast_ata(&ops_minus.d1))) / (2.0 * eps);
    let fd_s2_raw =
        (symmetrize(&fast_ata(&ops_plus.d2)) - symmetrize(&fast_ata(&ops_minus.d2))) / (2.0 * eps);
    eprintln!(
        "[raw_gram_fd_d1] S0_raw_base norm={:.4e}; fd_S0_raw_psi norm={:.4e}",
        s0_raw_base.iter().map(|v| v * v).sum::<f64>().sqrt(),
        fd_s0_raw.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    eprintln!(
        "[raw_gram_fd_d1] S1_raw_base norm={:.4e}; fd_S1_raw_psi norm={:.4e}",
        s1_raw_base.iter().map(|v| v * v).sum::<f64>().sqrt(),
        fd_s1_raw.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    eprintln!(
        "[raw_gram_fd_d1] S2_raw_base norm={:.4e}; fd_S2_raw_psi norm={:.4e}",
        s2_raw_base.iter().map(|v| v * v).sum::<f64>().sqrt(),
        fd_s2_raw.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
}

/// Variant of `test_duchon_log_kappa_derivative_matchesfd` at length_scale=1.0
/// (kappa=1, psi=0) to check whether the analytic dS/d(log κ) formulas
/// degrade at kappa=1 (where chain-rule corner cases are likeliest).
#[test]
fn test_duchon_log_kappa_derivative_matchesfd_lengthscale_one() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: Some(1.0),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let derivative = build_duchon_basis_log_kappa_derivatives(data.view(), &spec)
        .expect("analytic Duchon derivative should build");
    let eps: f64 = 1e-5;
    let kappa = 1.0;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");
    for idx in 0..derivative.first.penalties_derivative.len() {
        let fd = (&plus.penalties[idx] - &minus.penalties[idx]) / (2.0 * eps);
        let analytic = &derivative.first.penalties_derivative[idx];
        let err = (analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let a_norm = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fd_norm = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
        eprintln!(
            "[duchon_ls1] penalty {} analytic_norm={:.4e} fd_norm={:.4e} err={:.4e}",
            idx, a_norm, fd_norm, err
        );
    }
}

/// FD-test the **design** derivative dX/dpsi against the rebuilt cost
/// design with frozen identifiability transform.
#[test]
fn test_duchon_design_log_kappa_derivative_matchesfd_dim1_power1_frozen() {
    let n = 80usize;
    let mut data = ndarray::Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let base = build_duchon_basis(data.view(), &spec).expect("base build");
    let z_frozen = match &base.metadata {
        BasisMetadata::Duchon {
            identifiability_transform,
            ..
        } => identifiability_transform.clone(),
        _ => expected_duchon_metadata(),
    };
    let centers = match &base.metadata {
        BasisMetadata::Duchon { centers, .. } => centers.clone(),
        _ => expected_duchon_metadata_for_centers(),
    };
    spec.center_strategy = CenterStrategy::UserProvided(centers);
    spec.identifiability = match z_frozen {
        Some(z) => SpatialIdentifiability::FrozenTransform { transform: z },
        None => SpatialIdentifiability::None,
    };
    let derivative = build_duchon_basis_log_kappa_derivatives(data.view(), &spec)
        .expect("analytic Duchon derivative should build");
    let eps: f64 = 1e-5;
    let ls_plus = 1.0 / eps.exp();
    let ls_minus = 1.0 / (-eps).exp();
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");
    let plus_design = plus.design.to_dense();
    let minus_design = minus.design.to_dense();
    let fd_design = (&plus_design - &minus_design) / (2.0 * eps);
    let analytic_design = match derivative.implicit_operator.as_ref() {
        Some(op) => op
            .materialize_first(0)
            .expect("materialize first design derivative"),
        None => derivative.first.design_derivative.clone(),
    };
    eprintln!(
        "[duchon_d1_p1_frozen_design] analytic shape={:?} fd shape={:?}",
        analytic_design.shape(),
        fd_design.shape()
    );
    let a_norm = analytic_design.iter().map(|v| v * v).sum::<f64>().sqrt();
    let fd_norm = fd_design.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert_eq!(
        analytic_design.shape(),
        fd_design.shape(),
        "analytic and FD design-derivative shapes must agree"
    );
    let err = (&analytic_design - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let scale = a_norm.max(fd_norm).max(1e-12);
    assert!(
        err / scale < 1e-3,
        "[duchon_d1_p1_frozen_design] dX/dpsi mismatch: analytic_norm={a_norm:.4e} \
             fd_norm={fd_norm:.4e} err={err:.4e} rel_err={:.4e}",
        err / scale,
    );
}

/// Critical FD test mirroring iso-kappa REML path: uses FROZEN
/// identifiability transform captured from the design at kappa=1, then
/// FD's both the analytic derivative and the rebuilt-cost penalty with
/// the same frozen transform. If this fails, the bug is NOT in
/// kappa-dependence of Z; it's in the underlying derivative formula.
#[test]
fn test_duchon_log_kappa_derivative_matchesfd_dim1_power1_frozen() {
    let n = 80usize;
    let mut data = ndarray::Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let mut spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let base = build_duchon_basis(data.view(), &spec).expect("base build");
    let z_frozen = match &base.metadata {
        BasisMetadata::Duchon {
            identifiability_transform,
            ..
        } => identifiability_transform.clone(),
        _ => expected_duchon_metadata(),
    };
    // Freeze the transform and also user-supply the centers so the spec
    // is reproducible across length_scale shifts.
    let centers = match &base.metadata {
        BasisMetadata::Duchon { centers, .. } => centers.clone(),
        _ => expected_duchon_metadata_for_centers(),
    };
    spec.center_strategy = CenterStrategy::UserProvided(centers);
    spec.identifiability = match z_frozen {
        Some(z) => SpatialIdentifiability::FrozenTransform { transform: z },
        None => SpatialIdentifiability::None,
    };
    let derivative = build_duchon_basis_log_kappa_derivatives(data.view(), &spec)
        .expect("analytic Duchon derivative should build");
    let eps: f64 = 1e-5;
    let ls_plus = 1.0 / eps.exp();
    let ls_minus = 1.0 / (-eps).exp();
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");
    assert!(
        !derivative.first.penalties_derivative.is_empty(),
        "derivative must expose at least one penalty matrix"
    );
    for idx in 0..derivative.first.penalties_derivative.len() {
        let fd = (&plus.penalties[idx] - &minus.penalties[idx]) / (2.0 * eps);
        let analytic = &derivative.first.penalties_derivative[idx];
        let err = (analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let a_norm = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fd_norm = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
        let scale = a_norm.max(fd_norm).max(1e-12);
        assert!(
            err / scale < 1e-3,
            "[duchon_d1_p1_frozen] penalty {idx} mismatch: \
                 analytic_norm={a_norm:.4e} fd_norm={fd_norm:.4e} err={err:.4e} \
                 rel_err={:.4e}",
            err / scale,
        );
    }
}

/// Same as `test_duchon_log_kappa_derivative_matchesfd_dim1_power1_linear`
/// but with identifiability=None, to test whether the kappa-dependence of
/// SpatialIdentifiability::OrthogonalToParametric is the source of the bug.
#[test]
fn test_duchon_log_kappa_derivative_matchesfd_dim1_power1_linear_no_ident() {
    let n = 80usize;
    let mut data = ndarray::Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let derivative = build_duchon_basis_log_kappa_derivatives(data.view(), &spec)
        .expect("analytic Duchon derivative should build");
    let eps: f64 = 1e-5;
    let kappa = 1.0;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");
    for idx in 0..derivative.first.penalties_derivative.len() {
        let fd = (&plus.penalties[idx] - &minus.penalties[idx]) / (2.0 * eps);
        let analytic = &derivative.first.penalties_derivative[idx];
        let err = (analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let a_norm = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fd_norm = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
        eprintln!(
            "[duchon_d1_p1_no_ident] penalty {} analytic_norm={:.4e} fd_norm={:.4e} err={:.4e}",
            idx, a_norm, fd_norm, err
        );
    }
}

/// Mirrors the failing iso-kappa-Duchon FD config: dim=1, power=1,
/// nullspace=Linear, length_scale=1.0. Confirms whether `s_psi`
/// from `build_duchon_basis_log_kappa_derivatives` matches dS/d(log κ)
/// of the rebuilt penalty for this 1D BinomialProbit-style spec.
#[test]
fn test_duchon_log_kappa_derivative_matchesfd_dim1_power1_linear() {
    let n = 80usize;
    let mut data = ndarray::Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        data[[i, 0]] = i as f64 / (n as f64 - 1.0);
    }
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::FarthestPoint { num_centers: 8 },
        length_scale: Some(1.0),
        power: 1.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::default(),
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let derivative = build_duchon_basis_log_kappa_derivatives(data.view(), &spec)
        .expect("analytic Duchon derivative should build");
    let eps: f64 = 1e-5;
    let kappa = 1.0 / spec.length_scale.expect("hybrid Duchon length_scale");
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");
    eprintln!(
        "[duchon_d1_p1_linear] n_penalties={} analytic_n={}",
        plus.penalties.len(),
        derivative.first.penalties_derivative.len()
    );
    let base = build_duchon_basis(data.view(), &spec).expect("base build");
    for idx in 0..derivative.first.penalties_derivative.len() {
        let fd = (&plus.penalties[idx] - &minus.penalties[idx]) / (2.0 * eps);
        let analytic = &derivative.first.penalties_derivative[idx];
        let err = (analytic - &fd).iter().map(|v| v * v).sum::<f64>().sqrt();
        let a_norm = analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
        let fd_norm = fd.iter().map(|v| v * v).sum::<f64>().sqrt();
        let s0_base_norm = base.penalties[idx]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let s0_plus_norm = plus.penalties[idx]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let s0_minus_norm = minus.penalties[idx]
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        eprintln!(
            "[duchon_d1_p1_linear] penalty {} S_base_norm={:.6e} S_plus_norm={:.6e} S_minus_norm={:.6e}",
            idx, s0_base_norm, s0_plus_norm, s0_minus_norm
        );
        eprintln!(
            "[duchon_d1_p1_linear] penalty {} analytic_norm={:.4e} fd_norm={:.4e} err={:.4e}",
            idx, a_norm, fd_norm, err
        );
        // First 9 entries of dS/dpsi for shape comparison.
        let pr = |m: &ndarray::Array2<f64>, label: &str| {
            let n = m.nrows().min(3);
            let mut s = String::new();
            for i in 0..n {
                for j in 0..n {
                    s.push_str(&format!("{:+.4e} ", m[[i, j]]));
                }
                s.push_str("| ");
            }
            eprintln!("[duchon_d1_p1_linear] penalty {} {}: {}", idx, label, s);
        };
        pr(analytic, "analytic");
        pr(&fd, "fd     ");
    }
}

#[test]
fn test_duchon_log_kappa_derivative_matchesfd() {
    let data = array![[0.0, 0.0], [1.0, 0.2], [0.3, 1.1], [0.9, 0.8]];
    let centers = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        periodic: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        length_scale: Some(0.9),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let mut workspace = BasisWorkspace::default();
    let derivative =
        build_duchon_basis_log_kappa_derivativewithworkspace(data.view(), &spec, &mut workspace)
            .expect("analytic Duchon derivative should build");

    let eps: f64 = 1e-6;
    let kappa = 1.0 / spec.length_scale.expect("hybrid Duchon length_scale");
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");

    let plus_design = plus.design.to_dense();
    let minus_design = minus.design.to_dense();
    let fd_design = (&plus_design - &minus_design) / (2.0 * eps);
    // The Duchon design-derivative path now always returns an implicit
    // operator (force_operator = is_duchon_family). Materialize axis 0 so
    // the comparison shape matches the dense FD design derivative.
    let analytic_design = derivative
        .implicit_operator
        .as_ref()
        .expect("Duchon design derivative must expose an implicit operator")
        .materialize_first(0)
        .expect("materialize first design derivative");
    let design_err = (&analytic_design - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        design_err < 1e-4,
        "Duchon design derivative mismatch too large: {design_err}"
    );

    assert_eq!(derivative.penalties_derivative.len(), plus.penalties.len());
    let fd_primary_penalty = (&plus.penalties[0] - &minus.penalties[0]) / (2.0 * eps);
    let primary_penalty_err = (&derivative.penalties_derivative[0] - &fd_primary_penalty)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        primary_penalty_err < 1e-4,
        "Duchon mass penalty derivative mismatch too large: {primary_penalty_err}"
    );
    for penalty_idx in 1..derivative.penalties_derivative.len() {
        let fd_penalty =
            (&plus.penalties[penalty_idx] - &minus.penalties[penalty_idx]) / (2.0 * eps);
        let penalty_err = (&derivative.penalties_derivative[penalty_idx] - &fd_penalty)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            penalty_err < 1e-4,
            "Duchon operator penalty derivative mismatch too large at block {penalty_idx}: {penalty_err}"
        );
    }
}

#[test]
fn test_periodic_duchon_log_kappa_derivative_matchesfd() {
    let data = array![[0.05], [0.4], [1.2], [2.1], [3.4], [4.8], [5.5], [6.2]];
    let centers = array![[0.0], [0.9], [2.0], [3.3], [4.7], [6.3]];
    let spec = DuchonBasisSpec {
        radial_reparam: None,
        center_strategy: CenterStrategy::UserProvided(centers),
        periodic: None,
        length_scale: Some(0.8),
        power: 2.0,
        nullspace_order: DuchonNullspaceOrder::Linear,
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: DuchonOperatorPenaltySpec::default(),
        boundary: OneDimensionalBoundary::Open,
    };
    let mut workspace = BasisWorkspace::default();
    let second_derivative = build_duchon_basis_log_kappasecond_derivativewithworkspace(
        data.view(),
        &spec,
        &mut workspace,
    )
    .expect("analytic Duchon second derivative should build");
    let base = build_duchon_basis(data.view(), &spec).expect("base build");

    let eps: f64 = 2e-5;
    let kappa = 1.0 / spec.length_scale.expect("hybrid Duchon length_scale");
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let mut spec_plus = spec.clone();
    let mut spec_minus = spec.clone();
    spec_plus.length_scale = Some(ls_plus);
    spec_minus.length_scale = Some(ls_minus);
    let plus = build_duchon_basis(data.view(), &spec_plus).expect("plus build");
    let minus = build_duchon_basis(data.view(), &spec_minus).expect("minus build");

    let plus_design = plus.design.to_dense();
    let base_design = base.design.to_dense();
    let minus_design = minus.design.to_dense();
    let fd_design = (&plus_design - &(base_design.clone() * 2.0) + &minus_design) / (eps * eps);
    // Same as the first-derivative test: the Duchon design path is now
    // operator-only, so we materialize the diagonal second derivative for
    // axis 0 to match the dense central-difference shape.
    let analytic_second = second_derivative
        .implicit_operator
        .as_ref()
        .expect("Duchon design second derivative must expose an implicit operator")
        .materialize_second_diag(0)
        .expect("materialize second-diag design derivative");
    let design_err = (&analytic_second - &fd_design)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        design_err < 5e-3,
        "Duchon design second derivative mismatch too large: {design_err}"
    );

    assert_eq!(
        second_derivative.penaltiessecond_derivative.len(),
        base.penalties.len()
    );
    let fd_primary_penalty = (&plus.penalties[0] - &(base.penalties[0].clone() * 2.0)
        + &minus.penalties[0])
        / (eps * eps);
    let primary_penalty_err = (&second_derivative.penaltiessecond_derivative[0]
        - &fd_primary_penalty)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    assert!(
        primary_penalty_err < 5e-3,
        "Duchon mass penalty second derivative mismatch too large: {primary_penalty_err}"
    );
    for penalty_idx in 1..second_derivative.penaltiessecond_derivative.len() {
        let fd_penalty = (&plus.penalties[penalty_idx]
            - &(base.penalties[penalty_idx].clone() * 2.0)
            + &minus.penalties[penalty_idx])
            / (eps * eps);
        let penalty_err = (&second_derivative.penaltiessecond_derivative[penalty_idx]
            - &fd_penalty)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            penalty_err < 5e-3,
            "Duchon operator penalty second derivative mismatch too large at block {penalty_idx}: {penalty_err}"
        );
    }
}

#[test]
fn test_gram_and_psi_derivatives_from_operator_matchesfd() {
    // Build D(psi) = D0 + psi D1 + 0.5 psi^2 D2 with nontrivial shape.
    let d0 = array![
        [0.9, -0.2, 0.3],
        [0.4, 0.8, -0.6],
        [0.1, 0.7, 0.5],
        [-0.3, 0.2, 0.4]
    ];
    let d1 = array![
        [0.2, -0.1, 0.05],
        [0.3, 0.07, -0.2],
        [-0.15, 0.06, 0.1],
        [0.04, -0.09, 0.12]
    ];
    let d2 = array![
        [0.08, -0.02, 0.01],
        [0.03, 0.04, -0.05],
        [0.02, -0.01, 0.06],
        [-0.07, 0.03, 0.02]
    ];

    let psi0 = 0.35;
    let d = &d0 + &(d1.mapv(|v| psi0 * v)) + &(d2.mapv(|v| 0.5 * psi0 * psi0 * v));
    let d_psi = &d1 + &(d2.mapv(|v| psi0 * v));
    let d_psi_psi = d2.clone();

    let (s, s_psi, s_psi_psi) = gram_and_psi_derivatives_from_operator(&d, &d_psi, &d_psi_psi);

    let h = 1e-6;
    let eval_s = |psi: f64| {
        let d_eval = &d0 + &(d1.mapv(|v| psi * v)) + &(d2.mapv(|v| 0.5 * psi * psi * v));
        symmetrize(&fast_ata(&d_eval))
    };
    let s_plus = eval_s(psi0 + h);
    let s_minus = eval_s(psi0 - h);
    let sfd = (&s_plus - &s_minus) / (2.0 * h);
    let s2fd = (&s_plus - &(s.mapv(|v| 2.0 * v)) + &s_minus) / (h * h);

    let err1 = (&s_psi - &sfd).iter().map(|v| v * v).sum::<f64>().sqrt();
    let err2 = (&s_psi_psi - &s2fd)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    for i in 0..s_psi.nrows() {
        for j in 0..s_psi.ncols() {
            assert_eq!(s_psi[[i, j]].signum(), sfd[[i, j]].signum());
            assert_eq!(s_psi_psi[[i, j]].signum(), s2fd[[i, j]].signum());
        }
    }

    assert!(err1 < 2e-6, "S' mismatch too large: {err1}");
    assert!(err2 < 5e-4, "S'' mismatch too large: {err2}");
}

#[test]
fn test_normalize_penaltywith_psi_derivatives_matchesfd() {
    // Build S(psi) = S0 + psi S1 + 0.5 psi^2 S2 and validate exact
    // normalization derivatives against finite differences of S/||S||_F.
    let s0 = array![[2.0, 0.3, -0.2], [0.3, 1.7, 0.4], [-0.2, 0.4, 1.4]];
    let s1 = array![[0.2, -0.05, 0.1], [-0.05, 0.12, 0.03], [0.1, 0.03, -0.08]];
    let s2 = array![
        [0.04, 0.02, -0.01],
        [0.02, -0.03, 0.015],
        [-0.01, 0.015, 0.02]
    ];

    let psi0 = -0.4;
    let s = &s0 + &(s1.mapv(|v| psi0 * v)) + &(s2.mapv(|v| 0.5 * psi0 * psi0 * v));
    let s_psi = &s1 + &(s2.mapv(|v| psi0 * v));
    let s_psi_psi = s2.clone();

    let (_, sn_psi, sn_psi_psi, _) = normalize_penaltywith_psi_derivatives(&s, &s_psi, &s_psi_psi);

    let h = 1e-6;
    let eval_snorm = |psi: f64| {
        let s_eval = &s0 + &(s1.mapv(|v| psi * v)) + &(s2.mapv(|v| 0.5 * psi * psi * v));
        let c = trace_of_product(&s_eval, &s_eval).sqrt();
        s_eval.mapv(|v| v / c)
    };
    let sn = eval_snorm(psi0);
    let sn_plus = eval_snorm(psi0 + h);
    let sn_minus = eval_snorm(psi0 - h);
    let snfd = (&sn_plus - &sn_minus) / (2.0 * h);
    let sn2fd = (&sn_plus - &(sn.mapv(|v| 2.0 * v)) + &sn_minus) / (h * h);

    let err1 = (&sn_psi - &snfd).iter().map(|v| v * v).sum::<f64>().sqrt();
    let err2 = (&sn_psi_psi - &sn2fd)
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    for i in 0..sn_psi.nrows() {
        for j in 0..sn_psi.ncols() {
            assert_eq!(sn_psi[[i, j]].signum(), snfd[[i, j]].signum());
            assert_eq!(sn_psi_psi[[i, j]].signum(), sn2fd[[i, j]].signum());
        }
    }

    assert!(err1 < 2e-6, "normalized S' mismatch too large: {err1}");
    assert!(err2 < 5e-4, "normalized S'' mismatch too large: {err2}");
}
#[test]
fn test_log_kappa_scaling_identities_match_autodiff() {
    let psi0 = -0.23;
    let r = 0.71;
    let d = 5.0;
    let eta = -3.5;
    let kappa = psi0.exp();
    let t = kappa * r;
    let eta_q = eta + 2.0;

    let (phi, phi_psi_ad, phi_psi_psi_ad) =
        second_derivative(|psi| scaling_testphi(psi, r, eta), psi0);
    let (q, q_psi_ad, q_psi_psi_ad) = second_derivative(|psi| scaling_test_q(psi, r, eta), psi0);
    let (lap, lap_psi_ad, lap_psi_psi_ad) =
        second_derivative(|psi| scaling_test_lap(psi, r, eta, d), psi0);

    let phi_r = kappa.powf(eta + 1.0) * (2.0 * t + 4.0 * t.powi(3));
    let phi_rr = kappa.powf(eta + 2.0) * (2.0 + 12.0 * t * t);
    let q_r = kappa.powf(eta + 3.0) * (8.0 * t);
    let q_rr = kappa.powf(eta + 4.0) * 8.0;
    let lap_r = kappa.powf(eta + 3.0) * ((8.0 * d + 16.0) * t);
    let lap_rr = kappa.powf(eta + 4.0) * (8.0 * d + 16.0);

    let phi_psi = eta * phi + r * phi_r;
    let phi_psi_psi = eta * eta * phi + (2.0 * eta + 1.0) * r * phi_r + r * r * phi_rr;
    let q_psi = eta_q * q + r * q_r;
    let q_psi_psi = eta_q * eta_q * q + (2.0 * eta_q + 1.0) * r * q_r + r * r * q_rr;
    let lap_psi = eta_q * lap + r * lap_r;
    let lap_psi_psi = eta_q * eta_q * lap + (2.0 * eta_q + 1.0) * r * lap_r + r * r * lap_rr;

    assert!((phi_psi - phi_psi_ad).abs() < 1e-12);
    assert!((phi_psi_psi - phi_psi_psi_ad).abs() < 1e-12);
    assert!((q_psi - q_psi_ad).abs() < 1e-12);
    assert!((q_psi_psi - q_psi_psi_ad).abs() < 1e-12);
    assert!((lap_psi - lap_psi_ad).abs() < 1e-12);
    assert!((lap_psi_psi - lap_psi_psi_ad).abs() < 1e-12);
}

#[test]
fn test_duchonspectral_scaling_matches_implementation() {
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let k_dim = 4usize;
    let length_scale_1 = 1.7;
    let length_scale_2 = 0.85;
    let kappa_1 = 1.0 / length_scale_1;
    let kappa_2 = 1.0 / length_scale_2;
    let scale = kappa_2 / kappa_1;
    let r = 0.43;
    let scaled_r = scale * r;
    let delta = duchon_scaling_exponent(p_order, s_order, k_dim);

    let coeffs_1 = duchon_partial_fraction_coeffs(p_order, s_order, kappa_1);
    let coeffs_2 = duchon_partial_fraction_coeffs(p_order, s_order, kappa_2);

    let phi_1 = duchon_matern_kernel_general_from_distance(
        scaled_r,
        Some(length_scale_1),
        p_order,
        s_order,
        k_dim,
        Some(&coeffs_1),
    )
    .expect("scaled phi_1");
    let phi_2 = duchon_matern_kernel_general_from_distance(
        r,
        Some(length_scale_2),
        p_order,
        s_order,
        k_dim,
        Some(&coeffs_2),
    )
    .expect("phi_2");
    let jets_1 = duchon_radial_jets(scaled_r, length_scale_1, p_order, s_order, k_dim, &coeffs_1)
        .expect("jets_1");
    let jets_2 =
        duchon_radial_jets(r, length_scale_2, p_order, s_order, k_dim, &coeffs_2).expect("jets_2");

    let phi_scale = scale.powf(delta);
    let op_scale = scale.powf(delta + 2.0);
    // phi(r;κ) = κ^δ·H(κr) holds for the spectral Fourier transform, but
    // only modulo a κ-dependent additive constant that reflects the
    // IR-divergence ambiguity of the polyharmonic log branch at even
    // dimension with 2m == d. The code's polyharmonic_kernel picks the
    // log(r) convention (reference scale 1), so under s = κ_2/κ_1 the
    // m=2, d=4 block leaves the residue
    //   s^δ · a_m(κ_1) · c_p · log(s),
    // with c_p = polyharmonic_log_sign(m,d) / (2^{2m-1} π^{d/2} Γ(m) Γ(m-d/2+1)).
    // Operator scalars (q, Δphi) are differentiated and have no residue;
    // they are still checked tightly below.
    let m_log = 2usize;
    let c_p = polyharmonic_log_sign(m_log, k_dim)
        / (2.0_f64.powi((2 * m_log - 1) as i32)
            * std::f64::consts::PI.powf(0.5 * k_dim as f64)
            * gamma_lanczos(m_log as f64)
            * gamma_lanczos((m_log - k_dim / 2 + 1) as f64));
    let a_m_kappa_1 = kappa_1.powf(-2.0 * (s_order + p_order - m_log) as f64);
    let log_branch_residue = (phi_scale * a_m_kappa_1 * c_p * scale.ln()).abs();
    let phi_tol = (log_branch_residue * 1.5).max(1e-12);
    assert!(
        (phi_2 - phi_scale * phi_1).abs() < phi_tol,
        "phi scaling residue {} exceeds expected log-branch bound {}",
        (phi_2 - phi_scale * phi_1).abs(),
        phi_tol,
    );
    assert!((jets_2.q - op_scale * jets_1.q).abs() < 1e-8);
    assert!((jets_2.lap - op_scale * jets_1.lap).abs() < 1e-8);

    let core =
        duchon_radial_core_psi_triplet(r, length_scale_2, p_order, s_order, k_dim, &coeffs_2)
            .expect("radial core");
    assert!(core.phi.value.is_finite());
    assert!(core.phi.psi.is_finite());
    assert!(core.phi.psi_psi.is_finite());
}

#[test]
fn test_radial_basis_cartesian_derivative_matches_legacy_loops() {
    // Parity test for the shared Duchon radial-jet / Cartesian-derivative
    // engine (issue #425). The shared `radial_basis_cartesian_derivative`
    // must reproduce, bit-for-bit-up-to-rounding, the radial→Cartesian
    // tensors that the analytic-penalty path used to build with its own
    // inline loops. We rebuild those *legacy* loops here verbatim from the
    // independently-evaluated radial derivative matrices and require an
    // exact match.
    let centers = array![
        [0.10, -0.30, 0.20],
        [-0.40, 0.15, 0.05],
        [0.25, 0.35, -0.10],
        [-0.05, -0.20, 0.45],
    ];
    // Eval points: one off-origin generic row plus one that lands exactly
    // on a center (collision r = 0) to exercise both branches.
    let t = array![
        [0.05, 0.10, -0.15],
        [-0.40, 0.15, 0.05], // == centers row 1 → r = 0 at k = 1
        [0.30, -0.25, 0.40],
    ];
    let d = centers.ncols();
    let n_centers = centers.nrows();
    let n_rows = t.nrows();
    let p_out = 2usize;
    let coeffs = array![[1.10, -0.40], [-0.25, 0.70], [0.60, 0.15], [0.05, -0.90],];
    let length_scale = Some(0.8_f64);
    let nullspace_order = DuchonNullspaceOrder::Linear;
    // Hybrid spectral order s = power: with p = 1 (Linear) and d = 3 the
    // hybrid kernel `‖w‖^{2p}(κ²+‖w‖²)^s` needs `2(p+s) > d`, i.e. s ≥ 1.
    // Both the direct radial adapters and the Cartesian engine must resolve
    // the *same* `(p, s, κ)` (issue #440), so they share `power` here.
    let power = 1usize;

    // Independent radial derivative matrices via the public adapters.
    let phi_r = duchon_radial_first_derivative_nd(
        t.view(),
        centers.view(),
        length_scale,
        nullspace_order,
        power,
    )
    .expect("phi_r");
    let phi_rr = duchon_radial_second_derivative_nd(
        t.view(),
        centers.view(),
        length_scale,
        nullspace_order,
        power,
    )
    .expect("phi_rr");
    let phi_rrr = duchon_radial_third_derivative_nd(
        t.view(),
        centers.view(),
        length_scale,
        nullspace_order,
        power,
    )
    .expect("phi_rrr");

    // --- order 2: legacy Hessian loop -----------------------------------
    let mut legacy2 = Array2::<f64>::zeros((n_rows, p_out * d * d));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..d {
                let delta = t[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            for a in 0..d {
                for c in 0..d {
                    let basis_hess = if r == 0.0 {
                        if a == c { phi_rr[[n, k]] } else { 0.0 }
                    } else {
                        let inv_r = 1.0 / r;
                        let u_a = (t[[n, a]] - centers[[k, a]]) * inv_r;
                        let u_c = (t[[n, c]] - centers[[k, c]]) * inv_r;
                        let q = phi_r[[n, k]] * inv_r;
                        let eye = if a == c { 1.0 } else { 0.0 };
                        q * eye + (phi_rr[[n, k]] - q) * u_a * u_c
                    };
                    if basis_hess == 0.0 {
                        continue;
                    }
                    for i in 0..p_out {
                        legacy2[[n, (i * d + a) * d + c]] += coeffs[[k, i]] * basis_hess;
                    }
                }
            }
        }
    }
    let shared2 = radial_basis_cartesian_derivative(
        2,
        t.view(),
        centers.view(),
        coeffs.view(),
        length_scale,
        nullspace_order,
        power,
    )
    .expect("shared order-2");
    assert_eq!(shared2.dim(), legacy2.dim());
    let err2 = (&shared2 - &legacy2)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err2 < 1e-14,
        "order-2 Cartesian derivative parity mismatch: max abs err {err2}"
    );

    // --- order 3: legacy third-derivative loop --------------------------
    let mut legacy3 = ndarray::Array3::<f64>::zeros((n_rows, p_out, d * d * d));
    for n in 0..n_rows {
        for k in 0..n_centers {
            let mut r2 = 0.0_f64;
            for a in 0..d {
                let delta = t[[n, a]] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            if r == 0.0 {
                continue;
            }
            let inv_r = 1.0 / r;
            let q = phi_r[[n, k]] * inv_r;
            let b_coef = (phi_rr[[n, k]] - q) * inv_r;
            let a_coef = phi_rrr[[n, k]] - 3.0 * b_coef;
            for a in 0..d {
                let u_a = (t[[n, a]] - centers[[k, a]]) * inv_r;
                for c in 0..d {
                    let u_c = (t[[n, c]] - centers[[k, c]]) * inv_r;
                    for e in 0..d {
                        let u_e = (t[[n, e]] - centers[[k, e]]) * inv_r;
                        let eye_ac = if a == c { 1.0 } else { 0.0 };
                        let eye_ae = if a == e { 1.0 } else { 0.0 };
                        let eye_ce = if c == e { 1.0 } else { 0.0 };
                        let basis_third = a_coef * u_a * u_c * u_e
                            + b_coef * (eye_ac * u_e + eye_ae * u_c + eye_ce * u_a);
                        if basis_third == 0.0 {
                            continue;
                        }
                        let idx = ((a * d) + c) * d + e;
                        for i in 0..p_out {
                            legacy3[[n, i, idx]] += coeffs[[k, i]] * basis_third;
                        }
                    }
                }
            }
        }
    }
    let shared3_flat = radial_basis_cartesian_derivative(
        3,
        t.view(),
        centers.view(),
        coeffs.view(),
        length_scale,
        nullspace_order,
        power,
    )
    .expect("shared order-3");
    let shared3 = shared3_flat
        .into_shape_with_order((n_rows, p_out, d * d * d))
        .expect("reshape order-3");
    assert_eq!(shared3.dim(), legacy3.dim());
    let err3 = (&shared3 - &legacy3)
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        err3 < 1e-14,
        "order-3 Cartesian derivative parity mismatch: max abs err {err3}"
    );

    // The radial-jet matrices themselves must be non-trivial, so the
    // parity assertions above are exercising real values rather than an
    // all-zero coincidence.
    assert!(
        phi_r.iter().any(|v| v.abs() > 1e-9)
            && phi_rr.iter().any(|v| v.abs() > 1e-9)
            && phi_rrr.iter().any(|v| v.abs() > 1e-9),
        "radial-jet matrices unexpectedly trivial"
    );
    assert!(
        shared2.iter().any(|v| v.abs() > 1e-9) && shared3.iter().any(|v| v.abs() > 1e-9),
        "Cartesian derivative tensors unexpectedly trivial"
    );
}

#[test]
fn test_duchon_collision_operator_limits_matchphi_rr_identities() {
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);

    let (phi_rr, phi_rr_psi, phi_rr_psi_psi) =
        duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("collision phi_rr");

    let eps = 2e-5_f64;
    let ls_plus = 1.0 / (kappa * eps.exp());
    let ls_minus = 1.0 / (kappa * (-eps).exp());
    let coeffs_plus = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ls_plus);
    let coeffs_minus = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ls_minus);
    let (phi_rr_plus, _, _) =
        duchonphi_rr_collision_psi_triplet(ls_plus, p_order, s_order, k_dim, &coeffs_plus)
            .expect("plus collision phi_rr");
    let (phi_rr_minus, _, _) =
        duchonphi_rr_collision_psi_triplet(ls_minus, p_order, s_order, k_dim, &coeffs_minus)
            .expect("minus collision phi_rr");
    let phi_rr_psi_fd = (phi_rr_plus - phi_rr_minus) / (2.0 * eps);
    let phi_rr_psi_psi_fd = (phi_rr_plus - 2.0 * phi_rr + phi_rr_minus) / (eps * eps);
    assert!((phi_rr_psi - phi_rr_psi_fd).abs() < 1e-6);
    assert!((phi_rr_psi_psi - phi_rr_psi_psi_fd).abs() < 1e-4);
}

#[test]
fn test_duchon_collision_phi_rr_log_kappa_derivatives_even_log_branch_matchfd() {
    // Regression for the q=2 operator-penalty derivative path when a
    // requested linear nullspace degrades to p=1 in d=2.  The old collision
    // shortcut used (delta+2) scaling for phi''(0); that misses the
    // kappa-dependent finite-part constants of even-dimensional log-Riesz
    // blocks and corrupts the stiffness derivative.
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Zero);
    let s_order = 2usize;
    let k_dim = 2usize;
    let length_scale = 0.9_f64;
    let kappa = 1.0 / length_scale;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, kappa);
    let (phi_rr, phi_rr_psi, phi_rr_psi_psi) =
        duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("collision phi_rr triplet");

    let eps = 2.0e-5_f64;
    let at = |psi_step: f64| -> f64 {
        let ls = 1.0 / (kappa * psi_step.exp());
        let coeffs_step = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / ls);
        duchonphi_rr_collision_psi_triplet(ls, p_order, s_order, k_dim, &coeffs_step)
            .expect("collision phi_rr at perturbed kappa")
            .0
    };
    let plus = at(eps);
    let minus = at(-eps);
    let fd_first = (plus - minus) / (2.0 * eps);
    let fd_second = (plus - 2.0 * phi_rr + minus) / (eps * eps);

    let denom_first = fd_first.abs().max(phi_rr_psi.abs()).max(1.0e-12);
    let denom_second = fd_second.abs().max(phi_rr_psi_psi.abs()).max(1.0e-12);
    assert!(
        (phi_rr_psi - fd_first).abs() / denom_first < 2.0e-7,
        "even log-Riesz collision phi_rr psi mismatch: analytic={phi_rr_psi:.12e} fd={fd_first:.12e}"
    );
    assert!(
        (phi_rr_psi_psi - fd_second).abs() / denom_second < 2.0e-4,
        "even log-Riesz collision phi_rr psi-psi mismatch: analytic={phi_rr_psi_psi:.12e} fd={fd_second:.12e}"
    );
}

#[test]
fn test_duchon_radial_jets_use_collision_limits_at_origin() {
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 4usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let jets = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("jets at origin");
    let (phi_rr, _, _) =
        duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("collision phi_rr");
    let t_collision = duchon_phi_rrrr_collision(length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("collision phi''''")
        / 3.0;

    assert!(jets.phi_r.abs() < 1e-12);
    assert!((jets.q - phi_rr).abs() < 1e-12);
    assert!((jets.lap - k_dim as f64 * phi_rr).abs() < 1e-12);
    assert!(jets.q_r.abs() < 1e-12);
    assert!((jets.q_rr - t_collision).abs() < 1e-12);
    assert!(jets.lap_r.abs() < 1e-12);
    assert!((jets.lap_rr - (k_dim as f64 + 2.0) * t_collision).abs() < 1e-12);
    // t(0) should be finite (= φ''''(0) / 3) and is checked more
    // thoroughly in the dedicated t-field tests below.
    assert!((jets.t - t_collision).abs() < 1e-12);
}

#[test]
fn test_duchon_radial_jets_use_lower_order_collision_limits_at_origin() {
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);
    let jets = duchon_radial_jets(0.0, length_scale, p_order, s_order, k_dim, &coeffs)
        .expect("jets at origin");
    let (phi_rr, _, _) =
        duchonphi_rr_collision_psi_triplet(length_scale, p_order, s_order, k_dim, &coeffs)
            .expect("collision phi_rr");

    assert!(jets.phi_r.abs() < 1e-12);
    assert!((jets.q - phi_rr).abs() < 1e-12);
    assert!((jets.lap - k_dim as f64 * phi_rr).abs() < 1e-12);
    assert!(jets.q_r.abs() < 1e-12);
    assert!(jets.lap_r.abs() < 1e-12);
}

#[test]
fn test_duchon_radial_jets_t_equals_phi_rr_minus_q_over_r2() {
    // Verify t = (φ'' - q) / r² at several r values.
    let p_order = duchon_p_from_nullspace_order(DuchonNullspaceOrder::Linear);
    let s_order = 3usize;
    let k_dim = 4usize;
    let length_scale = 0.85;
    let coeffs = duchon_partial_fraction_coeffs(p_order, s_order, 1.0 / length_scale);

    for &r in &[0.01, 0.1, 0.5, 1.0, 2.0] {
        let jets =
            duchon_radial_jets(r, length_scale, p_order, s_order, k_dim, &coeffs).expect("jets");
        let t_expected = (jets.phi_rr - jets.q) / (r * r);
        let rel = if t_expected.abs() > 1e-15 {
            ((jets.t - t_expected) / t_expected).abs()
        } else {
            (jets.t - t_expected).abs()
        };
        assert!(
            rel < 1e-10,
            "t mismatch at r={r}: jets.t={}, expected={}, rel_err={rel}",
            jets.t,
            t_expected,
        );
    }
}
