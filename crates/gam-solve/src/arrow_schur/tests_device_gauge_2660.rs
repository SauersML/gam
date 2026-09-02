#![cfg(target_os = "linux")]

use super::*;
use ndarray::{Array1, Array2};

/// The #2660 production shape: 240 rows, five active latent coordinates per
/// row, and the composed narrow co-fit's 35 decoder coefficients.
///
/// The declared non-axis beta direction is an exact orbit of the row cross
/// blocks (`H_tbeta q = 0`), while `g_beta` deliberately contains a large orbit
/// component. An unquotiented solve therefore returns a visible
/// `q^T delta_beta`; the Faddeev--Popov solve must erase it while preserving the
/// identifiable complement.
///
/// A second, identifiable direction is deliberately indefinite in the reduced
/// Schur complement after `ridge_beta` is applied. Its right-hand side is zero,
/// so the canonical spectral floor changes only the collapsed operator
/// direction and leaves the raw-system backward error meaningful. This makes
/// the fixture discriminate the production conditioning contract: a raw device
/// POTRF must fail, while the canonical Direct factorization must floor that
/// direction and solve.
fn production_shaped_gauge_system(
    ridge_beta: f64,
) -> (ArrowSchurSystem, Array1<f64>) {
    let n = 240usize;
    let d = 5usize;
    let k = 35usize;
    let raw_direction = Array1::from_shape_fn(k, |index| 0.5 + ((index as f64 + 1.0) * 0.37).sin());
    let quotient =
        ArrowBetaGaugeQuotient::new(vec![raw_direction]).expect("independent beta gauge");
    let direction = quotient.directions[0].clone();
    let raw_collapsed_direction =
        Array1::from_shape_fn(k, |index| 0.4 + ((index as f64 + 2.0) * 0.23).cos());
    let mut collapsed_direction =
        quotient.project_complement(raw_collapsed_direction.view());
    let collapsed_norm = collapsed_direction.dot(&collapsed_direction).sqrt();
    assert!(
        collapsed_norm > 0.0 && collapsed_norm.is_finite(),
        "fixture collapsed direction must survive quotient projection"
    );
    collapsed_direction.mapv_inplace(|value| value / collapsed_norm);
    assert!(
        direction.dot(&collapsed_direction).abs() <= 2.0e-15,
        "fixture collapsed direction must be identifiable"
    );

    let mut sys = ArrowSchurSystem::new(n, d, k);
    for (row_index, row) in sys.rows.iter_mut().enumerate() {
        let mut htt = Array2::<f64>::zeros((d, d));
        for axis in 0..d {
            htt[[axis, axis]] = 3.0 + 0.2 * axis as f64;
            for other in 0..axis {
                let coupling = 0.01 / (1.0 + (axis - other) as f64);
                htt[[axis, other]] = coupling;
                htt[[other, axis]] = coupling;
            }
            row.gt[axis] = 0.03 * (((row_index + 1) * (axis + 2)) as f64 * 0.017).sin();
        }
        row.htt = htt;

        for axis in 0..d {
            let mut cross = Array1::from_shape_fn(k, |beta| {
                0.002 * (((row_index + 3) * (axis + 5) * (beta + 7)) as f64 * 0.0013).cos()
            });
            let orbit_component = cross.dot(&direction);
            cross.scaled_add(-orbit_component, &direction);
            let collapsed_component = cross.dot(&collapsed_direction);
            cross.scaled_add(-collapsed_component, &collapsed_direction);
            for beta in 0..k {
                row.htbeta[[axis, beta]] = cross[beta];
            }
            assert!(
                row.htbeta.row(axis).dot(&direction).abs() <= 2.0e-16,
                "fixture cross row must annihilate the declared beta orbit"
            );
            assert!(
                row.htbeta.row(axis).dot(&collapsed_direction).abs() <= 2.0e-16,
                "fixture cross row must annihilate the collapsed beta direction"
            );
        }
    }
    sys.hbb = Array2::from_diag(&Array1::from_elem(k, 12.0));
    // After adding `ridge_beta`, this direction has reduced eigenvalue -1e-3.
    // Every cross row annihilates it, so the Schur subtraction cannot obscure
    // the witness. All orthogonal identifiable directions retain curvature 12.
    let raw_collapsed_eigenvalue = -1.0e-3 - ridge_beta;
    let rank_one_scale = raw_collapsed_eigenvalue - 12.0;
    for row in 0..k {
        for column in 0..k {
            sys.hbb[[row, column]] +=
                rank_one_scale * collapsed_direction[row] * collapsed_direction[column];
        }
    }

    let raw_identifiable =
        Array1::from_shape_fn(k, |index| 0.04 * ((index as f64 + 1.0) * 0.11).cos());
    let mut identifiable = quotient.project_complement(raw_identifiable.view());
    let collapsed_gradient = identifiable.dot(&collapsed_direction);
    identifiable.scaled_add(-collapsed_gradient, &collapsed_direction);
    identifiable.scaled_add(0.5, &direction);
    sys.gb = identifiable;
    assert!(
        direction.dot(&sys.gb).abs() >= 0.49,
        "fixture needs a discriminating raw gauge-gradient component"
    );
    assert!(
        collapsed_direction.dot(&sys.gb).abs() <= 2.0e-15,
        "fixture collapsed direction must have zero reduced right-hand side"
    );
    sys.set_beta_gauge_quotient(quotient)
        .expect("quotient width matches production border");
    (sys, direction)
}

fn production_floor_cpu_oracle(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> (Array1<f64>, Array1<f64>, ArrowPcgDiagnostics) {
    let strict_cpu_options =
        ArrowSolveOptions::direct().with_gpu_policy(gam_gpu::GpuPolicy::Off);
    match solve_arrow_newton_step_core(sys, ridge_t, ridge_beta, &strict_cpu_options) {
        Err(ArrowSchurError::SchurFactorFailed { .. }) => {}
        // SAFETY: #2660 fixture oracle. The witness is only meaningful if the
        // strict Direct solve fails at the indefinite reduced Schur specifically;
        // any other error means the fixture no longer probes what it claims, and
        // silently continuing would certify against the wrong precondition.
        Err(error) => panic!(
            "#2660 strict Direct witness must fail specifically at the indefinite reduced Schur: \
             {error}"
        ),
        // SAFETY: as above -- a successful solve means the fixture stopped
        // activating the production spectral floor, so the comparison below
        // would measure a path the test is not about.
        Ok(_) => panic!(
            "#2660 fixture stopped activating the production spectral floor: strict Direct \
             unexpectedly solved the indefinite reduced Schur"
        ),
    }

    let cpu_options = ArrowSolveOptions::direct()
        .with_gpu_policy(gam_gpu::GpuPolicy::Off)
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR);
    solve_arrow_newton_step_core(sys, ridge_t, ridge_beta, &cpu_options)
        .expect("CPU quotient Direct solve with production spectral floor")
}

#[test]
fn device_reduced_factor_seam_is_the_canonical_direct_factor_2660() {
    // Eigenpairs are deliberately explicit:
    //   (1, 1, 0)/sqrt(2) -> 4,
    //   (1,-1, 0)/sqrt(2) -> -1e-3,
    //   (0, 0, 1)         -> 2.
    // The positive diagonal rules out a trivial bad-diagonal fixture; this is
    // the same collapsed-subspace shape that makes raw POTRF fail in practice.
    let schur = Array2::from_shape_vec(
        (3, 3),
        vec![
            1.9995, 2.0005, 0.0, 2.0005, 1.9995, 0.0, 0.0, 0.0, 2.0,
        ],
    )
    .expect("3x3 reduced witness");
    let mut schur_col_major = Vec::with_capacity(9);
    for column in 0..3 {
        for row in 0..3 {
            schur_col_major.push(schur[[row, column]]);
        }
    }

    let factorized =
        crate::gpu_kernels::arrow_schur::canonicalize_device_beta_factor(
            None,
            Some(SPECTRAL_DEFLATION_REL_FLOOR),
            3,
            &mut schur_col_major,
        )
        .expect("device reduced-factor seam must accept the canonical floor");
    assert!(
        factorized,
        "active production conditioning must replace raw device POTRF"
    );

    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR);
    let (_, expected_factor, _) =
        solve_dense_reduced_system(&schur, &Array1::zeros(3), &options, None)
            .expect("canonical CPU Direct factor");
    let expected_factor = expected_factor.expect("Direct returns its dense factor");
    for column in 0..3 {
        for row in 0..3 {
            assert_eq!(
                schur_col_major[column * 3 + row],
                expected_factor[[row, column]],
                "#2660 device seam drifted from the canonical Direct factor at ({row},{column})"
            );
        }
    }
}

#[test]
fn production_shaped_quotient_requires_canonical_spectral_floor_2660() {
    let ridge_t = 1.0e-7;
    let ridge_beta = 1.0e-6;
    let (sys, gauge_direction) = production_shaped_gauge_system(ridge_beta);
    let (cpu_t, cpu_beta, cpu_diag) =
        production_floor_cpu_oracle(&sys, ridge_t, ridge_beta);
    assert!(
        !cpu_diag.used_device_arrow,
        "GpuPolicy::Off CPU oracle must not report device execution"
    );
    let orbit_step = gauge_direction.dot(&cpu_beta).abs();
    assert!(
        orbit_step <= 1.0e-12,
        "#2660 CPU floor witness leaked beta-gauge motion: |q^T delta_beta|={orbit_step:e}"
    );
    let backward_error = arrow_quotient_backward_error_certificate(
        &sys,
        ridge_t,
        ridge_beta,
        cpu_t.view(),
        cpu_beta.view(),
    )
    .expect("CPU floor witness backward-error certificate");
    assert!(
        backward_error.is_finite(),
        "#2660 CPU floor witness raw-system backward error must remain finite"
    );
    assert!(
        backward_error > 1.0e-8,
        "#2660 fixture stopped materially activating the spectral floor: its raw-system \
         backward error is only {backward_error:e}"
    );
}

