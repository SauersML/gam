//! Closure oracle for #2307: the anisotropic Matérn structural-intercept
//! double penalty must differentiate the represented FUNCTION metric, including
//! first, diagonal-second, and mixed optimizer-coordinate derivatives.
//!
//! This test belongs to `gam-terms`, whose public Matérn construction contract
//! it exercises. Keeping it out of the top-level `gam` integration binary avoids
//! compiling and linking every model, CLI, prediction, and Python dependency to
//! validate one basis-level invariant.

use gam_terms::basis::{
    BasisWorkspace, CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu,
    PenaltySource, build_matern_basis_log_kappa_aniso_derivatives, build_matern_basiswithworkspace,
};
use ndarray::{Array2, array};

fn psi_to_length_scale_and_eta(psi: &[f64]) -> (f64, Vec<f64>) {
    let mean = psi.iter().sum::<f64>() / psi.len() as f64;
    (
        (-mean).exp(),
        psi.iter().map(|value| value - mean).collect(),
    )
}

fn fixture() -> (Array2<f64>, MaternBasisSpec, Vec<f64>) {
    let centers = array![
        [-1.20, -0.40],
        [-0.65, 0.85],
        [-0.10, -0.95],
        [0.35, 0.30],
        [0.90, 1.15],
        [1.45, -0.20],
    ];
    let data = array![
        [-1.20, -0.40],
        [-0.65, 0.85],
        [-0.10, -0.95],
        [0.35, 0.30],
        [0.90, 1.15],
        [1.45, -0.20],
        [0.15, 1.50],
        [1.25, 0.55],
    ];
    let psi = vec![0.55, -0.25];
    let (length_scale, eta) = psi_to_length_scale_and_eta(&psi);
    let spec = MaternBasisSpec {
        center_strategy: CenterStrategy::UserProvided(centers),
        periodic: None,
        length_scale,
        nu: MaternNu::FiveHalves,
        include_intercept: true,
        double_penalty: true,
        // No coefficient gauge: the positive Matérn columns have a strong,
        // directly observable center-measure overlap with the intercept.
        identifiability: MaternIdentifiability::None,
        aniso_log_scales: Some(eta),
    };
    (data, spec, psi)
}

fn realized_penalties_at_psi(
    data: &Array2<f64>,
    spec: &MaternBasisSpec,
    psi: &[f64],
) -> Vec<Array2<f64>> {
    let (length_scale, eta) = psi_to_length_scale_and_eta(psi);
    let mut trial = spec.clone();
    trial.length_scale = length_scale;
    trial.aniso_log_scales = Some(eta);
    build_matern_basiswithworkspace(data.view(), &trial, &mut BasisWorkspace::default())
        .expect("realized anisotropic Matérn value build")
        .active_penalties
        .iter()
        .map(|penalty| penalty.matrix.clone())
        .collect()
}

fn max_abs(matrix: &Array2<f64>) -> f64 {
    matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
}

fn assert_matrix_close(
    label: &str,
    analytic: &Array2<f64>,
    finite_difference: &Array2<f64>,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) {
    let error = (analytic - finite_difference)
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let analytic_scale = max_abs(analytic);
    let fd_scale = max_abs(finite_difference);
    let bound = absolute_tolerance + relative_tolerance * analytic_scale.max(fd_scale);
    assert!(
        error <= bound,
        "{label}: max error {error:.3e} exceeds {bound:.3e} (analytic scale {analytic_scale:.3e}, FD scale {fd_scale:.3e})"
    );
}

#[test]
fn matern_aniso_nonorthogonal_structural_ridge_jet_matches_central_finite_differences() {
    let (data, spec, psi) = fixture();
    let base = build_matern_basiswithworkspace(data.view(), &spec, &mut BasisWorkspace::default())
        .expect("base anisotropic Matérn value build");
    let ridge_index = base
        .active_penalties
        .iter()
        .position(|penalty| matches!(penalty.info.source, PenaltySource::DoublePenaltyNullspace))
        .expect("explicit intercept must emit an active structural ridge");
    let ridge = &base.active_penalties[ridge_index].matrix;
    let intercept_column = ridge.ncols() - 1;
    let kernel_intercept_overlap = (0..intercept_column)
        .map(|column| ridge[[column, intercept_column]].abs())
        .fold(0.0_f64, f64::max);
    assert!(
        kernel_intercept_overlap > 1.0e-3 * max_abs(ridge),
        "fixture must be nonorthogonal in function space; max kernel/intercept ridge entry={kernel_intercept_overlap:.3e}"
    );

    let derivatives = build_matern_basis_log_kappa_aniso_derivatives(data.view(), &spec)
        .expect("analytic anisotropic Matérn derivative build");
    assert_eq!(derivatives.penalties_first.len(), 2);
    assert_eq!(derivatives.penalties_second_diag.len(), 2);
    assert_eq!(
        derivatives.penalties_first[0].len(),
        base.active_penalties.len()
    );
    assert_eq!(
        derivatives.penalties_second_diag[0].len(),
        base.active_penalties.len()
    );
    assert!(derivatives.penalties_cross_pairs.contains(&(0, 1)));
    let analytic_cross = derivatives
        .penalties_cross_provider
        .as_ref()
        .expect("mixed structural-ridge derivative provider")
        .evaluate(0, 1)
        .expect("mixed structural-ridge derivative evaluation");
    assert_eq!(analytic_cross.len(), base.active_penalties.len());

    let first_step = 1.0e-6;
    let mut psi_first_plus = psi.clone();
    psi_first_plus[0] += first_step;
    let mut psi_first_minus = psi.clone();
    psi_first_minus[0] -= first_step;
    let first_plus = realized_penalties_at_psi(&data, &spec, &psi_first_plus);
    let first_minus = realized_penalties_at_psi(&data, &spec, &psi_first_minus);
    let fd_first = (&first_plus[ridge_index] - &first_minus[ridge_index]) / (2.0 * first_step);

    let second_step = 2.0e-4;
    let mut psi_second_plus = psi.clone();
    psi_second_plus[0] += second_step;
    let mut psi_second_minus = psi.clone();
    psi_second_minus[0] -= second_step;
    let second_plus = realized_penalties_at_psi(&data, &spec, &psi_second_plus);
    let second_minus = realized_penalties_at_psi(&data, &spec, &psi_second_minus);
    let fd_second = (&second_plus[ridge_index]
        - &(&base.active_penalties[ridge_index].matrix * 2.0)
        + &second_minus[ridge_index])
        / (second_step * second_step);

    let corner = |axis_a_sign: f64, axis_b_sign: f64| {
        let mut point = psi.clone();
        point[0] += axis_a_sign * second_step;
        point[1] += axis_b_sign * second_step;
        realized_penalties_at_psi(&data, &spec, &point)
    };
    let plus_plus = corner(1.0, 1.0);
    let plus_minus = corner(1.0, -1.0);
    let minus_plus = corner(-1.0, 1.0);
    let minus_minus = corner(-1.0, -1.0);
    let fd_cross = (&plus_plus[ridge_index] - &plus_minus[ridge_index] - &minus_plus[ridge_index]
        + &minus_minus[ridge_index])
        / (4.0 * second_step * second_step);

    let analytic_first = &derivatives.penalties_first[0][ridge_index];
    let analytic_second = &derivatives.penalties_second_diag[0][ridge_index];
    let analytic_mixed = &analytic_cross[ridge_index];
    assert_matrix_close(
        "structural ridge first derivative",
        analytic_first,
        &fd_first,
        2.0e-5,
        2.0e-9,
    );
    assert_matrix_close(
        "structural ridge diagonal second derivative",
        analytic_second,
        &fd_second,
        2.0e-3,
        2.0e-6,
    );
    assert_matrix_close(
        "structural ridge mixed derivative",
        analytic_mixed,
        &fd_cross,
        2.0e-3,
        2.0e-6,
    );

    for (label, analytic, finite_difference) in [
        ("first", analytic_first, &fd_first),
        ("diagonal second", analytic_second, &fd_second),
        ("mixed", analytic_mixed, &fd_cross),
    ] {
        assert!(
            max_abs(analytic) > 1.0e-4 && max_abs(finite_difference) > 1.0e-4,
            "structural-ridge {label} derivative must be materially nonzero; analytic={:.3e}, FD={:.3e}",
            max_abs(analytic),
            max_abs(finite_difference),
        );
    }
}
