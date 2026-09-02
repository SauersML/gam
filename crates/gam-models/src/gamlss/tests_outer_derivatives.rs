//! Model-owned outer-derivative gates for the concrete binomial
//! location-scale families.
//!
//! These tests live with the families they instantiate. Keeping them out of
//! `gam-custom-family` prevents a leaf numerical carrier from dev-depending
//! back upward on the model layer merely to exercise generic outer machinery.

use super::*;
use crate::custom_family::{CustomFamilyHyperLayout, OuterCriterionDiagnostics};
use ndarray::{Array1, Array2, array};

fn outerobjective_andgradient<F: CustomFamily + Clone + Send + Sync + 'static>(
    family: &F,
    specs: &[ParameterBlockSpec],
    options: &BlockwiseFitOptions,
    rho: &Array1<f64>,
) -> Result<(f64, Array1<f64>, ()), String> {
    let (objective, gradient, _, warm_start) = outerobjectivegradienthessian(
        family,
        specs,
        options,
        rho,
        gam_problem::EvalMode::ValueAndGradient,
    )?;
    Ok((objective, gradient, warm_start))
}

fn test_design_hyper_layout(
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
) -> CustomFamilyHyperLayout {
    let axis_count = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    CustomFamilyHyperLayout::new(
        derivative_blocks,
        Vec::new(),
        Array1::zeros(axis_count),
    )
    .expect("test design hyper layout")
}

pub(crate) struct BinomialLocationScaleWiggleOuterFixture {
    pub(crate) family: BinomialLocationScaleWiggleFamily,
    pub(crate) specs: Vec<ParameterBlockSpec>,
    pub(crate) rho: Array1<f64>,
    pub(crate) options: BlockwiseFitOptions,
}

/// Shared probit binomial-location-scale outer-derivative test fixture:
/// builds the (threshold, log_sigma) block specs, family, and outer options
/// that every `outer_laml*_binomial_location_scale_*` finite-difference test
/// constructs identically apart from `y` and the two block initial betas.
fn binomial_location_scale_outer_fixture(
    y: Array1<f64>,
    threshold_initial_beta: f64,
    log_sigma_initial_beta: f64,
) -> (
    BinomialLocationScaleFamily,
    Vec<ParameterBlockSpec>,
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
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: 1e-10,
        outer_max_iter: 1,
        ..BlockwiseFitOptions::default()
    };
    (family, specs, options)
}

#[test]
pub(crate) fn outer_lamlgradient_diagonal_binomial_location_scale_matchesfd() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let (family, specs, options) =
        binomial_location_scale_outer_fixture(y, 0.0, 0.0);
    let rho = array![0.0, 0.0];

    let (f0, g0, _) =
        outerobjective_andgradient(&family, &specs, &options, &rho)
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
            outerobjective_andgradient(&family, &specs, &options, &rho_p)
                .expect("objective+");
        let (fm, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &rho_m)
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
    let (family, specs, options) =
        binomial_location_scale_outer_fixture(y, 0.2, -0.1);
    let rho = array![0.15, -0.25];

    let (f0, g0, _) =
        outerobjective_andgradient(&family, &specs, &options, &rho)
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
            outerobjective_andgradient(&family, &specs, &options, &rho_p)
                .expect("objective+");
        let (fm, _, _) =
            outerobjective_andgradient(&family, &specs, &options, &rho_m)
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
    let (family, specs, options) =
        binomial_location_scale_outer_fixture(y, 0.15, -0.05);
    let rho = array![0.1, -0.2];

    let (_, _, h0_opt, _) = outerobjectivegradienthessian(
        &family,
        &specs,
        &options,
        &rho,
        gam_problem::EvalMode::ValueGradientHessian,
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
        let (_, gp, _, _) = outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &rho_p,
            gam_problem::EvalMode::ValueAndGradient,
        )
        .expect("objective/gradient +");
        let (_, gm, _, _) = outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &rho_m,
            gam_problem::EvalMode::ValueAndGradient,
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
pub(crate) fn outer_lamlhessian_joint_exact_binomial_location_scale_hard_case_matchesfd() {
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let (family, specs, options) =
        binomial_location_scale_outer_fixture(y, 0.2, -0.1);
    let rho = array![0.15, -0.25];

    let (_, _, h0_opt, _) = outerobjectivegradienthessian(
        &family,
        &specs,
        &options,
        &rho,
        gam_problem::EvalMode::ValueGradientHessian,
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
        let (_, gp, _, _) = outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &rho_p,
            gam_problem::EvalMode::ValueAndGradient,
        )
        .expect("objective/gradient +");
        let (_, gm, _, _) = outerobjectivegradienthessian(
            &family,
            &specs,
            &options,
            &rho_m,
            gam_problem::EvalMode::ValueAndGradient,
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
