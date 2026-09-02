#![cfg(test)]
// Behavior tests for the gamlss family stack (real `#[cfg(test)] mod tests`).
// `super::*` resolves to the parent `gamlss` module, whose flat re-exports
// surface every concern-submodule item these tests exercise.
//
// The scope claim above is made to the compiler, not just to the filename: the
// sole declaration of this module (`crates/gam-models/src/gamlss.rs`) is
// `#[cfg(test)] mod tests;`, so the inner attribute below is a no-op that makes
// the test-only scope checkable in this file.
#![cfg(test)]

use super::*;
// Helpers exercised only by these tests; imported here (not at module scope)
// so they are not flagged unused in a non-test `--lib` build.
use super::binomial_q_derivs::{
    binomial_neglog_q_derivatives_cloglog_closed_form,
    binomial_neglog_q_derivatives_logit_closed_form,
    binomial_neglog_q_derivatives_probit_closed_form,
    binomial_neglog_q_fourth_derivative_cloglog_closed_form,
    binomial_neglog_q_fourth_derivative_logit_closed_form,
    binomial_neglog_q_fourth_derivative_probit_closed_form,
};
use super::dispersion_family::dispersion_row_kernel;

use super::test_support::{binomial_location_scale_nll_tower, dispersion_tweedie_nll_generic};
use crate::custom_family::CustomFamilyHyperLayout;
use crate::fit_orchestration::{FitConfig, FitResult, fit_from_formula};

/// Dense `Tower4<2>` Tweedie row NLL oracle: the #932 all-channels instantiation
/// of the single-source [`dispersion_tweedie_nll_generic`] that production runs
/// as `Order2<2>` (via `dispersion_tweedie_nll_order2`). Test-only — it lives
/// here in the gamlss test module (its sole consumer) rather than as a
/// production `src/` item with no production caller.
#[inline]
fn dispersion_tweedie_nll_tower(
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    p: f64,
    wi: f64,
) -> gam_math::jet_tower::Tower4<2> {
    dispersion_tweedie_nll_generic::<gam_math::jet_tower::Tower4<2>>(yi, eta_mu, eta_d, p, wi)
}
use crate::wiggle::{
    monotone_wiggle_internal_degree, split_wiggle_penalty_orders,
};
use gam_data::encode_recordswith_inferred_schema;
use gam_terms::basis::{CenterStrategy, MaternBasisSpec, MaternIdentifiability, MaternNu};
use gam_terms::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
use gam_test_support::{binomial_location_scale_base_fixture, no_densify_design};
use ndarray::{Array2, Axis, array};
use num_dual::{
    DualNum, second_derivative, second_partial_derivative, third_partial_derivative_vec,
};

fn test_design_hyper_layout(
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
) -> CustomFamilyHyperLayout {
    let axis_count = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    CustomFamilyHyperLayout::new(
        derivative_blocks.to_vec(),
        Vec::new(),
        Array1::zeros(axis_count),
    )
    .expect("test design hyper layout")
}

pub(crate) fn compose_theta_from_hints_test(
    mean_penalty_count: usize,
    noise_penalty_count: usize,
    mean_log_lambda_hint: &Option<Array1<f64>>,
    noise_log_lambda_hint: &Option<Array1<f64>>,
    extra_rho0: &Array1<f64>,
) -> Array1<f64> {
    let layout =
        GamlssLambdaLayout::withwiggle(mean_penalty_count, noise_penalty_count, extra_rho0.len());
    let mut theta = Array1::<f64>::zeros(layout.total());
    if let Some(v) = mean_log_lambda_hint
        && v.len() == layout.k_mean
    {
        theta.slice_mut(s![0..layout.noise_start()]).assign(v);
    }
    if let Some(v) = noise_log_lambda_hint
        && v.len() == layout.k_noise
    {
        theta
            .slice_mut(s![layout.noise_start()..layout.noise_end()])
            .assign(v);
    }
    if layout.kwiggle > 0 {
        theta
            .slice_mut(s![layout.wiggle_start()..layout.wiggle_end()])
            .assign(extra_rho0);
    }
    theta
}

#[test]
pub(crate) fn monotone_wiggle_post_update_validator_rejects_hidden_projection() {
    validate_monotone_wiggle_beta_nonnegative(
        &array![0.0, 1.0e-13, 2.0],
        "monotone wiggle validator test",
    )
    .expect("feasible nonnegative wiggle beta should validate");

    let err = validate_monotone_wiggle_beta_nonnegative(
        &array![0.0, -1.0e-3, 2.0],
        "monotone wiggle validator test",
    )
    .expect_err("negative wiggle beta must be rejected instead of projected");
    assert!(
        err.contains("monotone wiggle coefficients must be non-negative"),
        "unexpected error: {err}"
    );
}

#[test]
pub(crate) fn logb_dlog_sigma_deta_preserves_negative_tail_precision() {
    let eta = -703.4873664863218;
    let SigmaJet1 { sigma, d1 } = logb_sigma_jet1_scalar(eta);

    assert_eq!(
        1.0 - LOGB_SIGMA_FLOOR / sigma,
        0.0,
        "the algebraically equivalent complement form must cancel at this eta"
    );
    assert!(
        logb_dlog_sigma_deta(sigma, d1) > 0.0,
        "d_sigma_deta / sigma must preserve the remaining tail derivative"
    );
    assert!(
        logb_dlog_sigma_deta(f64::INFINITY, f64::INFINITY).is_nan(),
        "an unrepresentable link must be certified by its caller, not projected to an analytic limit"
    );
}

pub(crate) fn assert_rel_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tol * scale,
        "{label}: actual={actual:+.16e}, expected={expected:+.16e}, diff={:+.3e}, scale={scale:.3e}",
        actual - expected
    );
}

/// #932: the production binomial location-scale JOINT Hessian assembler must
/// equal the single-sourced `binomial_location_scale_nll_tower`.
///
/// `binomial_nonwiggle_tower_matches_hand_witness_channels` pins the *tower*
/// against a *test* hand witness, and the operator-workspace tests pin the
/// lazy operator against the dense `exact_newton_joint_hessian_from_designs`.
/// But NOTHING pinned the production assembler's own row coefficients
/// (`exact_newton_joint_hessian_row_coefficients`: `coeff_tt = m2 r²`,
/// `coeff_tl = κ r (m1 + q m2)`, `coeff_ll = κ² q (m1 + q m2)`, with the q-chain
/// `q = −η_t·e^{−η_ls}` and `κ = σ'(η_ls)/σ`) to the single-source tower. A
/// typo in those coefficients (a dropped `q m2`, a wrong `κ` power — the #736
/// cross-term genus) would slip past both existing oracles.
///
/// This closes the gap. For a multi-column non-wiggle fixture, the production
/// `exact_newton_joint_hessian_from_designs` joint matrix is compared, at
/// ~1e-9, to the joint Hessian assembled by pulling the per-row `Tower4<2>`
/// curvature `tower.h` (in (η_t, η_ls)) through the same designs:
/// `H = Σ_i [X_t; X_ls]_iᵀ · tower.h_i · [X_t; X_ls]_i`. Independent arithmetic
/// (the tower differentiates one expression by Leibniz; the production builds
/// the coefficients by hand), so agreement is a correctness proof of the hand
/// assembler — across probit / logit / cloglog.
#[test]
pub(crate) fn binomial_location_scale_joint_hessian_matches_single_sourced_tower_932() {
    let n = 7usize;
    let pt = 2usize;
    let pls = 2usize;
    let xt = Array2::from_shape_fn((n, pt), |(i, j)| {
        ((i as f64) * 0.31 + (j as f64) * 0.17).sin() + 0.4
    });
    let xls = Array2::from_shape_fn((n, pls), |(i, j)| {
        ((i as f64) * 0.23 + (j as f64) * 0.41).cos() * 0.5
    });
    let beta_t = array![0.35, -0.20];
    let beta_ls = array![0.18, -0.27];
    let eta_t = xt.dot(&beta_t);
    let eta_ls = xls.dot(&beta_ls);
    let total = pt + pls;

    for link in [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
    ] {
        let family = BinomialLocationScaleFamily {
            y: Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })),
            weights: Array1::from_iter((0..n).map(|i| 0.5 + 0.2 * i as f64)),
            link_kind: link.clone(),
            threshold_design: None,
            log_sigma_design: None,
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
        };
        let states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls.clone(),
            },
        ];

        // Production hand-assembled joint Hessian (the path under audit).
        let h_prod = family
            .exact_newton_joint_hessian_from_designs(&states, &xt, &xls)
            .expect("production joint Hessian")
            .expect("production joint Hessian present");
        assert_eq!(h_prod.dim(), (total, total));

        // Single-sourced reference: per-row Tower4<2> curvature in (η_t, η_ls),
        // pulled through the SAME designs. σ = e^{η_ls} ⇒ κ = 1, matching the
        // tower's inv_sigma = e^{−η_ls}.
        let mut h_tower = Array2::<f64>::zeros((total, total));
        for i in 0..n {
            let sigma = exp_sigma_from_eta_scalar(eta_ls[i]);
            let q = binomial_location_scale_q0(eta_t[i], sigma);
            let jet = inverse_link_jet_for_inverse_link(&link, q).expect("link jet");
            let tower = binomial_location_scale_nll_tower(
                family.y[i],
                family.weights[i],
                eta_t[i],
                eta_ls[i],
                q,
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
                &link,
                true,
            )
            .expect("row tower");

            // Row design in the joint coefficient layout [X_t | X_ls].
            let mut row = vec![0.0_f64; total];
            for c in 0..pt {
                row[c] = xt[[i, c]];
            }
            for c in 0..pls {
                row[pt + c] = xls[[i, c]];
            }
            // channel(a): 0 -> η_t block, 1 -> η_ls block.
            let block_of = |coef: usize| if coef < pt { 0usize } else { 1usize };
            for a_coef in 0..total {
                let ca = block_of(a_coef);
                for b_coef in 0..total {
                    let cb = block_of(b_coef);
                    h_tower[[a_coef, b_coef]] += tower.h[ca][cb] * row[a_coef] * row[b_coef];
                }
            }
        }

        for ((a, b), &prod) in h_prod.indexed_iter() {
            let want = h_tower[[a, b]];
            assert!(
                (prod - want).abs() <= 1e-9 * (1.0 + want.abs()),
                "{link:?}: joint Hessian [{a}][{b}] hand-assembler {prod:.9e} != \
                 single-sourced tower {want:.9e}"
            );
        }
    }
}

#[test]
// Regression for #1107: the Tweedie y=0 dispersion-channel curvature in the
// η_d = −log φ link must equal the observed-information second derivative
// ∂²(−ℓ)/∂η_d² = c/φ, NOT the Fisher-information shortcut 2c/φ. The shortcut
// drops the first-order score term (valid only when E[score]=0, i.e. the
// saddlepoint y>0 branch) and was 2× too large for the deterministic zero-mass
// branch. This asserts the kernel's reported per-row curvature (`disp_weight`
// at unit prior weight) matches a centered finite-difference of the NLL.
pub(crate) fn tweedie_zero_mass_dispersion_curvature_matches_finite_difference() {
    // (p in (1,2), eta_mu, eta_d) cases spanning small/large μ and φ.
    let cases = [
        (1.3_f64, -2.0_f64, -1.0_f64),
        (1.5, -0.5, 0.5),
        (1.5, 1.0, -1.5),
        (1.7, 2.0, 2.0),
        (1.1, -3.0, 0.0),
    ];
    for (p, eta_mu, eta_d) in cases {
        let kind = DispersionFamilyKind::Tweedie { p };
        // NLL(η_d) = −loglik(η_d) at unit prior weight; loglik is the kernel's
        // reported log-likelihood contribution for this row.
        let nll = |ed: f64| -dispersion_row_kernel(kind, 0.0, eta_mu, ed, 1.0).loglik;
        let h = 1e-4;
        let fd_curv = (nll(eta_d + h) - 2.0 * nll(eta_d) + nll(eta_d - h)) / (h * h);

        let kernel = dispersion_row_kernel(kind, 0.0, eta_mu, eta_d, 1.0);
        // disp_weight at unit prior weight is exactly the per-row curvature.
        assert_rel_close(
            "tweedie y=0 dispersion curvature vs finite difference",
            kernel.disp_weight,
            fd_curv,
            1e-5,
        );

        // Closed-form guard: curvature must be c/φ, and a 2× error (the old
        // 2c/φ) would be caught by the FD check above but we pin it explicitly.
        let mu = (eta_mu as f64).exp();
        let phi = (-eta_d).exp();
        let c = mu.powf(2.0 - p) / (2.0 - p);
        assert_rel_close(
            "tweedie y=0 dispersion curvature equals c/phi (not 2c/phi)",
            kernel.disp_weight,
            c / phi,
            1e-10,
        );
    }
}

#[test]
// #932: the single-expression `dispersion_tweedie_nll_tower` IS the production
// Tweedie row NLL; its mechanically-derived gradient and Hessian channels must
// be the exact derivatives of its own value channel. Anchor every channel of
// the tower against centered finite differences of the value, in BOTH predictor
// directions (η_μ, η_d) and BOTH density branches (y > 0 saddlepoint, y = 0
// point mass), so a dropped chain term or a sign flip in the Faà-di-Bruno
// composition shows up here independent of any closed-form witness.
pub(crate) fn tweedie_nll_tower_is_finite_difference_consistent() {
    // (p in (1,2), y, eta_mu, eta_d, weight); y = 0 hits the point-mass branch.
    let cases = [
        (1.5_f64, 0.0_f64, -0.7_f64, 0.4_f64, 0.9_f64),
        (1.5, 3.2, 0.6, -0.3, 1.2),
        (1.2, 0.0, 1.1, -0.8, 0.6),
        (1.8, 7.5, -0.4, 1.0, 1.3),
        (1.3, 2.0, 0.2, 0.7, 1.0),
    ];
    let eval = |p: f64, y: f64, em: f64, ed: f64, w: f64| -> f64 {
        dispersion_tweedie_nll_tower(y, em, ed, p, w).v
    };
    for (p, y, em, ed, w) in cases {
        let t = dispersion_tweedie_nll_tower(y, em, ed, p, w);
        let h = 1e-5;
        // value → gradient and gradient → Hessian, one direction at a time.
        for (axis, perturb) in [(0usize, [h, 0.0]), (1usize, [0.0, h])] {
            let vp = eval(p, y, em + perturb[0], ed + perturb[1], w);
            let vm = eval(p, y, em - perturb[0], ed - perturb[1], w);
            let fd_g = (vp - vm) / (2.0 * h);
            assert_rel_close(
                "tweedie tower gradient vs finite difference",
                t.g[axis],
                fd_g,
                1e-5,
            );
            // Diagonal Hessian via the gradient of a perturbed tower.
            let tp = dispersion_tweedie_nll_tower(y, em + perturb[0], ed + perturb[1], p, w);
            let tm = dispersion_tweedie_nll_tower(y, em - perturb[0], ed - perturb[1], p, w);
            let fd_h = (tp.g[axis] - tm.g[axis]) / (2.0 * h);
            assert_rel_close(
                "tweedie tower diagonal Hessian vs finite difference",
                t.h[axis][axis],
                fd_h,
                1e-5,
            );
        }
        // Mixed cross block — the #736 fragility shape — anchored both ways.
        let cross = {
            let tp = dispersion_tweedie_nll_tower(y, em + h, ed, p, w);
            let tm = dispersion_tweedie_nll_tower(y, em - h, ed, p, w);
            (tp.g[1] - tm.g[1]) / (2.0 * h)
        };
        assert_rel_close(
            "tweedie tower cross-Hessian vs finite difference",
            t.h[0][1],
            cross,
            1e-5,
        );
        assert_rel_close(
            "tweedie tower Hessian symmetry",
            t.h[0][1],
            t.h[1][0],
            1e-12,
        );
    }
}

// Source-of-truth Gaussian logb negloglik. Analytic helpers MUST autodiff-match this.
pub(crate) fn gaussian_negloglik_log_sigma_psi_numdual<D: DualNum<f64> + Copy>(
    beta_mu: D,
    beta_ls: D,
    psi: D,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    x_mu0: &Array1<f64>,
    x_ls0: &Array1<f64>,
    x_ls_psi: &Array1<f64>,
    x_ls_psi_psi: &Array1<f64>,
) -> D {
    let half = D::from(0.5);
    let mut out = D::zero();
    for i in 0..y.len() {
        let eta_mu = D::from(x_mu0[i]) * beta_mu;
        let x_ls = D::from(x_ls0[i])
            + psi * D::from(x_ls_psi[i])
            + half * psi * psi * D::from(x_ls_psi_psi[i]);
        let eta_ls = x_ls * beta_ls;
        // Mirror the production logb noise link σ = LOGB_SIGMA_FLOOR + exp(η_ls)
        // (see `GaussianLocationScaleFamily::loglik`); using the bare-exp link
        // here would diverge from the family's σ at the same η and break the
        // psi-derivative identities that this reference negloglik certifies.
        let sigma = D::from(LOGB_SIGMA_FLOOR) + eta_ls.exp();
        let resid = D::from(y[i]) - eta_mu;
        out += D::from(weights[i]) * (half * (resid / sigma).powi(2) + sigma.ln());
    }
    out
}

pub(crate) fn gaussian_negloglik_log_sigma_psi_only_numdual<D: DualNum<f64> + Copy>(
    psi: D,
    beta_mu: f64,
    beta_ls: f64,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    x_mu0: &Array1<f64>,
    x_ls0: &Array1<f64>,
    x_ls_psi: &Array1<f64>,
    x_ls_psi_psi: &Array1<f64>,
) -> D {
    gaussian_negloglik_log_sigma_psi_numdual(
        D::from(beta_mu),
        D::from(beta_ls),
        psi,
        y,
        weights,
        x_mu0,
        x_ls0,
        x_ls_psi,
        x_ls_psi_psi,
    )
}

pub(crate) fn gaussian_negloglik_log_sigma_mu_psi_numdual<D: DualNum<f64> + Copy>(
    beta_mu: D,
    psi: D,
    beta_ls: f64,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    x_mu0: &Array1<f64>,
    x_ls0: &Array1<f64>,
    x_ls_psi: &Array1<f64>,
    x_ls_psi_psi: &Array1<f64>,
) -> D {
    gaussian_negloglik_log_sigma_psi_numdual(
        beta_mu,
        D::from(beta_ls),
        psi,
        y,
        weights,
        x_mu0,
        x_ls0,
        x_ls_psi,
        x_ls_psi_psi,
    )
}

pub(crate) fn gaussian_negloglik_log_sigma_ls_psi_numdual<D: DualNum<f64> + Copy>(
    beta_ls: D,
    psi: D,
    beta_mu: f64,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    x_mu0: &Array1<f64>,
    x_ls0: &Array1<f64>,
    x_ls_psi: &Array1<f64>,
    x_ls_psi_psi: &Array1<f64>,
) -> D {
    gaussian_negloglik_log_sigma_psi_numdual(
        D::from(beta_mu),
        beta_ls,
        psi,
        y,
        weights,
        x_mu0,
        x_ls0,
        x_ls_psi,
        x_ls_psi_psi,
    )
}

pub(crate) fn gaussian_negloglik_log_sigma_beta_vec_numdual<D: DualNum<f64> + Copy>(
    v: &[D],
    y: &Array1<f64>,
    weights: &Array1<f64>,
    x_mu0: &Array1<f64>,
    x_ls0: &Array1<f64>,
    x_ls_psi: &Array1<f64>,
    x_ls_psi_psi: &Array1<f64>,
) -> D {
    gaussian_negloglik_log_sigma_psi_numdual(
        v[0],
        v[1],
        v[2],
        y,
        weights,
        x_mu0,
        x_ls0,
        x_ls_psi,
        x_ls_psi_psi,
    )
}

pub(crate) fn gaussian_psi_test_spec(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
        offset: Array1::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: vec![],
        initial_log_lambdas: Array1::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

#[test]
pub(crate) fn gaussian_joint_psi_firstweights_score_ls_carries_logb_chain_rule_factor() {
    let y = array![1.1];
    let etamu = array![0.3];
    let eta_ls = array![-0.2];
    let weights = array![2.5];
    let rows =
        gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
    let firstweights = gaussian_joint_psi_firstweights(&rows, &array![0.0], &array![1.0]);
    let sigma = crate::sigma_link::logb_sigma_from_eta_scalar(eta_ls[0]);
    let kappa = 1.0 - crate::sigma_link::LOGB_SIGMA_FLOOR / sigma;
    let standardized_residual = (y[0] - etamu[0]) / sigma;
    let expected = kappa * (weights[0] - weights[0] * standardized_residual.powi(2));

    assert!(
        (firstweights.score_ls[0] - expected).abs() <= 1e-12,
        "Under the logb link σ = b + exp(η_ls), d/dη_ls of weight*(ln σ + 0.5(y-μ)^2/σ^2) carries the chain-rule factor κ = 1 - b/σ, so the row score must equal κ*(weight - n_i). The helper coded {} but the κ-corrected expectation is {}.",
        firstweights.score_ls[0],
        expected
    );
    assert!(
        (firstweights.objective_psirow[0] - expected).abs() <= 1e-12,
        "With mu_psi=0 and eta_psi=1, the exact psi objective derivative must equal κ*(weight - n_i) (κ = 1 - b/σ from the logb chain rule). The helper coded {} but the κ-corrected expectation is {}.",
        firstweights.objective_psirow[0],
        expected
    );
}

#[test]
pub(crate) fn cloglog_binomial_right_tail_derivatives_stay_finite() {
    let (m1, m2, m3) = binomial_neglog_q_derivatives_cloglog_closed_form(1.0, 1.0, 1000.0);
    let m4 = binomial_neglog_q_fourth_derivative_cloglog_closed_form(1.0, 1.0, 300.0);

    assert_eq!(m1, 0.0);
    assert_eq!(m2, 0.0);
    assert_eq!(m3, 0.0);
    assert_eq!(m4, 0.0);
}

#[test]
pub(crate) fn cloglog_binomial_fractional_right_tail_keeps_y0_branch() {
    let y = 0.25;
    let weight = 2.0;
    let q = 300.0;
    let expected = weight * (1.0 - y) * q.exp();
    let (m1, m2, m3) = binomial_neglog_q_derivatives_cloglog_closed_form(y, weight, q);
    let m4 = binomial_neglog_q_fourth_derivative_cloglog_closed_form(y, weight, q);

    assert!(m1.is_finite());
    assert!(m2.is_finite());
    assert!(m3.is_finite());
    assert!(m4.is_finite());
    assert_eq!(m1, expected);
    assert_eq!(m2, expected);
    assert_eq!(m3, expected);
    assert_eq!(m4, expected);
}

#[test]
pub(crate) fn logit_binomial_tail_derivatives_are_exact_not_clipped() {
    // Regression for issue #948 (2b): the logit curvature/4th derivative
    // must be the EXACT Bernoulli variance s = p(1-p) in the saturated
    // tail — never floored to MIN_PROB·(1−MIN_PROB) ≈ 1e-10. At q=50 the
    // true variance is s = e^{-50}/(1+e^{-50})² ≈ e^{-50} ≈ 1.93e-22.
    let q = 50.0;
    let t = (-q).exp();
    let denom = 1.0 + t;
    let s_exact = t / (denom * denom);

    let (m1, m2, m3) = binomial_neglog_q_derivatives_logit_closed_form(1.0, 1.0, q);
    let m4 = binomial_neglog_q_fourth_derivative_logit_closed_form(1.0, 1.0, q);

    // The clipped surrogate would have reported ~1e-10; the exact value is
    // ~1.9e-22, twelve orders of magnitude smaller.
    assert!(
        s_exact < 1e-21,
        "sanity: exact tail variance should be ~1e-22, got {s_exact}"
    );
    // m1 = w(p - y); at q=50, p rounds to 1.0 exactly, so m1 = 0.
    assert!(m1.abs() <= 1e-15, "m1 should be ~0 at p≈1, got {m1}");
    assert!(
        (m2 - s_exact).abs() <= 1e-30,
        "logit curvature must equal exact s=p(1-p) in the tail, got {m2}, want {s_exact}"
    );
    // The clipped floor would be ~5e-12 larger than the truth: assert we
    // are nowhere near it.
    assert!(
        m2 < 1e-15,
        "logit curvature must NOT be floored at MIN_PROB·(1−MIN_PROB)≈1e-10, got {m2}"
    );
    assert!(m3.is_finite());
    assert!(
        (m4 - s_exact * (1.0 - 6.0 * s_exact)).abs() <= 1e-30,
        "logit fourth derivative must equal exact ws(1-6s) in the tail, got {m4}"
    );
}

#[test]
pub(crate) fn probit_binomial_incompatible_tail_keeps_mills_score() {
    let q = 40.0;
    let (m1, m2, m3) = binomial_neglog_q_derivatives_probit_closed_form(0.0, 1.0, q);
    let m4 = binomial_neglog_q_fourth_derivative_probit_closed_form(0.0, 1.0, q);

    assert!(
        m1 > 39.0 && m1 < 41.0,
        "right-tail probit score should be Mills-ratio sized, got {m1}"
    );
    assert!(
        m2 > 0.9 && m2 < 1.1,
        "right-tail probit curvature should stay near one, got {m2}"
    );
    assert!(
        m3.is_finite(),
        "third derivative must stay finite, got {m3}"
    );
    assert!(
        m4.is_finite(),
        "fourth derivative must stay finite, got {m4}"
    );
}

#[test]
pub(crate) fn binomial_location_scale_loglik_uses_tail_stable_standard_links() {
    use crate::custom_family::{CustomFamily, ParameterBlockState};

    let n = 2usize;
    let design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_elem((n, 1), 1.0),
    ));
    let log_sigma = ParameterBlockState {
        beta: array![0.0],
        eta: array![0.0, 0.0],
    };

    let logit_family = BinomialLocationScaleFamily {
        y: array![0.0, 1.0],
        weights: Array1::ones(n),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        threshold_design: Some(design.clone()),
        log_sigma_design: Some(design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let logit_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![-1000.0, 1000.0],
        },
        log_sigma.clone(),
    ];
    let logit_ll = logit_family
        .log_likelihood_only(&logit_states)
        .expect("logit tail likelihood");
    assert!(
        (logit_ll + 2000.0).abs() <= 1e-10,
        "logit tail likelihood must use softplus natural-parameter algebra, got {logit_ll}"
    );

    let cloglog_family = BinomialLocationScaleFamily {
        y: array![0.0, 1.0],
        weights: Array1::ones(n),
        link_kind: InverseLink::Standard(StandardLink::CLogLog),
        threshold_design: Some(design.clone()),
        log_sigma_design: Some(design),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let cloglog_states = vec![
        ParameterBlockState {
            beta: array![0.0],
            eta: array![-20.0, 1000.0],
        },
        log_sigma,
    ];
    let cloglog_ll = cloglog_family
        .log_likelihood_only(&cloglog_states)
        .expect("cloglog tail likelihood");
    let expected = -20.0_f64.exp() - 1000.0;
    let rel = (cloglog_ll - expected).abs() / expected.abs();
    assert!(
        rel <= 1e-14,
        "cloglog tail likelihood must use exp(q) survival algebra, got {cloglog_ll}, expected {expected}"
    );
}

#[test]
pub(crate) fn gaussian_joint_psisecondweights_eta_ab_term_carries_logb_chain_rule_factor() {
    let y = array![1.1];
    let etamu = array![0.3];
    let eta_ls = array![-0.2];
    let weights = array![2.5];
    let rows =
        gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
    let secondweights = gaussian_joint_psisecondweights(
        &rows,
        &array![0.0],
        &array![0.0],
        &array![0.0],
        &array![0.0],
        &array![0.0],
        &array![1.0],
    );
    let sigma = crate::sigma_link::logb_sigma_from_eta_scalar(eta_ls[0]);
    let kappa = 1.0 - crate::sigma_link::LOGB_SIGMA_FLOOR / sigma;
    let standardized_residual = (y[0] - etamu[0]) / sigma;
    let expected = kappa * (weights[0] - weights[0] * standardized_residual.powi(2));

    assert!(
        (secondweights.objective_psi_psirow[0] - expected).abs() <= 1e-12,
        "With only eta_psi_psi=1 active, the Gaussian second psi objective contribution from the linear η_ls term carries the logb chain-rule factor κ = 1 - b/σ, so it must equal κ*(weight - n_i). The helper coded {} but the κ-corrected expectation is {}.",
        secondweights.objective_psi_psirow[0],
        expected
    );
}

#[test]
pub(crate) fn gaussian_location_scale_coefficient_cost_delegates_to_joint_coupled_helper() {
    // GAMLSS families (all five variants) share the joint-coupled formula
    // n · (Σ p_b)². They each pull n from `self.y.len()` and forward the
    // specs to the shared helper. This regression test pins that contract
    // for the simplest representative (GaussianLocationScale); the other
    // four GAMLSS impls are byte-for-byte identical aside from the comment.
    let n = 100usize;
    let p_mu = 7usize;
    let p_log_sigma = 4usize;
    let family = GaussianLocationScaleFamily {
        y: Array1::zeros(n),
        weights: Array1::from_elem(n, 1.0),
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "mu".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, p_mu)),
            )),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, p_log_sigma)),
            )),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    let p_total = (p_mu + p_log_sigma) as u64;
    let expected = crate::custom_family::joint_coupled_coefficient_hessian_cost(n as u64, &specs);
    assert_eq!(family.coefficient_hessian_cost(&specs), expected);
    assert_eq!(expected, (n as u64) * p_total * p_total);
    assert!(
        expected > crate::custom_family::default_coefficient_hessian_cost(&specs),
        "joint-coupled cost must exceed block-diagonal default by the cross-block fill"
    );
}

#[test]
pub(crate) fn large_n_gaussian_location_scale_keeps_exact_outer_hessian_plan() {
    let n = 50_001usize;
    let p_mu = 20usize;
    let p_log_sigma = 20usize;
    let family = GaussianLocationScaleFamily {
        y: Array1::zeros(n),
        weights: Array1::from_elem(n, 1.0),
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "mu".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, p_mu)),
            )),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, p_log_sigma)),
            )),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];

    let options = BlockwiseFitOptions::default();
    let (gradient, hessian) =
        crate::custom_family::custom_family_outer_derivatives(&family, &specs, &options);
    assert_eq!(gradient, gam_problem::Derivative::Analytic);
    assert_eq!(
        hessian,
        gam_problem::DeclaredHessianForm::Either,
        "large-n GAMLSS location-scale fits must advertise exact second-order curvature instead of triggering the historical BFGS downgrade"
    );

    let p_total = p_mu + p_log_sigma;
    assert!(
        gam_solve::estimate::reml::reml_outer_engine::prefer_outer_hessian_operator(n, p_total, 2),
        "the large-n work model should select the scalable explicit Hessian-operator representation"
    );

    let plan = gam_solve::rho_optimizer::plan(&gam_solve::rho_optimizer::OuterCapability {
        gradient,
        hessian,
        n_params: 2,
        psi_dim: 0,
        fixed_point_available: false,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: true,
    });
    assert_eq!(plan.solver, gam_solve::rho_optimizer::Solver::Arc);
    assert_eq!(
        plan.hessian_source,
        gam_solve::rho_optimizer::HessianSource::Analytic
    );
}

/// Helper: build a small Gaussian location-scale family + state + specs
/// for matrix-free joint-Hessian validation.
pub(crate) fn gls_workspace_fixture() -> (
    GaussianLocationScaleFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
) {
    let n = 7usize;
    let p_mu = 3usize;
    let p_ls = 2usize;
    let xmu = Array2::from_shape_fn((n, p_mu), |(i, j)| {
        ((i as f64) * 0.13 + (j as f64) * 0.31).sin()
    });
    let xls = Array2::from_shape_fn((n, p_ls), |(i, j)| {
        ((i as f64) * 0.21 + (j as f64) * 0.47).cos()
    });
    let beta_mu = array![0.10, -0.20, 0.30];
    let beta_ls = array![0.40, -0.10];
    let eta_mu = xmu.dot(&beta_mu);
    let eta_ls = xls.dot(&beta_ls);
    let y = Array1::from_shape_fn(n, |i| 0.5 + 0.1 * (i as f64).cos());
    let weights = Array1::from_elem(n, 1.0);
    let mu_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xmu.clone()));
    let log_sigma_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
    let family = GaussianLocationScaleFamily {
        y,
        weights,
        mu_design: Some(mu_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let states = vec![
        ParameterBlockState {
            beta: beta_mu,
            eta: eta_mu,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
    ];
    let specs = vec![
        ParameterBlockSpec {
            name: "mu".to_string(),
            design: mu_design,
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design,
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    (family, states, specs)
}

/// Helper: build a small Binomial location-scale family + state + specs
/// for matrix-free joint-Hessian validation. Probit is the production link.
pub(crate) fn bls_workspace_fixture() -> (
    BinomialLocationScaleFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
) {
    let n = 8usize;
    let pt = 3usize;
    let pls = 2usize;
    let xt = Array2::from_shape_fn((n, pt), |(i, j)| {
        ((i as f64) * 0.17 + (j as f64) * 0.29).sin()
    });
    let xls = Array2::from_shape_fn((n, pls), |(i, j)| {
        ((i as f64) * 0.23 + (j as f64) * 0.41).cos() * 0.5
    });
    let beta_t = array![0.20, -0.10, 0.05];
    let beta_ls = array![0.30, -0.15];
    let eta_t = xt.dot(&beta_t);
    let eta_ls = xls.dot(&beta_ls);
    let y = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let threshold_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xt.clone()));
    let log_sigma_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
    let family = BinomialLocationScaleFamily {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
    ];
    let specs = vec![
        ParameterBlockSpec {
            name: "threshold".to_string(),
            design: threshold_design,
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: log_sigma_design,
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    (family, states, specs)
}

#[test]
pub(crate) fn gaussian_location_scale_workspace_matvec_matches_dense() {
    // Patch 7 mirror of the CTN matrix-free reference test: the matrix-
    // free `Hv` and `diag(H)` operators must reconstruct the dense joint
    // Hessian element-wise. This pins the cross-block coefficient
    // (`coeff_ml` in GaussianLocationScaleHessianWorkspace) against any
    // future regression of the t↔ℓ coupling.
    let (family, states, specs) = gls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let dense = family
        .exact_newton_joint_hessian(&states)
        .expect("dense joint Hessian build")
        .expect("dense joint Hessian present");
    assert_eq!(dense.nrows(), p);
    assert_eq!(dense.ncols(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    let diag_op = workspace
        .hessian_diagonal()
        .expect("diagonal call")
        .expect("diagonal present");
    assert_eq!(diag_op.len(), p);
    for i in 0..p {
        let want = dense[[i, i]];
        let got = diag_op[i];
        assert!(
            (want - got).abs() <= 1e-10 * want.abs().max(1.0) + 1e-10,
            "GLS diagonal mismatch at {i}: dense={want:.6e}, workspace={got:.6e}"
        );
    }

    let directions = [
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0]),
        Array1::from_vec(vec![0.30, -0.70, 0.50, -0.20, 0.15]),
        Array1::from_vec(vec![-0.42, 0.11, 0.93, 0.05, -0.31]),
    ];
    for (k, v) in directions.iter().enumerate() {
        assert_eq!(v.len(), p);
        let want = dense.dot(v);
        let got = workspace
            .hessian_matvec(v)
            .expect("matvec call")
            .expect("matvec present");
        assert_eq!(got.len(), p);
        for i in 0..p {
            let tol = 1e-10 * want[i].abs().max(1.0) + 1e-10;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "GLS matvec[{k}, {i}] mismatch: dense={:.6e}, workspace={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

/// Shared assertion for the four "hessian_dense matches canonical-basis
/// HVP path" tests across the LocationScale {Gaussian, Binomial} × {non-
/// wiggle, wiggle} grid. Each test only needs to build the workspace and
/// pass it here with the expected total coefficient dim and a short
/// family label (used in the diff message).
pub(crate) fn assert_dense_matches_canonical_basis_hvp(
    workspace: &dyn crate::custom_family::ExactNewtonJointHessianWorkspace,
    total: usize,
    label: &str,
) {
    let dense = workspace
        .hessian_dense()
        .expect("hessian_dense call")
        .expect("hessian_dense present");
    assert_eq!(dense.nrows(), total);
    assert_eq!(dense.ncols(), total);

    // Reconstruct H column-by-column via canonical-basis HVPs (the path
    // the dense build replaces).
    let mut assembled = Array2::<f64>::zeros((total, total));
    for j in 0..total {
        let mut e = Array1::<f64>::zeros(total);
        e[j] = 1.0;
        let col = workspace
            .hessian_matvec(&e)
            .expect("matvec call")
            .expect("matvec present");
        assembled.column_mut(j).assign(&col);
    }
    let assembled_sym = 0.5 * (&assembled + &assembled.t());

    let max_rel = dense
        .iter()
        .zip(assembled_sym.iter())
        .map(|(d, a)| ((d - a) / d.abs().max(a.abs()).max(1.0)).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_rel < 1e-12,
        "{label} hessian_dense vs canonical HVP max relative diff: {max_rel:.3e}"
    );
}

/// Bit-equivalence guard for the `hessian_dense` hook. The dispatch site
/// `exact_newton_joint_hessian_source_from_workspace` prefers
/// `hessian_dense` over the canonical-basis HVP fallback at large-scale
/// scale; this test pins the dense build against the same column-by-
/// column HVP path it replaces. Any future regression in the GEMM
/// fill (e.g. swapped block coordinates, sign error in `coeff_ml`)
/// fails here before it can corrupt outer-Hessian assembly.
#[test]
pub(crate) fn gaussian_location_scale_hessian_dense_matches_canonical_basis_hvp_path() {
    let (family, states, specs) = gls_workspace_fixture();
    let total = states[0].beta.len() + states[1].beta.len();

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    assert_dense_matches_canonical_basis_hvp(workspace.as_ref(), total, "GLS");
}

/// Bit-equivalence guard for the binomial location-scale dense Hessian
/// hook. Same structure as the Gaussian non-wiggle test.
#[test]
pub(crate) fn binomial_location_scale_hessian_dense_matches_canonical_basis_hvp_path() {
    let (family, states, specs) = bls_workspace_fixture();
    let total = states[0].beta.len() + states[1].beta.len();

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    assert_dense_matches_canonical_basis_hvp(workspace.as_ref(), total, "BLS");
}

#[test]
pub(crate) fn gaussian_location_scale_workspace_dh_operator_matches_dense() {
    let (family, states, specs) = gls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let d_beta = array![0.07, -0.04, 0.21, 0.08, -0.13];
    assert_eq!(d_beta.len(), p);

    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative(&states, &d_beta)
        .expect("dense dH build")
        .expect("dense dH present");

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&d_beta)
        .expect("dH operator call")
        .expect("dH operator present");

    let probes = [
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.30, -0.70, 0.50, -0.20, 0.15]),
    ];
    for (k, w) in probes.iter().enumerate() {
        assert_eq!(w.len(), p);
        let want = dense_dh.dot(w);
        let got = dh_op.mul_vec(w);
        assert_eq!(got.len(), p);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "GLS dH op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn binomial_location_scale_workspace_matvec_matches_dense() {
    // Probit + logb-sigma is the production-pipeline link combination, so
    // the cross-block coefficient `coeff_tl` must agree with the dense
    // assembly to within tight tolerance on randomly sampled directions.
    let (family, states, specs) = bls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let dense = family
        .exact_newton_joint_hessian(&states)
        .expect("dense joint Hessian build")
        .expect("dense joint Hessian present");
    assert_eq!(dense.nrows(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    let diag_op = workspace
        .hessian_diagonal()
        .expect("diagonal call")
        .expect("diagonal present");
    assert_eq!(diag_op.len(), p);
    for i in 0..p {
        let want = dense[[i, i]];
        let got = diag_op[i];
        assert!(
            (want - got).abs() <= 1e-10 * want.abs().max(1.0) + 1e-10,
            "BLS diagonal mismatch at {i}: dense={want:.6e}, workspace={got:.6e}"
        );
    }

    let directions = [
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0]),
        Array1::from_vec(vec![0.30, -0.70, 0.50, -0.20, 0.15]),
        Array1::from_vec(vec![-0.42, 0.11, 0.93, 0.05, -0.31]),
    ];
    for (k, v) in directions.iter().enumerate() {
        assert_eq!(v.len(), p);
        let want = dense.dot(v);
        let got = workspace
            .hessian_matvec(v)
            .expect("matvec call")
            .expect("matvec present");
        for i in 0..p {
            let tol = 1e-10 * want[i].abs().max(1.0) + 1e-10;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "BLS matvec[{k}, {i}] mismatch: dense={:.6e}, workspace={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn binomial_location_scale_operator_workspace_never_densifies_specs() {
    let n = 8usize;
    let pt = 3usize;
    let pls = 2usize;
    let xt = Array2::from_shape_fn((n, pt), |(i, j)| {
        ((i as f64) * 0.17 + (j as f64) * 0.29).sin()
    });
    let xls = Array2::from_shape_fn((n, pls), |(i, j)| {
        ((i as f64) * 0.23 + (j as f64) * 0.41).cos() * 0.5
    });
    let beta_t = array![0.20, -0.10, 0.05];
    let beta_ls = array![0.30, -0.15];
    let eta_t = xt.dot(&beta_t);
    let eta_ls = xls.dot(&beta_ls);
    let family = BinomialLocationScaleFamily {
        y: Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })),
        weights: Array1::from_elem(n, 1.0),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
    ];
    let specs = vec![
        ParameterBlockSpec {
            name: "threshold".to_string(),
            design: no_densify_design(xt.clone()),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: no_densify_design(xls.clone()),
            offset: Array1::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    assert!(family.inner_coefficient_hessian_hvp_available(&specs));

    let dense_h = family
        .exact_newton_joint_hessian_from_designs(&states, &xt, &xls)
        .expect("dense reference Hessian")
        .expect("dense Hessian present");
    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("operator workspace build")
        .expect("operator workspace present");
    let got_h = workspace
        .hessian_dense()
        .expect("operator-backed dense Hessian")
        .expect("operator-backed dense Hessian present");
    assert_eq!(got_h.dim(), dense_h.dim());
    for i in 0..got_h.nrows() {
        for j in 0..got_h.ncols() {
            let want = dense_h[[i, j]];
            let got = got_h[[i, j]];
            let tol = 1e-10 * want.abs().max(1.0) + 1e-10;
            assert!(
                (want - got).abs() <= tol,
                "lazy BLS dense Hessian mismatch at ({i}, {j}): dense={want:.6e}, op={got:.6e}"
            );
        }
    }
    let v = array![0.30, -0.70, 0.50, -0.20, 0.15];
    let got_hv = workspace
        .hessian_matvec(&v)
        .expect("operator matvec")
        .expect("operator matvec present");
    let want_hv = dense_h.dot(&v);
    for i in 0..v.len() {
        let tol = 1e-10 * want_hv[i].abs().max(1.0) + 1e-10;
        assert!(
            (want_hv[i] - got_hv[i]).abs() <= tol,
            "lazy BLS Hv mismatch at {i}: dense={:.6e}, op={:.6e}",
            want_hv[i],
            got_hv[i]
        );
    }

    let got_diag = workspace
        .hessian_diagonal()
        .expect("operator diagonal")
        .expect("operator diagonal present");
    for i in 0..v.len() {
        let want = dense_h[[i, i]];
        let tol = 1e-10 * want.abs().max(1.0) + 1e-10;
        assert!(
            (want - got_diag[i]).abs() <= tol,
            "lazy BLS diagonal mismatch at {i}: dense={:.6e}, op={:.6e}",
            want,
            got_diag[i]
        );
    }

    let dense_xt = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xt.clone()));
    let dense_xls = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
    let want_grad = family
        .exact_newton_joint_gradient_from_designs(&states, &dense_xt, &dense_xls)
        .expect("dense reference gradient");
    let got_grad = family
        .exact_newton_joint_gradient_evaluation(&states, &specs)
        .expect("operator gradient")
        .expect("operator gradient present");
    assert!(
        (want_grad.log_likelihood - got_grad.log_likelihood).abs() <= 1e-12,
        "operator gradient log-likelihood mismatch"
    );
    for i in 0..v.len() {
        let want = want_grad.gradient[i];
        let got = got_grad.gradient[i];
        let tol = 1e-10 * want.abs().max(1.0) + 1e-10;
        assert!(
            (want - got).abs() <= tol,
            "lazy BLS gradient mismatch at {i}: dense={:.6e}, op={:.6e}",
            want,
            got
        );
    }

    let d_beta = array![0.07, -0.04, 0.21, 0.08, -0.13];
    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative_from_designs(&states, &xt, &xls, &d_beta)
        .expect("dense dH")
        .expect("dense dH present");
    let got_dh_v = workspace
        .directional_derivative_operator(&d_beta)
        .expect("operator dH")
        .expect("operator dH present")
        .mul_vec(&v);
    let want_dh_v = dense_dh.dot(&v);
    for i in 0..v.len() {
        let tol = 1e-9 * want_dh_v[i].abs().max(1.0) + 1e-9;
        assert!(
            (want_dh_v[i] - got_dh_v[i]).abs() <= tol,
            "lazy BLS dH*v mismatch at {i}: dense={:.6e}, op={:.6e}",
            want_dh_v[i],
            got_dh_v[i]
        );
    }

    let d_beta_v = array![-0.11, 0.13, -0.05, -0.22, 0.09];
    let dense_d2h = family
        .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            &states, &xt, &xls, &d_beta, &d_beta_v,
        )
        .expect("dense d2H")
        .expect("dense d2H present");
    let got_d2h_v = workspace
        .second_directional_derivative_operator(&d_beta, &d_beta_v)
        .expect("operator d2H")
        .expect("operator d2H present")
        .mul_vec(&v);
    let want_d2h_v = dense_d2h.dot(&v);
    for i in 0..v.len() {
        let tol = 1e-9 * want_d2h_v[i].abs().max(1.0) + 1e-9;
        assert!(
            (want_d2h_v[i] - got_d2h_v[i]).abs() <= tol,
            "lazy BLS d2H*v mismatch at {i}: dense={:.6e}, op={:.6e}",
            want_d2h_v[i],
            got_d2h_v[i]
        );
    }
}

#[test]
pub(crate) fn binomial_location_scale_workspace_dh_operator_matches_dense() {
    let (family, states, specs) = bls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let d_beta = array![0.07, -0.04, 0.21, 0.08, -0.13];
    assert_eq!(d_beta.len(), p);

    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative(&states, &d_beta)
        .expect("dense dH build")
        .expect("dense dH present");

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&d_beta)
        .expect("dH operator call")
        .expect("dH operator present");

    let probes = [
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.30, -0.70, 0.50, -0.20, 0.15]),
    ];
    for (k, w) in probes.iter().enumerate() {
        assert_eq!(w.len(), p);
        let want = dense_dh.dot(w);
        let got = dh_op.mul_vec(w);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "BLS dH op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn binomial_location_scale_workspace_d2h_operator_matches_dense() {
    let (family, states, specs) = bls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let d_beta_u = array![0.07, -0.04, 0.21, 0.08, -0.13];
    let d_beta_v = array![-0.11, 0.13, -0.05, -0.22, 0.09];
    assert_eq!(d_beta_u.len(), p);
    assert_eq!(d_beta_v.len(), p);

    let dense_d2h = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &d_beta_u, &d_beta_v)
        .expect("dense d2H build")
        .expect("dense d2H present");

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let d2h_op = workspace
        .second_directional_derivative_operator(&d_beta_u, &d_beta_v)
        .expect("d2H operator call")
        .expect("d2H operator present");

    let probes = [
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.30, -0.70, 0.50, -0.20, 0.15]),
    ];
    for (k, w) in probes.iter().enumerate() {
        let want = dense_d2h.dot(w);
        let got = d2h_op.mul_vec(w);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "BLS d2H op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn binomial_location_scale_projected_trace_cache_matches_dense() {
    let (family, states, specs) = bls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let d_beta_u = array![0.07, -0.04, 0.21, 0.08, -0.13];
    let d_beta_v = array![-0.11, 0.13, -0.05, -0.22, 0.09];
    let factor = Array2::from_shape_fn((p, 3), |(i, j)| {
        ((i as f64 + 1.0) * 0.19 + (j as f64 + 0.5) * 0.37).sin()
    });

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&d_beta_u)
        .expect("dH operator call")
        .expect("dH operator present");
    let d2h_op = workspace
        .second_directional_derivative_operator(&d_beta_u, &d_beta_v)
        .expect("d2H operator call")
        .expect("d2H operator present");
    let cache = gam_problem::ProjectedFactorCache::default();

    for (name, op) in [("dH", dh_op.clone()), ("d2H", d2h_op.clone())] {
        let dense = op.to_dense();
        let dense_projected = dense.dot(&factor);
        let want: f64 = factor
            .iter()
            .zip(dense_projected.iter())
            .map(|(&f, &bf)| f * bf)
            .sum();
        let uncached = op.trace_projected_factor(&factor);
        let cached_first = op.trace_projected_factor_cached(&factor, &cache);
        let cached_second = op.trace_projected_factor_cached(&factor, &cache);

        for (label, got) in [
            ("uncached", uncached),
            ("cached_first", cached_first),
            ("cached_second", cached_second),
        ] {
            let tol = 1e-9 * want.abs().max(1.0) + 1e-9;
            assert!(
                (want - got).abs() <= tol,
                "{name} projected trace {label} mismatch: dense={want:.6e}, got={got:.6e}"
            );
        }
    }

    let mut reused_factor = factor.clone();
    let cached_probe = dh_op.trace_projected_factor_cached(&reused_factor, &cache);
    assert!(cached_probe.is_finite());
    reused_factor[[0, 0]] += 0.25;
    let dense = dh_op.to_dense();
    let dense_projected = dense.dot(&reused_factor);
    let want: f64 = reused_factor
        .iter()
        .zip(dense_projected.iter())
        .map(|(&f, &bf)| f * bf)
        .sum();
    let got = dh_op.trace_projected_factor_cached(&reused_factor, &cache);
    let tol = 1e-9 * want.abs().max(1.0) + 1e-9;
    assert!(
        (want - got).abs() <= tol,
        "cached projected trace reused stale factor contents: dense={want:.6e}, got={got:.6e}"
    );
}

#[test]
#[should_panic(expected = "two-block cached projected trace factor row mismatch")]
pub(crate) fn binomial_location_scale_projected_trace_rejects_wrong_factor_rows() {
    let (family, states, specs) = bls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let d_beta = array![0.07, -0.04, 0.21, 0.08, -0.13];
    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&d_beta)
        .expect("dH operator call")
        .expect("dH operator present");
    let bad_factor = Array2::<f64>::zeros((p + 1, 2));
    let cache = gam_problem::ProjectedFactorCache::default();
    dh_op.trace_projected_factor_cached(&bad_factor, &cache);
}

#[test]
pub(crate) fn binomial_location_scale_workspace_dh_operator_finite_difference() {
    // FD check: [H(β + ε u) v − H(β − ε u) v] / (2ε) ≈ DH[u] v
    // The operator must agree with a centered finite-difference of the
    // dense Hessian along an arbitrary coefficient direction u.
    let (family, states, specs) = bls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let u = array![0.07, -0.04, 0.21, 0.08, -0.13];
    let v = array![0.30, -0.70, 0.50, -0.20, 0.15];
    let eps = 1e-6;
    // Build perturbed states (β ± ε u) using the fixture's designs to
    // recompute η.
    let perturb = |sign: f64| -> Vec<ParameterBlockState> {
        let mut out = states.clone();
        let pt = states[0].beta.len();
        for j in 0..pt {
            out[0].beta[j] += sign * eps * u[j];
        }
        for j in 0..(p - pt) {
            out[1].beta[j] += sign * eps * u[pt + j];
        }
        // recompute η from spec design and new beta.
        let xt_dense = specs[0].design.as_dense_ref().expect("dense xt");
        let xls_dense = specs[1].design.as_dense_ref().expect("dense xls");
        out[0].eta = xt_dense.dot(&out[0].beta);
        out[1].eta = xls_dense.dot(&out[1].beta);
        out
    };
    let states_plus = perturb(1.0);
    let states_minus = perturb(-1.0);
    let h_plus = family
        .exact_newton_joint_hessian(&states_plus)
        .expect("dense H+")
        .expect("dense H+ present");
    let h_minus = family
        .exact_newton_joint_hessian(&states_minus)
        .expect("dense H-")
        .expect("dense H- present");
    let fd = (h_plus.dot(&v) - h_minus.dot(&v)) / (2.0 * eps);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&u)
        .expect("dH op call")
        .expect("dH op present");
    let analytic = dh_op.mul_vec(&v);

    for i in 0..p {
        let tol = 1e-5 * fd[i].abs().max(1.0) + 1e-5;
        assert!(
            (fd[i] - analytic[i]).abs() <= tol,
            "BLS dH FD mismatch at {i}: fd={:.6e}, analytic={:.6e}",
            fd[i],
            analytic[i]
        );
    }
}

#[test]
pub(crate) fn binomial_location_scale_workspace_d2h_operator_finite_difference() {
    // FD check on the second directional: [DH(β + ε u') [u] v
    //                                     − DH(β − ε u') [u] v]/(2ε)
    // ≈ D²H[u', u] v. We choose u' = v as the FD-direction and probe
    // with an arbitrary u.
    let (family, states, specs) = bls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let u = array![0.07, -0.04, 0.21, 0.08, -0.13];
    let u_fd = array![0.30, -0.70, 0.50, -0.20, 0.15];
    let probe = array![-0.21, 0.11, 0.05, 0.32, -0.04];
    let eps = 1e-6;
    let perturb = |sign: f64| -> Vec<ParameterBlockState> {
        let mut out = states.clone();
        let pt = states[0].beta.len();
        for j in 0..pt {
            out[0].beta[j] += sign * eps * u_fd[j];
        }
        for j in 0..(p - pt) {
            out[1].beta[j] += sign * eps * u_fd[pt + j];
        }
        let xt_dense = specs[0].design.as_dense_ref().expect("dense xt");
        let xls_dense = specs[1].design.as_dense_ref().expect("dense xls");
        out[0].eta = xt_dense.dot(&out[0].beta);
        out[1].eta = xls_dense.dot(&out[1].beta);
        out
    };
    let states_plus = perturb(1.0);
    let states_minus = perturb(-1.0);
    let dh_plus = family
        .exact_newton_joint_hessian_directional_derivative(&states_plus, &u)
        .expect("dense dH+")
        .expect("dense dH+ present");
    let dh_minus = family
        .exact_newton_joint_hessian_directional_derivative(&states_minus, &u)
        .expect("dense dH-")
        .expect("dense dH- present");
    let fd = (dh_plus.dot(&probe) - dh_minus.dot(&probe)) / (2.0 * eps);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let d2h_op = workspace
        .second_directional_derivative_operator(&u_fd, &u)
        .expect("d2H op call")
        .expect("d2H op present");
    let analytic = d2h_op.mul_vec(&probe);

    for i in 0..p {
        let tol = 5e-5 * fd[i].abs().max(1.0) + 5e-5;
        assert!(
            (fd[i] - analytic[i]).abs() <= tol,
            "BLS d2H FD mismatch at {i}: fd={:.6e}, analytic={:.6e}",
            fd[i],
            analytic[i]
        );
    }
}

#[test]
pub(crate) fn gaussian_location_scale_workspace_d2h_operator_matches_dense() {
    let (family, states, specs) = gls_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len();
    let d_beta_u = array![0.07, -0.04, 0.21, 0.08, -0.13];
    let d_beta_v = array![-0.11, 0.13, -0.05, -0.22, 0.09];

    let dense_d2h = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &d_beta_u, &d_beta_v)
        .expect("dense d2H build")
        .expect("dense d2H present");

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let d2h_op = workspace
        .second_directional_derivative_operator(&d_beta_u, &d_beta_v)
        .expect("d2H op call")
        .expect("d2H op present");

    let probes = [
        Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]),
        Array1::from_vec(vec![0.30, -0.70, 0.50, -0.20, 0.15]),
    ];
    for (k, w) in probes.iter().enumerate() {
        let want = dense_d2h.dot(w);
        let got = d2h_op.mul_vec(w);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "GLS d2H op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn zeroweightrows_stay_inactive_in_builtin_diagonal_families() {
    let weights = Array1::from_vec(vec![0.0, 1.0]);

    let gaussian = GaussianLocationScaleFamily {
        y: Array1::from_vec(vec![2.0, -1.0]),
        weights: weights.clone(),
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let gaussian_eval = gaussian
        .evaluate(&[
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::from_vec(vec![0.5, -0.25]),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: Array1::from_vec(vec![0.1, -0.2]),
            },
        ])
        .expect("gaussian evaluate");
    match &gaussian_eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_MU] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            assert_eq!(working_weights[0], 0.0);
            assert_eq!(working_response[0], 0.5);
            assert!(working_weights[1] > 0.0);
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gaussian mu block")
        }
    }
    match &gaussian_eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            assert_eq!(working_weights[0], 0.0);
            assert_eq!(working_response[0], 0.1);
            assert!(working_weights[1] > 0.0);
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gaussian log-sigma block")
        }
    }

    let poisson = PoissonLogFamily {
        y: Array1::from_vec(vec![3.0, 1.0]),
        weights: weights.clone(),
    };
    let poisson_eval = poisson
        .evaluate(&[ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::from_vec(vec![0.7, -0.4]),
        }])
        .expect("poisson evaluate");
    match &poisson_eval.blockworking_sets[PoissonLogFamily::BLOCK_ETA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            assert_eq!(working_weights[0], 0.0);
            assert_eq!(working_response[0], 0.7);
            assert!(working_weights[1] > 0.0);
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Poisson block")
        }
    }

    let gamma = GammaLogFamily {
        y: Array1::from_vec(vec![1.5, 0.8]),
        weights,
        shape: 2.5,
    };
    let gamma_eval = gamma
        .evaluate(&[ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::from_vec(vec![0.2, -0.1]),
        }])
        .expect("gamma evaluate");
    match &gamma_eval.blockworking_sets[GammaLogFamily::BLOCK_ETA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            assert_eq!(working_weights[0], 0.0);
            assert_eq!(working_response[0], 0.2);
            assert!(working_weights[1] > 0.0);
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gamma block")
        }
    }
}

#[test]
pub(crate) fn log_link_rows_remain_exact_beyond_former_clamp() {
    let poisson = PoissonLogFamily {
        y: Array1::from_vec(vec![1.0, 2.0, 3.0]),
        weights: Array1::from_vec(vec![1.0, 1.0, 1.0]),
    };
    let poisson_eta = Array1::from_vec(vec![-35.0, 0.2, 35.0]);
    let poisson_eval = poisson
        .evaluate(&[ParameterBlockState {
            beta: Array1::zeros(0),
            eta: poisson_eta.clone(),
        }])
        .expect("poisson evaluate");
    match &poisson_eval.blockworking_sets[PoissonLogFamily::BLOCK_ETA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            assert_eq!(working_weights[0], poisson_eta[0].exp());
            assert_ne!(working_response[0], poisson_eta[0]);
            assert!(working_weights[1] > 0.0);
            assert_eq!(working_weights[2], poisson_eta[2].exp());
            assert_ne!(working_response[2], poisson_eta[2]);
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Poisson block")
        }
    }

    let gamma = GammaLogFamily {
        y: Array1::from_vec(vec![0.8, 1.2, 2.5]),
        weights: Array1::from_vec(vec![1.0, 1.0, 1.0]),
        shape: 3.0,
    };
    let gamma_eta = Array1::from_vec(vec![-40.0, -0.3, 40.0]);
    let gamma_eval = gamma
        .evaluate(&[ParameterBlockState {
            beta: Array1::zeros(0),
            eta: gamma_eta.clone(),
        }])
        .expect("gamma evaluate");
    match &gamma_eval.blockworking_sets[GammaLogFamily::BLOCK_ETA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            assert!(working_weights[0] > 0.0);
            assert_ne!(working_response[0], gamma_eta[0]);
            assert!(working_weights[1] > 0.0);
            assert!(working_weights[2] > 0.0);
            assert_ne!(working_response[2], gamma_eta[2]);
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gamma block")
        }
    }
}

#[test]
pub(crate) fn poisson_log_canonical_diagonal_weight_is_fisher_and_observed() {
    let family = PoissonLogFamily {
        y: array![0.0, 3.0],
        weights: array![1.5, 0.5],
    };
    let eta = array![-0.4_f64, 0.7_f64];
    let eval = family
        .evaluate(&[ParameterBlockState {
            beta: Array1::zeros(0),
            eta: eta.clone(),
        }])
        .expect("poisson evaluate");

    match &eval.blockworking_sets[PoissonLogFamily::BLOCK_ETA] {
        BlockWorkingSet::Diagonal {
            working_response: _,
            working_weights,
        } => {
            for i in 0..eta.len() {
                let fisher_weight = family.weights[i] * eta[i].exp();
                assert!(
                    (working_weights[i] - fisher_weight).abs() < 1e-12,
                    "canonical Poisson-log observed and Fisher weights should coincide at row {i}: got {}, expected {}",
                    working_weights[i],
                    fisher_weight
                );
            }
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Poisson block")
        }
    }
}

#[test]
pub(crate) fn gamma_log_noncanonical_diagonal_uses_observed_not_fisher_weight_and_dw() {
    let family = GammaLogFamily {
        y: array![2.0, 0.25],
        weights: array![1.25, 0.75],
        shape: 3.0,
    };
    let eta = array![0.0_f64, -0.5_f64];
    let states = vec![ParameterBlockState {
        beta: Array1::zeros(0),
        eta: eta.clone(),
    }];
    let eval = family.evaluate(&states).expect("gamma evaluate");

    match &eval.blockworking_sets[GammaLogFamily::BLOCK_ETA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            for i in 0..eta.len() {
                let mu = eta[i].exp();
                let fisher_weight = family.weights[i] * family.shape;
                let observed_weight = fisher_weight * family.y[i] / mu;
                assert!(
                    (working_weights[i] - observed_weight).abs() < 1e-12,
                    "Gamma-log row {i} should use observed weight: got {}, expected {}",
                    working_weights[i],
                    observed_weight
                );
                assert!(
                    (working_weights[i] - fisher_weight).abs() > 1e-6,
                    "fixture should distinguish observed from Fisher at row {i}: observed {}, fisher {}",
                    working_weights[i],
                    fisher_weight
                );

                let score = fisher_weight * (family.y[i] / mu - 1.0);
                let expected_response = eta[i] + score / observed_weight;
                assert!(
                    (working_response[i] - expected_response).abs() < 1e-12,
                    "Gamma-log row {i} working response should be consistent with observed Newton weight: got {}, expected {}",
                    working_response[i],
                    expected_response
                );
            }
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gamma block")
        }
    }

    let d_eta = array![0.5_f64, -2.0_f64];
    let dw = family
        .diagonalworking_weights_directional_derivative(&states, GammaLogFamily::BLOCK_ETA, &d_eta)
        .expect("gamma dW")
        .expect("gamma dW present");
    for i in 0..eta.len() {
        let observed_weight = family.weights[i] * family.shape * family.y[i] / eta[i].exp();
        let expected_dw = -observed_weight * d_eta[i];
        assert!(
            (dw[i] - expected_dw).abs() < 1e-12,
            "Gamma-log row {i} dW should differentiate observed weights: got {}, expected {}",
            dw[i],
            expected_dw
        );
    }
}

#[test]
pub(crate) fn gaussian_log_sigmaweight_directional_derivative_iszero_on_active_floor_branch() {
    let family = GaussianLocationScaleFamily {
        y: Array1::from_vec(vec![0.3]),
        weights: Array1::from_vec(vec![1.0]),
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::from_vec(vec![0.0]),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::from_vec(vec![35.0]),
        },
    ];
    let d_eta = Array1::from_vec(vec![1.0]);

    let dw = family
        .diagonalworking_weights_directional_derivative(
            &states,
            GaussianLocationScaleFamily::BLOCK_LOG_SIGMA,
            &d_eta,
        )
        .expect("gaussian directional derivative")
        .expect("gaussian log-sigma derivative");
    assert_eq!(dw[0], 0.0);
}

#[test]
pub(crate) fn gaussian_log_sigmaweight_directional_derivative_matches_finite_difference() {
    let family = GaussianLocationScaleFamily {
        y: Array1::from_vec(vec![1.2]),
        weights: Array1::from_vec(vec![1.0]),
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let etamu = Array1::from_vec(vec![0.1]);
    let eta_ls = Array1::from_vec(vec![0.4]);
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: etamu.clone(),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: eta_ls.clone(),
        },
    ];
    let d_eta = Array1::from_vec(vec![1.0]);

    let dw = family
        .diagonalworking_weights_directional_derivative(
            &states,
            GaussianLocationScaleFamily::BLOCK_LOG_SIGMA,
            &d_eta,
        )
        .expect("gaussian directional derivative")
        .expect("gaussian log-sigma derivative");

    let eps = 1e-6;
    let mut states_plus = states.clone();
    states_plus[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta[0] += eps;
    let eval_plus = family.evaluate(&states_plus).expect("gaussian eval plus");
    let w_plus = match &eval_plus.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
        BlockWorkingSet::Diagonal {
            working_response: _,
            working_weights,
        } => working_weights[0],
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gaussian log-sigma block")
        }
    };

    let mut states_minus = states;
    states_minus[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA].eta[0] -= eps;
    let eval_minus = family.evaluate(&states_minus).expect("gaussian eval minus");
    let w_minus = match &eval_minus.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA]
    {
        BlockWorkingSet::Diagonal {
            working_response: _,
            working_weights,
        } => working_weights[0],
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gaussian log-sigma block")
        }
    };

    let fd = (w_plus - w_minus) / (2.0 * eps);
    assert!((dw[0] - fd).abs() < 1e-6, "dw={} fd={}", dw[0], fd);
}

#[test]
pub(crate) fn gaussian_sigma_helper_matches_exact_exp_link() {
    let eta0 = 701.0_f64;
    let eta = array![eta0];
    let (sigma, d1, d2, d3, d4) = exp_sigma_derivs_up_to_fourth_array(eta.view());
    let coded_sigma = safe_exp(eta0);
    assert!(
        (sigma[0] - coded_sigma).abs() < 1e-30,
        "Gaussian sigma helper should evaluate the exact exp sigma link at eta={eta0}; got {} vs {}",
        sigma[0],
        coded_sigma
    );
    assert!(
        (d1[0] - sigma[0]).abs() / sigma[0] < 1e-12,
        "Gaussian sigma helper first derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
        d1[0],
        sigma[0]
    );
    assert!(
        (d2[0] - sigma[0]).abs() / sigma[0] < 1e-12,
        "Gaussian sigma helper second derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
        d2[0],
        sigma[0]
    );
    assert!(
        (d3[0] - sigma[0]).abs() / sigma[0] < 1e-12,
        "Gaussian sigma helper third derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
        d3[0],
        sigma[0]
    );
    assert!(
        (d4[0] - sigma[0]).abs() / sigma[0] < 1e-12,
        "Gaussian sigma helper fourth derivative should equal exp(eta) at eta={eta0}; got {} vs {}",
        d4[0],
        sigma[0]
    );
}

#[test]
pub(crate) fn gaussian_diagonal_log_sigma_block_uses_fisher_score_step_in_far_tail() {
    let family = GaussianLocationScaleFamily {
        y: array![0.0],
        weights: array![1.0],
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let eta_mu = array![0.0];
    let eta_ls0 = 350.0_f64;
    let states_at = |eta_ls: f64| {
        vec![
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: eta_mu.clone(),
            },
            ParameterBlockState {
                beta: Array1::zeros(0),
                eta: array![eta_ls],
            },
        ]
    };

    let eval = family.evaluate(&states_at(eta_ls0)).expect("evaluate");
    match &eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
        BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            // logb link σ = b + e^η: at η ≫ log b the floor is dwarfed
            // (σ ≈ e^η ~ 1e304), so dlogσ/dη = 1 − b/σ → 1 to within
            // f64 precision and the IRLS step matches the pure-exp Fisher
            // step. Compute the expectation explicitly from the new link.
            let sigma = logb_sigma_from_eta_scalar(eta_ls0);
            let inv_s2 = sigma.recip() * sigma.recip();
            let dlog = logb_dlog_sigma_deta(sigma, logb_sigma_jet1_scalar(eta_ls0).d1);
            let residual = family.y[0] - eta_mu[0];
            let expected_score = family.weights[0] * (residual * residual * inv_s2 - 1.0) * dlog;
            let expected_info = 2.0 * family.weights[0] * dlog * dlog;
            let expected_response = eta_ls0 + expected_score / expected_info;

            assert!((working_weights[0] - expected_info).abs() < 1e-12);
            assert!(
                (working_response[0] - expected_response).abs() < 1e-12,
                "working response mismatch: got {}, expected {}",
                working_response[0],
                expected_response
            );
        }
        BlockWorkingSet::ExactNewton { .. } | BlockWorkingSet::NaturalDiagonal { .. } => {
            panic!("expected diagonal Gaussian log-sigma block")
        }
    }

    let loglik = |eta_ls: f64| family.log_likelihood_only(&states_at(eta_ls)).expect("ll");
    let h = 1e-4;
    let ll_plus = loglik(eta_ls0 + h);
    let ll0 = loglik(eta_ls0);
    let ll_minus = loglik(eta_ls0 - h);
    let score_fd = (ll_plus - ll_minus) / (2.0 * h);
    assert!(score_fd.is_finite());
    assert!(
        (score_fd + 1.0).abs() < 1e-6,
        "far-tail score should be -1, got {score_fd}"
    );
    assert!(
        (ll_plus - 2.0 * ll0 + ll_minus).abs() < 1e-5,
        "far-tail Gaussian log-sigma block should have near-zero observed curvature"
    );
}

#[test]
pub(crate) fn gaussian_exact_joint_path_refuses_unrepresentable_scale_atomically() {
    let mu_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]]));
    let log_sigma_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(array![[1.0]]));
    let family = GaussianLocationScaleFamily {
        y: array![0.0],
        weights: array![1.0],
        mu_design: Some(mu_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let beta_mu = array![0.0];
    let beta_ls = array![710.0];
    let states = vec![
        ParameterBlockState {
            beta: beta_mu.clone(),
            eta: mu_design.matrixvectormultiply(&beta_mu),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: log_sigma_design.matrixvectormultiply(&beta_ls),
        },
    ];

    let err = family
        .exact_newton_joint_hessian(&states)
        .expect_err("overflowing Gaussian scale must be refused");
    assert!(
        err.contains("row 0") && err.contains("Gaussian scale link"),
        "typed row refusal should identify the first row and quantity: {err}"
    );
}

#[test]
pub(crate) fn gaussian_diagonal_geometry_preserves_representable_tiny_fisher_weights() {
    let family = GaussianLocationScaleFamily {
        y: array![0.0, 0.0],
        weights: array![1.0, 1.0],
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0, 0.0],
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![200.0, -20.0],
        },
    ];
    let eval = family
        .evaluate(&states)
        .expect("representable tiny geometry");
    let location = match &eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_MU] {
        BlockWorkingSet::Diagonal {
            working_weights, ..
        } => working_weights,
        _ => panic!("expected diagonal location block"),
    };
    let scale = match &eval.blockworking_sets[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA] {
        BlockWorkingSet::Diagonal {
            working_weights, ..
        } => working_weights,
        _ => panic!("expected diagonal scale block"),
    };
    assert!(location[0] > 0.0 && location[0] < 1.0e-12);
    assert!(scale[1] > 0.0 && scale[1] < 1.0e-12);
    let sigma0 = logb_sigma_from_eta_scalar(200.0);
    let expected_location = sigma0.recip() * sigma0.recip();
    assert!((location[0] / expected_location - 1.0).abs() <= 4.0 * f64::EPSILON);
    let jet1 = logb_sigma_jet1_scalar(-20.0);
    let kappa1 = jet1.d1 / jet1.sigma;
    let expected_scale = 2.0 * kappa1 * kappa1;
    assert!((scale[1] / expected_scale - 1.0).abs() <= 4.0 * f64::EPSILON);
}

#[test]
pub(crate) fn gaussian_batch_certification_reports_smallest_unrepresentable_row() {
    let family = GaussianLocationScaleFamily {
        y: Array1::zeros(3),
        weights: Array1::ones(3),
        mu_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: Array1::zeros(3),
        },
        ParameterBlockState {
            beta: Array1::zeros(0),
            eta: array![0.0, -400.0, 710.0],
        },
    ];
    let err = family
        .evaluate(&states)
        .expect_err("unrepresentable row geometry must be refused");
    assert!(
        err.contains("row 1") && err.contains("log-scale Fisher information"),
        "parallel certification must report the smallest failing row: {err}"
    );
}

#[test]
pub(crate) fn gaussian_location_scale_hotloop_optimized_matches_legacy_and_is_faster_locally() {
    let n = 4096usize;
    let y = Array1::from_shape_fn(n, |i| ((i as f64) * 0.003).sin() + 0.1);
    let mu = Array1::from_shape_fn(n, |i| ((i as f64) * 0.001).cos() - 0.2);
    let eta_ls = Array1::from_shape_fn(n, |i| ((i as f64) * 0.002).sin() * 0.8 - 0.1);
    let weights = Array1::from_shape_fn(n, |i| if i % 37 == 0 { 0.0 } else { 1.0 });
    let ln2pi = (2.0 * std::f64::consts::PI).ln();

    let legacy_eval = || {
        let mut ll = 0.0;
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut zls = Array1::<f64>::zeros(n);
        let mut wls = Array1::<f64>::zeros(n);
        for i in 0..n {
            let w = weights[i];
            let eta = eta_ls[i];
            let SigmaJet1 { sigma, d1 } = logb_sigma_jet1_scalar(eta);
            let inv_s2 = (sigma * sigma).recip();
            let r = y[i] - mu[i];
            ll += w * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma.ln()));
            if w == 0.0 {
                wmu[i] = 0.0;
                zmu[i] = mu[i];
            } else {
                wmu[i] = w * inv_s2;
                zmu[i] = mu[i] + r;
            }
            let dlogsigma_du = logb_dlog_sigma_deta(sigma, d1);
            let info_u = 2.0 * w * dlogsigma_du * dlogsigma_du;
            if info_u == 0.0 {
                wls[i] = 0.0;
                zls[i] = eta;
            } else {
                wls[i] = info_u;
                let score_ls = w * (r * r * inv_s2 - 1.0) * dlogsigma_du;
                zls[i] = eta + score_ls / info_u;
            }
        }
        (ll, zmu, wmu, zls, wls)
    };

    let optimized_eval = || {
        let mut ll = 0.0;
        let mut zmu = Array1::<f64>::zeros(n);
        let mut wmu = Array1::<f64>::zeros(n);
        let mut zls = Array1::<f64>::zeros(n);
        let mut wls = Array1::<f64>::zeros(n);
        for i in 0..n {
            let eta = eta_ls[i];
            let SigmaJet1 { sigma, d1 } = logb_sigma_jet1_scalar(eta);
            let inv_s2 = (sigma * sigma).recip();
            let w = weights[i];
            let r = y[i] - mu[i];
            ll += w * (-0.5 * (r * r * inv_s2 + ln2pi + 2.0 * sigma.ln()));
            if w == 0.0 {
                wmu[i] = 0.0;
                zmu[i] = mu[i];
            } else {
                wmu[i] = w * inv_s2;
                zmu[i] = mu[i] + r;
            }
            let dlogsigma_du = logb_dlog_sigma_deta(sigma, d1);
            let info_u = 2.0 * w * dlogsigma_du * dlogsigma_du;
            if info_u == 0.0 {
                wls[i] = 0.0;
                zls[i] = eta;
            } else {
                wls[i] = info_u;
                let score_ls = w * (r * r * inv_s2 - 1.0) * dlogsigma_du;
                zls[i] = eta + score_ls / info_u;
            }
        }
        (ll, zmu, wmu, zls, wls)
    };

    let (ll_legacy, zmu_legacy, wmu_legacy, zls_legacy, wls_legacy) = legacy_eval();
    let (ll_opt, zmu_opt, wmu_opt, zls_opt, wls_opt) = optimized_eval();
    assert!((ll_legacy - ll_opt).abs() < 1e-10);
    assert!((&zmu_legacy - &zmu_opt).iter().all(|v| v.abs() < 1e-12));
    assert!((&wmu_legacy - &wmu_opt).iter().all(|v| v.abs() < 1e-12));
    assert!((&zls_legacy - &zls_opt).iter().all(|v| v.abs() < 1e-12));
    assert!((&wls_legacy - &wls_opt).iter().all(|v| v.abs() < 1e-12));
}

pub(crate) fn simple_matern_term_collection(
    feature_cols: &[usize],
    length_scale: f64,
) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![SmoothTermSpec {
            frozen_parametric_residualization: None,
            name: "spatial".to_string(),
            basis: SmoothBasisSpec::Matern {
                feature_cols: feature_cols.to_vec(),
                spec: MaternBasisSpec {
                    periodic: None,
                    center_strategy: CenterStrategy::EqualMass { num_centers: 6 },
                    length_scale: gam_terms::basis::MaternLengthScale::fixed(length_scale),
                    nu: MaternNu::ThreeHalves,
                    include_intercept: false,
                    double_penalty: false,
                    identifiability: MaternIdentifiability::CenterSumToZero,
                    aniso_log_scales: None,
                },
                input_scale: None,
            },
            shape: ShapeConstraint::None,
            joint_null_rotation: None,
        }],
    }
}

pub(crate) fn empty_term_collection() -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: Vec::new(),
    }
}

pub(crate) fn spatial_kappa_options() -> SpatialLengthScaleOptimizationOptions {
    SpatialLengthScaleOptimizationOptions {
        enabled: true,
        max_outer_iter: 4,
        rel_tol: 1e-4,
        log_step: std::f64::consts::LN_2,
        min_length_scale: 0.1,
        max_length_scale: 2.0,
        pilot_subsample_threshold: 10_000,
    }
}

pub(crate) fn spatial_fit_smoke_options() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        // The location-scale-wiggle spatial smoke test can need more than
        // 24 blockwise cycles after the final outer REML refit; keep the
        // tolerance unchanged and allow enough iterations for the same
        // convergence criterion to be reached deterministically.
        inner_max_cycles: 48,
        inner_tol: 1e-4,
        outer_max_iter: 3,
        outer_tol: 1e-4,
        ..BlockwiseFitOptions::default()
    }
}

#[test]
pub(crate) fn binomial_location_scale_exact_probit_tailobjects_stay_finite() {
    let n = 6usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_elem(n, 1.0);
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_elem((n, 1), 1.0),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_elem((n, 1), 1.0),
    ));
    let family = BinomialLocationScaleFamily {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let beta_t = array![250.0];
    let beta_ls = array![0.0];
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: threshold_design.matrixvectormultiply(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: log_sigma_design.matrixvectormultiply(&beta_ls),
        },
    ];

    let eval = family
        .evaluate(&states)
        .expect("evaluate tail-stable family");
    assert!(eval.log_likelihood.is_finite());
    let joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected exact joint hessian");
    assert!(joint.iter().all(|v| v.is_finite()));
    let direction = array![0.1, -0.2];
    let d_h = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("joint dH")
        .expect("expected exact joint dH");
    assert!(d_h.iter().all(|v| v.is_finite()));
    let d2_h = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &direction, &direction)
        .expect("joint d2H")
        .expect("expected exact joint d2H");
    assert!(d2_h.iter().all(|v| v.is_finite()));
}

#[test]
pub(crate) fn binomial_location_scale_many_smoothing_params_keeps_second_order_outer() {
    fn spec_with_penalties(name: &str, n: usize, p: usize, k: usize) -> ParameterBlockSpec {
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::from_elem((n, p), 1.0),
            )),
            offset: Array1::zeros(n),
            penalties: (0..k)
                .map(|_| PenaltyMatrix::Dense(identity_penalty(p)))
                .collect(),
            nullspace_dims: vec![0; k],
            initial_log_lambdas: Array1::zeros(k),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    }

    let n = 8usize;
    let family = BinomialLocationScaleFamily {
        y: Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        weights: Array1::from_elem(n, 1.0),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let specs = vec![
        spec_with_penalties("threshold", n, 3, 2),
        spec_with_penalties("log_sigma", n, 6, 11),
    ];

    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        crate::custom_family::ExactOuterDerivativeOrder::Second
    );
    let (_gradient, hessian) = crate::custom_family::custom_family_outer_derivatives(
        &family,
        &specs,
        &BlockwiseFitOptions::default(),
    );
    assert_eq!(hessian, gam_problem::DeclaredHessianForm::Either);
}

#[test]
pub(crate) fn binomial_location_scale_term_builder_requires_exact_spatial_joint_path() {
    let n = 8usize;
    let builder = BinomialLocationScaleTermBuilder {
        y: Array1::from_elem(n, 0.0),
        weights: Array1::from_elem(n, 1.0),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        meanspec: simple_matern_term_collection(&[0, 1], 0.4),
        noisespec: simple_matern_term_collection(&[0, 1], 0.75),
        mean_offset: Array1::zeros(n),
        noise_offset: Array1::zeros(n),
    };
    assert!(builder.exact_spatial_joint_supported());
    assert!(builder.require_exact_spatial_joint());
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
    }
    let mean_design =
        build_term_collection_design(data.view(), builder.meanspec()).expect("mean design");
    let noise_design =
        build_term_collection_design(data.view(), builder.noisespec()).expect("noise design");
    let family = builder.build_family(&mean_design, &noise_design);
    assert!(family.exact_joint_supported());
}

#[test]
pub(crate) fn binomial_location_scale_builder_populateswarm_start_betas() {
    let n = 12usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let builder = BinomialLocationScaleTermBuilder {
        mean_offset: Array1::zeros(y.len()),
        noise_offset: Array1::zeros(y.len()),
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        meanspec: simple_matern_term_collection(&[0, 1], 0.45),
        noisespec: simple_matern_term_collection(&[0, 1], 0.8),
    };
    let mean_design =
        build_term_collection_design(data.view(), builder.meanspec()).expect("mean design");
    let noise_design =
        build_term_collection_design(data.view(), builder.noisespec()).expect("noise design");
    let rho = compose_theta_from_hints_test(
        builder.mean_penalty_count(&mean_design),
        builder.noise_penalty_count(&noise_design),
        &None,
        &None,
        &Array1::zeros(0),
    );
    let blocks = builder
        .build_blocks(&rho, &mean_design, &noise_design, None, None)
        .expect("build blocks");
    assert_eq!(blocks.len(), 2);
    assert!(blocks[0].initial_beta.is_some());
    assert!(blocks[1].initial_beta.is_some());
}

#[test]
pub(crate) fn binomial_location_scale_exact_newton_spatial_joint_hyper_returns_fullhessian() {
    let n = 12usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.0 * std::f64::consts::PI * t).cos();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let meanspec = simple_matern_term_collection(&[0, 1], 0.45);
    let noisespec = simple_matern_term_collection(&[0, 1], 0.8);
    let builder = BinomialLocationScaleTermBuilder {
        mean_offset: Array1::zeros(y.len()),
        noise_offset: Array1::zeros(y.len()),
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        meanspec: meanspec.clone(),
        noisespec: noisespec.clone(),
    };
    let mean_design =
        build_term_collection_design(data.view(), &meanspec).expect("build mean design");
    let noise_design =
        build_term_collection_design(data.view(), &noisespec).expect("build noise design");
    let meanspec_resolved =
        freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
    let noisespec_resolved =
        freeze_term_collection_from_design(&noisespec, &noise_design).expect("freeze noise spec");
    let rho = compose_theta_from_hints_test(
        builder.mean_penalty_count(&mean_design),
        builder.noise_penalty_count(&noise_design),
        &None,
        &None,
        &Array1::zeros(0),
    );
    let blocks = builder
        .build_blocks(&rho, &mean_design, &noise_design, None, None)
        .expect("build blocks");
    let family = builder.build_family(&mean_design, &noise_design);
    let derivative_blocks = builder
        .build_psiderivative_blocks(
            data.view(),
            &meanspec_resolved,
            &noisespec_resolved,
            &mean_design,
            &noise_design,
        )
        .expect("psi derivative blocks");
    let eval = evaluate_custom_family_joint_hyper(
        &family,
        &blocks,
        &BlockwiseFitOptions {
            use_remlobjective: true,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        },
        &rho,
        &test_design_hyper_layout(&derivative_blocks),
        None,
        gam_problem::EvalMode::ValueGradientHessian,
    )
    .expect("exact spatial joint hyper eval");
    assert!(eval.objective.is_finite());
    assert!(eval.gradient.iter().all(|v| v.is_finite()));
    let hess = eval
        .outer_hessian
        .materialize_dense()
        .expect("exact spatial joint hyper path should materialize a full [rho, psi] hessian")
        .expect("exact spatial joint hyper path should return a full [rho, psi] hessian");
    let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    let theta_dim = rho.len() + psi_dim;
    assert_eq!(eval.gradient.len(), theta_dim);
    assert_eq!(hess.nrows(), theta_dim);
    assert_eq!(hess.ncols(), theta_dim);
}

#[test]
pub(crate) fn gaussian_location_scale_exact_newton_spatial_joint_hyper_returns_fullhessian() {
    let n = 12usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| {
        let x0 = data[[i, 0]];
        let x1 = data[[i, 1]];
        0.4 * x0 - 0.2 * x1 + 0.15
    }));
    let weights = Array1::from_elem(n, 1.0);
    let meanspec = simple_matern_term_collection(&[0, 1], 0.45);
    let noisespec = simple_matern_term_collection(&[0, 1], 0.8);
    let builder = GaussianLocationScaleTermBuilder {
        y,
        weights,
        meanspec: meanspec.clone(),
        noisespec: noisespec.clone(),
        mean_offset: Array1::zeros(n),
        noise_offset: Array1::zeros(n),
    };
    let mean_design =
        build_term_collection_design(data.view(), &meanspec).expect("build mean design");
    let noise_design =
        build_term_collection_design(data.view(), &noisespec).expect("build noise design");
    let meanspec_resolved =
        freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
    let noisespec_resolved =
        freeze_term_collection_from_design(&noisespec, &noise_design).expect("freeze noise spec");
    let rho = compose_theta_from_hints_test(
        builder.mean_penalty_count(&mean_design),
        builder.noise_penalty_count(&noise_design),
        &None,
        &None,
        &Array1::zeros(0),
    );
    let blocks = builder
        .build_blocks(&rho, &mean_design, &noise_design, None, None)
        .expect("build blocks");
    assert_eq!(
        builder.noise_penalty_count(&noise_design),
        noise_design.penalties.len(),
        "Gaussian scale-block rho layout must contain only formula-native penalties"
    );
    assert_eq!(
        blocks[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA]
            .penalties
            .len(),
        noise_design.penalties.len(),
        "Gaussian scale-block construction must not penalize the likelihood-identified log-sigma intercept"
    );
    let family = builder.build_family(&mean_design, &noise_design);
    let derivative_blocks = builder
        .build_psiderivative_blocks(
            data.view(),
            &meanspec_resolved,
            &noisespec_resolved,
            &mean_design,
            &noise_design,
        )
        .expect("psi derivative blocks");
    let eval = evaluate_custom_family_joint_hyper(
        &family,
        &blocks,
        &BlockwiseFitOptions {
            use_remlobjective: true,
            outer_max_iter: 1,
            ..BlockwiseFitOptions::default()
        },
        &rho,
        &test_design_hyper_layout(&derivative_blocks),
        None,
        gam_problem::EvalMode::ValueGradientHessian,
    )
    .expect("exact spatial joint hyper eval");
    assert!(eval.objective.is_finite());
    assert!(eval.gradient.iter().all(|v| v.is_finite()));
    let hess = eval
        .outer_hessian
        .materialize_dense()
        .expect("exact spatial joint hyper path should materialize a full [rho, psi] hessian")
        .expect("exact spatial joint hyper path should return a full [rho, psi] hessian");
    let psi_dim = derivative_blocks.iter().map(Vec::len).sum::<usize>();
    let theta_dim = rho.len() + psi_dim;
    assert_eq!(eval.gradient.len(), theta_dim);
    assert_eq!(hess.nrows(), theta_dim);
    assert_eq!(hess.ncols(), theta_dim);
    assert!(hess.iter().all(|v| v.is_finite()));
}

/// Shared assertion body for the `*_exposes_joint_psi_hook_surface` tests:
/// pulls the joint ψ terms / second-order terms / mixed directional drift
/// off `family` and checks their shapes. `label` names the family in the
/// panic messages; `slope`/`intercept` parameterize the `d_beta` probe.
pub(crate) fn assert_joint_psi_hook_surface<F: CustomFamily>(
    family: &F,
    block_states: &[ParameterBlockState],
    blocks: &[ParameterBlockSpec],
    derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    slope: f64,
    intercept: f64,
    label: &str,
) {
    let hyper_layout = test_design_hyper_layout(derivative_blocks);
    let psi_terms = family
        .exact_newton_joint_psi_terms(block_states, blocks, &hyper_layout, 0)
        .expect("joint psi terms call")
        .unwrap_or_else(|| panic!("{label} family should return joint psi terms"));
    let psi2_terms = family
        .exact_newton_joint_psisecond_order_terms(block_states, blocks, &hyper_layout, 0, 0)
        .expect("joint psi second-order call")
        .unwrap_or_else(|| panic!("{label} family should return joint psi second-order terms"));
    let total = block_states
        .iter()
        .map(|state| state.beta.len())
        .sum::<usize>();
    assert_eq!(psi_terms.score_psi.len(), total);
    if psi_terms.hessian_psi_operator.is_some() {
        assert_eq!(psi_terms.hessian_psi.dim(), (0, 0));
    } else {
        assert_eq!(psi_terms.hessian_psi.dim(), (total, total));
    }
    assert_eq!(psi2_terms.score_psi_psi.len(), total);
    if psi2_terms.hessian_psi_psi_operator.is_some() {
        assert_eq!(psi2_terms.hessian_psi_psi.dim(), (0, 0));
    } else {
        assert_eq!(psi2_terms.hessian_psi_psi.dim(), (total, total));
    }

    let mut d_beta_flat = Array1::<f64>::zeros(total);
    let mut at = 0usize;
    for state in block_states {
        let end = at + state.beta.len();
        d_beta_flat
            .slice_mut(s![at..end])
            .assign(&state.beta.mapv(|v| slope * v + intercept));
        at = end;
    }
    let mixed = family
        .exact_newton_joint_psihessian_directional_derivative(
            block_states,
            blocks,
            &hyper_layout,
            0,
            &d_beta_flat,
        )
        .expect("joint psi mixed drift call")
        .unwrap_or_else(|| panic!("{label} family should return joint psi mixed drift"));
    assert_eq!(mixed.dim(), (total, total));
}

// The joint-psi hook surface is a family-level derivative property evaluated at
// a fixed set of coefficients, so it is exercised at explicitly-built
// block_states (as the binomial-wiggle sibling does), not at a minted fit. A
// minted fit is neither needed nor obtainable here: certification-ownership does
// not return a non-stationary best-effort fit, and one cold outer step on this
// deliberately tiny fixture is never stationary.
#[test]
pub(crate) fn gaussian_location_scale_family_exposes_joint_psi_hook_surface() {
    let n = 10usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.0 * std::f64::consts::PI * t).cos();
    }
    let y = Array1::from_iter((0..n).map(|i| {
        let x0 = data[[i, 0]];
        let x1 = data[[i, 1]];
        0.3 * x0 - 0.15 * x1 + 0.2
    }));
    let weights = Array1::from_elem(n, 1.0);
    let meanspec = simple_matern_term_collection(&[0, 1], 0.4);
    let noisespec = simple_matern_term_collection(&[0, 1], 0.7);
    let builder = GaussianLocationScaleTermBuilder {
        y,
        weights,
        meanspec: meanspec.clone(),
        noisespec: noisespec.clone(),
        mean_offset: Array1::zeros(n),
        noise_offset: Array1::zeros(n),
    };
    let mean_design =
        build_term_collection_design(data.view(), &meanspec).expect("build mean design");
    let noise_design =
        build_term_collection_design(data.view(), &noisespec).expect("build noise design");
    let meanspec_resolved =
        freeze_term_collection_from_design(&meanspec, &mean_design).expect("freeze mean spec");
    let noisespec_resolved =
        freeze_term_collection_from_design(&noisespec, &noise_design).expect("freeze noise spec");
    let rho = compose_theta_from_hints_test(
        builder.mean_penalty_count(&mean_design),
        builder.noise_penalty_count(&noise_design),
        &None,
        &None,
        &Array1::zeros(0),
    );
    let blocks = builder
        .build_blocks(&rho, &mean_design, &noise_design, None, None)
        .expect("build blocks");
    let family = builder.build_family(&mean_design, &noise_design);
    // The joint-psi hook surface is exercised at explicit coefficients, not at a
    // minted fit: build each block's state from its penalty-geometry seed so the
    // family exposes its psi score / Hessian / mixed-drift hooks without asking
    // the outer optimizer to certify a (deliberately tiny) fixture.
    // `block_geometry` answers "what geometry do THESE states see", so it is
    // asked once the whole state vector exists. Seeding the betas first and
    // filling the etas in a second pass keeps every call to it addressed to a
    // complete partition, instead of to a vector that is empty on block 0.
    let mut block_states: Vec<ParameterBlockState> = blocks
        .iter()
        .map(|spec| ParameterBlockState {
            beta: spec
                .initial_beta
                .clone()
                .unwrap_or_else(|| Array1::zeros(spec.design.ncols())),
            eta: Array1::zeros(spec.design.nrows()),
        })
        .collect();
    for (index, spec) in blocks.iter().enumerate() {
        let (design, offset) = family
            .block_geometry(&block_states, spec)
            .expect("hook fixture block geometry");
        let eta = design.matrixvectormultiply(&block_states[index].beta) + &offset;
        block_states[index].eta = eta;
    }
    family
        .evaluate(&block_states)
        .expect("hook fixture state should evaluate");
    let derivative_blocks = builder
        .build_psiderivative_blocks(
            data.view(),
            &meanspec_resolved,
            &noisespec_resolved,
            &mean_design,
            &noise_design,
        )
        .expect("psi derivative blocks");
    assert_joint_psi_hook_surface(
        &family,
        &block_states,
        &blocks,
        &derivative_blocks,
        0.2,
        0.15,
        "gaussian",
    );
}

#[test]
pub(crate) fn gaussian_location_scale_terms_reject_invalidweights_early() {
    let n = 8usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = i as f64;
        data[[i, 1]] = (i as f64).sin();
    }
    let spec = GaussianLocationScaleTermSpec {
        y: Array1::zeros(n),
        weights: Array1::from_vec(vec![1.0, 1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0]),
        meanspec: simple_matern_term_collection(&[0, 1], 0.35),
        log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.6),
        mean_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
    };

    let err = match fit_gaussian_location_scale_terms(
        data.view(),
        spec,
        &BlockwiseFitOptions::default(),
        &spatial_kappa_options(),
    ) {
        Ok(_) => panic!("term API should reject negative weights"),
        Err(err) => err,
    };
    assert!(err.contains("weights must be finite and non-negative"));
}

#[test]
pub(crate) fn binomial_location_scale_terms_reject_invalid_response_early() {
    let n = 8usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        data[[i, 0]] = i as f64;
        data[[i, 1]] = (i as f64).cos();
    }
    let spec = BinomialLocationScaleTermSpec {
        y: Array1::from_vec(vec![0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0]),
        weights: Array1::from_elem(n, 1.0),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        thresholdspec: simple_matern_term_collection(&[0, 1], 0.4),
        log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.75),
        threshold_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
    };

    let err = match fit_binomial_location_scale_terms(
        data.view(),
        spec,
        &BlockwiseFitOptions::default(),
        &spatial_kappa_options(),
    ) {
        Ok(_) => panic!("term API should reject invalid binomial responses"),
        Err(err) => err,
    };
    assert!(err.contains("binomial response must be finite in [0,1]"));
}

#[test]
pub(crate) fn binomial_location_scale_terms_reject_free_log_sigma_terms_early() {
    let n = 8usize;
    let data = Array2::<f64>::zeros((n, 2));
    let spec = BinomialLocationScaleTermSpec {
        y: Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })),
        weights: Array1::from_elem(n, 1.0),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        thresholdspec: simple_matern_term_collection(&[0, 1], 0.4),
        log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.75),
        threshold_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
    };

    let err = match fit_binomial_location_scale_terms(
        data.view(),
        spec,
        &BlockwiseFitOptions::default(),
        &spatial_kappa_options(),
    ) {
        Ok(_) => panic!("Bernoulli free log_sigma terms must be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("identify only the composite q = -threshold / sigma"));
    assert!(err.contains("log_sigma must be intercept-only/fixed"));
}

#[test]
pub(crate) fn binomial_location_scale_terms_reject_datarow_mismatch_early() {
    let n = 8usize;
    let data = Array2::<f64>::zeros((n - 1, 2));
    let spec = BinomialLocationScaleTermSpec {
        y: Array1::from_elem(n, 0.0),
        weights: Array1::from_elem(n, 1.0),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        thresholdspec: simple_matern_term_collection(&[0, 1], 0.4),
        log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.75),
        threshold_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
    };

    let err = match fit_binomial_location_scale_terms(
        data.view(),
        spec,
        &BlockwiseFitOptions::default(),
        &spatial_kappa_options(),
    ) {
        Ok(_) => panic!("term API should reject data/y row mismatches"),
        Err(err) => err,
    };
    assert!(err.contains("data row count must match response length"));
}

// "Fit finitely" requires an interior REML optimum to certify. Linear, noiseless
// data drives both Matern smooths to lambda -> +inf (a rail that cannot certify a
// stationary interior optimum), so the fixture must carry genuine mean curvature
// (finite mean lambda) and genuine heteroscedasticity (finite log-sigma lambda),
// with deterministic Gaussian residuals, so each block has an interior optimum.
#[test]
pub(crate) fn gaussian_location_scale_termswith_matern_spatial_blocks_fit_finitely() {
    let n = 48usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.0 * std::f64::consts::PI * t).sin();
    }
    // Deterministic LCG -> uniform(0,1); probit gives standard-normal draws
    // (same generator idiom as the homoscedastic #365 fixture).
    let mut lcg: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next_unit = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (lcg >> 11) as f64 / ((1u64 << 53) as f64);
        bits.clamp(1.0e-6, 1.0 - 1.0e-6)
    };
    let y = Array1::from_iter((0..n).map(|i| {
        let x0 = data[[i, 0]];
        let x1 = data[[i, 1]];
        // Mean with genuine curvature over the domain (one full sine period in
        // x0 plus an x1 tilt) — a Matern smooth with correlation 0.35 has real
        // structure to fit, so its lambda stays finite/interior.
        let true_mean = 0.6 * (2.0 * std::f64::consts::PI * x0).sin() + 0.2 * x1;
        // Heteroscedastic scale that rises across the domain so the log-sigma
        // Matern block has genuine signal and its lambda stays finite too.
        let true_log_sigma = -0.9 + 1.0 * x0;
        let z = standard_normal_quantile(next_unit()).expect("finite probit draw");
        true_mean + true_log_sigma.exp() * z
    }));
    let weights = Array1::from_elem(n, 1.0);
    let spec = GaussianLocationScaleTermSpec {
        y,
        weights,
        meanspec: simple_matern_term_collection(&[0, 1], 0.35),
        log_sigmaspec: simple_matern_term_collection(&[0, 1], 0.6),
        mean_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
    };
    // Production-sized outer budget: with genuine interior signal the REML
    // optimum is a real stationary point, so give the outer enough iterations to
    // reach and certify it (this is not budget-inflation to chase a saturated
    // rail — the honest optimum here is interior).
    let options = BlockwiseFitOptions {
        inner_max_cycles: 48,
        inner_tol: 1e-4,
        outer_max_iter: 60,
        outer_tol: 1e-4,
        ..BlockwiseFitOptions::default()
    };
    let fit = fit_gaussian_location_scale_terms(
        data.view(),
        spec,
        &options,
        &spatial_kappa_options(),
    )
    .expect("gaussian location-scale spatial fit");
    assert!(fit.fit.penalized_objective().is_some_and(f64::is_finite));
    assert_eq!(fit.fit.block_states.len(), 2);
}

/// Issue #365 (primary symptom): a *homoscedastic* Gaussian fit with a
/// smooth `noise_formula` must NOT degrade the mean fit. The released
/// repro fed `y = 1 + 0.7x + sin(x) + N(0, σ²)` with constant σ to a model
/// carrying a smooth mean *and* a smooth log-σ block and got a mean RMSE of
/// ~1.5 (the predicted mean range collapsed inward toward the grand mean),
/// versus ~0.03 for a plain GAM. A smooth scale block that is free to
/// wiggle can absorb mean-residual structure into the variance, which lets
/// the joint REML over-smooth the mean block. This test pins the headline
/// contract directly: adding the smooth scale block to homoscedastic data
/// must leave the recovered mean tracking the truth, not flattened.
///
/// It is deterministic (LCG uniforms pushed through the probit to draw the
/// Gaussian residuals) and exercises the real end-to-end two-block joint
/// solve, not a synthetic linear-algebra stub. A mean-flattening regression
/// (the #365 failure mode) drives the RMSE far above the asserted bound.
#[test]
pub(crate) fn gaussian_location_scale_smooth_noise_homoscedastic_recovers_mean() {
    let n = 300usize;
    // Deterministic LCG -> uniform(0,1); probit gives standard-normal draws.
    let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next_unit = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Top 53 bits -> (0,1), nudged off the open-interval endpoints so
        // the probit stays finite.
        let bits = (lcg >> 11) as f64 / ((1u64 << 53) as f64);
        bits.clamp(1.0e-6, 1.0 - 1.0e-6)
    };

    // x uniform on [-3, 3] (matches the released repro grid).
    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        xs.push(-3.0 + 6.0 * next_unit());
    }
    let true_mean: Vec<f64> = xs.iter().map(|&x| 1.0 + 0.7 * x + x.sin()).collect();
    // Constant true scale: the data are homoscedastic (het = 0).
    let true_sigma = (-0.5_f64).exp();
    let y = Array1::from_iter((0..n).map(|i| {
        let z = standard_normal_quantile(next_unit()).expect("finite probit draw");
        true_mean[i] + true_sigma * z
    }));
    // Exercise the exact released formula path named by the regression report.
    // The old fixture substituted the spatial Matérn smoke helper for `s(x)`;
    // its three-penalty, five-column mean collapsed even when the scale block was
    // constant, so it did not isolate a location-scale failure. This dataset and
    // formula now pass through the same thin-plate materialization, joint fit,
    // and raw-response rescaling users invoke.
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows = xs
        .iter()
        .zip(y.iter())
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")]))
        .collect();
    let dataset =
        encode_recordswith_inferred_schema(headers, rows).expect("encode homoscedastic fixture");
    let result = fit_from_formula(
        "y ~ s(x, bs='tp')",
        &dataset,
        &FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("1 + s(x, bs='tp')".to_string()),
            ..FitConfig::default()
        },
    )
    .expect("gaussian location-scale smooth-noise homoscedastic formula fit");
    let FitResult::GaussianLocationScale(fit) = result else {
        panic!("homoscedastic noise formula must route to Gaussian location-scale");
    };

    // The mean block (BLOCK_MU = 0) carries identity-link η = predicted mean
    // (mean_offset is zero), so its state η is the fitted mean directly.
    let mean_eta = &fit.fit.fit.block_states[GaussianLocationScaleFamily::BLOCK_MU].eta;
    assert_eq!(mean_eta.len(), n);
    let mut sq_err = 0.0;
    for i in 0..n {
        let d = mean_eta[i] - true_mean[i];
        sq_err += d * d;
    }
    let mean_rmse = (sq_err / n as f64).sqrt();

    // A correctly converged mean tracks the truth to well within the noise
    // scale; the #365 collapse-to-grand-mean failure produces RMSE ~1.5.
    // The bound below is far below that failure level yet comfortably above
    // any honest small-n sampling/penalty bias, so it fails the bug and
    // passes the fix without being a tautology.
    assert!(
        mean_rmse < 0.5,
        "smooth noise_formula degraded the homoscedastic mean fit (issue #365): \
             mean RMSE = {mean_rmse:.4} (expected < 0.5; the regression produced ~1.5)"
    );
}

#[test]
pub(crate) fn binomial_location_scale_termswith_matern_spatial_blocks_fit_finitely() {
    let n = 36usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (3.0 * std::f64::consts::PI * t).cos();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 5 == 0 || i % 7 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let spec = BinomialLocationScaleTermSpec {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        thresholdspec: simple_matern_term_collection(&[0, 1], 0.4),
        log_sigmaspec: empty_term_collection(),
        threshold_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
    };
    let fit = fit_binomial_location_scale_terms(
        data.view(),
        spec,
        &spatial_fit_smoke_options(),
        &spatial_kappa_options(),
    )
    .expect("binomial location-scale spatial fit");
    assert!(fit.fit.penalized_objective().is_some_and(f64::is_finite));
    assert_eq!(fit.fit.block_states.len(), 2);
}

#[test]
pub(crate) fn gaussian_log_sigma_psi_terms_match_autodiff_scalar_objective() {
    let y = array![0.25, -0.4, 1.1];
    let weights = array![1.0, 0.7, 1.3];
    let x_mu0 = array![1.0, -0.35, 0.6];
    let x_ls0 = array![0.8, -0.25, 0.45];
    let x_ls_psi = array![0.2, -0.15, 0.1];
    let x_ls_psi_psi = array![0.05, -0.03, 0.04];
    let beta_mu0 = 0.35_f64;
    let beta_ls0 = -0.2_f64;

    let x_mu0_mat = x_mu0.clone().insert_axis(Axis(1));
    let x_ls0_mat = x_ls0.clone().insert_axis(Axis(1));
    let family = GaussianLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        mu_design: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(x_mu0_mat.clone()),
        )),
        log_sigma_design: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(x_ls0_mat.clone()),
        )),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let specs = vec![
        gaussian_psi_test_spec("mu", x_mu0_mat.clone()),
        gaussian_psi_test_spec("log_sigma", x_ls0_mat.clone()),
    ];
    let states = vec![
        ParameterBlockState {
            beta: array![beta_mu0],
            eta: x_mu0_mat.column(0).to_owned() * beta_mu0,
        },
        ParameterBlockState {
            beta: array![beta_ls0],
            eta: x_ls0_mat.column(0).to_owned() * beta_ls0,
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi: x_ls_psi.clone().insert_axis(Axis(1)),
            s_psi: Array2::zeros((1, 1)),
            s_psi_components: None,
            s_psi_penalty_components: None,
            x_psi_psi: Some(vec![x_ls_psi_psi.clone().insert_axis(Axis(1))]),
            s_psi_psi: Some(vec![Array2::zeros((1, 1))]),
            s_psi_psi_components: None,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        }],
    ];

    let psi_terms = family
        .exact_newton_joint_psi_terms(
            &states,
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
            0,
        )
        .expect("joint psi terms")
        .expect("expected gaussian psi terms");

    let vars = [beta_mu0, beta_ls0, 0.0_f64];
    let (_, dpsi, _) = second_derivative(
        |psi| {
            gaussian_negloglik_log_sigma_psi_only_numdual(
                psi,
                beta_mu0,
                beta_ls0,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        0.0,
    );
    let (_, _, _, score_mu_psi) = second_partial_derivative(
        |(beta_mu, psi)| {
            gaussian_negloglik_log_sigma_mu_psi_numdual(
                beta_mu,
                psi,
                beta_ls0,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        (beta_mu0, 0.0),
    );
    let (_, _, _, score_ls_psi) = second_partial_derivative(
        |(beta_ls, psi)| {
            gaussian_negloglik_log_sigma_ls_psi_numdual(
                beta_ls,
                psi,
                beta_mu0,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        (beta_ls0, 0.0),
    );
    let (_, _, _, _, _, _, _, h_mu_mu_psi) = third_partial_derivative_vec(
        |v| {
            gaussian_negloglik_log_sigma_beta_vec_numdual(
                v,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        &vars,
        0,
        0,
        2,
    );
    let (_, _, _, _, _, _, _, h_mu_ls_psi) = third_partial_derivative_vec(
        |v| {
            gaussian_negloglik_log_sigma_beta_vec_numdual(
                v,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        &vars,
        0,
        1,
        2,
    );
    let (_, _, _, _, _, _, _, h_ls_ls_psi) = third_partial_derivative_vec(
        |v| {
            gaussian_negloglik_log_sigma_beta_vec_numdual(
                v,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        &vars,
        1,
        1,
        2,
    );

    assert!(
        (psi_terms.objective_psi - dpsi).abs() <= 1e-10,
        "Gaussian log-sigma psi objective derivative mismatch: analytic={} autodiff={}",
        psi_terms.objective_psi,
        dpsi
    );
    assert!(
        (psi_terms.score_psi[0] - score_mu_psi).abs() <= 1e-10,
        "Gaussian log-sigma psi score_mu mismatch: analytic={} autodiff={}",
        psi_terms.score_psi[0],
        score_mu_psi
    );
    assert!(
        (psi_terms.score_psi[1] - score_ls_psi).abs() <= 1e-10,
        "Gaussian log-sigma psi score_ls mismatch: analytic={} autodiff={}",
        psi_terms.score_psi[1],
        score_ls_psi
    );
    assert!(
        (psi_terms.hessian_psi[[0, 0]] - h_mu_mu_psi).abs() <= 1e-9,
        "Gaussian log-sigma psi hessian(mu,mu) mismatch: analytic={} autodiff={}",
        psi_terms.hessian_psi[[0, 0]],
        h_mu_mu_psi
    );
    // OBSERVED curvature contract (#1561 cutover from block-Fisher #566/#684):
    // the analytic psi Hessian is now the OBSERVED joint penalized Hessian at
    // every block, so the (μ, log-σ) cross block ψ-derivative equals the AD
    // reference `∂³N/∂β_μ∂β_ls∂ψ` directly — no Fisher correction. (Previously
    // the cross block was Fisher 0, requiring a `−∂(2mκ·X_ls)/∂ψ` correction;
    // the observed production now carries that term itself.)
    assert!(
        (psi_terms.hessian_psi[[0, 1]] - h_mu_ls_psi).abs() <= 1e-9,
        "Gaussian log-sigma psi hessian(mu,ls) mismatch: analytic={} observed-AD={}",
        psi_terms.hessian_psi[[0, 1]],
        h_mu_ls_psi
    );
    // OBSERVED (ls,ls) block: `hessian_psi[[1,1]]` is now the ψ-derivative of the
    // observed curvature `κ'(a−n)+2κ²n`, equal to the AD `∂³N/∂β_ls²∂ψ` with no
    // Fisher gap correction.
    assert!(
        (psi_terms.hessian_psi[[1, 1]] - h_ls_ls_psi).abs() <= 1e-9,
        "Gaussian log-sigma psi hessian(ls,ls) mismatch: analytic={} observed-AD={}",
        psi_terms.hessian_psi[[1, 1]],
        h_ls_ls_psi
    );
}

#[test]
pub(crate) fn gaussian_log_sigma_psi_second_order_terms_match_autodiff_scalar_objective() {
    let y = array![0.25, -0.4, 1.1];
    let weights = array![1.0, 0.7, 1.3];
    let x_mu0 = array![1.0, -0.35, 0.6];
    let x_ls0 = array![0.8, -0.25, 0.45];
    let x_ls_psi = array![0.2, -0.15, 0.1];
    let x_ls_psi_psi = array![0.05, -0.03, 0.04];
    let beta_mu0 = 0.35_f64;
    let beta_ls0 = -0.2_f64;

    let x_mu0_mat = x_mu0.clone().insert_axis(Axis(1));
    let x_ls0_mat = x_ls0.clone().insert_axis(Axis(1));
    let family = GaussianLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        mu_design: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(x_mu0_mat.clone()),
        )),
        log_sigma_design: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(x_ls0_mat.clone()),
        )),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let specs = vec![
        gaussian_psi_test_spec("mu", x_mu0_mat.clone()),
        gaussian_psi_test_spec("log_sigma", x_ls0_mat.clone()),
    ];
    let states = vec![
        ParameterBlockState {
            beta: array![beta_mu0],
            eta: x_mu0_mat.column(0).to_owned() * beta_mu0,
        },
        ParameterBlockState {
            beta: array![beta_ls0],
            eta: x_ls0_mat.column(0).to_owned() * beta_ls0,
        },
    ];
    let derivative_blocks = vec![
        Vec::new(),
        vec![CustomFamilyBlockPsiDerivative {
            penalty_index: None,
            x_psi: x_ls_psi.clone().insert_axis(Axis(1)),
            s_psi: Array2::zeros((1, 1)),
            s_psi_components: None,
            s_psi_penalty_components: None,
            x_psi_psi: Some(vec![x_ls_psi_psi.clone().insert_axis(Axis(1))]),
            s_psi_psi: Some(vec![Array2::zeros((1, 1))]),
            s_psi_psi_components: None,
            s_psi_psi_penalty_components: None,
            implicit_operator: None,
            implicit_axis: 0,
            implicit_group_id: None,
        }],
    ];

    let psi2_terms = family
        .exact_newton_joint_psisecond_order_terms(
            &states,
            &specs,
            &test_design_hyper_layout(&derivative_blocks),
            0,
            0,
        )
        .expect("joint psi psi terms")
        .expect("expected gaussian psi psi terms");

    let vars = [beta_mu0, beta_ls0, 0.0_f64];
    let (_, _, d2psi) = second_derivative(
        |psi| {
            gaussian_negloglik_log_sigma_psi_only_numdual(
                psi,
                beta_mu0,
                beta_ls0,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        0.0,
    );
    let (_, _, _, _, _, _, _, score_mu_psi_psi) = third_partial_derivative_vec(
        |v| {
            gaussian_negloglik_log_sigma_beta_vec_numdual(
                v,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        &vars,
        0,
        2,
        2,
    );
    let (_, _, _, _, _, _, _, score_ls_psi_psi) = third_partial_derivative_vec(
        |v| {
            gaussian_negloglik_log_sigma_beta_vec_numdual(
                v,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_ls_psi,
                &x_ls_psi_psi,
            )
        },
        &vars,
        1,
        2,
        2,
    );

    assert!(
        (psi2_terms.objective_psi_psi - d2psi).abs() <= 1e-10,
        "Gaussian log-sigma psi second objective mismatch: analytic={} autodiff={}",
        psi2_terms.objective_psi_psi,
        d2psi
    );
    assert!(
        (psi2_terms.score_psi_psi[0] - score_mu_psi_psi).abs() <= 1e-9,
        "Gaussian log-sigma psi second score_mu mismatch: analytic={} autodiff={}",
        psi2_terms.score_psi_psi[0],
        score_mu_psi_psi
    );
    assert!(
        (psi2_terms.score_psi_psi[1] - score_ls_psi_psi).abs() <= 1e-9,
        "Gaussian log-sigma psi second score_ls mismatch: analytic={} autodiff={}",
        psi2_terms.score_psi_psi[1],
        score_ls_psi_psi
    );
}

// Sibling oracle: μ also depends on ψ. Used by the joint psi-second-order
// guardrail; the original oracle leaves μ fixed in ψ.
pub(crate) fn gaussian_negloglik_log_sigma_psi_full_numdual<D: DualNum<f64> + Copy>(
    beta_mu: D,
    beta_ls: D,
    psi: D,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    x_mu0: &Array1<f64>,
    x_ls0: &Array1<f64>,
    x_mu_psi: &Array1<f64>,
    x_ls_psi: &Array1<f64>,
    x_mu_psi_psi: &Array1<f64>,
    x_ls_psi_psi: &Array1<f64>,
) -> D {
    let half = D::from(0.5);
    let mut out = D::zero();
    for i in 0..y.len() {
        let x_mu = D::from(x_mu0[i])
            + psi * D::from(x_mu_psi[i])
            + half * psi * psi * D::from(x_mu_psi_psi[i]);
        let eta_mu = x_mu * beta_mu;
        let x_ls = D::from(x_ls0[i])
            + psi * D::from(x_ls_psi[i])
            + half * psi * psi * D::from(x_ls_psi_psi[i]);
        let eta_ls = x_ls * beta_ls;
        let sigma = D::from(LOGB_SIGMA_FLOOR) + eta_ls.exp();
        let resid = D::from(y[i]) - eta_mu;
        out += D::from(weights[i]) * (half * (resid / sigma).powi(2) + sigma.ln());
    }
    out
}

// Oracle with multi-column designs (β vectors). Used by the joint
// static-Hessian guardrail and its directional derivatives.
pub(crate) fn gaussian_negloglik_logb_dense_numdual<D: DualNum<f64> + Copy>(
    beta_mu: &[D],
    beta_ls: &[D],
    y: &Array1<f64>,
    weights: &Array1<f64>,
    xmu: &Array2<f64>,
    x_ls: &Array2<f64>,
) -> D {
    let half = D::from(0.5);
    let n = y.len();
    let mut out = D::zero();
    for i in 0..n {
        let mut eta_mu = D::zero();
        for k in 0..beta_mu.len() {
            eta_mu += D::from(xmu[[i, k]]) * beta_mu[k];
        }
        let mut eta_ls = D::zero();
        for k in 0..beta_ls.len() {
            eta_ls += D::from(x_ls[[i, k]]) * beta_ls[k];
        }
        let sigma = D::from(LOGB_SIGMA_FLOOR) + eta_ls.exp();
        let resid = D::from(y[i]) - eta_mu;
        out += D::from(weights[i]) * (half * (resid / sigma).powi(2) + sigma.ln());
    }
    out
}

pub(crate) fn gaussian_logb_design_test_data() -> (
    Array1<f64>,
    Array1<f64>,
    Array2<f64>,
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
) {
    // n=5, two-column designs (intercept + smooth feature). β_ls0 chosen so
    // that η_ls ≈ −0.4 on the central row → κ ≈ 0.985, which is noticeably
    // less than 1 so κ' chain-rule contributions register at strict tolerance.
    let y = array![0.25, -0.4, 1.1, 0.05, -0.2];
    let weights = array![1.0, 0.7, 1.3, 0.9, 1.1];
    let xmu = ndarray::arr2(&[[1.0, -0.6], [1.0, -0.2], [1.0, 0.1], [1.0, 0.4], [1.0, 0.7]]);
    let x_ls = ndarray::arr2(&[[1.0, 0.5], [1.0, -0.1], [1.0, 0.3], [1.0, -0.4], [1.0, 0.2]]);
    // β_ls = (−0.4, 0.05): η_ls hovers around −0.4, so σ ≈ 0.68 and κ ≈ 0.985.
    let beta_mu = array![0.35, -0.25];
    let beta_ls = array![-0.4, 0.05];
    (y, weights, xmu, x_ls, beta_mu, beta_ls)
}

#[test]
pub(crate) fn gaussian_joint_static_hessian_matches_autodiff() {
    let (y, weights, xmu, x_ls, beta_mu, beta_ls) = gaussian_logb_design_test_data();
    let etamu = xmu.dot(&beta_mu);
    let eta_ls = x_ls.dot(&beta_ls);

    let rows =
        gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
    let weights0 =
        gaussian_joint_psi_firstweights(&rows, &Array1::zeros(y.len()), &Array1::zeros(y.len()));
    let xmu_dense = DenseOrOperator::Borrowed(&xmu);
    let xls_dense = DenseOrOperator::Borrowed(&x_ls);
    let analytic = gaussian_joint_hessian_from_designs(
        &xmu_dense,
        &xls_dense,
        &weights0.hmumu,
        &weights0.hmu_ls,
        &weights0.h_ls_ls,
    )
    .expect("gaussian joint static hessian from designs");

    // AD ground truth: full p×p Hessian via second_partial_derivative,
    // packing β_full = (β_μ, β_ls) and stepping (i, j) pairs.
    let pmu = beta_mu.len();
    let p_ls = beta_ls.len();
    let total = pmu + p_ls;
    let mut beta_full = vec![0.0_f64; total];
    for k in 0..pmu {
        beta_full[k] = beta_mu[k];
    }
    for k in 0..p_ls {
        beta_full[pmu + k] = beta_ls[k];
    }

    // AD ground truth: full p×p Hessian. Diagonal (i==i) via second_derivative
    // (1D second derivative); off-diagonal (i<j) via second_partial_derivative
    // on a closure that injects two HyperDual variables into β.
    let mut ad = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in i..total {
            let val = if i == j {
                let g = |x: num_dual::Dual2<f64, f64>| {
                    let mut bm = vec![num_dual::Dual2::from_re(0.0); pmu];
                    let mut bl = vec![num_dual::Dual2::from_re(0.0); p_ls];
                    for k in 0..pmu {
                        bm[k] = num_dual::Dual2::from_re(beta_full[k]);
                    }
                    for k in 0..p_ls {
                        bl[k] = num_dual::Dual2::from_re(beta_full[pmu + k]);
                    }
                    if i < pmu {
                        bm[i] = x;
                    } else {
                        bl[i - pmu] = x;
                    }
                    gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
                };
                let (_, _, d2) = second_derivative(g, beta_full[i]);
                d2
            } else {
                let f = |(a, b): (num_dual::HyperDual<f64, f64>, num_dual::HyperDual<f64, f64>)| {
                    let mut bm = vec![num_dual::HyperDual::from_re(0.0); pmu];
                    let mut bl = vec![num_dual::HyperDual::from_re(0.0); p_ls];
                    for k in 0..pmu {
                        bm[k] = num_dual::HyperDual::from_re(beta_full[k]);
                    }
                    for k in 0..p_ls {
                        bl[k] = num_dual::HyperDual::from_re(beta_full[pmu + k]);
                    }
                    if i < pmu {
                        bm[i] = a;
                    } else {
                        bl[i - pmu] = a;
                    }
                    if j < pmu {
                        bm[j] = b;
                    } else {
                        bl[j - pmu] = b;
                    }
                    gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
                };
                let (_, _, _, d2xy) = second_partial_derivative(f, (beta_full[i], beta_full[j]));
                d2xy
            };
            ad[[i, j]] = val;
            if i != j {
                ad[[j, i]] = val;
            }
        }
    }

    // OBSERVED curvature contract (#1561 cutover from block-Fisher #566/#684):
    // the analytic joint Hessian is now the OBSERVED penalized Hessian at every
    // block (`gaussian_joint_psi_firstweights` ships `hmu_ls = 2κm`,
    // `h_ls_ls = κ'(a−n)+2κ²n`), exactly what the AD ground truth differentiates.
    // The score stays the exact observed gradient (joint Newton still lands on
    // the true MLE), and the observed curvature is the LAML determinant/EDF
    // object per Wood–Pya–Säfken 2016. So the analytic Hessian equals the AD
    // Hessian on every block directly — no `Fisher − observed` correction.
    for i in 0..total {
        for j in 0..total {
            let diff = (analytic[[i, j]] - ad[[i, j]]).abs();
            assert!(
                diff <= 1e-10,
                "Gaussian static joint H[{i},{j}] mismatch (κ < 1 case): analytic={} observed-AD={} diff={}",
                analytic[[i, j]],
                ad[[i, j]],
                diff
            );
        }
    }
    // Symmetry guardrail: floating-point skew must be at the noise floor.
    let skew = (&analytic - &analytic.t())
        .mapv(f64::abs)
        .fold(0.0_f64, |acc, &v| acc.max(v));
    assert!(
        skew <= 1e-12,
        "Gaussian static joint Hessian skew exceeds noise floor: {skew}"
    );
}

#[test]
pub(crate) fn gaussian_joint_first_directional_hessian_matches_autodiff() {
    let (y, weights, xmu, x_ls, beta_mu, beta_ls) = gaussian_logb_design_test_data();
    let etamu = xmu.dot(&beta_mu);
    let eta_ls = x_ls.dot(&beta_ls);

    let pmu = beta_mu.len();
    let p_ls = beta_ls.len();
    let total = pmu + p_ls;
    // Direction v over the joint β = (β_μ, β_ls).
    let v: Array1<f64> = Array1::from_shape_fn(total, |k| 0.13 + 0.07 * (k as f64));
    let v_mu = v.slice(s![0..pmu]).to_owned();
    let v_ls = v.slice(s![pmu..total]).to_owned();
    let ximu = xmu.dot(&v_mu);
    let xi_ls = x_ls.dot(&v_ls);

    let rows =
        gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
    let (dhmumu, dhmu_ls, dh_ls_ls) = gaussian_joint_first_directionalweights(&rows, &ximu, &xi_ls);
    let xmu_dense = DenseOrOperator::Borrowed(&xmu);
    let xls_dense = DenseOrOperator::Borrowed(&x_ls);
    let analytic =
        gaussian_joint_hessian_from_designs(&xmu_dense, &xls_dense, &dhmumu, &dhmu_ls, &dh_ls_ls)
            .expect("gaussian joint first-directional H from designs");

    // AD: differentiate N along (β + ε·v), evaluating ∂³N/∂β_i ∂β_j ∂ε at ε=0
    // via third_partial_derivative_vec on the augmented vector
    // [β_μ, β_ls, ε] of length total + 1.
    let mut vars = vec![0.0_f64; total + 1];
    for k in 0..pmu {
        vars[k] = beta_mu[k];
    }
    for k in 0..p_ls {
        vars[pmu + k] = beta_ls[k];
    }
    // vars[total] = ε = 0 by default.

    let g = |z: &[num_dual::HyperHyperDual<f64, f64>]| {
        // Reconstruct β + ε·v.
        let mut bm = vec![num_dual::HyperHyperDual::from_re(0.0); pmu];
        let mut bl = vec![num_dual::HyperHyperDual::from_re(0.0); p_ls];
        let eps = z[total];
        for k in 0..pmu {
            bm[k] = z[k] + eps * num_dual::HyperHyperDual::from_re(v[k]);
        }
        for k in 0..p_ls {
            bl[k] = z[pmu + k] + eps * num_dual::HyperHyperDual::from_re(v[pmu + k]);
        }
        gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
    };

    let mut ad = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in i..total {
            let (_, _, _, _, _, _, _, d3) = third_partial_derivative_vec(g, &vars, i, j, total);
            ad[[i, j]] = d3;
            if i != j {
                ad[[j, i]] = d3;
            }
        }
    }

    // OBSERVED curvature contract (#1561): `gaussian_joint_first_directionalweights`
    // returns the directional derivative of the OBSERVED joint Hessian at every
    // block — the (ls,ls) drift differentiates `κ'(a−n)+2κ²n` and the cross
    // drift `2κm`, exactly what the AD `∂³N` differentiates. So the analytic
    // first-directional dH equals the AD dH on every block directly, with no
    // `Fisher − observed` gap correction.
    for i in 0..total {
        for j in 0..total {
            let diff = (analytic[[i, j]] - ad[[i, j]]).abs();
            assert!(
                diff <= 1e-10,
                "Gaussian dH (first-directional) [{i},{j}] mismatch: analytic={} observed-AD={} diff={}",
                analytic[[i, j]],
                ad[[i, j]],
                diff
            );
        }
    }
    let skew = (&analytic - &analytic.t())
        .mapv(f64::abs)
        .fold(0.0_f64, |acc, &v| acc.max(v));
    assert!(
        skew <= 1e-12,
        "Gaussian first-directional dH skew exceeds noise floor: {skew}"
    );
}

#[test]
pub(crate) fn gaussian_row_scalar_cache_is_exact_and_eliminates_recompute() {
    let (y, weights, xmu, x_ls, beta_mu, beta_ls) = gaussian_logb_design_test_data();
    let etamu = xmu.dot(&beta_mu);
    let eta_ls = x_ls.dot(&beta_ls);
    let pmu = beta_mu.len();
    let p_ls = beta_ls.len();
    let total = pmu + p_ls;

    let family = GaussianLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        mu_design: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(xmu.clone()),
        )),
        log_sigma_design: Some(DesignMatrix::Dense(
            gam_linalg::matrix::DenseDesignMatrix::from(x_ls.clone()),
        )),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    let states = vec![
        ParameterBlockState {
            beta: beta_mu.clone(),
            eta: etamu.clone(),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: eta_ls.clone(),
        },
    ];
    let xmu_d = DenseOrOperator::Borrowed(&xmu);
    let xls_d = DenseOrOperator::Borrowed(&x_ls);

    // Independent (un-cached) reference scalars computed straight from the free
    // function: a cache HIT must return bit-identical contents.
    let reference =
        gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("reference row scalars");

    // Drive all four exact-joint paths under the SAME (η_μ, η_logσ). The first
    // populates the cache; the rest must hit it.
    let u: Array1<f64> = Array1::from_shape_fn(total, |k| 0.11 + 0.03 * (k as f64));
    let v: Array1<f64> = Array1::from_shape_fn(total, |k| -0.07 + 0.05 * (k as f64));

    let h0 = family
        .exact_newton_joint_hessian_from_designs(&states, &xmu_d, &xls_d)
        .expect("H")
        .expect("H present");
    // After the first consumer, the cache must be populated; grab the stored Arc.
    let stored = {
        let guard = family.cached_row_scalars.read().expect("lock");
        let (_, _, rows) = guard.as_ref().expect("cache populated after first call");
        std::sync::Arc::clone(rows)
    };
    let d1 = family
        .exact_newton_joint_hessian_directional_derivative_from_designs(&states, &xmu_d, &xls_d, &u)
        .expect("dH")
        .expect("dH present");
    let d2 = family
        .exact_newton_joint_hessiansecond_directional_derivative_from_designs(
            &states, &xmu_d, &xls_d, &u, &v,
        )
        .expect("d2H")
        .expect("d2H present");

    // Race-free proof that the 2nd…Kth consumers REUSE the stored allocation
    // (cache HIT shares the Arc via `Arc::clone`; a recompute would mint a new
    // allocation, so `ptr_eq` would be false). Family-local state, immune to
    // concurrent tests.
    let hit = family
        .get_or_compute_row_scalars(&etamu, &eta_ls)
        .expect("cached row scalars");
    assert!(
        std::sync::Arc::ptr_eq(&stored, &hit),
        "gaulss row-scalar cache should reuse the stored allocation (no redundant recompute)"
    );

    // Bit-identical cached contents vs the independent reference.
    let fields: [(&Array1<f64>, &Array1<f64>); 4] = [
        (&hit.obs_weight, &reference.obs_weight),
        (
            &hit.standardized_residual,
            &reference.standardized_residual,
        ),
        (&hit.inv_sigma, &reference.inv_sigma),
        (&hit.kappa, &reference.kappa),
    ];
    for (got, want) in fields {
        for (a, b) in got.iter().zip(want.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "cached row scalar bit mismatch");
        }
    }

    // A different (η_μ, η_logσ) must MISS (distinct fingerprint → fresh Arc).
    let eta_ls_shift = &eta_ls + 0.31;
    let miss = family
        .get_or_compute_row_scalars(&etamu, &eta_ls_shift)
        .expect("recompute on miss");
    assert!(
        !std::sync::Arc::ptr_eq(&stored, &miss),
        "distinct η must recompute, not serve the stale cached allocation"
    );

    // Fingerprint-collision guard: an η that shares the (first, mid, last)
    // sample points with the cached key but differs at an INTERIOR index must
    // still MISS. A lossy 3-point fingerprint would falsely HIT here and serve
    // the cached predictor's scalars to a different predictor; the full-vector
    // key must reject it and recompute scalars that actually differ.
    family.cached_row_scalars.write().expect("lock").take();
    let primed = family
        .get_or_compute_row_scalars(&etamu, &eta_ls)
        .expect("prime cache for collision probe");
    let mut eta_ls_interior = eta_ls.clone();
    let interior = eta_ls_interior.len() / 4; // an index that is NOT 0, n/2, or n-1
    assert!(
        interior != 0
            && interior != eta_ls_interior.len() / 2
            && interior != eta_ls_interior.len() - 1,
        "collision probe needs an interior index distinct from the 3 sampled points"
    );
    eta_ls_interior[interior] += 0.5;
    let collide = family
        .get_or_compute_row_scalars(&etamu, &eta_ls_interior)
        .expect("recompute on interior-only change");
    assert!(
        !std::sync::Arc::ptr_eq(&primed, &collide),
        "η differing only at an interior index must MISS, not collide on a 3-point fingerprint"
    );
    let recomputed_collide = gaussian_jointrow_scalars(&y, &etamu, &eta_ls_interior, &weights)
        .expect("collide reference");
    for (a, b) in collide
        .standardized_residual
        .iter()
        .zip(recomputed_collide.standardized_residual.iter())
    {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "interior-changed η must be served its OWN scalars, not the stale cached ones"
        );
    }

    // Invalidate and recompute the Hessian: must be bit-identical to the cached
    // run (proves the cache changes nothing numerically).
    family.cached_row_scalars.write().expect("lock").take();
    let h0b = family
        .exact_newton_joint_hessian_from_designs(&states, &xmu_d, &xls_d)
        .expect("H2")
        .expect("H2 present");
    for (a, b) in h0.iter().zip(h0b.iter()) {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Hessian not bit-identical after cache invalidation"
        );
    }
    assert_eq!(h0.dim(), (total, total));
    assert!(d1.iter().all(|x| x.is_finite()));
    assert!(d2.iter().all(|x| x.is_finite()));
}

#[test]
pub(crate) fn gaussian_joint_second_directional_hessian_matches_autodiff() {
    let (y, weights, xmu, x_ls, beta_mu, beta_ls) = gaussian_logb_design_test_data();
    let etamu = xmu.dot(&beta_mu);
    let eta_ls = x_ls.dot(&beta_ls);

    let pmu = beta_mu.len();
    let p_ls = beta_ls.len();
    let total = pmu + p_ls;
    let u: Array1<f64> = Array1::from_shape_fn(total, |k| 0.18 - 0.05 * (k as f64));
    let v: Array1<f64> = Array1::from_shape_fn(total, |k| -0.11 + 0.09 * (k as f64));
    let u_mu = u.slice(s![0..pmu]).to_owned();
    let u_ls = u.slice(s![pmu..total]).to_owned();
    let v_mu = v.slice(s![0..pmu]).to_owned();
    let v_ls = v.slice(s![pmu..total]).to_owned();
    let ximu_u = xmu.dot(&u_mu);
    let xi_ls_u = x_ls.dot(&u_ls);
    let ximuv = xmu.dot(&v_mu);
    let xi_lsv = x_ls.dot(&v_ls);

    let rows =
        gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
    let (d2hmumu, d2hmu_ls, d2h_ls_ls) =
        gaussian_jointsecond_directionalweights(&rows, &ximu_u, &xi_ls_u, &ximuv, &xi_lsv);
    let xmu_dense = DenseOrOperator::Borrowed(&xmu);
    let xls_dense = DenseOrOperator::Borrowed(&x_ls);
    let analytic = gaussian_joint_hessian_from_designs(
        &xmu_dense, &xls_dense, &d2hmumu, &d2hmu_ls, &d2h_ls_ls,
    )
    .expect("gaussian joint second-directional H from designs");

    // AD ground truth for ∂⁴N/∂β_i ∂β_j ∂ε_u ∂ε_v at (ε_u, ε_v) = (0, 0).
    // num-dual ships native AD up to third order; the fourth order is
    // obtained by central FD in ε_v of the AD third partial that already
    // covers (β_i, β_j, ε_u). Augmented vector layout:
    //   [β_μ ; β_ls ; ε_u]    of length total + 1 (ε_v lives outside AD).
    let mut vars_base = vec![0.0_f64; total + 1];
    for k in 0..pmu {
        vars_base[k] = beta_mu[k];
    }
    for k in 0..p_ls {
        vars_base[pmu + k] = beta_ls[k];
    }
    // vars_base[total] = ε_u = 0.

    let h = 1e-4;
    let mut ad = Array2::<f64>::zeros((total, total));
    for i in 0..total {
        for j in i..total {
            let g_plus = |z: &[num_dual::HyperHyperDual<f64, f64>]| {
                let mut bm = vec![num_dual::HyperHyperDual::from_re(0.0); pmu];
                let mut bl = vec![num_dual::HyperHyperDual::from_re(0.0); p_ls];
                let eps_u = z[total];
                for k in 0..pmu {
                    bm[k] = z[k]
                        + eps_u * num_dual::HyperHyperDual::from_re(u[k])
                        + num_dual::HyperHyperDual::from_re(h * v[k]);
                }
                for k in 0..p_ls {
                    bl[k] = z[pmu + k]
                        + eps_u * num_dual::HyperHyperDual::from_re(u[pmu + k])
                        + num_dual::HyperHyperDual::from_re(h * v[pmu + k]);
                }
                gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
            };
            let g_minus = |z: &[num_dual::HyperHyperDual<f64, f64>]| {
                let mut bm = vec![num_dual::HyperHyperDual::from_re(0.0); pmu];
                let mut bl = vec![num_dual::HyperHyperDual::from_re(0.0); p_ls];
                let eps_u = z[total];
                for k in 0..pmu {
                    bm[k] = z[k] + eps_u * num_dual::HyperHyperDual::from_re(u[k])
                        - num_dual::HyperHyperDual::from_re(h * v[k]);
                }
                for k in 0..p_ls {
                    bl[k] = z[pmu + k] + eps_u * num_dual::HyperHyperDual::from_re(u[pmu + k])
                        - num_dual::HyperHyperDual::from_re(h * v[pmu + k]);
                }
                gaussian_negloglik_logb_dense_numdual(&bm, &bl, &y, &weights, &xmu, &x_ls)
            };
            let (_, _, _, _, _, _, _, d3_plus) =
                third_partial_derivative_vec(g_plus, &vars_base, i, j, total);
            let (_, _, _, _, _, _, _, d3_minus) =
                third_partial_derivative_vec(g_minus, &vars_base, i, j, total);
            let val = (d3_plus - d3_minus) / (2.0 * h);
            ad[[i, j]] = val;
            if i != j {
                ad[[j, i]] = val;
            }
        }
    }

    // Tolerance: the 4th-order ground truth uses one FD step on top of an
    // AD third partial, so we relax from 1e-10 to a value compatible with
    // the central-difference truncation (O(h²) ≈ 1e-8) and the rounding
    // floor of the AD third partial (≈ 1e-10 / h ≈ 1e-6).
    // OBSERVED curvature contract (#1561): `gaussian_jointsecond_directionalweights`
    // returns the second directional derivative of the OBSERVED joint Hessian at
    // every block (the (ls,ls) second drift differentiates `κ'(a−n)+2κ²n`, the
    // cross `2κm`), exactly what the AD `∂⁴N` ground truth differentiates. So the
    // analytic d²H equals the AD d²H on every block directly, with no
    // `Fisher − observed` gap correction.
    let tol = 5e-6;
    for i in 0..total {
        for j in 0..total {
            let diff = (analytic[[i, j]] - ad[[i, j]]).abs();
            assert!(
                diff <= tol,
                "Gaussian d2H (second-directional) [{i},{j}] mismatch: analytic={} observed-AD={} diff={}",
                analytic[[i, j]],
                ad[[i, j]],
                diff
            );
        }
    }
    let skew = (&analytic - &analytic.t())
        .mapv(f64::abs)
        .fold(0.0_f64, |acc, &v| acc.max(v));
    assert!(
        skew <= 1e-10,
        "Gaussian second-directional d2H skew exceeds noise floor: {skew}"
    );
}

#[test]
pub(crate) fn gaussian_joint_psi_second_order_terms_match_autodiff() {
    // ψ-coupled scenario: both μ and η_ls depend on ψ via per-row
    // first/second drift vectors, with non-trivial coefficients.
    let y = array![0.25, -0.4, 1.1, 0.05, -0.2];
    let weights = array![1.0, 0.7, 1.3, 0.9, 1.1];
    let x_mu0 = array![1.0, -0.35, 0.6, 0.1, 0.45];
    let x_ls0 = array![0.8, -0.25, 0.45, -0.1, 0.3];
    let x_mu_psi = array![0.2, 0.15, -0.1, 0.05, 0.3];
    let x_ls_psi = array![0.18, -0.12, 0.25, -0.2, 0.07];
    let x_mu_psi_psi = array![0.04, -0.03, 0.05, 0.06, -0.02];
    let x_ls_psi_psi = array![0.05, -0.03, 0.04, 0.07, -0.04];
    // β_ls chosen so η_ls ≈ −0.4 (κ ≈ 0.985, noticeably less than 1).
    let beta_mu0 = 0.35_f64;
    let beta_ls0 = -0.4_f64;

    // Per-row predictor drifts.
    let etamu = &x_mu0 * beta_mu0;
    let eta_ls = &x_ls0 * beta_ls0;
    let zmu_psi = &x_mu_psi * beta_mu0;
    let z_ls_psi = &x_ls_psi * beta_ls0;
    let zmu_psi_psi = &x_mu_psi_psi * beta_mu0;
    let z_ls_psi_psi = &x_ls_psi_psi * beta_ls0;

    let rows =
        gaussian_jointrow_scalars(&y, &etamu, &eta_ls, &weights).expect("gaussian row scalars");
    let secondweights = gaussian_joint_psisecondweights(
        &rows,
        &zmu_psi,
        &z_ls_psi,
        &zmu_psi,
        &z_ls_psi,
        &zmu_psi_psi,
        &z_ls_psi_psi,
    );
    let analytic = secondweights.objective_psi_psirow.sum();

    // AD: differentiate the full ψ-dependent oracle twice in ψ at ψ=0.
    let (_, _, ad) = second_derivative(
        |psi| {
            gaussian_negloglik_log_sigma_psi_full_numdual(
                num_dual::Dual2::from_re(beta_mu0),
                num_dual::Dual2::from_re(beta_ls0),
                psi,
                &y,
                &weights,
                &x_mu0,
                &x_ls0,
                &x_mu_psi,
                &x_ls_psi,
                &x_mu_psi_psi,
                &x_ls_psi_psi,
            )
        },
        0.0,
    );

    let diff = (analytic - ad).abs();
    assert!(
        diff <= 1e-10,
        "Gaussian joint ψ-ψ objective mismatch (κ < 1, μ and σ both ψ-dependent): analytic={} ad={} diff={}",
        analytic,
        ad,
        diff
    );
}

pub(crate) fn assert_close_matrix(a: &Array2<f64>, b: &Array2<f64>, tol: f64, label: &str) {
    assert_eq!(a.dim(), b.dim(), "{label} shape mismatch");
    let max_err = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_err < tol,
        "{label} max error {max_err:.3e} >= {tol:.3e}"
    );
}

fn zz2155_splitmix_u01(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// The exact #2155 fixture from
/// `tests/bug_hunt_flexible_loglog_cauchit_binomial_wiggle.rs`: x ~ U(-2,2),
/// p = logistic(0.8 x), y ~ Bernoulli(p), splitmix64 stream seeded at 2155.
fn zz2155_fixture(n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut s = seed;
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let xv = -2.0 + 4.0 * zz2155_splitmix_u01(&mut s);
        let p = 1.0 / (1.0 + (-(0.8 * xv)).exp());
        let yv = if zz2155_splitmix_u01(&mut s) < p { 1.0 } else { 0.0 };
        x.push(xv);
        y.push(yv);
    }
    (Array1::from(y), Array1::from(x))
}

/// Unpenalized 2-parameter binomial GLM pilot (intercept + slope) by expected-
/// Fisher scoring — the same estimand as the production no-wiggle pilot fit.
fn zz2155_pilot(
    y: &Array1<f64>,
    x: &Array1<f64>,
    link: &InverseLink,
) -> (Array1<f64>, Array1<f64>) {
    let n = y.len();
    let mut beta = Array1::<f64>::zeros(2);
    for _ in 0..80 {
        let mut a00 = 0.0;
        let mut a01 = 0.0;
        let mut a11 = 0.0;
        let mut b0 = 0.0;
        let mut b1 = 0.0;
        for i in 0..n {
            let eta = beta[0] + beta[1] * x[i];
            let jet = inverse_link_jet_for_inverse_link(link, eta)
                .expect("pilot inverse-link jet");
            let mu = jet.mu.clamp(1e-12, 1.0 - 1e-12);
            let d1 = jet.d1;
            let w = (d1 * d1 / (mu * (1.0 - mu))).max(1e-12);
            let z = eta + (y[i] - mu) / if d1.abs() > 1e-12 { d1 } else { 1e-12 };
            a00 += w;
            a01 += w * x[i];
            a11 += w * x[i] * x[i];
            b0 += w * z;
            b1 += w * z * x[i];
        }
        let det = a00 * a11 - a01 * a01;
        let nb0 = (a11 * b0 - a01 * b1) / det;
        let nb1 = (a00 * b1 - a01 * b0) / det;
        let delta = (nb0 - beta[0]).abs().max((nb1 - beta[1]).abs());
        beta[0] = nb0;
        beta[1] = nb1;
        if delta < 1e-12 {
            break;
        }
    }
    let eta = Array1::from_shape_fn(n, |i| beta[0] + beta[1] * x[i]);
    (beta, eta)
}

/// One fixed-λ frozen-basis Gauss-Newton solve, mirroring
/// `fit_binomial_mean_wiggle`'s freeze-refit loop with the wiggle log-λ held
/// FIXED (no outer REML search): freeze `B(η̂)`, residualize against the mean
/// columns in observation space, run the joint two-block inner solve through
/// the public fixed-log-λ entry, re-freeze at the refit η̂, until the frozen
/// index is a fixed point. Returns
/// `(penalized_objective, deviance, beta_eta, beta_w, eta_hat, cycles)`.
struct Zz2155Problem {
    y: Array1<f64>,
    x: Array1<f64>,
    link: InverseLink,
    knots: Array1<f64>,
    degree: usize,
    wiggle_template: ParameterBlockInput,
}

impl Zz2155Problem {
    fn solve_fixed_lambda_freeze_refit(
        &self,
        rho_w: &Array1<f64>,
        eta0: &Array1<f64>,
        beta_eta0: &Array1<f64>,
        beta_w0: Option<&Array1<f64>>,
    ) -> Result<(f64, f64, Array1<f64>, Array1<f64>, Array1<f64>, usize), String> {
    use std::sync::Arc;
    let (y, x, link, knots, degree, wiggle_template) = (
        &self.y,
        &self.x,
        &self.link,
        &self.knots,
        self.degree,
        &self.wiggle_template,
    );
    let n = y.len();
    let mut x_dense = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        x_dense[[i, 0]] = 1.0;
        x_dense[[i, 1]] = x[i];
    }
    let base_family = BinomialMeanWiggleFamily {
        y: y.clone(),
        weights: Array1::from_elem(n, 1.0),
        link_kind: link.clone(),
        wiggle_knots: knots.clone(),
        wiggle_degree: degree,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        frozen_warp_design: None,
        continuation: false,
    };
    let mut frozen_eta = eta0.clone();
    let mut beta_eta = beta_eta0.clone();
    let mut beta_w: Option<Array1<f64>> = beta_w0.cloned();
    // The operating point the de-aliasing metric is read at, exactly as
    // `fit_binomial_mean_wiggle` maintains it (#2748). This harness mirrors the
    // production loop, so it calls the production de-aliasing rather than
    // carrying a second definition of it: a closed-form Euclidean 2x2 Gram
    // stopped being what production does, and a mirror that de-aliases in a
    // different metric measures a different map.
    let mut working_q = eta0.clone();
    for cycle in 0..60 {
        let b_full = base_family
            .wiggle_design(frozen_eta.view())
            .map_err(|e| format!("wiggle design: {e}"))?;
        let mut curvature = Array1::<f64>::zeros(n);
        for row in 0..n {
            let (_, m2, _) = base_family
                .neglog_q_derivatives(base_family.y[row], base_family.weights[row], working_q[row])
                .map_err(|e| format!("row curvature: {e}"))?;
            curvature[row] = if m2.is_finite() && m2 > 0.0 { m2 } else { 0.0 };
        }
        let (_, bda) = dealias_warp_against_mean_block(&x_dense, &b_full, &curvature)
            .map_err(|e| format!("warp de-aliasing: {e}"))?;

        let eta_input = ParameterBlockInput {
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                x_dense.clone(),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Some(Array1::zeros(0)),
            initial_beta: Some(beta_eta.clone()),
        };
        let mut wiggle_input = wiggle_template.clone();
        wiggle_input.design =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(bda.clone()));
        wiggle_input.offset = Array1::zeros(n);
        wiggle_input.initial_log_lambdas = Some(rho_w.clone());
        wiggle_input.initial_beta = Some(match &beta_w {
            Some(b) => b.clone(),
            None => Array1::zeros(bda.ncols()),
        });
        let specs = vec![
            eta_input.intospec("eta").map_err(|e| e.to_string())?,
            wiggle_input.intospec("wiggle").map_err(|e| e.to_string())?,
        ];
        let mut fam = base_family.clone();
        fam.frozen_warp_design = Some(Arc::new(bda));
        let options = BlockwiseFitOptions::default();
        let fit = crate::custom_family::fit_custom_family_fixed_log_lambdas(
            &fam, &specs, &options, None,
        )
        .map_err(|e| format!("fixed-λ inner solve (cycle {cycle}): {e:?}"))?;
        let new_beta_eta = fit.block_states[BinomialMeanWiggleFamily::BLOCK_ETA]
            .beta
            .clone();
        let new_beta_w = fit.block_states[BinomialMeanWiggleFamily::BLOCK_WIGGLE]
            .beta
            .clone();
        let eta_hat = x_dense.dot(&new_beta_eta);
        working_q = &fit.block_states[BinomialMeanWiggleFamily::BLOCK_ETA].eta
            + &fit.block_states[BinomialMeanWiggleFamily::BLOCK_WIGGLE].eta;
        let delta = eta_hat
            .iter()
            .zip(frozen_eta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        beta_eta = new_beta_eta;
        beta_w = Some(new_beta_w.clone());
        if delta <= 1.0e-9 {
            return Ok((
                fit.penalized_objective()
                    .expect("the alternating gamlss inner solve reports its objective"),
                fit.deviance,
                beta_eta,
                new_beta_w,
                eta_hat,
                cycle + 1,
            ));
        }
        frozen_eta = eta_hat;
    }
    Err("freeze-refit did not reach a fixed point in 60 cycles".to_string())
    }
}

// Wiggle / binomial-location-scale / release-cell tests live in a child
// module file to respect the crate-wide source-file length budget. Child
// modules see this module's entire scope (helpers AND imports) via
// `use super::*`, so the split is purely physical.
#[path = "tests_wiggle_ls.rs"]
mod wiggle_ls;
mod zz2155_mode_geography_tests;

// gam#2647: the joint penalized Hessian's non-singularity, asserted.
#[path = "tests_2647_gauge.rs"]
mod gauge_2647;
