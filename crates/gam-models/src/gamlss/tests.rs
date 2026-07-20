// Behavior tests for the gamlss family stack (real `#[cfg(test)] mod tests`).
// `super::*` resolves to the parent `gamlss` module, whose flat re-exports
// surface every concern-submodule item these tests exercise.

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
use super::dispersion_family::{DispersionRowKernel, dispersion_row_kernel};
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
    initializewiggle_knots_from_seed, monotone_wiggle_internal_degree, split_wiggle_penalty_orders,
};
use gam_data::encode_recordswith_inferred_schema;
use gam_terms::basis::{
    CenterStrategy, Dense, KnotSource, MaternBasisSpec, MaternIdentifiability, MaternNu,
    create_basis,
};
use gam_terms::smooth::{ShapeConstraint, SmoothBasisSpec, SmoothTermSpec};
use gam_test_support::{binomial_location_scale_base_fixture, no_densify_design};
use ndarray::{Array2, Axis, array};
use num_dual::{
    DualNum, second_derivative, second_partial_derivative, third_partial_derivative_vec,
};
use statrs::function::gamma::ln_gamma;

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

pub(crate) fn intercept_block(n: usize) -> ParameterBlockInput {
    ParameterBlockInput {
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
            Array2::from_elem((n, 1), 1.0),
        )),
        offset: Array1::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: vec![],
        initial_log_lambdas: None,
        initial_beta: None,
    }
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

pub(crate) struct HandNonWiggleQDirectional {
    pub(crate) delta_q: f64,
    pub(crate) delta_q_t: f64,
    pub(crate) delta_q_ls: f64,
    pub(crate) delta_q_tl: f64,
    pub(crate) delta_q_ll: f64,
}

pub(crate) fn hand_binomial_q_directional(
    q: NonWiggleQDerivs,
    d_eta_t: f64,
    d_eta_ls: f64,
) -> HandNonWiggleQDirectional {
    HandNonWiggleQDirectional {
        delta_q: q.q_t * d_eta_t + q.q_ls * d_eta_ls,
        delta_q_t: q.q_tl * d_eta_ls,
        delta_q_ls: q.q_tl * d_eta_t + q.q_ll * d_eta_ls,
        delta_q_tl: q.q_tl_ls * d_eta_ls,
        delta_q_ll: q.q_tl_ls * d_eta_t + q.q_ll_ls * d_eta_ls,
    }
}

pub(crate) fn hand_binomial_second_q_directional(
    q: NonWiggleQDerivs,
    u_t: f64,
    u_ls: f64,
    v_t: f64,
    v_ls: f64,
) -> (f64, f64, f64, f64, f64) {
    let d2q = q.q_tl * (u_t * v_ls + v_t * u_ls) + q.q_ll * u_ls * v_ls;
    let d2q_t = q.q_tl_ls * u_ls * v_ls;
    let d2q_ls = q.q_tl_ls * (u_ls * v_t + v_ls * u_t) + q.q_ll_ls * u_ls * v_ls;
    let d2q_tl = -q.q_tl_ls * u_ls * v_ls;
    let d2q_ll = d2q;
    (d2q, d2q_t, d2q_ls, d2q_tl, d2q_ll)
}

#[test]
pub(crate) fn binomial_nonwiggle_tower_matches_hand_witness_channels() {
    let links = [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
    ];
    let dirs = [([0.7, -0.4], [-0.2, 0.9]), ([1.3, 0.2], [0.5, -1.1])];
    for link in links {
        for y in [0.0, 1.0] {
            for weight in [0.25, 2.0] {
                for q_target in [-8.0, -6.0, -1.25, 0.0, 1.4, 6.0, 8.0] {
                    for eta_ls in [-0.7_f64, 0.0, 0.9] {
                        let sigma = eta_ls.exp();
                        let eta_t = -q_target * sigma;
                        let jet = inverse_link_jet_for_inverse_link(&link, q_target)
                            .expect("binomial link jet");
                        let tower = binomial_location_scale_nll_tower(
                            y, weight, eta_t, eta_ls, q_target, jet.mu, jet.d1, jet.d2, jet.d3,
                            &link, true,
                        )
                        .expect("binomial row tower");
                        let (m1, m2, m3) = binomial_neglog_q_derivatives_dispatch(
                            y, weight, q_target, jet.mu, jet.d1, jet.d2, jet.d3, &link,
                        );
                        let m4 = binomial_neglog_q_fourth_derivative_dispatch(
                            y, weight, q_target, jet.mu, jet.d1, jet.d2, jet.d3, &link,
                        )
                        .expect("binomial m4");
                        let q = nonwiggle_q_derivs(eta_t, sigma);
                        let h_tt = hessian_coeff_fromobjective_q_terms(m1, m2, q.q_t, q.q_t, 0.0);
                        let h_tl =
                            hessian_coeff_fromobjective_q_terms(m1, m2, q.q_t, q.q_ls, q.q_tl);
                        let h_ll =
                            hessian_coeff_fromobjective_q_terms(m1, m2, q.q_ls, q.q_ls, q.q_ll);
                        assert_rel_close("binomial h_tt", tower.h[0][0], h_tt, 1e-12);
                        assert_rel_close("binomial h_tl", tower.h[0][1], h_tl, 1e-12);
                        assert_rel_close("binomial h_ll", tower.h[1][1], h_ll, 1e-12);

                        for (u, v) in dirs {
                            let du = hand_binomial_q_directional(q, u[0], u[1]);
                            let t3 = tower.third_contracted(&u);
                            let dh_tt = directionalhessian_coeff_fromobjective_q_terms(
                                m1,
                                m2,
                                m3,
                                du.delta_q,
                                q.q_t,
                                q.q_t,
                                0.0,
                                du.delta_q_t,
                                du.delta_q_t,
                                0.0,
                            );
                            let dh_tl = directionalhessian_coeff_fromobjective_q_terms(
                                m1,
                                m2,
                                m3,
                                du.delta_q,
                                q.q_t,
                                q.q_ls,
                                q.q_tl,
                                du.delta_q_t,
                                du.delta_q_ls,
                                du.delta_q_tl,
                            );
                            let dh_ll = directionalhessian_coeff_fromobjective_q_terms(
                                m1,
                                m2,
                                m3,
                                du.delta_q,
                                q.q_ls,
                                q.q_ls,
                                q.q_ll,
                                du.delta_q_ls,
                                du.delta_q_ls,
                                du.delta_q_ll,
                            );
                            assert_rel_close("binomial dh_tt", t3[0][0], dh_tt, 1e-12);
                            assert_rel_close("binomial dh_tl", t3[0][1], dh_tl, 1e-12);
                            assert_rel_close("binomial dh_ll", t3[1][1], dh_ll, 1e-12);

                            let dv = hand_binomial_q_directional(q, v[0], v[1]);
                            let (d2q, d2q_t, d2q_ls, d2q_tl, d2q_ll) =
                                hand_binomial_second_q_directional(q, u[0], u[1], v[0], v[1]);
                            let t4 = tower.fourth_contracted(&u, &v);
                            let d2h_tt = second_directionalhessian_coeff_fromobjective_q_terms(
                                m1,
                                m2,
                                m3,
                                m4,
                                du.delta_q,
                                dv.delta_q,
                                d2q,
                                q.q_t,
                                q.q_t,
                                0.0,
                                du.delta_q_t,
                                dv.delta_q_t,
                                du.delta_q_t,
                                dv.delta_q_t,
                                d2q_t,
                                d2q_t,
                                0.0,
                                0.0,
                                0.0,
                            );
                            let d2h_tl = second_directionalhessian_coeff_fromobjective_q_terms(
                                m1,
                                m2,
                                m3,
                                m4,
                                du.delta_q,
                                dv.delta_q,
                                d2q,
                                q.q_t,
                                q.q_ls,
                                q.q_tl,
                                du.delta_q_t,
                                dv.delta_q_t,
                                du.delta_q_ls,
                                dv.delta_q_ls,
                                d2q_t,
                                d2q_ls,
                                du.delta_q_tl,
                                dv.delta_q_tl,
                                d2q_tl,
                            );
                            let d2h_ll = second_directionalhessian_coeff_fromobjective_q_terms(
                                m1,
                                m2,
                                m3,
                                m4,
                                du.delta_q,
                                dv.delta_q,
                                d2q,
                                q.q_ls,
                                q.q_ls,
                                q.q_ll,
                                du.delta_q_ls,
                                dv.delta_q_ls,
                                du.delta_q_ls,
                                dv.delta_q_ls,
                                d2q_ls,
                                d2q_ls,
                                du.delta_q_ll,
                                dv.delta_q_ll,
                                d2q_ll,
                            );
                            assert_rel_close("binomial d2h_tt", t4[0][0], d2h_tt, 1e-12);
                            assert_rel_close("binomial d2h_tl", t4[0][1], d2h_tl, 1e-12);
                            assert_rel_close("binomial d2h_ll", t4[1][1], d2h_ll, 1e-12);
                        }
                    }
                }
            }
        }
    }
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

/// #932: at βw = 0 the WIGGLE joint-Hessian assembler must reduce EXACTLY to
/// the (already tower-pinned) non-wiggle assembler on the (η_t, η_ls) block.
///
/// The canonical wiggle row program differentiates the composed index `q = q0 +
/// Σ_j βw_j·B_j(q0)` through `m = B'·βw + 1` and `g2 = B''·βw`; its
/// `coeff_tw_*` / `coeff_lw_*` / `coeff_ww` cross blocks are the #736 genus. A
/// full wiggle tower is
/// a larger unit (#932 comment), but one structurally-certain invariant is
/// cheap and independent: at `βw = 0` we have `m = 1`, `g2 = 0`, `etaw = 0`, so
/// `q = q0` and the wiggle base coefficients collapse to the non-wiggle ones
/// (`coeff_tt = hessian_coeff(m1, m2, q0_t, q0_t, 0)`, etc.). Therefore the
/// wiggle joint Hessian's top-left `(pt+pls)` block must equal the non-wiggle
/// `exact_newton_joint_hessian_from_designs` joint matrix built from the SAME
/// data — two INDEPENDENT hand assemblers (the non-wiggle one is itself pinned
/// to the single-source tower by
/// `binomial_location_scale_joint_hessian_matches_single_sourced_tower_932`),
/// so agreement transitively pins the wiggle base block to the tower and would
/// catch a typo in the wiggle base chain. Across probit / logit / cloglog.
#[test]
pub(crate) fn binomial_wiggle_joint_hessian_reduces_to_nonwiggle_at_zero_betaw_932() {
    let n = 10usize;
    let pt = 3usize;
    let pls = 2usize;
    let xt = Array2::from_shape_fn((n, pt), |(i, j)| {
        ((i as f64) * 0.17 + (j as f64) * 0.29).sin() * 0.4 + 0.2
    });
    let xls = Array2::from_shape_fn((n, pls), |(i, j)| {
        ((i as f64) * 0.23 + (j as f64) * 0.41).cos() * 0.3
    });
    let beta_t = array![0.20, -0.10, 0.05];
    let beta_ls = array![0.30, -0.15];
    let eta_t = xt.dot(&beta_t);
    let eta_ls = xls.dot(&beta_ls);
    let y = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_iter((0..n).map(|i| 0.5 + 0.2 * i as f64));
    let q_seed = Array1::linspace(-1.0, 1.0, n);
    let (_wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");

    for link in [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
    ] {
        let threshold_design =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xt.clone()));
        let log_sigma_design =
            DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
        let wiggle_family = BinomialLocationScaleWiggleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: link.clone(),
            threshold_design: Some(threshold_design),
            log_sigma_design: Some(log_sigma_design),
            wiggle_knots: knots.clone(),
            wiggle_degree: 2,
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
        };
        // βw = 0 ⇒ etaw = 0, m = 1, g2 = 0, q = q0.
        let q0 = Array1::from_iter(
            eta_t
                .iter()
                .zip(eta_ls.iter())
                .map(|(&t, &l)| binomial_location_scale_q0(t, exp_sigma_from_eta_scalar(l))),
        );
        let wiggle_design_current = wiggle_family
            .wiggle_design(q0.view())
            .expect("current wiggle basis");
        let pw = wiggle_design_current.ncols();
        let beta_w = Array1::<f64>::zeros(pw);
        let eta_w = Array1::<f64>::zeros(n);
        let wiggle_states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls.clone(),
            },
            ParameterBlockState {
                beta: beta_w,
                eta: eta_w,
            },
        ];
        let h_wiggle = wiggle_family
            .exact_newton_joint_hessian(&wiggle_states)
            .expect("wiggle joint Hessian")
            .expect("wiggle joint Hessian present");
        assert_eq!(h_wiggle.dim(), (pt + pls + pw, pt + pls + pw));

        // Non-wiggle reference on identical data / states.
        let nonwiggle_family = BinomialLocationScaleFamily {
            y: y.clone(),
            weights: weights.clone(),
            link_kind: link.clone(),
            threshold_design: None,
            log_sigma_design: None,
            policy: gam_runtime::resource::ResourcePolicy::default_library(),
        };
        let nonwiggle_states = vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t.clone(),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls.clone(),
            },
        ];
        let h_nonwiggle = nonwiggle_family
            .exact_newton_joint_hessian_from_designs(&nonwiggle_states, &xt, &xls)
            .expect("non-wiggle joint Hessian")
            .expect("non-wiggle joint Hessian present");
        assert_eq!(h_nonwiggle.dim(), (pt + pls, pt + pls));

        // The wiggle (η_t, η_ls) top-left block must equal the non-wiggle joint
        // Hessian exactly (both are analytic; βw = 0 makes them the same model).
        for a in 0..(pt + pls) {
            for b in 0..(pt + pls) {
                let w = h_wiggle[[a, b]];
                let nw = h_nonwiggle[[a, b]];
                assert!(
                    (w - nw).abs() <= 1e-9 * (1.0 + nw.abs()),
                    "{link:?}: wiggle (β_w=0) joint Hessian [{a}][{b}] {w:.9e} != \
                     non-wiggle {nw:.9e}"
                );
            }
        }
    }
}

pub(crate) fn hand_trigamma(x: f64) -> f64 {
    gam_math::jet_tower::trigamma_derivative_stack(x)[0]
}

pub(crate) fn hand_dispersion_row_kernel(
    kind: DispersionFamilyKind,
    yi: f64,
    eta_mu: f64,
    eta_d: f64,
    prior_weight: f64,
) -> DispersionRowKernel {
    let wi = prior_weight;
    let em = eta_mu;
    let ed = eta_d;
    match kind {
        DispersionFamilyKind::NegativeBinomial => {
            let mu = em.exp();
            let theta = ed.exp();
            let tpm = theta + mu;
            let tpy = theta + yi;
            let loglik = wi
                * (ln_gamma(yi + theta) - ln_gamma(theta) - ln_gamma(yi + 1.0)
                    + theta * theta.ln()
                    - theta * tpm.ln()
                    + yi * mu.ln()
                    - yi * tpm.ln());
            let mean_weight = wi * mu * theta / tpm;
            let mean_response = em + (yi - mu) / mu;
            let s_theta = statrs::function::gamma::digamma(yi + theta)
                - statrs::function::gamma::digamma(theta)
                + theta.ln()
                + 1.0
                - tpm.ln()
                - tpy / tpm;
            // #1606: production now scores the NB dispersion block with the
            // EXPECTED (Fisher) information in θ (Jensen plug-in for the per-row
            // E[ψ′(θ+Y)]), not the per-row observed Hessian channel, so the
            // block curvature stays positive-definite and the inner solve no
            // longer stalls. The hand oracle mirrors that exact closed form:
            //   I(θ) = ψ′(θ) − ψ′(θ+μ) − 1/θ + 1/(θ+μ) > 0.
            // The score `s_theta` is unchanged (the working response carries the
            // exact score, so the optimum is identical). `tpy` is retained only
            // for the score; the observed-curvature terms are dropped.
            let info_theta = hand_trigamma(theta) - hand_trigamma(tpm) - 1.0 / theta + 1.0 / tpm;
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight: wi * theta * theta * info_theta,
                disp_response: ed + s_theta / (theta * info_theta),
            }
        }
        DispersionFamilyKind::Gamma => {
            let mu = em.exp();
            let nu = ed.exp();
            let y_pos = yi;
            let loglik = wi
                * (nu * nu.ln() - nu * mu.ln() - ln_gamma(nu) + (nu - 1.0) * y_pos.ln()
                    - nu * yi / mu);
            let s_nu = nu.ln() + 1.0 - mu.ln() - statrs::function::gamma::digamma(nu) + y_pos.ln()
                - yi / mu;
            let info_nu = hand_trigamma(nu) - 1.0 / nu;
            DispersionRowKernel {
                loglik,
                mean_weight: wi * nu,
                mean_response: em + (yi - mu) / mu,
                disp_weight: wi * nu * nu * info_nu,
                disp_response: ed + s_nu / (nu * info_nu),
            }
        }
        DispersionFamilyKind::Beta => {
            let logit = gam_solve::mixture_link::logit_inverse_link_jet5(em);
            let mu = logit.mu;
            let phi = ed.exp();
            let q = logit.d1;
            let yc = yi;
            let a = mu * phi;
            let b = (1.0 - mu) * phi;
            let loglik = wi
                * (ln_gamma(phi) - ln_gamma(a) - ln_gamma(b)
                    + (a - 1.0) * yc.ln()
                    + (b - 1.0) * (1.0 - yc).ln());
            let score_mu = phi
                * (statrs::function::gamma::digamma(b) - statrs::function::gamma::digamma(a)
                    + yc.ln()
                    - (1.0 - yc).ln());
            let info_mu = phi * phi * (hand_trigamma(a) + hand_trigamma(b));
            let s_phi = statrs::function::gamma::digamma(phi)
                - mu * statrs::function::gamma::digamma(a)
                - (1.0 - mu) * statrs::function::gamma::digamma(b)
                + mu * yc.ln()
                + (1.0 - mu) * (1.0 - yc).ln();
            let info_phi = mu * mu * hand_trigamma(a) + (1.0 - mu) * (1.0 - mu) * hand_trigamma(b)
                - hand_trigamma(phi);
            DispersionRowKernel {
                loglik,
                mean_weight: wi * q * q * info_mu,
                mean_response: em + score_mu / (q * info_mu),
                disp_weight: wi * phi * phi * info_phi,
                disp_response: ed + s_phi / (phi * info_phi),
            }
        }
        DispersionFamilyKind::Tweedie { p } => {
            // Independent hand derivation of the Tweedie working set in the
            // predictor coordinates `(η_μ, η_d)` with `μ = exp(η_μ)`,
            // `φ = exp(−η_d)` — derived separately from the production tower so
            // a tower-composition bug (a dropped chain term, a sign flip in the
            // φ = exp(−η_d) curvature) shows up as a disagreement. Observed
            // information `∂²NLL/∂η_d²` is used uniformly (the same Newton
            // curvature the production arm reads off `tower.h[1][1]`), matching
            // the NB/Gamma/Beta dispersion arms.
            let mu = em.exp();
            let phi = (-ed).exp();
            let two_minus_p = 2.0 - p;
            let one_minus_p = 1.0 - p;
            let mean_weight = wi * mu.powf(two_minus_p) / phi;
            let mean_response = em + (yi - mu) / mu;
            let (loglik, s_eta, info_eta) = if yi > 0.0 {
                let dev = 2.0
                    * (yi.powf(two_minus_p) / (one_minus_p * two_minus_p)
                        - yi * mu.powf(one_minus_p) / one_minus_p
                        + mu.powf(two_minus_p) / two_minus_p);
                let loglik = wi
                    * (-dev / (2.0 * phi)
                        - 0.5 * (2.0 * std::f64::consts::PI * phi).ln()
                        - 0.5 * p * yi.ln());
                // NLL(η_d) = w·[ (dev/2)·exp(η_d) − ½η_d + const ].
                let s_eta = 0.5 - dev / (2.0 * phi);
                let info_eta = dev / (2.0 * phi);
                (loglik, s_eta, info_eta)
            } else {
                let c = mu.powf(two_minus_p) / two_minus_p;
                let loglik = wi * (-c / phi);
                // NLL(η_d) = w·c·exp(η_d).
                let s_eta = -c / phi;
                let info_eta = c / phi;
                (loglik, s_eta, info_eta)
            };
            let curvature_eta = if yi > 0.0 { 0.5 } else { info_eta };
            DispersionRowKernel {
                loglik,
                mean_weight,
                mean_response,
                disp_weight: wi * curvature_eta,
                disp_response: ed + s_eta / curvature_eta,
            }
        }
    }
}

#[test]
pub(crate) fn dispersion_row_towers_match_hand_witnesses() {
    let cases = [
        (
            DispersionFamilyKind::NegativeBinomial,
            0.0,
            -1.5,
            -25.0,
            0.7,
        ),
        (DispersionFamilyKind::NegativeBinomial, 6.0, 2.0, 3.0, 1.3),
        (DispersionFamilyKind::Gamma, 0.2, -2.0, -25.0, 0.9),
        (DispersionFamilyKind::Gamma, 9.0, 1.7, 25.0, 1.1),
        (DispersionFamilyKind::Beta, 0.02, -3.0, -20.0, 0.8),
        (DispersionFamilyKind::Beta, 0.98, 3.0, 20.0, 1.4),
        // Tweedie #932: the previously-unmechanized arm. Both density branches
        // (y = 0 exact point mass, y > 0 saddlepoint), two powers, and the
        // η-clamp boundary so the clamped natural parameters are exercised.
        (
            DispersionFamilyKind::Tweedie { p: 1.5 },
            0.0,
            -0.7,
            0.4,
            0.9,
        ),
        (
            DispersionFamilyKind::Tweedie { p: 1.5 },
            3.2,
            0.6,
            -0.3,
            1.2,
        ),
        (
            DispersionFamilyKind::Tweedie { p: 1.2 },
            0.0,
            1.1,
            -0.8,
            0.6,
        ),
        (
            DispersionFamilyKind::Tweedie { p: 1.8 },
            7.5,
            -0.4,
            1.0,
            1.3,
        ),
        (
            DispersionFamilyKind::Tweedie { p: 1.5 },
            2.0,
            -25.0,
            25.0,
            1.0,
        ),
    ];
    for (kind, y, eta_mu, eta_d, weight) in cases {
        let actual = dispersion_row_kernel(kind, y, eta_mu, eta_d, weight);
        let expected = hand_dispersion_row_kernel(kind, y, eta_mu, eta_d, weight);
        assert_rel_close("dispersion loglik", actual.loglik, expected.loglik, 1e-10);
        assert_rel_close(
            "dispersion mean_weight",
            actual.mean_weight,
            expected.mean_weight,
            1e-10,
        );
        assert_rel_close(
            "dispersion mean_response",
            actual.mean_response,
            expected.mean_response,
            1e-10,
        );
        assert_rel_close(
            "dispersion disp_weight",
            actual.disp_weight,
            expected.disp_weight,
            1e-10,
        );
        assert_rel_close(
            "dispersion disp_response",
            actual.disp_response,
            expected.disp_response,
            1e-10,
        );
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

pub(crate) fn logistic_numdual<D: DualNum<f64> + Copy>(x: D) -> D {
    D::one() / (D::one() + (-x).exp())
}

pub(crate) fn bspline_basis_scalar_numdual<D: DualNum<f64> + Copy>(
    x: D,
    knots: &Array1<f64>,
    degree: usize,
) -> Vec<D> {
    let n_basis = knots.len() - degree - 1;
    let x_real = x.re();
    let mut basis = vec![D::zero(); n_basis];
    let last_knot = knots[knots.len() - 1];
    for j in 0..n_basis {
        let left = knots[j];
        let right = knots[j + 1];
        let active = if x_real == last_knot {
            j + 1 == n_basis
        } else {
            left <= x_real && x_real < right
        };
        if active {
            basis[j] = D::one();
        }
    }
    for k in 1..=degree {
        let mut next = vec![D::zero(); n_basis];
        for j in 0..n_basis {
            let mut acc = D::zero();
            let left_denom = knots[j + k] - knots[j];
            if left_denom > 0.0 {
                acc += ((x - D::from(knots[j])) / D::from(left_denom)) * basis[j];
            }
            if j + 1 < n_basis {
                let right_denom = knots[j + k + 1] - knots[j + 1];
                if right_denom > 0.0 {
                    acc += ((D::from(knots[j + k + 1]) - x) / D::from(right_denom)) * basis[j + 1];
                }
            }
            next[j] = acc;
        }
        basis = next;
    }
    basis
}

pub(crate) fn monotone_wiggle_basis_scalar_numdual<D: DualNum<f64> + Copy>(
    x: D,
    knots: &Array1<f64>,
    degree: usize,
) -> Array1<D> {
    let bs_degree = monotone_wiggle_internal_degree(degree).expect("monotone wiggle degree") + 1;
    let left = knots[bs_degree];
    let full = bspline_basis_scalar_numdual(x, knots, bs_degree);
    let left_full = bspline_basis_scalar_numdual(D::from(left), knots, bs_degree);
    let mut out = Array1::<D>::from_elem(full.len().saturating_sub(1), D::zero());
    let mut running = D::zero();
    let mut left_running = D::zero();
    for j in (1..full.len()).rev() {
        running += full[j];
        left_running += left_full[j];
        out[j - 1] = running - left_running;
    }
    out
}

pub(crate) fn wiggle_negloglik_threshold_numdual<D: DualNum<f64> + Copy>(
    beta_t: D,
    beta_ls: f64,
    betaw: &Array1<f64>,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> D {
    let sigma = D::from(beta_ls).exp();
    let q0 = -beta_t / sigma;
    let basis = monotone_wiggle_basis_scalar_numdual(q0, knots, degree);
    let mut etaw = D::zero();
    for j in 0..betaw.len() {
        etaw += basis[j] * D::from(betaw[j]);
    }
    let q = q0 + etaw;
    let mu = logistic_numdual(q);
    let one_minusmu = D::one() - mu;
    let mut out = D::zero();
    for i in 0..y.len() {
        out -= D::from(weights[i])
            * (D::from(y[i]) * mu.ln() + D::from(1.0 - y[i]) * one_minusmu.ln());
    }
    out
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

/// Bit-equivalence guard for the Gaussian location-scale-wiggle dense
/// Hessian hook. Pins all six wiggle GEMMs (h_mm, h_ml, h_ll, h_mw_b,
/// h_mw_d, h_lw, h_ww — note the GLS wiggle only has a single
/// ls↔wiggle GEMM because σ-chain doesn't enter the wiggle term)
/// against the canonical-basis HVP path.
#[test]
pub(crate) fn gaussian_location_scale_wiggle_hessian_dense_matches_canonical_basis_hvp_path() {
    let (family, states, specs, _xmu, _xls, _xw) = gls_wiggle_workspace_fixture();
    let total = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    assert_dense_matches_canonical_basis_hvp(workspace.as_ref(), total, "GLSW");
}

/// Bit-equivalence guard for the binomial location-scale-wiggle dense
/// Hessian hook. Pins all eight wiggle GEMMs (h_tt, h_tl, h_ll,
/// h_tw_b, h_tw_d, h_lw_b, h_lw_d, h_ww) against the canonical-basis
/// HVP path.
#[test]
pub(crate) fn binomial_location_scale_wiggle_hessian_dense_matches_canonical_basis_hvp_path() {
    let (family, states, specs, _xt, _xls, _xw) = bls_wiggle_workspace_fixture();
    let total = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    assert_dense_matches_canonical_basis_hvp(workspace.as_ref(), total, "BLSW");
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
pub(crate) fn binomial_location_scale_wiggle_workspace_matvec_matches_dense() {
    // Probit + linkwiggle is the production-pipeline supervised link.
    // This is the load-bearing cross-block test: it pins the b/d wiggle
    // coefficients (`coeff_tw_b/d`, `coeff_lw_b/d`, `coeff_ww`) and the
    // t↔ℓ block against the dense assembly used by
    // `exact_newton_joint_hessian` for the wiggle variant.
    let (family, states, specs, _xt, _xls, wiggle_design_current) = bls_wiggle_workspace_fixture();
    let pt = 3usize;
    let pls = 2usize;
    let pw = wiggle_design_current.ncols();

    let p = pt + pls + pw;
    let dense = family
        .exact_newton_joint_hessian(&states)
        .expect("dense joint Hessian build")
        .expect("dense joint Hessian present");
    assert_eq!(dense.nrows(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    let directions = vec![
        // Axis-aligned probes per block:
        Array1::from_shape_fn(p, |i| if i == 0 { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pt { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pt + pls { 1.0 } else { 0.0 }),
        // Mixed direction across all three blocks:
        Array1::from_shape_fn(p, |i| 0.1 * ((i + 1) as f64).cos()),
    ];
    for (k, v) in directions.iter().enumerate() {
        let want = dense.dot(v);
        let got = workspace
            .hessian_matvec(v)
            .expect("matvec call")
            .expect("matvec present");
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "BLSW matvec[{k}, {i}] mismatch: dense={:.6e}, workspace={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

/// Helper: build a BLS Wiggle family + states + specs fixture
/// (mirrors the inline structure of
/// `binomial_location_scale_wiggle_workspace_matvec_matches_dense`).
pub(crate) fn bls_wiggle_workspace_fixture() -> (
    BinomialLocationScaleWiggleFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    bls_wiggle_workspace_fixture_with_link(InverseLink::Standard(StandardLink::Probit))
}

/// Link-parametrized variant of [`bls_wiggle_workspace_fixture`]. The expected
/// Fisher information derivative machinery is link-agnostic (it consumes only
/// `f, f1, f2` from the shared q-information kernel plus the warp geometry), so
/// exercising a second link — e.g. logit — is an independent angle on the same
/// assembly (gam#2353).
pub(crate) fn bls_wiggle_workspace_fixture_with_link(
    link_kind: InverseLink,
) -> (
    BinomialLocationScaleWiggleFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let n = 10usize;
    let pt = 3usize;
    let pls = 2usize;
    let xt = Array2::from_shape_fn((n, pt), |(i, j)| {
        ((i as f64) * 0.17 + (j as f64) * 0.29).sin() * 0.4
    });
    let xls = Array2::from_shape_fn((n, pls), |(i, j)| {
        ((i as f64) * 0.23 + (j as f64) * 0.41).cos() * 0.3
    });
    let beta_t = array![0.20, -0.10, 0.05];
    let beta_ls = array![0.30, -0.15];
    let eta_t = xt.dot(&beta_t);
    let eta_ls = xls.dot(&beta_ls);
    let q_seed = Array1::linspace(-1.0, 1.0, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");
    let y = Array1::from_iter((0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let threshold_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xt.clone()));
    let log_sigma_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
    let family = BinomialLocationScaleWiggleFamily {
        y,
        weights,
        link_kind,
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(eta_ls.iter())
            .map(|(&eta_t_i, &eta_ls_i)| {
                binomial_location_scale_q0(eta_t_i, exp_sigma_from_eta_scalar(eta_ls_i))
            }),
    );
    let wiggle_design_current = family
        .wiggle_design(q0.view())
        .expect("current wiggle basis");
    let pw = wiggle_design_current.ncols();
    let beta_w = Array1::from_shape_fn(pw, |j| 0.05 * ((j + 1) as f64).cos());
    let eta_w = wiggle_design_current.dot(&beta_w);
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: beta_w,
            eta: eta_w,
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
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design,
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
    (family, states, specs, xt, xls, wiggle_design_current)
}

#[test]
pub(crate) fn binomial_location_scale_wiggle_workspace_dh_operator_matches_dense() {
    let (family, states, specs, _xt, _xls, _xw) = bls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let d_beta = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).cos());

    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative(&states, &d_beta)
        .expect("dense dH build")
        .expect("dense dH present");
    assert_eq!(dense_dh.nrows(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&d_beta)
        .expect("dH op call")
        .expect("dH op present");

    let probes = [
        Array1::from_shape_fn(p, |i| if i == 0 { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == states[0].beta.len() { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| {
            if i == states[0].beta.len() + states[1].beta.len() {
                1.0
            } else {
                0.0
            }
        }),
        Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).sin()),
    ];
    for (k, w) in probes.iter().enumerate() {
        let want = dense_dh.dot(w);
        let got = dh_op.mul_vec(w);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "BLSW dH op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn binomial_location_scale_wiggle_workspace_dh_operator_finite_difference() {
    let (family, states, specs, xt, xls, _xw) = bls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let u = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).cos());
    let v = Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).sin());
    let pt = states[0].beta.len();
    let pls = states[1].beta.len();
    let eps = 1e-5;
    let perturb = |sign: f64| -> Vec<ParameterBlockState> {
        let mut out = states.clone();
        for j in 0..pt {
            out[0].beta[j] += sign * eps * u[j];
        }
        for j in 0..pls {
            out[1].beta[j] += sign * eps * u[pt + j];
        }
        for j in 0..(p - pt - pls) {
            out[2].beta[j] += sign * eps * u[pt + pls + j];
        }
        out[0].eta = xt.dot(&out[0].beta);
        out[1].eta = xls.dot(&out[1].beta);
        let q0 = Array1::from_iter(out[0].eta.iter().zip(out[1].eta.iter()).map(
            |(&eta_t, &eta_ls)| {
                binomial_location_scale_q0(eta_t, exp_sigma_from_eta_scalar(eta_ls))
            },
        ));
        out[2].eta = family
            .wiggle_design(q0.view())
            .expect("perturbed wiggle basis")
            .dot(&out[2].beta);
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
        let tol = 5e-5 * fd[i].abs().max(1.0) + 5e-5;
        assert!(
            (fd[i] - analytic[i]).abs() <= tol,
            "BLSW dH FD mismatch at {i}: fd={:.6e}, analytic={:.6e}",
            fd[i],
            analytic[i]
        );
    }
}

#[test]
pub(crate) fn binomial_location_scale_wiggle_expected_info_derivatives_match_finite_difference() {
    check_bls_wiggle_expected_info_derivatives_match_fd(bls_wiggle_workspace_fixture());
}

/// Same expected-info directional/second-directional FD acceptance gate on the
/// LOGIT link — an independent angle on the link-agnostic derivative assembly.
/// A regression of the gam#2353 spurious-curvature term would surface here too
/// because logit changes every `f, f1, f2` yet reuses the identical warp
/// geometry and coefficient assembly.
#[test]
pub(crate) fn binomial_location_scale_wiggle_expected_info_derivatives_match_finite_difference_logit()
{
    check_bls_wiggle_expected_info_derivatives_match_fd(bls_wiggle_workspace_fixture_with_link(
        InverseLink::Standard(StandardLink::Logit),
    ));
}

fn check_bls_wiggle_expected_info_derivatives_match_fd(
    fixture: (
        BinomialLocationScaleWiggleFamily,
        Vec<ParameterBlockState>,
        Vec<ParameterBlockSpec>,
        Array2<f64>,
        Array2<f64>,
        Array2<f64>,
    ),
) {
    let (family, states, specs, xt, xls, xw_at_base) = fixture;
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let pt = states[0].beta.len();
    let pls = states[1].beta.len();
    let pw = states[2].beta.len();
    assert_eq!(p, pt + pls + pw);
    assert_eq!(xw_at_base.ncols(), pw);
    // gam#1020: the expected-information override must disarm the
    // observed-Hessian "Jeffreys skippable" matvec pre-checks.
    assert!(!family.joint_jeffreys_information_matches_observed_hessian());
    let u = Array1::from_shape_fn(p, |i| 0.04 * ((i + 1) as f64).cos());
    let v = Array1::from_shape_fn(p, |i| -0.03 * ((i + 2) as f64).sin());
    let eps = 1e-5;
    let perturb = |direction: &Array1<f64>, scale: f64| -> Vec<ParameterBlockState> {
        let mut out = states.clone();
        for j in 0..pt {
            out[0].beta[j] += scale * direction[j];
        }
        for j in 0..pls {
            out[1].beta[j] += scale * direction[pt + j];
        }
        for j in 0..pw {
            out[2].beta[j] += scale * direction[pt + pls + j];
        }
        out[0].eta = xt.dot(&out[0].beta);
        out[1].eta = xls.dot(&out[1].beta);
        let q0 = Array1::from_iter(out[0].eta.iter().zip(out[1].eta.iter()).map(
            |(&eta_t, &eta_ls)| {
                binomial_location_scale_q0(eta_t, exp_sigma_from_eta_scalar(eta_ls))
            },
        ));
        out[2].eta = family
            .wiggle_design(q0.view())
            .expect("perturbed wiggle basis")
            .dot(&out[2].beta);
        out
    };

    let h_plus = family
        .joint_jeffreys_information_with_specs(&perturb(&u, eps), &specs)
        .expect("expected I plus")
        .expect("expected I plus present");
    let h_minus = family
        .joint_jeffreys_information_with_specs(&perturb(&u, -eps), &specs)
        .expect("expected I minus")
        .expect("expected I minus present");
    let fd_first = (&h_plus - &h_minus) / (2.0 * eps);
    let analytic_first = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &u)
        .expect("expected dI")
        .expect("expected dI present");
    assert_close_matrix(&analytic_first, &fd_first, 2e-7, "wiggle expected dI");

    let states_plus = perturb(&v, eps);
    let states_minus = perturb(&v, -eps);
    let d_plus = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states_plus, &specs, &u)
        .expect("expected dI plus")
        .expect("expected dI plus present");
    let d_minus = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states_minus, &specs, &u)
        .expect("expected dI minus")
        .expect("expected dI minus present");
    let fd_second = (&d_plus - &d_minus) / (2.0 * eps);
    let analytic_second = family
        .joint_jeffreys_information_second_directional_derivative_with_specs(
            &states, &specs, &u, &v,
        )
        .expect("expected d2I")
        .expect("expected d2I present");
    assert_close_matrix(&analytic_second, &fd_second, 2e-7, "wiggle expected d2I");
}

#[test]
pub(crate) fn binomial_location_scale_wiggle_workspace_d2h_operator_matches_dense() {
    let (family, states, specs, _xt, _xls, _xw) = bls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let d_beta_u = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).cos());
    let d_beta_v = Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).sin());

    let dense_d2h = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &d_beta_u, &d_beta_v)
        .expect("dense d2H build")
        .expect("dense d2H present");
    assert_eq!(dense_d2h.nrows(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let d2h_op = workspace
        .second_directional_derivative_operator(&d_beta_u, &d_beta_v)
        .expect("d2H op call")
        .expect("d2H op present");

    let pt = states[0].beta.len();
    let pls = states[1].beta.len();
    let probes = [
        Array1::from_shape_fn(p, |i| if i == 0 { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pt { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pt + pls { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| 0.07 * ((i + 3) as f64).cos()),
    ];
    for (k, w) in probes.iter().enumerate() {
        let want = dense_d2h.dot(w);
        let got = d2h_op.mul_vec(w);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "BLSW d2H op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn binomial_location_scale_wiggle_workspace_d2h_operator_finite_difference() {
    // FD check: [DH(β + ε u_fd) [u] v − DH(β − ε u_fd) [u] v] / (2ε)
    // ≈ D²H[u_fd, u] v.
    let (family, states, specs, xt, xls, xw) = bls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let u_fd = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).cos());
    let u = Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).sin());
    let probe = Array1::from_shape_fn(p, |i| 0.04 * ((i + 3) as f64).sin());
    let pt = states[0].beta.len();
    let pls = states[1].beta.len();
    let eps = 1e-5;
    let perturb = |sign: f64| -> Vec<ParameterBlockState> {
        let mut out = states.clone();
        for j in 0..pt {
            out[0].beta[j] += sign * eps * u_fd[j];
        }
        for j in 0..pls {
            out[1].beta[j] += sign * eps * u_fd[pt + j];
        }
        for j in 0..(p - pt - pls) {
            out[2].beta[j] += sign * eps * u_fd[pt + pls + j];
        }
        out[0].eta = xt.dot(&out[0].beta);
        out[1].eta = xls.dot(&out[1].beta);
        out[2].eta = xw.dot(&out[2].beta);
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
            "BLSW d2H FD mismatch at {i}: fd={:.6e}, analytic={:.6e}",
            fd[i],
            analytic[i]
        );
    }
}

#[test]
pub(crate) fn gaussian_location_scale_wiggle_workspace_matvec_matches_dense() {
    let n = 10usize;
    let p_mu = 3usize;
    let p_ls = 2usize;
    let xmu = Array2::from_shape_fn((n, p_mu), |(i, j)| {
        ((i as f64) * 0.13 + (j as f64) * 0.31).sin() * 0.4
    });
    let xls = Array2::from_shape_fn((n, p_ls), |(i, j)| {
        ((i as f64) * 0.21 + (j as f64) * 0.47).cos() * 0.3
    });
    let beta_mu = array![0.10, -0.20, 0.30];
    let beta_ls = array![0.40, -0.10];
    let eta_mu = xmu.dot(&beta_mu);
    let eta_ls = xls.dot(&beta_ls);

    let q_seed = Array1::linspace(-1.0, 1.0, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");
    let wiggle_design_dense = match wiggle_block.design.as_dense_ref() {
        Some(d) => d.clone(),
        None => panic!("wiggle design must be dense for this test fixture"),
    };
    let pw = wiggle_design_dense.ncols();
    let beta_w = Array1::from_shape_fn(pw, |j| 0.05 * ((j + 1) as f64).sin());
    let eta_w = wiggle_design_dense.dot(&beta_w);

    let y = Array1::from_shape_fn(n, |i| 0.5 + 0.1 * (i as f64).cos());
    let weights = Array1::from_elem(n, 1.0);
    let mu_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xmu.clone()));
    let log_sigma_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
    let family = GaussianLocationScaleWiggleFamily {
        y,
        weights,
        mu_design: Some(mu_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 2,
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
        ParameterBlockState {
            beta: beta_w,
            eta: eta_w,
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
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design,
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

    let p = p_mu + p_ls + pw;
    let dense = family
        .exact_newton_joint_hessian(&states)
        .expect("dense joint Hessian build")
        .expect("dense joint Hessian present");
    assert_eq!(dense.nrows(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");

    let directions = [
        Array1::from_shape_fn(p, |i| if i == 0 { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == p_mu { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == p_mu + p_ls { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| 0.1 * ((i + 1) as f64).sin()),
    ];
    for (k, v) in directions.iter().enumerate() {
        let want = dense.dot(v);
        let got = workspace
            .hessian_matvec(v)
            .expect("matvec call")
            .expect("matvec present");
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "GLSW matvec[{k}, {i}] mismatch: dense={:.6e}, workspace={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

/// Helper: build a GLS Wiggle family + states + specs fixture
/// (mirrors the inline structure of
/// `gaussian_location_scale_wiggle_workspace_matvec_matches_dense`).
pub(crate) fn gls_wiggle_workspace_fixture() -> (
    GaussianLocationScaleWiggleFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let n = 10usize;
    let p_mu = 3usize;
    let p_ls = 2usize;
    let xmu = Array2::from_shape_fn((n, p_mu), |(i, j)| {
        ((i as f64) * 0.13 + (j as f64) * 0.31).sin() * 0.4
    });
    let xls = Array2::from_shape_fn((n, p_ls), |(i, j)| {
        ((i as f64) * 0.21 + (j as f64) * 0.47).cos() * 0.3
    });
    let beta_mu = array![0.10, -0.20, 0.30];
    let beta_ls = array![0.40, -0.10];
    let eta_mu = xmu.dot(&beta_mu);
    let eta_ls = xls.dot(&beta_ls);
    let q_seed = Array1::linspace(-1.0, 1.0, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");
    let pw = wiggle_block.design.ncols();
    let beta_w = Array1::from_shape_fn(pw, |j| 0.05 * ((j + 1) as f64).sin());
    let y = Array1::from_shape_fn(n, |i| 0.5 + 0.1 * (i as f64).cos());
    let weights = Array1::from_elem(n, 1.0);
    let mu_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xmu.clone()));
    let log_sigma_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(xls.clone()));
    let family = GaussianLocationScaleWiggleFamily {
        y,
        weights,
        mu_design: Some(mu_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        cached_row_scalars: std::sync::RwLock::new(None),
    };
    // The wiggle block has dynamic geometry (q0-dependent basis): the
    // model is q = q0 + B(q0)·β_w, so η_w must be evaluated at the
    // *current* q0, not at the spec's static seed grid. Mirror what
    // `refresh_all_block_etas` does at fit time so the fixture state
    // satisfies the analytical formula's invariant.
    let xw_at_q0 = family
        .wiggle_design(eta_mu.view())
        .expect("wiggle basis at q0");
    let eta_w = xw_at_q0.dot(&beta_w);
    let states = vec![
        ParameterBlockState {
            beta: beta_mu,
            eta: eta_mu,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: beta_w,
            eta: eta_w,
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
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: wiggle_block.design,
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
    (family, states, specs, xmu, xls, xw_at_q0)
}

#[test]
pub(crate) fn gaussian_location_scale_wiggle_workspace_dh_operator_matches_dense() {
    let (family, states, specs, _xmu, _xls, _xw) = gls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let d_beta = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).sin());

    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative(&states, &d_beta)
        .expect("dense dH build")
        .expect("dense dH present");
    assert_eq!(dense_dh.nrows(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let dh_op = workspace
        .directional_derivative_operator(&d_beta)
        .expect("dH op call")
        .expect("dH op present");

    let pmu = states[0].beta.len();
    let pls = states[1].beta.len();
    let probes = [
        Array1::from_shape_fn(p, |i| if i == 0 { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pmu { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pmu + pls { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).cos()),
    ];
    for (k, w) in probes.iter().enumerate() {
        let want = dense_dh.dot(w);
        let got = dh_op.mul_vec(w);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "GLSW dH op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn gaussian_location_scale_wiggle_workspace_dh_operator_finite_difference() {
    let (family, states, specs, xmu, xls, _xw) = gls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let u = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).cos());
    let v = Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).sin());
    let pmu = states[0].beta.len();
    let pls = states[1].beta.len();
    let eps = 1e-5;
    let perturb = |sign: f64| -> Vec<ParameterBlockState> {
        let mut out = states.clone();
        for j in 0..pmu {
            out[0].beta[j] += sign * eps * u[j];
        }
        for j in 0..pls {
            out[1].beta[j] += sign * eps * u[pmu + j];
        }
        for j in 0..(p - pmu - pls) {
            out[2].beta[j] += sign * eps * u[pmu + pls + j];
        }
        out[0].eta = xmu.dot(&out[0].beta);
        out[1].eta = xls.dot(&out[1].beta);
        // Wiggle geometry is dynamic: η_w = B(q0)·β_w at the perturbed
        // q0, matching what `refresh_all_block_etas` would produce. Using
        // a static spec design here would compute the FD of a different
        // model than the analytical dH formula assumes (which carries
        // dq/dq0 = 1 + B'(q0)·β_w through the chain rule).
        let xw_perturbed = family
            .wiggle_design(out[0].eta.view())
            .expect("wiggle basis at perturbed q0");
        out[2].eta = xw_perturbed.dot(&out[2].beta);
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
        let tol = 5e-5 * fd[i].abs().max(1.0) + 5e-5;
        assert!(
            (fd[i] - analytic[i]).abs() <= tol,
            "GLSW dH FD mismatch at {i}: fd={:.6e}, analytic={:.6e}",
            fd[i],
            analytic[i]
        );
    }
}

#[test]
pub(crate) fn gaussian_location_scale_wiggle_workspace_d2h_operator_matches_dense() {
    let (family, states, specs, _xmu, _xls, _xw) = gls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let d_beta_u = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).sin());
    let d_beta_v = Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).cos());

    let dense_d2h = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &d_beta_u, &d_beta_v)
        .expect("dense d2H build")
        .expect("dense d2H present");
    assert_eq!(dense_d2h.nrows(), p);

    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace build")
        .expect("workspace present");
    let d2h_op = workspace
        .second_directional_derivative_operator(&d_beta_u, &d_beta_v)
        .expect("d2H op call")
        .expect("d2H op present");

    let pmu = states[0].beta.len();
    let pls = states[1].beta.len();
    let probes = [
        Array1::from_shape_fn(p, |i| if i == 0 { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pmu { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| if i == pmu + pls { 1.0 } else { 0.0 }),
        Array1::from_shape_fn(p, |i| 0.07 * ((i + 3) as f64).cos()),
    ];
    for (k, w) in probes.iter().enumerate() {
        let want = dense_d2h.dot(w);
        let got = d2h_op.mul_vec(w);
        for i in 0..p {
            let tol = 1e-9 * want[i].abs().max(1.0) + 1e-9;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "GLSW d2H op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                want[i],
                got[i]
            );
        }
    }
}

#[test]
pub(crate) fn gaussian_location_scale_wiggle_workspace_d2h_operator_finite_difference() {
    let (family, states, specs, xmu, xls, _xw) = gls_wiggle_workspace_fixture();
    let p = states[0].beta.len() + states[1].beta.len() + states[2].beta.len();
    let u_fd = Array1::from_shape_fn(p, |i| 0.05 * ((i + 1) as f64).cos());
    let u = Array1::from_shape_fn(p, |i| 0.07 * ((i + 2) as f64).sin());
    let probe = Array1::from_shape_fn(p, |i| 0.04 * ((i + 3) as f64).sin());
    let pmu = states[0].beta.len();
    let pls = states[1].beta.len();
    let eps = 1e-5;
    let perturb = |sign: f64| -> Vec<ParameterBlockState> {
        let mut out = states.clone();
        for j in 0..pmu {
            out[0].beta[j] += sign * eps * u_fd[j];
        }
        for j in 0..pls {
            out[1].beta[j] += sign * eps * u_fd[pmu + j];
        }
        for j in 0..(p - pmu - pls) {
            out[2].beta[j] += sign * eps * u_fd[pmu + pls + j];
        }
        out[0].eta = xmu.dot(&out[0].beta);
        out[1].eta = xls.dot(&out[1].beta);
        // Wiggle geometry is dynamic: η_w = B(q0)·β_w at the perturbed q0.
        let xw_perturbed = family
            .wiggle_design(out[0].eta.view())
            .expect("wiggle basis at perturbed q0");
        out[2].eta = xw_perturbed.dot(&out[2].beta);
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
            "GLSW d2H FD mismatch at {i}: fd={:.6e}, analytic={:.6e}",
            fd[i],
            analytic[i]
        );
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
        BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Gaussian mu block"),
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
        BlockWorkingSet::ExactNewton { .. } => {
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
        BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Poisson block"),
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
        BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Gamma block"),
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
        BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Poisson block"),
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
        BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Gamma block"),
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
        BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Poisson block"),
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
        BlockWorkingSet::ExactNewton { .. } => panic!("expected diagonal Gamma block"),
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
        BlockWorkingSet::ExactNewton { .. } => {
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
        BlockWorkingSet::ExactNewton { .. } => {
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
pub(crate) fn gaussian_log_sigma_design_keeps_shared_mean_basis() {
    let shared = array![[1.0, -1.5], [1.0, -0.25], [1.0, 0.75], [1.0, 2.0],];
    let shared_design =
        DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(shared.clone()));
    let prepared = prepared_gaussian_log_sigma_design(&shared_design, &shared_design)
        .expect("gaussian log-sigma design should accept shared columns");
    let prepared_dense = prepared.as_dense_cow();

    for i in 0..shared.nrows() {
        for j in 0..shared.ncols() {
            assert!(
                (prepared_dense[[i, j]] - shared[[i, j]]).abs() < 1e-12,
                "gaussian log-sigma design should preserve shared basis at ({i}, {j}): got {}, expected {}",
                prepared_dense[[i, j]],
                shared[[i, j]]
            );
        }
    }
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
        BlockWorkingSet::ExactNewton { .. } => {
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
pub(crate) fn binomial_location_scalewiggle_term_builder_requires_exact_spatial_joint_path() {
    let n = 8usize;
    let q_seed = Array1::linspace(-1.25, 1.25, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");
    let builder = BinomialLocationScaleWiggleTermBuilder {
        y: Array1::from_elem(n, 0.0),
        weights: Array1::from_elem(n, 1.0),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        meanspec: simple_matern_term_collection(&[0, 1], 0.4),
        noisespec: simple_matern_term_collection(&[0, 1], 0.75),
        mean_offset: Array1::zeros(n),
        noise_offset: Array1::zeros(n),
        wiggle_knots: knots,
        wiggle_degree: 2,
        wiggle_block,
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
    assert!(family.requires_joint_outer_hyper_path());
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
pub(crate) fn binomial_location_scalewiggle_builder_populateswarm_start_betas() {
    let n = 12usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.0 * std::f64::consts::PI * t).cos();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let q_seed = Array1::linspace(-1.25, 1.25, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");
    let builder = BinomialLocationScaleWiggleTermBuilder {
        mean_offset: Array1::zeros(y.len()),
        noise_offset: Array1::zeros(y.len()),
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        meanspec: simple_matern_term_collection(&[0, 1], 0.45),
        noisespec: simple_matern_term_collection(&[0, 1], 0.8),
        wiggle_knots: knots,
        wiggle_degree: 2,
        wiggle_block,
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
        &builder.extra_rho0().expect("extra rho0"),
    );
    let blocks = builder
        .build_blocks(&rho, &mean_design, &noise_design, None, None)
        .expect("build blocks");
    assert_eq!(blocks.len(), 3);
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
pub(crate) fn binomial_location_scalewiggle_exact_newton_spatial_joint_hyper_returns_fullhessian() {
    let n = 14usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.25 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 3 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let meanspec = simple_matern_term_collection(&[0, 1], 0.45);
    let noisespec = simple_matern_term_collection(&[0, 1], 0.8);
    let q_seed = Array1::linspace(-1.5, 1.5, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");
    let builder = BinomialLocationScaleWiggleTermBuilder {
        mean_offset: Array1::zeros(y.len()),
        noise_offset: Array1::zeros(y.len()),
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        meanspec: meanspec.clone(),
        noisespec: noisespec.clone(),
        wiggle_knots: knots,
        wiggle_degree: 2,
        wiggle_block,
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
        &builder.extra_rho0().expect("wiggle rho0"),
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
    .expect("exact wiggle spatial joint hyper eval");
    assert!(eval.objective.is_finite());
    assert!(eval.gradient.iter().all(|v| v.is_finite()));
    let hess = eval
        .outer_hessian
        .materialize_dense()
        .expect(
            "exact wiggle spatial joint hyper path should materialize a full [rho, psi] hessian",
        )
        .expect("exact wiggle spatial joint hyper path should return a full [rho, psi] hessian");
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

#[test]
pub(crate) fn binomial_location_scalewiggle_family_exposes_joint_psi_hook_surface() {
    let n = 12usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (1.75 * std::f64::consts::PI * t).cos();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 5 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let meanspec = simple_matern_term_collection(&[0, 1], 0.4);
    let noisespec = simple_matern_term_collection(&[0, 1], 0.7);
    let q_seed = Array1::linspace(-1.25, 1.25, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");
    let builder = BinomialLocationScaleWiggleTermBuilder {
        mean_offset: Array1::zeros(y.len()),
        noise_offset: Array1::zeros(y.len()),
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        meanspec: meanspec.clone(),
        noisespec: noisespec.clone(),
        wiggle_knots: knots,
        wiggle_degree: 2,
        wiggle_block,
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
        &builder.extra_rho0().expect("wiggle rho0"),
    );
    let blocks = builder
        .build_blocks(&rho, &mean_design, &noise_design, None, None)
        .expect("build blocks");
    let family = builder.build_family(&mean_design, &noise_design);
    let mut block_states = Vec::<ParameterBlockState>::with_capacity(blocks.len());
    for (block_idx, spec) in blocks.iter().enumerate() {
        let mut beta = spec
            .initial_beta
            .clone()
            .unwrap_or_else(|| Array1::zeros(spec.design.ncols()));
        if block_idx == BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE {
            beta.fill(0.04);
        }
        let (design, offset) = family
            .block_geometry(&block_states, spec)
            .expect("hook fixture block geometry");
        let eta = design.matrixvectormultiply(&beta) + &offset;
        block_states.push(ParameterBlockState { beta, eta });
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
        0.25,
        0.1,
        "wiggle",
    );
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
    let mut block_states = Vec::<ParameterBlockState>::with_capacity(blocks.len());
    for spec in blocks.iter() {
        let beta = spec
            .initial_beta
            .clone()
            .unwrap_or_else(|| Array1::zeros(spec.design.ncols()));
        let (design, offset) = family
            .block_geometry(&block_states, spec)
            .expect("hook fixture block geometry");
        let eta = design.matrixvectormultiply(&beta) + &offset;
        block_states.push(ParameterBlockState { beta, eta });
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
    assert!(fit.fit.penalized_objective.is_finite());
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
    assert!(fit.fit.penalized_objective.is_finite());
    assert_eq!(fit.fit.block_states.len(), 2);
}

#[test]
pub(crate) fn binomial_location_scalewiggle_termswith_matern_spatial_blocks_fit_finitely() {
    let n = 30usize;
    let mut data = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        let t = i as f64 / (n as f64 - 1.0);
        data[[i, 0]] = t;
        data[[i, 1]] = (2.5 * std::f64::consts::PI * t).sin();
    }
    let y = Array1::from_iter((0..n).map(|i| if i % 4 == 0 || i % 9 == 0 { 1.0 } else { 0.0 }));
    let weights = Array1::from_elem(n, 1.0);
    let q_seed = Array1::linspace(-1.5, 1.5, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 4, 2, false)
            .expect("wiggle block");
    let spec = BinomialLocationScaleWiggleTermSpec {
        y,
        weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        thresholdspec: simple_matern_term_collection(&[0, 1], 0.45),
        log_sigmaspec: empty_term_collection(),
        threshold_offset: Array1::zeros(n),
        log_sigma_offset: Array1::zeros(n),
        wiggle_knots: knots,
        wiggle_degree: 2,
        wiggle_block,
    };
    let fit = fit_binomial_location_scalewiggle_terms(
        data.view(),
        spec,
        &spatial_fit_smoke_options(),
        &spatial_kappa_options(),
    )
    .expect("binomial location-scale wiggle spatial fit");
    assert!(fit.fit.penalized_objective.is_finite());
    assert_eq!(fit.fit.block_states.len(), 3);
}

#[test]
pub(crate) fn wiggle_family_evaluate_returns_exact_newton_blocks() {
    let n = 6usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.5, 1.5, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design),
        log_sigma_design: Some(log_sigma_design),
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let eta_t = Array1::from_vec(vec![0.4; n]);
    let eta_ls = Array1::from_vec(vec![-0.2; n]);
    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.05; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);
    let eval = family
        .evaluate(&[
            ParameterBlockState {
                beta: Array1::from_vec(vec![0.4]),
                eta: eta_t,
            },
            ParameterBlockState {
                beta: Array1::from_vec(vec![-0.2]),
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ])
        .expect("evaluate");

    assert_eq!(eval.blockworking_sets.len(), 3);
    match &eval.blockworking_sets[0] {
        BlockWorkingSet::ExactNewton { gradient, hessian } => {
            let hessian = hessian.to_dense();
            assert_eq!(gradient.len(), 1);
            assert_eq!(hessian.dim(), (1, 1));
            assert!(gradient[0].is_finite());
            assert!(hessian[[0, 0]].is_finite());
        }
        BlockWorkingSet::Diagonal { .. } => panic!("threshold block should be exact newton"),
    }
    match &eval.blockworking_sets[1] {
        BlockWorkingSet::ExactNewton { gradient, hessian } => {
            let hessian = hessian.to_dense();
            assert_eq!(gradient.len(), 1);
            assert_eq!(hessian.dim(), (1, 1));
            assert!(gradient[0].is_finite());
            assert!(hessian[[0, 0]].is_finite());
        }
        BlockWorkingSet::Diagonal { .. } => panic!("log-sigma block should be exact newton"),
    }
    match &eval.blockworking_sets[2] {
        BlockWorkingSet::ExactNewton { gradient, hessian } => {
            let hessian = hessian.to_dense();
            assert_eq!(gradient.len(), betaw.len());
            assert_eq!(hessian.nrows(), betaw.len());
            assert_eq!(hessian.ncols(), betaw.len());
            assert!(gradient.iter().all(|v| v.is_finite()));
            assert!(hessian.iter().all(|v| v.is_finite()));
        }
        BlockWorkingSet::Diagonal { .. } => panic!("wiggle block should be exact newton"),
    }
}

#[test]
pub(crate) fn wiggle_family_exact_newton_directional_derivative_matches_finite_difference() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let eta_t = threshold_design.matrixvectormultiply(&beta_t);
    let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);

    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: eta_t.clone(),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: eta_ls.clone(),
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw.clone(),
        },
    ];

    let extract = |eval: FamilyEvaluation, idx: usize| -> Array2<f64> {
        match &eval.blockworking_sets[idx] {
            BlockWorkingSet::ExactNewton {
                gradient: _,
                hessian,
            } => hessian.to_dense(),
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
        }
    };

    let base_eval = family.evaluate(&states).expect("base eval");
    let eps = 1e-6;
    for block_idx in 0..3 {
        let d_beta = Array1::ones(states[block_idx].beta.len());
        let analytic = family
            .exact_newton_hessian_directional_derivative(&states, block_idx, &d_beta)
            .expect("analytic dH")
            .expect("expected derivative");

        let mut plus_states = states.clone();
        plus_states[block_idx].beta = &plus_states[block_idx].beta + &(eps * &d_beta);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta = threshold_design
            .matrixvectormultiply(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta);
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta = log_sigma_design
            .matrixvectormultiply(
                &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta,
            );
        let plus_core_q0 = binomial_location_scale_core(
            &y,
            &weights,
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta,
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta,
            None,
            &family.link_kind,
        )
        .expect("plus core q0");
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta = family
            .wiggle_design(plus_core_q0.q0.view())
            .expect("plus wiggle design")
            .dot(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta);

        let h_plus = extract(family.evaluate(&plus_states).expect("plus eval"), block_idx);
        let h_base = extract(base_eval.clone(), block_idx);
        let fd = (h_plus - h_base) / eps;
        gam_test_support::assert_matrix_derivativefd(
            &fd,
            &analytic,
            5e-4,
            &format!("block {} dH", block_idx),
        );
    }
}

#[test]
pub(crate) fn wiggle_threshold_block_exacthessian_matches_autodiffobjective() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots.clone(),
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let beta_t0 = 0.25;
    let beta_ls0 = -0.15;
    let beta_t = array![beta_t0];
    let beta_ls = array![beta_ls0];
    let eta_t = threshold_design.matrixvectormultiply(&beta_t);
    let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw,
        },
    ];

    let eval = family.evaluate(&states).expect("evaluate wiggle family");
    let blockhessian = match &eval.blockworking_sets[BinomialLocationScaleWiggleFamily::BLOCK_T] {
        BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
        BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton threshold block"),
    };
    let (_, _, hess_ad) = second_derivative(
        |bt| wiggle_negloglik_threshold_numdual(bt, beta_ls0, &betaw, &y, &weights, &knots, 3),
        beta_t0,
    );
    assert!(
        (blockhessian[[0, 0]] - hess_ad).abs() <= 5e-6,
        "wiggle threshold exact hessian mismatch: evaluate()={} autodiff={}",
        blockhessian[[0, 0]],
        hess_ad
    );
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

#[test]
pub(crate) fn wiggle_family_block_hessians_match_jointhessian_principal_blocks() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let eta_t = threshold_design.matrixvectormultiply(&beta_t);
    let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw,
        },
    ];

    let eval = family.evaluate(&states).expect("evaluate wiggle family");
    let joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let beta_layout = GamlssBetaLayout::withwiggle(beta_t.len(), beta_ls.len(), betaw.len());
    let ranges = [
        (0usize, beta_layout.pt),
        (beta_layout.pt, beta_layout.pt + beta_layout.pls),
        (
            beta_layout.pt + beta_layout.pls,
            beta_layout.pt + beta_layout.pls + beta_layout.pw,
        ),
    ];

    for (block_idx, (start, end)) in ranges.into_iter().enumerate() {
        let blockhessian = match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
        };
        let joint_block = joint.slice(s![start..end, start..end]).to_owned();
        gam_test_support::assert_matrix_derivativefd(
            &joint_block,
            &blockhessian,
            1e-10,
            &format!("wiggle block {block_idx} principal block"),
        );
    }
}

/// Build the nontrivial-design BLS Wiggle family + designs + wiggle block
/// shared by the FD-gradient and FD-joint-Hessian tests below.
pub(crate) fn wiggle_nontrivial_fixture() -> (
    BinomialLocationScaleWiggleFamily,
    DesignMatrix,
    DesignMatrix,
    ParameterBlockInput,
    Array1<f64>,
    Array1<f64>,
) {
    let n = 9usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let t_grid = Array1::linspace(0.0, 1.0, n);
    let threshold_x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => 1.0,
        1 => t_grid[i] - 0.5,
        2 => (2.0 * std::f64::consts::PI * t_grid[i]).sin(),
        _ => unreachable!(),
    });
    let log_sigma_x = Array2::from_shape_fn((n, 2), |(i, j)| match j {
        0 => 1.0,
        1 => (3.0 * std::f64::consts::PI * t_grid[i]).cos(),
        _ => unreachable!(),
    });
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        threshold_x.clone(),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        log_sigma_x.clone(),
    ));
    let q_seed = Array1::linspace(-1.3, 1.1, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    (
        family,
        threshold_design,
        log_sigma_design,
        wiggle_block,
        y,
        weights,
    )
}

/// Rebuild the three-block state for the nontrivial-design wiggle fixture.
pub(crate) fn rebuild_wiggle_nontrivial_states(
    family: &BinomialLocationScaleWiggleFamily,
    threshold_design: &DesignMatrix,
    log_sigma_design: &DesignMatrix,
    y: &Array1<f64>,
    weights: &Array1<f64>,
    beta_t: &Array1<f64>,
    beta_ls: &Array1<f64>,
    betaw: &Array1<f64>,
) -> Vec<ParameterBlockState> {
    let eta_t = threshold_design.matrixvectormultiply(beta_t);
    let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
    let core_q0 =
        binomial_location_scale_core(y, weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let etaw = family
        .wiggle_design(core_q0.q0.view())
        .expect("wiggle design")
        .dot(betaw);
    vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw,
        },
    ]
}

/// Extract the exact-Newton gradient for one block of a wiggle evaluation.
pub(crate) fn extract_wiggle_gradient(eval: &FamilyEvaluation, block_idx: usize) -> Array1<f64> {
    match &eval.blockworking_sets[block_idx] {
        BlockWorkingSet::ExactNewton {
            gradient,
            hessian: _,
        } => gradient.clone(),
        BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
    }
}

#[test]
pub(crate) fn wiggle_familygradients_match_finite_differencewith_nontrivial_designs() {
    let (family, threshold_design, log_sigma_design, wiggle_block, y, weights) =
        wiggle_nontrivial_fixture();

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        rebuild_wiggle_nontrivial_states(
            &family,
            &threshold_design,
            &log_sigma_design,
            &y,
            &weights,
            beta_t,
            beta_ls,
            betaw,
        )
    };

    let objective = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>, betaw: &Array1<f64>| {
        let states = rebuild_states(beta_t, beta_ls, betaw);
        -family.evaluate(&states).expect("evaluate").log_likelihood
    };

    let extractgradient = extract_wiggle_gradient;

    let beta_t = Array1::from_vec(vec![0.15, -0.3, 0.2]);
    let beta_ls = Array1::from_vec(vec![-0.2, 0.1]);
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);
    let eval = family.evaluate(&states).expect("evaluate");
    let eps = 1e-6;

    for block_idx in 0..3 {
        let analytic = extractgradient(&eval, block_idx);
        let mut fd = Array1::<f64>::zeros(analytic.len());
        for j in 0..analytic.len() {
            let mut beta_t_plus = beta_t.clone();
            let mut beta_ls_plus = beta_ls.clone();
            let mut betaw_plus = betaw.clone();
            let mut beta_t_minus = beta_t.clone();
            let mut beta_ls_minus = beta_ls.clone();
            let mut betaw_minus = betaw.clone();
            match block_idx {
                BinomialLocationScaleWiggleFamily::BLOCK_T => {
                    beta_t_plus[j] += eps;
                    beta_t_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                    beta_ls_plus[j] += eps;
                    beta_ls_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                    betaw_plus[j] += eps;
                    betaw_minus[j] -= eps;
                }
                _ => unreachable!(),
            }
            let f_plus = objective(&beta_t_plus, &beta_ls_plus, &betaw_plus);
            let f_minus = objective(&beta_t_minus, &beta_ls_minus, &betaw_minus);
            fd[j] = (f_plus - f_minus) / (2.0 * eps);
        }
        gam_test_support::assert_matrix_derivativefd(
            &fd.insert_axis(Axis(1)),
            &(-&analytic).insert_axis(Axis(1)),
            2e-4,
            &format!("wiggle block {block_idx} score"),
        );
    }
}

#[test]
pub(crate) fn wiggle_family_joint_hessian_matches_fd_gradients_with_nontrivial_designs() {
    let (family, threshold_design, log_sigma_design, wiggle_block, y, weights) =
        wiggle_nontrivial_fixture();

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        rebuild_wiggle_nontrivial_states(
            &family,
            &threshold_design,
            &log_sigma_design,
            &y,
            &weights,
            beta_t,
            beta_ls,
            betaw,
        )
    };

    let extractgradient = extract_wiggle_gradient;

    let beta_t = Array1::from_vec(vec![0.15, -0.3, 0.2]);
    let beta_ls = Array1::from_vec(vec![-0.2, 0.1]);
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);
    let h_joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let pt = beta_t.len();
    let pls = beta_ls.len();
    let eps = 1e-6;
    let total = pt + pls + betaw.len();
    let mut fd = Array2::<f64>::zeros((total, total));
    let source_offsets = [0usize, pt, pt + pls];

    for source_block in 0..3 {
        let source_len = states[source_block].beta.len();
        for j in 0..source_len {
            let mut beta_t_plus = beta_t.clone();
            let mut beta_ls_plus = beta_ls.clone();
            let mut betaw_plus = betaw.clone();
            let mut beta_t_minus = beta_t.clone();
            let mut beta_ls_minus = beta_ls.clone();
            let mut betaw_minus = betaw.clone();
            match source_block {
                BinomialLocationScaleWiggleFamily::BLOCK_T => {
                    beta_t_plus[j] += eps;
                    beta_t_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                    beta_ls_plus[j] += eps;
                    beta_ls_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                    betaw_plus[j] += eps;
                    betaw_minus[j] -= eps;
                }
                _ => unreachable!(),
            }
            let eval_plus = family
                .evaluate(&rebuild_states(&beta_t_plus, &beta_ls_plus, &betaw_plus))
                .expect("eval plus");
            let eval_minus = family
                .evaluate(&rebuild_states(&beta_t_minus, &beta_ls_minus, &betaw_minus))
                .expect("eval minus");

            let mut row_offset = 0usize;
            for target_block in 0..3 {
                let grad_plus = extractgradient(&eval_plus, target_block);
                let grad_minus = extractgradient(&eval_minus, target_block);
                let col = (&grad_plus - &grad_minus).mapv(|v| -v / (2.0 * eps));
                let col_idx = source_offsets[source_block] + j;
                fd.slice_mut(s![
                    row_offset..row_offset + grad_plus.len(),
                    col_idx..col_idx + 1
                ])
                .assign(&col.insert_axis(Axis(1)));
                row_offset += grad_plus.len();
            }
        }
    }

    gam_test_support::assert_matrix_derivativefd(&fd, &h_joint, 4e-4, "wiggle joint hessian");
}

#[test]
pub(crate) fn wiggle_family_joint_exacthessian_directional_derivative_matches_finite_difference() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let eta_t = threshold_design.matrixvectormultiply(&beta_t);
    let eta_ls = log_sigma_design.matrixvectormultiply(&beta_ls);
    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);
    let states = vec![
        ParameterBlockState {
            beta: beta_t,
            eta: eta_t,
        },
        ParameterBlockState {
            beta: beta_ls,
            eta: eta_ls,
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw,
        },
    ];

    let base_h = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let direction = Array1::ones(base_h.nrows());
    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("joint dH")
        .expect("expected joint exact dH");

    let eps = 1e-6;
    let mut plus_states = states.clone();
    let beta_layout = GamlssBetaLayout::withwiggle(
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T]
            .beta
            .len(),
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
            .beta
            .len(),
        plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE]
            .beta
            .len(),
    );
    let (dir_t, dir_ls, dirw) = beta_layout
        .split_three(&direction, "wiggle test direction split")
        .expect("split wiggle test direction");
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta =
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta + &(eps * dir_t);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta =
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta + &(eps * dir_ls);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta =
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta + &(eps * dirw);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta = threshold_design
        .matrixvectormultiply(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].beta);
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta = log_sigma_design
        .matrixvectormultiply(
            &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].beta,
        );
    let plus_core_q0 = binomial_location_scale_core(
        &y,
        &weights,
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta,
        &plus_states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta,
        None,
        &family.link_kind,
    )
    .expect("plus core q0");
    plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta = family
        .wiggle_design(plus_core_q0.q0.view())
        .expect("plus wiggle design")
        .dot(&plus_states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta);

    let h_plus = family
        .exact_newton_joint_hessian(&plus_states)
        .expect("plus joint hessian")
        .expect("expected plus joint hessian");
    let fd = (h_plus - base_h) / eps;
    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 2e-3, "joint dH");
}

#[test]
pub(crate) fn wiggle_family_joint_exacthessiansecond_directional_derivative_matches_finite_difference()
 {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 4, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 4,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        let eta_t = threshold_design.matrixvectormultiply(beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
        let core_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let etaw = family
            .wiggle_design(core_q0.q0.view())
            .expect("wiggle design")
            .dot(betaw);
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t,
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ]
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let betaw = Array1::from_vec(vec![0.03; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);

    let pt = beta_t.len();
    let pls = beta_ls.len();
    let pw = betaw.len();
    let total = pt + pls + pw;
    let direction_u = Array1::from_shape_fn(total, |k| 0.2 + 0.1 * (k as f64));
    let directionv = Array1::from_shape_fn(total, |k| -0.15 + 0.07 * (k as f64));

    let analytic = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &direction_u, &directionv)
        .expect("joint d2H")
        .expect("expected joint exact d2H");

    let eps = 1e-6;
    let beta_layout = GamlssBetaLayout::withwiggle(pt, pls, pw);
    let (step_t, step_ls, stepw) = beta_layout
        .split_three(&directionv, "wiggle d2H test directionv")
        .expect("split wiggle test direction");

    let states_plus = rebuild_states(
        &(&beta_t + &(eps * &step_t)),
        &(&beta_ls + &(eps * &step_ls)),
        &(&betaw + &(eps * &stepw)),
    );
    let states_minus = rebuild_states(
        &(&beta_t - &(eps * &step_t)),
        &(&beta_ls - &(eps * &step_ls)),
        &(&betaw - &(eps * &stepw)),
    );
    let d_h_plus = family
        .exact_newton_joint_hessian_directional_derivative(&states_plus, &direction_u)
        .expect("joint dH plus")
        .expect("expected joint exact dH plus");
    let d_h_minus = family
        .exact_newton_joint_hessian_directional_derivative(&states_minus, &direction_u)
        .expect("joint dH minus")
        .expect("expected joint exact dH minus");
    let fd = (d_h_plus - d_h_minus) / (2.0 * eps);

    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 4e-3, "joint d2H");
}

#[test]
pub(crate) fn wiggle_family_joint_hessian_cross_blocks_match_finite_difference_of_gradients() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_block = intercept_block(n);
    let log_sigma_block = intercept_block(n);
    let q_seed = Array1::linspace(-1.4, 1.4, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 3, 4, 2, false)
            .expect("wiggle block");
    let threshold_design = threshold_block.design.clone();
    let log_sigma_design = log_sigma_block.design.clone();
    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        wiggle_knots: knots,
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let rebuild_states = |beta_t: &Array1<f64>,
                          beta_ls: &Array1<f64>,
                          betaw: &Array1<f64>|
     -> Vec<ParameterBlockState> {
        let eta_t = threshold_design.matrixvectormultiply(beta_t);
        let eta_ls = log_sigma_design.matrixvectormultiply(beta_ls);
        let core_q0 =
            binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
                .expect("core q0");
        let etaw = family
            .wiggle_design(core_q0.q0.view())
            .expect("wiggle design")
            .dot(betaw);
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: eta_t,
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: eta_ls,
            },
            ParameterBlockState {
                beta: betaw.clone(),
                eta: etaw,
            },
        ]
    };

    let extractgradient = |eval: &FamilyEvaluation, block_idx: usize| -> Array1<f64> {
        match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton {
                gradient,
                hessian: _,
            } => gradient.clone(),
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton"),
        }
    };

    let beta_t = Array1::from_vec(vec![0.25]);
    let beta_ls = Array1::from_vec(vec![-0.15]);
    let betaw = Array1::from_vec(vec![0.04; wiggle_block.design.ncols()]);
    let states = rebuild_states(&beta_t, &beta_ls, &betaw);

    let h_joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");

    let pt = beta_t.len();
    let pls = beta_ls.len();
    let pw = betaw.len();
    let eps = 1e-6;

    let fd_cross_block = |target_block: usize, source_block: usize| -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((
            states[target_block].beta.len(),
            states[source_block].beta.len(),
        ));
        for j in 0..states[source_block].beta.len() {
            let mut beta_t_plus = beta_t.clone();
            let mut beta_ls_plus = beta_ls.clone();
            let mut betaw_plus = betaw.clone();
            let mut beta_t_minus = beta_t.clone();
            let mut beta_ls_minus = beta_ls.clone();
            let mut betaw_minus = betaw.clone();
            match source_block {
                BinomialLocationScaleWiggleFamily::BLOCK_T => {
                    beta_t_plus[j] += eps;
                    beta_t_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA => {
                    beta_ls_plus[j] += eps;
                    beta_ls_minus[j] -= eps;
                }
                BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE => {
                    betaw_plus[j] += eps;
                    betaw_minus[j] -= eps;
                }
                _ => panic!("unexpected block"),
            }

            let eval_plus = family
                .evaluate(&rebuild_states(&beta_t_plus, &beta_ls_plus, &betaw_plus))
                .expect("eval plus");
            let eval_minus = family
                .evaluate(&rebuild_states(&beta_t_minus, &beta_ls_minus, &betaw_minus))
                .expect("eval minus");
            let grad_plus = extractgradient(&eval_plus, target_block);
            let grad_minus = extractgradient(&eval_minus, target_block);
            let col = (&grad_plus - &grad_minus).mapv(|v| -v / (2.0 * eps));
            out.slice_mut(ndarray::s![.., j]).assign(&col);
        }
        out
    };

    let fd_t_ls = fd_cross_block(
        BinomialLocationScaleWiggleFamily::BLOCK_T,
        BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
    );
    let fd_tw = fd_cross_block(
        BinomialLocationScaleWiggleFamily::BLOCK_T,
        BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
    );
    let fd_lsw = fd_cross_block(
        BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA,
        BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE,
    );

    let h_t_ls = h_joint.slice(ndarray::s![0..pt, pt..pt + pls]).to_owned();
    let h_tw = h_joint
        .slice(ndarray::s![0..pt, pt + pls..pt + pls + pw])
        .to_owned();
    let h_lsw = h_joint
        .slice(ndarray::s![pt..pt + pls, pt + pls..pt + pls + pw])
        .to_owned();

    gam_test_support::assert_matrix_derivativefd(&fd_t_ls, &h_t_ls, 2e-4, "H_t_ls");
    gam_test_support::assert_matrix_derivativefd(&fd_tw, &h_tw, 4e-4, "H_tw");
    gam_test_support::assert_matrix_derivativefd(&fd_lsw, &h_lsw, 6e-4, "H_lsw");
}

#[test]
pub(crate) fn nonwiggle_family_evaluate_returns_exact_newton_blockswhen_designs_are_present() {
    let n = 6usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).cos(),
                _ => unreachable!(),
            }
        }),
    ));
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let beta_t = array![0.2, -0.15];
    let beta_ls = array![-0.1, 0.05];
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

    let eval = family.evaluate(&states).expect("evaluate nonwiggle family");
    assert_eq!(eval.blockworking_sets.len(), 2);
    let joint = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let pt = beta_t.len();
    let pls = beta_ls.len();

    for (block_idx, (start, end)) in [(0usize, pt), (pt, pt + pls)].into_iter().enumerate() {
        let blockhessian = match &eval.blockworking_sets[block_idx] {
            BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
            BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
        };
        let joint_block = joint.slice(s![start..end, start..end]).to_owned();
        gam_test_support::assert_matrix_derivativefd(
            &joint_block,
            &blockhessian,
            1e-10,
            &format!("nonwiggle block {block_idx} principal block"),
        );
    }
}

#[test]
pub(crate) fn nonwiggle_family_joint_exacthessian_directional_derivative_matches_finite_difference()
{
    let n = 8usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                _ => unreachable!(),
            }
        }),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }),
    ));
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let rebuild_states = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: threshold_design.matrixvectormultiply(beta_t),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: log_sigma_design.matrixvectormultiply(beta_ls),
            },
        ]
    };

    let beta_t = array![0.2, -0.1];
    let beta_ls = array![-0.15, 0.08];
    let states = rebuild_states(&beta_t, &beta_ls);
    let base_h = family
        .exact_newton_joint_hessian(&states)
        .expect("joint hessian")
        .expect("expected joint exact hessian");
    let direction = array![0.2, 0.3, -0.15, 0.1];
    let analytic = family
        .exact_newton_joint_hessian_directional_derivative(&states, &direction)
        .expect("joint dH")
        .expect("expected joint exact dH");

    let eps = 1e-6;
    let dir_t = direction.slice(s![0..beta_t.len()]).to_owned();
    let dir_ls = direction.slice(s![beta_t.len()..]).to_owned();
    let states_plus = rebuild_states(&(&beta_t + &(eps * &dir_t)), &(&beta_ls + &(eps * &dir_ls)));
    let h_plus = family
        .exact_newton_joint_hessian(&states_plus)
        .expect("plus joint hessian")
        .expect("expected plus joint hessian");
    let fd = (h_plus - base_h) / eps;
    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 2e-3, "nonwiggle joint dH");
}

#[test]
pub(crate) fn nonwiggle_family_joint_exacthessiansecond_directional_derivative_matches_finite_difference()
 {
    let n = 8usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let threshold_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => (2.0 * std::f64::consts::PI * t).sin(),
                _ => unreachable!(),
            }
        }),
    ));
    let log_sigma_design = DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
        Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            match j {
                0 => 1.0,
                1 => t - 0.5,
                _ => unreachable!(),
            }
        }),
    ));
    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(threshold_design.clone()),
        log_sigma_design: Some(log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let rebuild_states = |beta_t: &Array1<f64>, beta_ls: &Array1<f64>| {
        vec![
            ParameterBlockState {
                beta: beta_t.clone(),
                eta: threshold_design.matrixvectormultiply(beta_t),
            },
            ParameterBlockState {
                beta: beta_ls.clone(),
                eta: log_sigma_design.matrixvectormultiply(beta_ls),
            },
        ]
    };

    let beta_t = array![0.2, -0.1];
    let beta_ls = array![-0.15, 0.08];
    let states = rebuild_states(&beta_t, &beta_ls);
    let direction_u = array![0.2, 0.3, -0.15, 0.1];
    let directionv = array![-0.05, 0.12, 0.08, -0.09];
    let analytic = family
        .exact_newton_joint_hessiansecond_directional_derivative(&states, &direction_u, &directionv)
        .expect("joint d2H")
        .expect("expected joint exact d2H");

    let eps = 1e-6;
    let step_t = directionv.slice(s![0..beta_t.len()]).to_owned();
    let step_ls = directionv.slice(s![beta_t.len()..]).to_owned();
    let states_plus = rebuild_states(
        &(&beta_t + &(eps * &step_t)),
        &(&beta_ls + &(eps * &step_ls)),
    );
    let states_minus = rebuild_states(
        &(&beta_t - &(eps * &step_t)),
        &(&beta_ls - &(eps * &step_ls)),
    );
    let d_h_plus = family
        .exact_newton_joint_hessian_directional_derivative(&states_plus, &direction_u)
        .expect("joint dH plus")
        .expect("expected joint exact dH plus");
    let d_h_minus = family
        .exact_newton_joint_hessian_directional_derivative(&states_minus, &direction_u)
        .expect("joint dH minus")
        .expect("expected joint exact dH minus");
    let fd = (d_h_plus - d_h_minus) / (2.0 * eps);
    gam_test_support::assert_matrix_derivativefd(&fd, &analytic, 4e-3, "nonwiggle joint d2H");
}

#[test]
pub(crate) fn wiggle_basis_is_structurally_monotone_for_nonnegative_coefficients() {
    let q_seed = Array1::linspace(-2.0, 2.0, 17);
    let degree = 3usize;
    let num_internal_knots = 6usize;
    let penalty_order = 2usize;

    let (block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
        q_seed.view(),
        degree,
        num_internal_knots,
        penalty_order,
        false,
    )
    .expect("wiggle block");
    let design = match &block.design {
        DesignMatrix::Dense(x) => x.to_dense_arc(),
        DesignMatrix::Sparse(_) => panic!("expected dense wiggle design"),
    };
    let beta = Array1::from_elem(design.ncols(), 0.2);
    let derivative = monotone_wiggle_basis_with_derivative_order(q_seed.view(), &knots, degree, 1)
        .expect("wiggle derivative basis")
        .dot(&beta);
    assert!(
        derivative.iter().all(|&value| value >= -1e-12),
        "I-spline wiggle derivative must stay non-negative for non-negative coefficients: min={}",
        derivative.iter().fold(f64::INFINITY, |acc, &v| acc.min(v))
    );
}

#[test]
pub(crate) fn degeneratewiggle_seed_uses_broad_fallback_domain() {
    let q_seed = Array1::zeros(9);
    let degree = 3usize;
    let knots = initializewiggle_knots_from_seed(q_seed.view(), degree, 5)
        .expect("initialize degenerate wiggle knots");
    let bs_degree = monotone_wiggle_internal_degree(degree).expect("cubic wiggle degree") + 1;
    let domain_min = knots[bs_degree];
    let domain_max = knots[knots.len() - bs_degree - 1];
    assert!(
        domain_min <= -2.9,
        "unexpected left fallback boundary: {domain_min}"
    );
    assert!(
        domain_max >= 2.9,
        "unexpected right fallback boundary: {domain_max}"
    );
}

#[test]
pub(crate) fn wiggle_block_design_matches_ispline_basis() {
    let q_seed = Array1::linspace(-1.0, 1.0, 11);
    let degree = 2usize;
    let num_internal_knots = 4usize;
    let penalty_order = 2usize;

    let (block, knots) = BinomialLocationScaleWiggleFamily::buildwiggle_block_input(
        q_seed.view(),
        degree,
        num_internal_knots,
        penalty_order,
        false,
    )
    .expect("wiggle block");
    let (basis, _) = create_basis::<Dense>(
        q_seed.view(),
        KnotSource::Provided(knots.view()),
        monotone_wiggle_internal_degree(degree).expect("wiggle degree"),
        BasisOptions::i_spline(),
    )
    .expect("I-spline basis");
    let expected = (*basis).clone();

    let got = match &block.design {
        DesignMatrix::Dense(x) => x.to_dense_arc(),
        DesignMatrix::Sparse(_) => panic!("expected dense wiggle design"),
    };
    assert_eq!(got.dim(), expected.dim());
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            assert!(
                (got[[i, j]] - expected[[i, j]]).abs() < 1e-10,
                "wiggle design mismatch at ({}, {}): got {}, expected {}",
                i,
                j,
                got[[i, j]],
                expected[[i, j]]
            );
        }
    }
}

#[test]
pub(crate) fn split_wiggle_penalty_orders_uses_requested_order_one_as_primary() {
    let (primary, extras) =
        split_wiggle_penalty_orders(2, &[1, 2, 3, 3]).expect("valid derivative orders");
    assert_eq!(primary, 1);
    assert_eq!(extras, vec![2, 3]);
}

#[test]
pub(crate) fn selected_wiggle_function_penalties_keep_order_one() {
    let q_seed = Array1::linspace(-1.0, 1.0, 11);
    let degree = 3usize;
    let num_internal_knots = 5usize;
    let cfg = WiggleBlockConfig {
        degree,
        num_internal_knots,
        penalty_order: 1,
        double_penalty: false,
    };
    let selected =
        select_wiggle_basis_from_seed(q_seed.view(), &cfg, &[1, 3]).expect("selected wiggle basis");

    assert_eq!(selected.block.penalties.len(), 2);
    assert_eq!(selected.block.nullspace_dims, vec![0, 2]);
}

#[test]
pub(crate) fn binomial_location_scale_generative_matches_coremu() {
    let n = 7usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let eta_t = Array1::from_vec(vec![0.8, -0.4, 0.2, -1.1, 0.0, 0.5, -0.7]);
    let eta_ls = Array1::from_vec(vec![-3.0, -1.2, -0.1, 0.3, 1.1, 2.0, 4.0]);

    let family = BinomialLocationScaleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: None,
        log_sigma_design: None,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_t.clone(),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_ls.clone(),
        },
    ];
    let spec = family.generativespec(&states).expect("generative spec");
    let core = binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
        .expect("core");
    for i in 0..n {
        assert!(
            (spec.mean[i] - core.mu[i]).abs() < 1e-7,
            "mean mismatch at {i}: got {}, expected {}",
            spec.mean[i],
            core.mu[i]
        );
    }
}

#[test]
pub(crate) fn wiggle_geometry_and_generative_use_same_sigma_link_as_core() {
    let n = 8usize;
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    let weights = Array1::from_vec(vec![1.0; n]);
    let eta_t = Array1::from_vec(vec![0.5, -0.6, 0.1, -0.3, 0.9, -0.2, 0.4, -0.8]);
    let eta_ls = Array1::from_vec(vec![-2.5, -1.5, -0.5, 0.0, 0.7, 1.4, 2.2, 3.0]);

    let q_seed = Array1::linspace(-1.5, 1.5, n);
    let (wiggle_block, knots) =
        BinomialLocationScaleWiggleFamily::buildwiggle_block_input(q_seed.view(), 2, 3, 2, false)
            .expect("wiggle block");

    let family = BinomialLocationScaleWiggleFamily {
        y: y.clone(),
        weights: weights.clone(),
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: None,
        log_sigma_design: None,
        wiggle_knots: knots,
        wiggle_degree: 2,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let core_for_q0 =
        binomial_location_scale_core(&y, &weights, &eta_t, &eta_ls, None, &family.link_kind)
            .expect("core q0");
    let betaw = Array1::from_vec(vec![0.15; wiggle_block.design.ncols()]);
    let etaw = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("wiggle design")
        .dot(&betaw);

    let states = vec![
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_t.clone(),
        },
        ParameterBlockState {
            beta: Array1::zeros(1),
            eta: eta_ls.clone(),
        },
        ParameterBlockState {
            beta: betaw.clone(),
            eta: etaw.clone(),
        },
    ];

    let wigglespec = wiggle_block
        .clone()
        .intospec("wiggle")
        .expect("wiggle spec");
    let (geom_x, _) = family
        .block_geometry(&states, &wigglespec)
        .expect("block geometry");
    let geom = match geom_x {
        DesignMatrix::Dense(x) => x.to_dense(),
        DesignMatrix::Sparse(_) => panic!("expected dense wiggle geometry design"),
    };
    let expected_geom = family
        .wiggle_design(core_for_q0.q0.view())
        .expect("expected wiggle geometry");
    assert_eq!(geom.dim(), expected_geom.dim());
    for i in 0..geom.nrows() {
        for j in 0..geom.ncols() {
            assert!(
                (geom[[i, j]] - expected_geom[[i, j]]).abs() < 1e-12,
                "geometry mismatch at ({i}, {j}): got {}, expected {}",
                geom[[i, j]],
                expected_geom[[i, j]]
            );
        }
    }

    let generated = family.generativespec(&states).expect("generative spec");
    let core = binomial_location_scale_core(
        &y,
        &weights,
        &eta_t,
        &eta_ls,
        Some(&etaw),
        &family.link_kind,
    )
    .expect("core with wiggle");
    for i in 0..n {
        assert!(
            (generated.mean[i] - core.mu[i]).abs() < 1e-7,
            "wiggle mean mismatch at {i}: got {}, expected {}",
            generated.mean[i],
            core.mu[i]
        );
    }
}

#[test]
pub(crate) fn poisson_extreme_eta_uses_exact_exp_and_refuses_only_unrepresentable_geometry() {
    use crate::custom_family::{CustomFamily, ParameterBlockState};
    let poisson = PoissonLogFamily {
        y: Array1::from_vec(vec![1.0, 2.0, 3.0]),
        weights: Array1::from_vec(vec![1.0, 1.0, 1.0]),
    };
    let extreme_eta = Array1::from_vec(vec![0.5, 709.0, -0.3]);
    let eval_result = poisson.evaluate(&[ParameterBlockState {
        beta: Array1::zeros(0),
        eta: extreme_eta,
    }]);
    let eval =
        eval_result.expect("Poisson evaluate must succeed while exact geometry is representable");
    match &eval.blockworking_sets[0] {
        crate::custom_family::BlockWorkingSet::Diagonal {
            working_response,
            working_weights,
        } => {
            let all_finite = working_response.iter().all(|v| v.is_finite())
                && working_weights.iter().all(|v| v.is_finite())
                && eval.log_likelihood.is_finite();
            assert!(
                all_finite,
                "Poisson evaluate should produce finite outputs for all eta, \
                     but got non-finite values: ll={}, z={:?}, w={:?}",
                eval.log_likelihood, working_response, working_weights
            );
        }
        _ => panic!("expected Diagonal block"),
    }

    let refused = match poisson.evaluate(&[ParameterBlockState {
        beta: Array1::zeros(0),
        eta: Array1::from_vec(vec![0.5, 710.0, -0.3]),
    }]) {
        Ok(_) => panic!("overflowing exact exp geometry must be refused"),
        Err(err) => err,
    };
    assert!(refused.contains("row 1"), "unexpected refusal: {refused}");
}

/// The batched outer-gradient override on `BinomialLocationScaleFamily`
/// must produce a gradient that agrees with the central finite
/// difference of the same family's outer cost. This is the strongest
/// available correctness property: it does not depend on whether the
/// generic per-coordinate path is reachable in this build, only on the
/// scale-invariant identity `g_k = (V(ρ + h e_k) − V(ρ − h e_k)) / (2h)`
/// at converged β̂. Because the unified evaluator already routes
/// `ValueAndGradient` calls through the batched override (custom_family.rs
/// at the `batched_outer_gradient_terms` call site), this also pins the
/// wiring: any future regression that detaches the override from the
/// dispatcher will trip the FD check via stale (zero) gradients.
#[test]
pub(crate) fn binomial_location_scale_batched_gradient_matches_finite_difference() {
    use crate::custom_family::BlockwiseFitOptions;

    // 7-row, two-block intercept-only problem with a unit-Identity
    // penalty per block. Larger n risks PIRLS taking many iterations and
    // amplifying FD round-off; small p keeps the leverage-block sizes
    // (p_t = 1, p_ls = 1) tiny so the manual reference is trivial to
    // sanity-check.
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design),
        log_sigma_design: Some(base.log_sigma_design),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };

    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let rho = array![0.05, -0.15];
    let options = BlockwiseFitOptions {
        use_remlobjective: true,
        ridge_floor: 1e-10,
        outer_max_iter: 1,
        ..BlockwiseFitOptions::default()
    };

    let eval_outer = |rho: &Array1<f64>| {
        let derivative_blocks = vec![Vec::<CustomFamilyBlockPsiDerivative>::new(); specs.len()];
        let result = evaluate_custom_family_joint_hyper(
            &family,
            &specs,
            &options,
            rho,
            &test_design_hyper_layout(&derivative_blocks),
            None,
            gam_problem::EvalMode::ValueAndGradient,
        )
        .expect("objective+gradient at rho");
        (result.objective, result.gradient)
    };

    let (f0, g0) = eval_outer(&rho);
    assert!(f0.is_finite(), "outer cost must be finite at rho");
    assert_eq!(g0.len(), rho.len());

    let h = 1e-5;
    // Same noise-floor convention as the existing wiggle-family FD test
    // (custom_family.rs `outer_lamlgradient_matches_finite_differencewhen_joint_exact_path_is_active`):
    // below floor `EPS·|cost|/h`, the FD estimator can't resolve the
    // true gradient.
    let cost_magnitude = f0.abs().max(1.0);
    let noise_floor = (10.0 * f64::EPSILON * cost_magnitude / h).max(1e-9);

    for k in 0..rho.len() {
        let mut rho_p = rho.clone();
        let mut rho_m = rho.clone();
        rho_p[k] += h;
        rho_m[k] -= h;
        let (fp, _) = eval_outer(&rho_p);
        let (fm, _) = eval_outer(&rho_m);
        let gfd = (fp - fm) / (2.0 * h);
        let both_in_noise = g0[k].abs() < noise_floor && gfd.abs() < noise_floor;
        if !both_in_noise {
            let abs_err = (g0[k] - gfd).abs();
            let rel_err = abs_err / gfd.abs().max(g0[k].abs()).max(1e-12);
            assert!(
                rel_err < 1e-3 || abs_err < 1e-6,
                "batched gradient mismatch at coord {k}: \
                     batched={:.6e}, fd={:.6e}, abs_err={:.3e}, rel_err={:.3e}",
                g0[k],
                gfd,
                abs_err,
                rel_err,
            );
        }
    }
}

pub(crate) fn binomial_mean_wiggle_operator_fixture() -> (
    BinomialMeanWiggleFamily,
    Vec<ParameterBlockState>,
    Vec<ParameterBlockSpec>,
    Array2<f64>,
) {
    let x_eta = array![
        [1.0, -0.9],
        [1.0, -0.45],
        [1.0, -0.1],
        [1.0, 0.2],
        [1.0, 0.55],
        [1.0, 0.9],
    ];
    let beta_eta = array![-0.15, 0.7];
    let eta = x_eta.dot(&beta_eta);
    let degree = 3usize;
    let knots = initializewiggle_knots_from_seed(eta.view(), degree, 4).expect("mean-wiggle knots");
    let family = BinomialMeanWiggleFamily {
        y: array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        weights: array![1.0, 0.8, 1.2, 1.0, 0.7, 1.1],
        link_kind: InverseLink::Standard(StandardLink::Logit),
        wiggle_knots: knots,
        wiggle_degree: degree,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        frozen_warp_design: None,
    };
    let basis = family.wiggle_design(eta.view()).expect("wiggle basis");
    let beta_w = Array1::from_iter((0..basis.ncols()).map(|j| 0.015 * (j as f64 + 1.0)));
    let etaw = basis.dot(&beta_w);
    let states = vec![
        ParameterBlockState {
            beta: beta_eta,
            eta: eta.clone(),
        },
        ParameterBlockState {
            beta: beta_w,
            eta: etaw,
        },
    ];
    let specs = vec![
        ParameterBlockSpec {
            name: "eta".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(x_eta.clone())),
            offset: Array1::zeros(eta.len()),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(basis)),
            offset: Array1::zeros(eta.len()),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    (family, states, specs, x_eta)
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

#[test]
pub(crate) fn binomial_location_scale_expected_info_derivatives_match_finite_difference() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    // gam#1020: the expected-information override must disarm the
    // observed-Hessian "Jeffreys skippable" matvec pre-checks.
    assert!(!family.joint_jeffreys_information_matches_observed_hessian());
    let beta_t = Array1::from_iter((0..x_t.ncols()).map(|j| 0.12 - 0.03 * j as f64));
    let beta_ls = Array1::from_iter((0..x_ls.ncols()).map(|j| -0.08 + 0.02 * j as f64));
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: x_t.dot(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: x_ls.dot(&beta_ls),
        },
    ];
    let total = beta_t.len() + beta_ls.len();
    let u = Array1::from_iter((0..total).map(|j| 0.03 * (j as f64 + 0.4).sin()));
    let v = Array1::from_iter((0..total).map(|j| -0.02 * (j as f64 + 0.7).cos()));

    let info = |direction: &Array1<f64>, scale: f64| {
        let mut next = states.clone();
        let pt = beta_t.len();
        next[BinomialLocationScaleFamily::BLOCK_T]
            .beta
            .scaled_add(scale, &direction.slice(s![0..pt]));
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
            .beta
            .scaled_add(scale, &direction.slice(s![pt..total]));
        next[BinomialLocationScaleFamily::BLOCK_T].eta =
            x_t.dot(&next[BinomialLocationScaleFamily::BLOCK_T].beta);
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
            x_ls.dot(&next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
        family
            .joint_jeffreys_information_with_specs(&next, &specs)
            .expect("expected information")
            .expect("expected information available")
    };

    let h0 = family
        .joint_jeffreys_information_with_specs(&states, &specs)
        .expect("expected information")
        .expect("expected information available");
    assert_close_matrix(&info(&u, 0.0), &h0, 1e-12, "expected information value");

    let eps = 1e-5;
    let hp = info(&u, eps);
    let hm = info(&u, -eps);
    let fd_first = (&hp - &hm) / (2.0 * eps);
    let analytic_first = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &u)
        .expect("expected dI")
        .expect("expected dI available");
    assert_close_matrix(&analytic_first, &fd_first, 1e-7, "expected dI");

    let mut states_plus = states.clone();
    let pt = beta_t.len();
    states_plus[BinomialLocationScaleFamily::BLOCK_T]
        .beta
        .scaled_add(eps, &v.slice(s![0..pt]));
    states_plus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .scaled_add(eps, &v.slice(s![pt..total]));
    states_plus[BinomialLocationScaleFamily::BLOCK_T].eta =
        x_t.dot(&states_plus[BinomialLocationScaleFamily::BLOCK_T].beta);
    states_plus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
        x_ls.dot(&states_plus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
    let d_plus = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states_plus, &specs, &u)
        .expect("expected dI plus")
        .expect("expected dI plus available");

    let mut states_minus = states.clone();
    states_minus[BinomialLocationScaleFamily::BLOCK_T]
        .beta
        .scaled_add(-eps, &v.slice(s![0..pt]));
    states_minus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .scaled_add(-eps, &v.slice(s![pt..total]));
    states_minus[BinomialLocationScaleFamily::BLOCK_T].eta =
        x_t.dot(&states_minus[BinomialLocationScaleFamily::BLOCK_T].beta);
    states_minus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
        x_ls.dot(&states_minus[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
    let d_minus = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states_minus, &specs, &u)
        .expect("expected dI minus")
        .expect("expected dI minus available");
    let fd_second = (&d_plus - &d_minus) / (2.0 * eps);
    let analytic_second = family
        .joint_jeffreys_information_second_directional_derivative_with_specs(
            &states, &specs, &u, &v,
        )
        .expect("expected d2I")
        .expect("expected d2I available");
    assert_close_matrix(&analytic_second, &fd_second, 1e-7, "expected d2I");
}

/// Layer-5 deliverable (gam#979 / gam#1020): the Tier-B Jeffreys term built
/// on the EXPECTED Fisher information must NOT reward probit saturation,
/// whereas the OBSERVED-information Jeffreys term DOES — which is the long
/// quasi-flat descent valley that made the constrained-wiggle inner solve
/// walk `|β|→∞`.
///
/// Mechanism. For probit `q ↦ Φ(q)`, drive the threshold predictor `η_t`
/// (hence `q`) into saturation. The OBSERVED per-row curvature
/// `−∂²ℓ/∂q² = w·(z·q′ + …)` carries the misclassification term that GROWS
/// like `q²` on rows the saturated mean gets wrong, so `½log det H_obs`
/// climbs without bound — Φ_obs rewards walking toward saturation. The
/// EXPECTED Fisher weight `w^F = φ(q)²/(p(1−p))` DECAYS as `q→±∞` (the
/// Gaussian pdf `φ` kills the numerator faster than `p(1−p)→0` shrinks the
/// denominator), so `½log det H_exp` is bounded above — Φ_exp has no valley.
///
/// The assertion: across a saturation sweep, Φ on the expected information
/// stays bounded (and ultimately decreases), while Φ on the observed
/// information grows past it — the exact sign that the expected-information
/// hook removes the gam#979 saturation reward. Both Φ are evaluated through
/// the SAME `joint_jeffreys_term` value path on the FULL identifiable span
/// (`Z_J = I`), differing only in the information matrix consumed.
#[test]
pub(crate) fn expected_info_jeffreys_does_not_reward_probit_saturation() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    let total = x_t.ncols() + x_ls.ncols();
    let z = Array2::<f64>::eye(total);

    // Φ on a supplied information matrix at threshold β_t (log-σ fixed at 0,
    // so σ = 1 and q = -β_t scans the probit argument across saturation).
    let phi_on = |info: &Array2<f64>| -> f64 {
        let (phi, _grad, _hphi) =
            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
                info.view(),
                z.view(),
                |_axis: &Array1<f64>| Ok(None),
            )
            .expect("jeffreys term value");
        phi
    };
    let states_at = |beta_t: f64| -> Vec<ParameterBlockState> {
        let bt = Array1::from_elem(x_t.ncols(), beta_t);
        let bls = Array1::zeros(x_ls.ncols());
        vec![
            ParameterBlockState {
                eta: x_t.dot(&bt),
                beta: bt,
            },
            ParameterBlockState {
                eta: x_ls.dot(&bls),
                beta: bls,
            },
        ]
    };

    // Sweep the threshold into deep probit saturation.
    let betas = [1.0_f64, 2.0, 3.0, 4.0, 6.0, 8.0];
    let mut phi_obs = Vec::with_capacity(betas.len());
    let mut phi_exp = Vec::with_capacity(betas.len());
    for &b in betas.iter() {
        let states = states_at(b);
        let obs = family
            .exact_newton_joint_hessian_with_specs(&states, &specs)
            .expect("observed hessian")
            .expect("observed hessian available");
        let exp = family
            .joint_jeffreys_information_with_specs(&states, &specs)
            .expect("expected information")
            .expect("expected information available");
        phi_obs.push(phi_on(&obs));
        phi_exp.push(phi_on(&exp));
    }

    // (1) The expected-information Jeffreys term is BOUNDED across the sweep
    // (no runaway reward); concretely it does not increase from its
    // mild-saturation value to its deepest-saturation value — the valley is
    // gone (decaying expected information).
    let exp_first = phi_exp[0];
    let exp_last = *phi_exp.last().expect("nonempty");
    assert!(
        exp_last <= exp_first + 1e-9,
        "expected-info Jeffreys Φ rewarded saturation: Φ_exp went {exp_first:.6} → {exp_last:.6} \
             across β_t {:?} (full sweep {phi_exp:?})",
        betas
    );

    // (2) The observed-information Jeffreys term, in contrast, GROWS into
    // saturation and overtakes the expected one — the genuine valley the
    // layer-5 hook exists to remove. This makes the test a real
    // discriminator: it fails if the family silently reverts to observed
    // information.
    let obs_last = *phi_obs.last().expect("nonempty");
    assert!(
        obs_last > exp_last + 0.5,
        "observed-info Jeffreys Φ did not exhibit the saturation valley the \
             expected-info hook removes: Φ_obs_last={obs_last:.6} vs Φ_exp_last={exp_last:.6} \
             (Φ_obs sweep {phi_obs:?}, Φ_exp sweep {phi_exp:?})"
    );
}

#[test]
pub(crate) fn binomial_location_scale_expected_info_contracted_trace_matches_second_directional() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    let beta_t = Array1::from_iter((0..x_t.ncols()).map(|j| 0.11 - 0.02 * j as f64));
    let beta_ls = Array1::from_iter((0..x_ls.ncols()).map(|j| -0.07 + 0.03 * j as f64));
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: x_t.dot(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: x_ls.dot(&beta_ls),
        },
    ];
    let total = beta_t.len() + beta_ls.len();
    let weight = Array2::from_shape_fn((total, total), |(i, j)| {
        0.03 * ((i + 2 * j + 1) as f64).sin()
    });
    let contracted = family
        .joint_jeffreys_information_contracted_trace_hessian_with_specs(&states, &specs, &weight)
        .expect("contracted trace")
        .expect("contracted trace present");
    let mut expected = Array2::<f64>::zeros((total, total));
    for a in 0..total {
        let mut axis_a = Array1::<f64>::zeros(total);
        axis_a[a] = 1.0;
        for b in a..total {
            let mut axis_b = Array1::<f64>::zeros(total);
            axis_b[b] = 1.0;
            let second = family
                .joint_jeffreys_information_second_directional_derivative_with_specs(
                    &states, &specs, &axis_a, &axis_b,
                )
                .expect("expected d2I")
                .expect("expected d2I present");
            let mut trace = 0.0;
            for row in 0..total {
                for col in 0..total {
                    trace += weight[[row, col]] * second[[col, row]];
                }
            }
            expected[[a, b]] = trace;
            expected[[b, a]] = trace;
        }
    }
    assert_close_matrix(&contracted, &expected, 1e-9, "expected contracted trace");
}

#[test]
pub(crate) fn binomial_location_scale_expected_hphi_drift_matches_finite_difference() {
    let base = binomial_location_scale_base_fixture();
    let family = BinomialLocationScaleFamily {
        y: base.y,
        weights: base.weights,
        link_kind: InverseLink::Standard(StandardLink::Probit),
        threshold_design: Some(base.threshold_design.clone()),
        log_sigma_design: Some(base.log_sigma_design.clone()),
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
    };
    let specs = vec![base.threshold_spec, base.log_sigma_spec];
    let x_t = specs[BinomialLocationScaleFamily::BLOCK_T]
        .design
        .as_dense_ref()
        .expect("threshold dense design");
    let x_ls = specs[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
        .design
        .as_dense_ref()
        .expect("log-sigma dense design");
    let beta_t = Array1::from_iter((0..x_t.ncols()).map(|j| 0.09 - 0.02 * j as f64));
    let beta_ls = Array1::from_iter((0..x_ls.ncols()).map(|j| -0.06 + 0.04 * j as f64));
    let states = vec![
        ParameterBlockState {
            beta: beta_t.clone(),
            eta: x_t.dot(&beta_t),
        },
        ParameterBlockState {
            beta: beta_ls.clone(),
            eta: x_ls.dot(&beta_ls),
        },
    ];
    let total = beta_t.len() + beta_ls.len();
    // IDENTIFIABLE-SPAN Jeffreys subspace `Z_J`. The binomial location-scale map
    // `q = −η_t/σ` carries an EXACT threshold↔scale gauge degeneracy: the
    // direction `(δη_t = η_t, δη_ls = 1)` gives `q̇ = q_t·η_t + q_ls = −η_t/σ +
    // η_t/σ = 0`, so the per-row q-gradient — hence the whole expected Fisher
    // information `I(β)` — is rank-deficient by exactly one along this gauge
    // axis. On the constant-design fixture every row is proportional, so `I` is
    // rank 1 and its smallest eigenvalue is structurally ZERO. Differencing the
    // floored-pseudo-inverse `H_Φ` over the FULL span (`Z = I`) therefore
    // central-differences a quantity whose near-zero-eigenvalue eigenvector is
    // arbitrary up to numerical noise: the FD is meaningless (it swings by
    // O(1/floor) with the eps choice) even though the analytic drift is exact.
    // Production never runs the Jeffreys term on the raw gauge-degenerate span;
    // it reduces to the identifiable coordinates first. We mirror that here by
    // taking `Z_J` to be the eigenvectors of the base information with
    // non-negligible eigenvalue, so the reduced `H_Φ` is well-conditioned and
    // its central difference converges to the analytic directional derivative at
    // the 1e-7 bar. (The dropped gauge axis carries no identifiable curvature, so
    // restricting to it loses nothing the objective ever uses.)
    let z = {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;
        let base_info = family
            .joint_jeffreys_information_with_specs(&states, &specs)
            .expect("base expected info")
            .expect("base expected info present");
        let mut sym = Array2::<f64>::zeros((total, total));
        for i in 0..total {
            for j in 0..total {
                sym[[i, j]] = 0.5 * (base_info[[i, j]] + base_info[[j, i]]);
            }
        }
        let (evals, evecs) = sym.eigh(Side::Lower).expect("base info eigendecomposition");
        let lambda_max = evals.iter().cloned().fold(0.0_f64, f64::max);
        // Keep the identifiable directions (curvature ≥ a tiny fraction of the
        // dominant eigenvalue); drop the structural gauge null space.
        let keep: Vec<usize> = (0..total)
            .filter(|&i| evals[i] > lambda_max * 1e-8)
            .collect();
        assert!(
            !keep.is_empty(),
            "base information must have an identifiable direction"
        );
        let mut z = Array2::<f64>::zeros((total, keep.len()));
        for (col, &i) in keep.iter().enumerate() {
            z.column_mut(col).assign(&evecs.column(i));
        }
        z
    };
    let direction = Array1::from_shape_fn(total, |i| 0.03 * ((i + 1) as f64).sin());
    let perturb = |scale: f64| {
        let mut next = states.clone();
        let pt = beta_t.len();
        next[BinomialLocationScaleFamily::BLOCK_T]
            .beta
            .scaled_add(scale, &direction.slice(s![0..pt]));
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA]
            .beta
            .scaled_add(scale, &direction.slice(s![pt..total]));
        next[BinomialLocationScaleFamily::BLOCK_T].eta =
            x_t.dot(&next[BinomialLocationScaleFamily::BLOCK_T].beta);
        next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].eta =
            x_ls.dot(&next[BinomialLocationScaleFamily::BLOCK_LOG_SIGMA].beta);
        next
    };
    let hphi_at = |block_states: &[ParameterBlockState]| {
        let info = family
            .joint_jeffreys_information_with_specs(block_states, &specs)
            .expect("expected info")
            .expect("expected info present");
        let (_phi, _grad, hphi) =
            gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_term(
                info.view(),
                z.view(),
                |axis: &Array1<f64>| {
                    family.joint_jeffreys_information_directional_derivative_with_specs(
                        block_states,
                        &specs,
                        axis,
                    )
                },
            )
            .expect("hphi term");
        hphi
    };
    let eps = 1e-5;
    let h_plus = hphi_at(&perturb(eps));
    let h_minus = hphi_at(&perturb(-eps));
    let fd = (&h_plus - &h_minus) / (2.0 * eps);
    let info = family
        .joint_jeffreys_information_with_specs(&states, &specs)
        .expect("expected info")
        .expect("expected info present");
    // Mode-response drift `D_β H_Φ[δ]` via the production-level perturbation core
    // (the `joint_jeffreys_hphi_directional_derivative` oracle is a thin wrapper
    // over this: `Hdot[δ]` once, then the perturbation derivative). Calling the
    // core directly keeps the oracle private to its own `#[cfg(test)]` module.
    let pert_h = family
        .joint_jeffreys_information_directional_derivative_with_specs(&states, &specs, &direction)
        .expect("Hdot[delta]")
        .expect("Hdot[delta] present");
    let analytic =
        gam_solve::estimate::reml::jeffreys_subspace::joint_jeffreys_hphi_perturbation_derivative(
            info.view(),
            z.view(),
            |axis: &Array1<f64>| {
                family.joint_jeffreys_information_directional_derivative_with_specs(
                    &states, &specs, axis,
                )
            },
            &pert_h,
            |axis: &Array1<f64>| {
                family.joint_jeffreys_information_second_directional_derivative_with_specs(
                    &states, &specs, &direction, axis,
                )
            },
        )
        .expect("hphi drift");
    assert_close_matrix(&analytic, &fd, 1e-7, "expected H_phi drift");
}

#[test]
pub(crate) fn binomial_mean_wiggle_hessian_operators_match_dense_derivatives() {
    let (family, states, specs, x_eta) = binomial_mean_wiggle_operator_fixture();
    let p_eta = x_eta.ncols();
    let pw = states[BinomialMeanWiggleFamily::BLOCK_WIGGLE].beta.len();
    let total = p_eta + pw;
    let dir_u = Array1::from_iter((0..total).map(|j| 0.03 * (j as f64 + 1.0).sin()));
    let dir_v = Array1::from_iter((0..total).map(|j| -0.02 * (j as f64 + 0.5).cos()));

    let dense_h = family
        .exact_newton_joint_hessian_with_specs(&states, &specs)
        .expect("dense H")
        .expect("dense H available");
    let workspace = family
        .exact_newton_joint_hessian_workspace(&states, &specs)
        .expect("workspace")
        .expect("workspace available");
    let h_columns = Array2::from_shape_fn((total, total), |(i, j)| if i == j { 1.0 } else { 0.0 });
    let op_h = gam_problem::HyperOperator::mul_mat(
        family
            .bmw_static_hessian_operator(&states, Arc::new(x_eta.clone()))
            .expect("static op")
            .as_ref(),
        &h_columns,
    );
    assert_close_matrix(&op_h, &dense_h, 1e-10, "static H operator");
    let hv = workspace
        .hessian_matvec(&dir_u)
        .expect("workspace HVP")
        .expect("workspace HVP available");
    let hv_dense = dense_h.dot(&dir_u);
    let hv_err = (&hv - &hv_dense).mapv(f64::abs).sum();
    assert!(hv_err < 1e-10, "workspace HVP mismatch {hv_err:.3e}");

    let dense_dh = family
        .exact_newton_joint_hessian_directional_derivative_with_specs(&states, &specs, &dir_u)
        .expect("dense dH")
        .expect("dense dH available");
    let op_dh = workspace
        .directional_derivative_operator(&dir_u)
        .expect("dH operator")
        .expect("dH operator available")
        .to_dense();
    assert_close_matrix(&op_dh, &dense_dh, 1e-10, "directional dH operator");

    let dense_d2h = family
        .exact_newton_joint_hessian_second_directional_derivative_with_specs(
            &states, &specs, &dir_u, &dir_v,
        )
        .expect("dense d2H")
        .expect("dense d2H available");
    let op_d2h = workspace
        .second_directional_derivative_operator(&dir_u, &dir_v)
        .expect("d2H operator")
        .expect("d2H operator available")
        .to_dense();
    assert_close_matrix(
        &op_d2h,
        &dense_d2h,
        1e-10,
        "second directional d2H operator",
    );
}

#[test]
pub(crate) fn binomial_mean_wiggle_planner_keeps_second_order_at_large_n() {
    let n = 50_001usize;
    let family = BinomialMeanWiggleFamily {
        y: Array1::zeros(n),
        weights: Array1::ones(n),
        link_kind: InverseLink::Standard(StandardLink::Logit),
        wiggle_knots: initializewiggle_knots_from_seed(Array1::linspace(-1.0, 1.0, 9).view(), 3, 4)
            .expect("large-n knots"),
        wiggle_degree: 3,
        policy: gam_runtime::resource::ResourcePolicy::default_library(),
        frozen_warp_design: None,
    };
    let specs = vec![
        ParameterBlockSpec {
            name: "eta".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, 2)),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
        ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(
                Array2::zeros((n, 34)),
            )),
            offset: Array1::zeros(n),
            penalties: vec![],
            nullspace_dims: vec![],
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        },
    ];
    assert!(family.inner_coefficient_hessian_hvp_available(&specs));
    assert_eq!(
        family.exact_outer_derivative_order(&specs, &BlockwiseFitOptions::default()),
        crate::custom_family::ExactOuterDerivativeOrder::Second
    );
}

/// #2387 contract for the observed joint Hessian and its ψ tower (supersedes
/// the #684 Fisher-cross-zero pin). Production deliberately builds the
/// OBSERVED Wood–Pya–Säfken LAML curvature (`gaussian_locscale_observed_joint_
/// row_coeffs`: `mm = w`, `ml = 2κm`, `ll = κ′(a−n) + 2κ²n`; see #1561 — the
/// old block-Fisher object zeroed `ml`, dropped the cross-block Schur deficit
/// and biased λ̂_σ upward). The fixtures carry NONZERO residuals, so the
/// observed `2κm` cross is genuinely nonzero here and a Fisher-zero pin is
/// wrong by design. What IS invariant, and what this test pins:
///
///  * the dense joint Hessian equals the independent row-sum assembly
///    `Σ_i x_iᵀ · [[mm, ml], [ml, ll]]_i · x_i` of those observed
///    coefficients (content pin, tight tolerance);
///  * every ψ-layer builder (`exact_newton_joint_psi*`: 1st-order, 2nd-order,
///    mixed β·ψ) is the exact matrix derivative of that SAME observed object,
///    pinned against full-matrix central finite differences in which the
///    ψ-perturbed family is REBUILT at every probe point (fresh row-scalar
///    cache — the FD gate never freezes the cache), with the design model
///    `X(ψ) = X + ψ·X_ψ + ½ψ²·X_ψψ` matching the derivative payloads.
///
/// A regression to the Fisher object (or any dropped observed cross term)
/// fails the content pin; a ψ-builder that differentiates a different object
/// than the dense path fails the FD gates. Both families (plain and wiggle)
/// are covered; the wiggle FD rebuilds the q0-dependent basis at each probe
/// exactly as `refresh_all_block_etas` does at fit time.
#[test]
pub(crate) fn gaussian_location_scale_joint_hessian_is_observed_and_psi_layers_match_matrix_fd_2387()
{
    use gam_problem::HyperOperator;

    // Materialize an `ExactNewtonJointPsiTerms` joint Hessian regardless of
    // whether the family returns it dense or operator-backed.
    fn materialize(
        dense: &Array2<f64>,
        operator: Option<&dyn HyperOperator>,
        total: usize,
    ) -> Array2<f64> {
        match operator {
            Some(op) => op.to_dense(),
            None => {
                assert_eq!(dense.dim(), (total, total));
                dense.clone()
            }
        }
    }

    // Max |entry| over the rectangular block H[r0..r1, c0..c1].
    fn block_max_abs(h: &Array2<f64>, r0: usize, r1: usize, c0: usize, c1: usize) -> f64 {
        let mut m = 0.0_f64;
        for r in r0..r1 {
            for c in c0..c1 {
                m = m.max(h[[r, c]].abs());
            }
        }
        m
    }

    // Full-matrix closeness against a reference, scaled by the reference's
    // magnitude so O(1) and O(10⁻³) entries are judged uniformly.
    fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64, label: &str) {
        assert_eq!(actual.dim(), expected.dim(), "{label}: dimension mismatch");
        let scale = 1.0 + expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for r in 0..expected.nrows() {
            for c in 0..expected.ncols() {
                let a = actual[[r, c]];
                let e = expected[[r, c]];
                assert!(
                    (a - e).abs() <= tol * scale,
                    "{label}: entry ({r},{c}) diverged: got {a:.12e}, expected {e:.12e} \
                     (tol {tol:.1e} × scale {scale:.3e})"
                );
            }
        }
    }

    // ---- Non-wiggle GaussianLocationScaleFamily ----------------------
    {
        let (family, states, specs) = gls_workspace_fixture();
        let p_mu = states[GaussianLocationScaleFamily::BLOCK_MU].beta.len();
        let p_ls = states[GaussianLocationScaleFamily::BLOCK_LOG_SIGMA]
            .beta
            .len();
        let total = p_mu + p_ls;
        let n = family.y.len();
        let xmu = specs[0]
            .design
            .as_dense_ref()
            .expect("dense xmu")
            .to_owned();
        let xls = specs[1]
            .design
            .as_dense_ref()
            .expect("dense xls")
            .to_owned();

        // ψ design-Jacobian payloads on the MEAN (μ) block, with a 2nd-order
        // payload so the 2nd-order builder is exercised too.
        let x_mu_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.2 + 0.11 * ((i as f64) * 0.37 + (j as f64) * 0.53).sin()
        });
        let x_mu_psi_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.07 * ((i as f64) * 0.19 + (j as f64) * 0.23).cos()
        });
        let derivative_blocks = vec![
            vec![CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: x_mu_psi.clone(),
                s_psi: Array2::zeros((p_mu, p_mu)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: Some(vec![x_mu_psi_psi.clone()]),
                s_psi_psi: Some(vec![Array2::zeros((p_mu, p_mu))]),
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: None,
                implicit_axis: 0,
                implicit_group_id: None,
            }],
            Vec::new(),
        ];

        // (A) Content pin: the dense joint Hessian IS the observed object —
        // the independent row-sum assembly of the single-source coefficients.
        let dense_h = family
            .exact_newton_joint_hessian(&states)
            .expect("dense joint Hessian build")
            .expect("dense joint Hessian present");
        let rows = family
            .get_or_compute_row_scalars(&states[0].eta, &states[1].eta)
            .expect("row scalars");
        let (mm, ml, ll) = gaussian_locscale_observed_joint_row_coeffs(&rows);
        let mut expected = Array2::<f64>::zeros((total, total));
        for i in 0..n {
            for a in 0..p_mu {
                for b in 0..p_mu {
                    expected[[a, b]] += xmu[[i, a]] * mm[i] * xmu[[i, b]];
                }
                for b in 0..p_ls {
                    let v = xmu[[i, a]] * ml[i] * xls[[i, b]];
                    expected[[a, p_mu + b]] += v;
                    expected[[p_mu + b, a]] += v;
                }
            }
            for a in 0..p_ls {
                for b in 0..p_ls {
                    expected[[p_mu + a, p_mu + b]] += xls[[i, a]] * ll[i] * xls[[i, b]];
                }
            }
        }
        assert_matrix_close(
            &dense_h,
            &expected,
            1e-10,
            "#2387: dense joint Hessian must equal the observed row-sum assembly",
        );
        // Non-vacuity: the observed 2κm cross is genuinely nonzero on this
        // fixture (a Fisher regression would zero it and fail the content pin
        // loudly, not vacuously).
        let cross_mag = block_max_abs(&dense_h, 0, p_mu, p_mu, total);
        assert!(
            cross_mag > 1e-1,
            "#2387: fixture must exercise a nonzero observed 2κm cross, got {cross_mag:.3e}"
        );

        // Dense observed H with the μ design perturbed along the ψ payloads:
        // X(ψ) = X + ψ·X_ψ + ½ψ²·X_ψψ, family REBUILT per probe (fresh cache).
        let dense_h_at = |t: f64| -> Array2<f64> {
            let xmu_t = &xmu + &(t * &x_mu_psi) + &((0.5 * t * t) * &x_mu_psi_psi);
            let fam_t = GaussianLocationScaleFamily {
                y: family.y.clone(),
                weights: family.weights.clone(),
                mu_design: Some(DesignMatrix::Dense(
                    gam_linalg::matrix::DenseDesignMatrix::from(xmu_t.clone()),
                )),
                log_sigma_design: family.log_sigma_design.clone(),
                policy: gam_runtime::resource::ResourcePolicy::default_library(),
                cached_row_scalars: std::sync::RwLock::new(None),
            };
            let states_t = vec![
                ParameterBlockState {
                    beta: states[0].beta.clone(),
                    eta: xmu_t.dot(&states[0].beta),
                },
                states[1].clone(),
            ];
            fam_t
                .exact_newton_joint_hessian(&states_t)
                .expect("perturbed dense joint Hessian build")
                .expect("perturbed dense joint Hessian present")
        };

        // (B) 1st-order ψ builder == central FD of the dense observed H.
        let layout = test_design_hyper_layout(&derivative_blocks);
        let psi = family
            .exact_newton_joint_psi_terms(&states, &specs, &layout, 0)
            .expect("psi terms call")
            .expect("gaussian psi terms present");
        let h_psi = materialize(&psi.hessian_psi, psi.hessian_psi_operator.as_deref(), total);
        let h1 = 1e-5;
        let fd1 = (&dense_h_at(h1) - &dense_h_at(-h1)) / (2.0 * h1);
        assert_matrix_close(
            &h_psi,
            &fd1,
            1e-6,
            "#2387: 1st-order ψ joint Hessian must be the matrix FD of the dense observed H",
        );

        // (C) 2nd-order ψ builder == second central FD.
        let psi2 = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &layout, 0, 0)
            .expect("psi 2nd-order call")
            .expect("gaussian psi 2nd-order present");
        let h_psi2 = materialize(
            &psi2.hessian_psi_psi,
            psi2.hessian_psi_psi_operator.as_deref(),
            total,
        );
        let h2 = 1e-3;
        let fd2 = (&(&dense_h_at(h2) + &dense_h_at(-h2)) - &(2.0 * &dense_h)) / (h2 * h2);
        assert_matrix_close(
            &h_psi2,
            &fd2,
            5e-4,
            "#2387: 2nd-order ψ joint Hessian must be the second matrix FD of the dense observed H",
        );

        // (D) Mixed β·ψ builder == central FD in β (along d_beta) of the
        // 1st-order ψ-Hessian, states rebuilt (η from designs) per probe.
        let d_beta = Array1::from_shape_fn(total, |i| 0.05 + 0.13 * ((i + 1) as f64).sin());
        let psi_h_at_beta = |t: f64| -> Array2<f64> {
            let mut st = states.clone();
            for j in 0..p_mu {
                st[0].beta[j] += t * d_beta[j];
            }
            for j in 0..p_ls {
                st[1].beta[j] += t * d_beta[p_mu + j];
            }
            st[0].eta = xmu.dot(&st[0].beta);
            st[1].eta = xls.dot(&st[1].beta);
            let psi_t = family
                .exact_newton_joint_psi_terms(&st, &specs, &layout, 0)
                .expect("perturbed psi terms call")
                .expect("perturbed psi terms present");
            materialize(&psi_t.hessian_psi, psi_t.hessian_psi_operator.as_deref(), total)
        };
        let mixed = family
            .exact_newton_joint_psihessian_directional_derivative(&states, &specs, &layout, 0, &d_beta)
            .expect("psi mixed-drift call")
            .expect("gaussian psi mixed-drift present");
        assert_eq!(mixed.dim(), (total, total));
        let hb = 1e-5;
        let fdm = (&psi_h_at_beta(hb) - &psi_h_at_beta(-hb)) / (2.0 * hb);
        assert_matrix_close(
            &mixed,
            &fdm,
            1e-5,
            "#2387: mixed β·ψ ψ-Hessian must be the β-directional FD of the 1st-order ψ-Hessian",
        );
    }

    // ---- Wiggle GaussianLocationScaleWiggleFamily --------------------
    {
        let (family, states, specs, xmu, xls, _xw_seed) = gls_wiggle_workspace_fixture();
        let p_mu = states[GaussianLocationScaleWiggleFamily::BLOCK_MU]
            .beta
            .len();
        let p_ls = states[GaussianLocationScaleWiggleFamily::BLOCK_LOG_SIGMA]
            .beta
            .len();
        let p_w = states[GaussianLocationScaleWiggleFamily::BLOCK_WIGGLE]
            .beta
            .len();
        let total = p_mu + p_ls + p_w;
        let n = family.y.len();
        let ls0 = p_mu;
        let ls1 = p_mu + p_ls;
        let w0 = p_mu + p_ls;
        let w1 = total;

        // ψ design-Jacobian on the MEAN (μ) block (psi_index 0), exercising
        // both mean⊥scale crosses and their derivatives.
        let x_mu_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.18 + 0.09 * ((i as f64) * 0.41 + (j as f64) * 0.29).sin()
        });
        let x_mu_psi_psi = Array2::from_shape_fn((n, p_mu), |(i, j)| {
            0.06 * ((i as f64) * 0.17 + (j as f64) * 0.31).cos()
        });
        let derivative_blocks = vec![
            vec![CustomFamilyBlockPsiDerivative {
                penalty_index: None,
                x_psi: x_mu_psi.clone(),
                s_psi: Array2::zeros((p_mu, p_mu)),
                s_psi_components: None,
                s_psi_penalty_components: None,
                x_psi_psi: Some(vec![x_mu_psi_psi.clone()]),
                s_psi_psi: Some(vec![Array2::zeros((p_mu, p_mu))]),
                s_psi_psi_components: None,
                s_psi_psi_penalty_components: None,
                implicit_operator: None,
                implicit_axis: 0,
                implicit_group_id: None,
            }],
            Vec::new(),
            Vec::new(),
        ];

        // Dense observed H with the μ design perturbed along the ψ payloads.
        // The wiggle block has dynamic geometry (q0-dependent basis), so the
        // rebuild recomputes η_w from the perturbed q0 exactly as
        // `refresh_all_block_etas` does at fit time — a frozen wiggle basis
        // would silently drop the basis-drift channel from the FD reference.
        let dense_h_at = |t: f64| -> Array2<f64> {
            let xmu_t = &xmu + &(t * &x_mu_psi) + &((0.5 * t * t) * &x_mu_psi_psi);
            let fam_t = GaussianLocationScaleWiggleFamily {
                y: family.y.clone(),
                weights: family.weights.clone(),
                mu_design: Some(DesignMatrix::Dense(
                    gam_linalg::matrix::DenseDesignMatrix::from(xmu_t.clone()),
                )),
                log_sigma_design: family.log_sigma_design.clone(),
                wiggle_knots: family.wiggle_knots.clone(),
                wiggle_degree: family.wiggle_degree,
                policy: gam_runtime::resource::ResourcePolicy::default_library(),
                cached_row_scalars: std::sync::RwLock::new(None),
            };
            let eta_mu_t = xmu_t.dot(&states[0].beta);
            let eta_w_t = fam_t
                .wiggle_design(eta_mu_t.view())
                .expect("perturbed wiggle basis at q0")
                .dot(&states[2].beta);
            let states_t = vec![
                ParameterBlockState {
                    beta: states[0].beta.clone(),
                    eta: eta_mu_t,
                },
                states[1].clone(),
                ParameterBlockState {
                    beta: states[2].beta.clone(),
                    eta: eta_w_t,
                },
            ];
            fam_t
                .exact_newton_joint_hessian(&states_t)
                .expect("perturbed wiggle dense joint Hessian build")
                .expect("perturbed wiggle dense joint Hessian present")
        };

        let dense_h = dense_h_at(0.0);
        assert_eq!(dense_h.dim(), (total, total));
        // Non-vacuity: both mean⊥scale observed crosses are present (the old
        // Fisher pin asserted these were zero; observed they are not).
        let c_ml = block_max_abs(&dense_h, 0, p_mu, ls0, ls1);
        let c_wl = block_max_abs(&dense_h, w0, w1, ls0, ls1);
        assert!(
            c_ml > 1e-3,
            "#2387 (wiggle): μ↔logσ observed cross must be nonzero on this fixture, got {c_ml:.3e}"
        );
        assert!(
            c_wl > 1e-4,
            "#2387 (wiggle): wiggle↔logσ observed cross must be nonzero on this fixture, got {c_wl:.3e}"
        );

        // 1st-order ψ builder == central FD.
        let layout = test_design_hyper_layout(&derivative_blocks);
        let psi = family
            .exact_newton_joint_psi_terms(&states, &specs, &layout, 0)
            .expect("wiggle psi terms call")
            .expect("wiggle psi terms present");
        let h_psi = materialize(&psi.hessian_psi, psi.hessian_psi_operator.as_deref(), total);
        let h1 = 1e-5;
        let fd1 = (&dense_h_at(h1) - &dense_h_at(-h1)) / (2.0 * h1);
        assert_matrix_close(
            &h_psi,
            &fd1,
            1e-6,
            "#2387 (wiggle): 1st-order ψ joint Hessian must be the matrix FD of the dense observed H",
        );

        // 2nd-order ψ builder == second central FD.
        let psi2 = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &layout, 0, 0)
            .expect("wiggle psi 2nd-order call")
            .expect("wiggle psi 2nd-order present");
        let h_psi2 = materialize(
            &psi2.hessian_psi_psi,
            psi2.hessian_psi_psi_operator.as_deref(),
            total,
        );
        let h2 = 1e-3;
        let fd2 = (&(&dense_h_at(h2) + &dense_h_at(-h2)) - &(2.0 * &dense_h)) / (h2 * h2);
        assert_matrix_close(
            &h_psi2,
            &fd2,
            5e-4,
            "#2387 (wiggle): 2nd-order ψ joint Hessian must be the second matrix FD of the dense observed H",
        );

        // Mixed β·ψ builder == central FD in β of the 1st-order ψ-Hessian.
        // The β perturbation moves ALL THREE blocks; η_w is rebuilt from the
        // perturbed q0 AND the perturbed wiggle coefficients.
        let d_beta = Array1::from_shape_fn(total, |i| 0.04 + 0.1 * ((i + 1) as f64).cos());
        let psi_h_at_beta = |t: f64| -> Array2<f64> {
            let mut st = states.clone();
            for j in 0..p_mu {
                st[0].beta[j] += t * d_beta[j];
            }
            for j in 0..p_ls {
                st[1].beta[j] += t * d_beta[p_mu + j];
            }
            for j in 0..p_w {
                st[2].beta[j] += t * d_beta[w0 + j];
            }
            st[0].eta = xmu.dot(&st[0].beta);
            st[1].eta = xls.dot(&st[1].beta);
            st[2].eta = family
                .wiggle_design(st[0].eta.view())
                .expect("perturbed-β wiggle basis at q0")
                .dot(&st[2].beta);
            let psi_t = family
                .exact_newton_joint_psi_terms(&st, &specs, &layout, 0)
                .expect("perturbed-β wiggle psi terms call")
                .expect("perturbed-β wiggle psi terms present");
            materialize(&psi_t.hessian_psi, psi_t.hessian_psi_operator.as_deref(), total)
        };
        let mixed = family
            .exact_newton_joint_psihessian_directional_derivative(&states, &specs, &layout, 0, &d_beta)
            .expect("wiggle psi mixed-drift call")
            .expect("wiggle psi mixed-drift present");
        assert_eq!(mixed.dim(), (total, total));
        let hb = 1e-5;
        let fdm = (&psi_h_at_beta(hb) - &psi_h_at_beta(-hb)) / (2.0 * hb);
        assert_matrix_close(
            &mixed,
            &fdm,
            1e-5,
            "#2387 (wiggle): mixed β·ψ ψ-Hessian must be the β-directional FD of the 1st-order ψ-Hessian",
        );
    }
}

/// #932 exact-tower oracle for the canonical binomial location-scale WIGGLE
/// order-two row program.
///
/// `BinomialLocationScaleWiggleFamily::wiggle_order2_rows` lowers the shared
/// row expression into per-row joint-Hessian coefficients for the composed
/// index
/// `q = q0(η_t, η_ls) + Σ_j βw_j·B_j(q0)` via the chain factors `m = B'·βw + 1`,
/// `g2 = B''·βw`. The cross-block coefficients (`coeff_tw_*`, `coeff_lw_*`,
/// `coeff_ww`: threshold/log-sigma × wiggle and wiggle × wiggle) are exactly the
/// #736 dropped/sign-flipped cross-term genus, and until now no exact oracle
/// pinned them to an independent tower — only an operator-vs-dense check and
/// an FD approximation covered them.
///
/// This is the #932 single-source guard. For each row `i` and basis column `j`
/// we build an INDEPENDENT order-2 jet `Tower2<3>` over `(η_t, η_ls, βw_j)`
/// (the other `βw_k` held at their fixed values), compose the wiggle basis onto
/// the non-wiggle index tower
/// (`q = q0_tower + Σ_k coef_k · q0_tower.compose_unary([B_k, B'_k, B''_k])`),
/// then compose the binomial neglog objective onto `q`
/// (`nll = q.compose_unary([·, m1, m2])`). The resulting `3×3` Hessian block IS
/// every `coeff_*` mechanically:
///   `h[0][0]=coeff_tt`, `h[0][1]=coeff_tl`, `h[1][1]=coeff_ll`,
///   `h[0][2]=coeff_tw_b·B_j + coeff_tw_d·B'_j`,
///   `h[1][2]=coeff_lw_b·B_j + coeff_lw_d·B'_j`,
///   `h[2][2]=coeff_ww·B_j²`.
/// A dropped or sign-flipped generated coefficient shifts a block well outside
/// 1e-9
/// and fails loudly, for probit / logit / cloglog. The value channel is
/// irrelevant to the Hessian (`compose_unary`'s `h` reads only `f'`/`f''`), so a
/// placeholder `0.0` is passed for the objective value.
#[test]
pub(crate) fn binomial_location_scale_wiggle_order2_rows_match_jet_tower_932() {
    use super::binomial_q_derivs::binomial_neglog_q_derivatives_dispatch;
    use gam_math::jet_tower::Tower2;

    let (probit_family, states, _specs, _xt, _xls, _wd) = bls_wiggle_workspace_fixture();
    let n = probit_family.y.len();

    for link in [
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::CLogLog),
    ] {
        // Designs, knots and the block etas are link-independent (`q0` and the
        // wiggle basis do not depend on the binomial link), so reuse them and
        // swap only `link_kind` for each arm.
        let family = BinomialLocationScaleWiggleFamily {
            y: probit_family.y.clone(),
            weights: probit_family.weights.clone(),
            link_kind: link.clone(),
            threshold_design: probit_family.threshold_design.clone(),
            log_sigma_design: probit_family.log_sigma_design.clone(),
            wiggle_knots: probit_family.wiggle_knots.clone(),
            wiggle_degree: probit_family.wiggle_degree,
            policy: probit_family.policy.clone(),
        };

        let pieces = family
            .wiggle_order2_rows(&states)
            .expect("canonical wiggle order-two rows");

        let eta_t = &states[BinomialLocationScaleWiggleFamily::BLOCK_T].eta;
        let eta_ls = &states[BinomialLocationScaleWiggleFamily::BLOCK_LOG_SIGMA].eta;
        let etaw = &states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].eta;
        let betaw = &states[BinomialLocationScaleWiggleFamily::BLOCK_WIGGLE].beta;

        let core0 = binomial_location_scale_core(
            &family.y,
            &family.weights,
            eta_t,
            eta_ls,
            Some(etaw),
            &family.link_kind,
        )
        .expect("binomial location-scale core");

        // Same basis tensors the canonical row program consumes:
        // pieces.{b0,d0} are exactly B and B' it used; recompute B'' for the
        // order-2 composition.
        let b0 = &pieces.b0;
        let d0 = &pieces.d0;
        let dd0 = family
            .wiggle_basiswith_options(core0.q0.view(), BasisOptions::second_derivative())
            .expect("wiggle second-derivative basis");
        let pw = b0.ncols();

        for i in 0..n {
            let qi = core0.q0[i] + etaw[i];
            let (m1, m2, _m3) = binomial_neglog_q_derivatives_dispatch(
                family.y[i],
                family.weights[i],
                qi,
                core0.mu[i],
                core0.dmu_dq[i],
                core0.d2mu_dq2[i],
                core0.d3mu_dq3[i],
                &family.link_kind,
            );

            // Non-wiggle index q0 = -η_t · exp(-η_ls) over axes (η_t, η_ls);
            // axis 2 is reserved for the per-column wiggle amplitude.
            let eta_t_t = Tower2::<3>::variable(eta_t[i], 0);
            let eta_ls_t = Tower2::<3>::variable(eta_ls[i], 1);

            let q0_tower = (eta_t_t * -1.0) * (eta_ls_t * -1.0).exp();

            for j in 0..pw {
                let mut q = q0_tower;
                for k in 0..pw {
                    let coef = if k == j {
                        Tower2::<3>::variable(betaw[j], 2)
                    } else {
                        Tower2::<3>::constant(betaw[k])
                    };
                    let basis_k = q0_tower.compose_unary([b0[[i, k]], d0[[i, k]], dd0[[i, k]]]);
                    q = q + coef * basis_k;
                }
                let nll = q.compose_unary([0.0, m1, m2]);
                let h = nll.h;

                let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);

                assert!(
                    close(h[0][0], pieces.coeff_tt[i]),
                    "{link:?} coeff_tt[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[0][0],
                    pieces.coeff_tt[i]
                );
                assert!(
                    close(h[0][1], pieces.coeff_tl[i]),
                    "{link:?} coeff_tl[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[0][1],
                    pieces.coeff_tl[i]
                );
                assert!(
                    close(h[1][1], pieces.coeff_ll[i]),
                    "{link:?} coeff_ll[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[1][1],
                    pieces.coeff_ll[i]
                );

                let tw = pieces.coeff_tw_b[i] * b0[[i, j]] + pieces.coeff_tw_d[i] * d0[[i, j]];
                let lw = pieces.coeff_lw_b[i] * b0[[i, j]] + pieces.coeff_lw_d[i] * d0[[i, j]];
                let ww = pieces.coeff_ww[i] * b0[[i, j]] * b0[[i, j]];
                assert!(
                    close(h[0][2], tw),
                    "{link:?} (η_t,βw) cross[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[0][2],
                    tw
                );
                assert!(
                    close(h[1][2], lw),
                    "{link:?} (η_ls,βw) cross[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[1][2],
                    lw
                );
                assert!(
                    close(h[2][2], ww),
                    "{link:?} (βw,βw)[{i},{j}]: tower={:.9e} canonical={:.9e}",
                    h[2][2],
                    ww
                );
            }
        }
    }
}

// ── #1606: NB location-scale (GAMLSS-style joint mean/dispersion) inner solve ──
//
// Regression for gam#1606: a negative-binomial location-scale fit
// (`family="nb"` + a dispersion smooth) ABORTED at fit time with an
// `IntegrationError` on well-posed heteroscedastic count data, while every
// sibling path (plain NB, Gaussian-LS, Gamma-LS) fit the same design. Root
// cause: the NB dispersion (log-θ) block assembled its IRLS curvature from the
// per-row OBSERVED Hessian channel `−∂²ℓ/∂θ²`, which carries the row-specific
// `ψ′(θ+y)` term and goes NEGATIVE for every row whose count sits below its
// current fitted precision. Replacing each negative row by an arbitrary
// epsilon then divides the exact score by ~0 in the
// working response, producing O(1e10) IRLS targets that explode the dispersion
// step and stall the inner block-cyclic solve, whose non-convergence is then
// escalated to a hard error. The fix switches the dispersion curvature to the
// EXPECTED (Fisher) information `ψ′(θ)−ψ′(θ+μ)−1/θ+1/(θ+μ) > 0` (Fisher
// scoring; the working RESPONSE still carries the exact score, so the penalized
// optimum is unchanged — only the inner conditioning improves), matching the
// mean block, which always used its closed-form expected info.

// Direct root-cause regression (the fail-before / pass-after gate): at a
// heteroscedastic iterate whose fitted precision sits ABOVE the data's true
// overdispersion (the regime the inner solve traverses), the NB dispersion
// working set must stay well-conditioned. With the pre-fix OBSERVED curvature
// the per-row information is negative for these rows, gets epsilon-clamped,
// and the working response `disp_response` blows up
// to O(1e9)+ (the exact score divided by ~0). With the EXPECTED (Fisher)
// curvature the response stays O(1) and the per-row IRLS weight reflects
// genuine positive curvature. This asserts the bounded, well-conditioned
// behaviour — it FAILS on the observed-curvature code (huge |disp_response|)
// and PASSES on the Fisher-curvature fix.
#[test]
fn nb_dispersion_working_set_stays_bounded_above_optimum_1606() {
    use super::dispersion_family::dispersion_row_kernel;

    // Overdispersed rows (true θ small, large counts) evaluated at a high fitted
    // precision η_d = ln(8): there μ²/θ_true ≫ μ, so y ≫ μ for many rows while
    // the model currently believes the precision is large — exactly where
    // `−∂²ℓ/∂θ²` goes negative.
    let mu = 20.0_f64;
    let eta_mu = mu.ln();
    let eta_d = 8.0_f64.ln(); // fitted θ = 8, well above the true overdispersion
    // A spread of counts straddling μ, including the small/zero counts that
    // drive the observed information negative.
    let counts = [0.0_f64, 2.0, 4.0, 6.0, 8.0, 22.0, 27.0, 40.0, 63.0, 95.0];
    let mut saw_overdispersed_row = false;
    for &yi in &counts {
        let row = dispersion_row_kernel(
            DispersionFamilyKind::NegativeBinomial,
            yi,
            eta_mu,
            eta_d,
            1.0,
        );
        // The working response is `η_d + score/(θ·info)`. With the Fisher
        // information it is O(1); with the floored observed information it is
        // O(1e9)+. Pin a generous-but-decisive bound: anything below 1e6 is the
        // well-conditioned Fisher path, anything above is the broken floored
        // observed path (the real failures are ~1e10).
        assert!(
            row.disp_response.is_finite() && row.disp_response.abs() < 1.0e6,
            "NB dispersion working response must stay bounded at an above-optimum \
             iterate (gam#1606): y={yi}, disp_response={:.6e} (an O(1e9)+ value is the \
             pre-fix floored-observed-curvature blow-up)",
            row.disp_response,
        );
        // The per-row IRLS weight must be a genuine positive curvature, not the
        // ~0 floor that the negative observed information collapses to.
        assert!(
            row.disp_weight.is_finite() && row.disp_weight >= 0.0,
            "NB dispersion working weight must be a finite non-negative curvature: \
             y={yi}, disp_weight={:.6e}",
            row.disp_weight,
        );
        if yi < mu {
            saw_overdispersed_row = true;
            // These are precisely the rows whose OBSERVED information is negative
            // (count below fitted precision); the Fisher weight keeps them at a
            // strictly positive, finite curvature.
            assert!(
                row.disp_weight > 0.0,
                "below-fitted-precision rows must still carry positive Fisher \
                 curvature: y={yi}, disp_weight={:.6e}",
                row.disp_weight,
            );
        }
    }
    assert!(
        saw_overdispersed_row,
        "fixture must include rows below the fitted precision (the negative-observed-info regime)"
    );
}

// End-to-end contract check: the documented NB location-scale fit drives the
// two-block custom-family inner solve through the public fixed-log-λ entry
// point (`fit_custom_family_fixed_log_lambdas`, which runs `inner_blockwise_fit`
// and returns `Err` when the inner solve fails to converge — the same
// non-convergence the profile-objective evaluator escalates), and must converge
// and predict finite, strictly positive per-row means. The Python repro
// (`bug_hunt_nb_location_scale_inner_solve_abort_test`) cannot run under the
// build.rs author-guard deadlock, so this Rust-level fit stands in for it.
#[test]
fn nb_location_scale_inner_solve_converges_on_heteroscedastic_counts() {
    use super::dispersion_family::{DispersionFamilyKind, DispersionGlmLocationScaleFamily};
    use crate::custom_family::fit_custom_family_fixed_log_lambdas;

    // Deterministic LCG so the synthetic data (and thread-independent inner
    // path) is byte-reproducible — the issue noted order/thread-state-dependent
    // flips, so the fixture must not depend on any global RNG state.
    struct Lcg(u64);
    impl Lcg {
        fn next_u01(&mut self) -> f64 {
            // Numerical Recipes LCG constants.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // top 53 bits → [0,1)
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn next_gamma_shape_ge1(&mut self, shape: f64) -> f64 {
            // Marsaglia–Tsang for shape ≥ 1 (we only call with shape ≥ 1).
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                // crude standard normal via sum of 12 uniforms − 6
                let mut z = -6.0;
                for _ in 0..12 {
                    z += self.next_u01();
                }
                let v = (1.0 + c * z).powi(3);
                if v <= 0.0 {
                    continue;
                }
                let u = self.next_u01();
                if u.ln() < 0.5 * z * z + d - d * v + d * (v).ln() {
                    return d * v;
                }
            }
        }
        fn next_poisson(&mut self, lambda: f64) -> f64 {
            // Knuth, fine for the moderate λ here.
            let l = (-lambda).exp();
            let mut k = 0.0;
            let mut p = 1.0;
            loop {
                k += 1.0;
                p *= self.next_u01();
                if p <= l {
                    return k - 1.0;
                }
            }
        }
        fn next_nb(&mut self, mu: f64, theta: f64) -> f64 {
            // Gamma–Poisson mixture: λ ~ Gamma(theta, mu/theta), Y ~ Pois(λ).
            let lam = if theta >= 1.0 {
                self.next_gamma_shape_ge1(theta) * (mu / theta)
            } else {
                // boost shape by 1 then scale down (Stuart's method)
                let g = self.next_gamma_shape_ge1(theta + 1.0);
                let u = self.next_u01().max(1e-300);
                g * u.powf(1.0 / theta) * (mu / theta)
            };
            self.next_poisson(lam.max(1e-9))
        }
    }

    let n = 600usize;
    let p = 6usize;
    // The mean and dispersion smooths ride on TWO DISTINCT covariates (x for the
    // mean, z for the dispersion). Each channel's design is a sum-to-zero,
    // column-orthonormal polynomial basis in its OWN covariate built from the
    // monomials t¹..tᵖ (NO constant column): the per-channel level is carried by
    // a constant `offset`, exactly as a centered production `s(x)` smooth plus a
    // gauge-fixed intercept. Dropping the constant column is what keeps the flat
    // pre-fit identifiability audit happy — a single-channel custom family sees
    // both channels' designs as ordinary columns, and two identical all-ones
    // intercept columns across blocks would alias (overlap 1.0) and fail the
    // audit. With no constant column and two different covariates, the
    // concatenated [mean | log_precision] joint design is full-rank and
    // alias-free, while the constant offsets still let each η reach its level.
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();
    let zs: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.6180339887) % 1.0) // golden-ratio low-discrepancy spread
        .collect();
    // Modified Gram–Schmidt over the monomials t¹, t², … (skip t⁰), each column
    // first centered to mean-zero so it is orthogonal to the constant direction
    // too. Every resulting column is a distinct, mutually-orthonormal,
    // sum-to-zero direction; none is the constant, and across two different
    // covariates none coincides cross-block.
    let build_design = |t: &[f64]| -> Array2<f64> {
        let mut cols: Vec<Array1<f64>> = Vec::with_capacity(p);
        for j in 0..p {
            // monomial t^(j+1), centered to mean zero.
            let mut v = Array1::from_shape_fn(n, |i| t[i].powi((j + 1) as i32));
            let mean = v.sum() / (n as f64);
            v.mapv_inplace(|e| e - mean);
            for c in &cols {
                let proj = v.dot(c);
                v.scaled_add(-proj, c);
            }
            let nrm = v.dot(&v).sqrt().max(1e-12);
            v.mapv_inplace(|e| e / nrm);
            cols.push(v);
        }
        let mut d = Array2::<f64>::zeros((n, p));
        for (j, c) in cols.iter().enumerate() {
            d.column_mut(j).assign(c);
        }
        d
    };
    let mean_x = build_design(&xs);
    let disp_x = build_design(&zs);
    // Per-channel constant level carried by the offset (centered smooth + level).
    let mean_offset = Array1::from_elem(n, 1.4_f64);
    let disp_offset = Array1::from_elem(n, 0.5_f64);

    // True surfaces: mean μ(x) = exp(η_μ) sweeps a moderate count range, and the
    // dispersion log θ(z) sweeps from high overdispersion (small θ) to near-
    // Poisson (large θ) — the heteroscedastic regime that drives the dispersion
    // block's η_d across the negative-observed-info zone.
    let eta_mu_true: Vec<f64> = xs.iter().map(|&x| 1.4 + 1.1 * (2.2 * x).sin()).collect();
    let log_theta_true: Vec<f64> = zs.iter().map(|&z| -1.2 + 3.4 * z).collect();

    let mut rng = Lcg(0x1606_2024_dead_beef);
    let y = Array1::from_shape_fn(n, |i| {
        let mu = eta_mu_true[i].exp();
        let theta = log_theta_true[i].exp();
        rng.next_nb(mu, theta)
    });
    // Sanity: the response must be a non-degenerate count vector.
    assert!(
        y.iter().any(|&v| v > 0.0) && y.iter().all(|&v| v >= 0.0 && v.fract() == 0.0),
        "synthetic NB response must be non-negative integer counts with positive mass"
    );

    let weights = Array1::from_elem(n, 1.0);
    let family = DispersionGlmLocationScaleFamily {
        kind: DispersionFamilyKind::NegativeBinomial,
        y: y.clone(),
        weights,
    };

    // Each block: a wiggliness penalty that shrinks the higher-order
    // (orthonormal) polynomial columns, leaving the two lowest-order columns
    // (the 2-dim penalty nullspace) free. This gives the smooth genuine
    // shrinkage at a moderate fixed smoothing parameter, mirroring the `s(x)`
    // production path.
    let make_penalty = || {
        let mut pmat = Array2::<f64>::zeros((p, p));
        for j in 2..p {
            // Increasing penalty weight on higher-order columns.
            pmat[[j, j]] = (j as f64 - 1.0).powi(2);
        }
        PenaltyMatrix::Dense(pmat)
    };

    let mk_spec = |name: &str, design: Array2<f64>, offset: Array1<f64>| ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(gam_linalg::matrix::DenseDesignMatrix::from(design)),
        offset,
        penalties: vec![make_penalty()],
        // Penalty nullspace = the two lowest-order columns = 2 unpenalized dirs.
        nullspace_dims: vec![2],
        initial_log_lambdas: Array1::from_elem(1, (0.5_f64).ln()),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let specs = vec![
        mk_spec("mean", mean_x, mean_offset),
        mk_spec("log_precision", disp_x, disp_offset),
    ];

    let options = BlockwiseFitOptions::default();

    // The fixed-log-λ fit runs `inner_blockwise_fit` and returns `Err` exactly
    // when the inner solve fails to converge — the non-convergence the profile
    // objective escalates to the fatal abort. Before the fix this returns
    // `Err(Optimization{ "...inner solve did not converge..." })`.
    let result = fit_custom_family_fixed_log_lambdas(&family, &specs, &options, None);
    let fit = result.unwrap_or_else(|e| {
        panic!(
            "NB location-scale inner solve must converge on heteroscedastic count data \
             (gam#1606); instead the inner blockwise solve aborted: {e:?}"
        )
    });

    // Predicted per-row means must be finite and strictly positive (the contract
    // the issue requires: the NB LS fit predicts finite positive per-row means).
    // The mean-channel predictor η_μ is block 0's converged `eta`.
    let eta_mu = &fit.block_states[DispersionGlmLocationScaleFamily::BLOCK_MEAN].eta;
    assert_eq!(eta_mu.len(), n, "mean predictor must cover every row");
    assert!(
        eta_mu.iter().all(|&e| e.is_finite()),
        "fitted mean predictor must be finite on every row"
    );
    assert!(
        eta_mu.iter().all(|&e| e.exp().is_finite() && e.exp() > 0.0),
        "fitted per-row means must be finite and strictly positive"
    );
}
