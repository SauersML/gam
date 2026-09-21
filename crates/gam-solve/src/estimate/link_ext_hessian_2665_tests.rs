//! #2665 — the link–link (ψψ) block of the flexible-link outer Hessian.
//!
//! `evaluate_unified_with_link_ext` installs two pair callbacks. The ρ–link one
//! is legitimately zero (penalties do not depend on link-shape parameters); the
//! link–link one used to carry the *same* zeros under the *same* comment, which
//! asserts `−∂²ℓ/∂θ_i∂θ_j = 0` — that the log-likelihood has no second
//! derivative in the link parameters. It has one, it dominates the block, and
//! its absence reads as a sign flip: the SAS fixtures certify stationarity and
//! are then refused on `hessian_psd=NO` with `λ_min = −1721.5` where the true
//! directional curvature is `+121.6`.
//!
//! The block had **no gate at all** — the only tests reaching this evaluator
//! ask for `ValueAndGradient` and carry `hessian: HessianValue::Unavailable`, so
//! a structurally-zero block cost nothing, produced no NaN and failed no
//! finiteness check. This module is that gate: it central-differences the
//! analytic θ-gradient of the same evaluator at the same point and compares the
//! result block by block.
//!
//! The ρρ and ρψ blocks are the **control arms**. They were already accurate
//! (1.8e-3 / 7.1e-3 relative on the reproducer) with the ψψ block wrong by more
//! than its own magnitude, which is what proves the instrument reads the
//! criterion rather than its own noise. A gate that only watched ψψ could pass
//! by the whole Hessian collapsing.
#![cfg(test)]

use super::*;
use crate::mixture_link::state_from_sasspec;
use gam_problem::{InverseLink, LikelihoodSpec, ResponseFamily, SasLinkSpec};
use ndarray::{Array1, Array2};

/// A ρ away from both rails, so the criterion is smooth in every coordinate the
/// finite difference walks and no barrier term dominates the comparison.
const PROBE_RHO: f64 = 1.0;
/// Link parameters well inside both SAS domain bounds, so the
/// smooth-bound maps are in their identity region and the block under test is
/// the genuine SAS curvature rather than a clamp's.
const PROBE_EPSILON: f64 = 0.3;
const PROBE_LOG_DELTA: f64 = 0.2;

fn tiny_design(n: usize) -> Array2<f64> {
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64;
        let x1 = -1.5 + 3.0 * t;
        x[[i, 0]] = 1.0;
        x[[i, 1]] = x1;
        x[[i, 2]] = (2.1 * x1).sin();
    }
    x
}

fn binomial_response(n: usize) -> Array1<f64> {
    Array1::from_iter((0..n).map(|i| if (i * 7 + 3) % 5 < 2 { 1.0 } else { 0.0 }))
}

fn sas_config_at(epsilon: f64, log_delta: f64) -> RemlConfig {
    let sas_state = state_from_sasspec(SasLinkSpec {
        initial_epsilon: epsilon,
        initial_log_delta: log_delta,
    })
    .expect("SAS state at the probe point");
    RemlConfig::external(
        gam_spec::GlmLikelihoodSpec::canonical(LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Sas(sas_state),
        )),
        1e-10,
        false,
    )
}

fn sas_state<'a>(
    y: &'a Array1<f64>,
    w: &'a Array1<f64>,
    offset: &'a Array1<f64>,
    x: &Array2<f64>,
    cfg: &'a RemlConfig,
) -> crate::estimate::reml::RemlState<'a> {
    let p = x.ncols();
    let mut s = Array2::<f64>::zeros((p, p));
    for j in 1..p {
        s[[j, j]] = 1.0;
    }
    let (canonical_penalties, active_nullspace_dims) =
        gam_terms::construction::canonicalize_penalty_specs(
            &[PenaltySpec::Dense(s)],
            &[1],
            p,
            "#2665 link-ext Hessian fixture",
        )
        .expect("canonicalize the one-penalty fixture");
    let mut state = crate::estimate::reml::RemlState::newwith_offset(
        y.view(),
        x.clone(),
        w.view(),
        offset.view(),
        canonical_penalties,
        p,
        cfg,
        Some(active_nullspace_dims),
        None,
        None,
    )
    .expect("build the SAS-link REML state");
    state.set_link_states(None, cfg.link_kind.sas_state().copied());
    state
}

/// The θ-gradient of the link-ext criterion at `(rho, epsilon, log_delta)`.
///
/// A fresh `RemlState` per evaluation: the link parameters live on the state
/// (and on the config the state reads its link from), not in the `rho` argument
/// the evaluator caches on, so mutating them in place would risk answering from
/// a bundle cached at the unperturbed point — a finite difference of zero.
fn theta_gradient(rho: f64, epsilon: f64, log_delta: f64) -> Array1<f64> {
    criterion_and_gradient(rho, epsilon, log_delta).1
}

/// The analytic θ-Hessian of the link-ext criterion at the probe point.
fn analytic_hessian() -> Array2<f64> {
    analytic_hessian_at([PROBE_RHO, PROBE_EPSILON, PROBE_LOG_DELTA])
}

/// The analytic θ-Hessian of the link-ext criterion at `[rho, epsilon, log_delta]`.
fn analytic_hessian_at(base: [f64; 3]) -> Array2<f64> {
    let n = 60usize;
    let x = tiny_design(n);
    let y = binomial_response(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let cfg = sas_config_at(base[1], base[2]);
    let state = sas_state(&y, &w, &offset, &x, &cfg);
    let rho_vec = Array1::from_elem(1, base[0]);
    let evaluation = state
        .evaluate_unified_with_link_ext(
            &rho_vec,
            crate::estimate::reml::reml_outer_engine::EvalMode::ValueGradientHessian,
        )
        .expect("the SAS link-ext criterion must evaluate with a Hessian");
    match evaluation.hessian {
        gam_problem::HessianValue::Dense(hessian) => hessian,
        gam_problem::HessianValue::Operator(_) => panic!(
            "this gate needs a DENSE analytic outer Hessian on the link-ext path; got an \
             operator. A gate that cannot obtain the matrix it audits passes by not \
             running, which is the failure mode that produced #2665."
        ),
        gam_problem::HessianValue::Unavailable => panic!(
            "the link-ext evaluator returned no Hessian in ValueGradientHessian mode. A gate \
             that cannot obtain the matrix it audits passes by not running, which is the \
             failure mode that produced #2665."
        ),
    }
}

/// Central finite difference of the θ-gradient, one column per θ coordinate.
///
/// `theta[0]` is ρ, `theta[1]` is ε, `theta[2]` is raw log δ — the same order
/// the evaluator appends its link coordinates in.
fn fd_hessian(step: f64) -> Array2<f64> {
    fd_hessian_at([PROBE_RHO, PROBE_EPSILON, PROBE_LOG_DELTA], step)
}

/// Central finite difference of the θ-gradient at `base`.
fn fd_hessian_at(base: [f64; 3], step: f64) -> Array2<f64> {
    let dim = base.len();
    let mut out = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        let mut plus = base;
        let mut minus = base;
        plus[i] += step;
        minus[i] -= step;
        let g_plus = theta_gradient(plus[0], plus[1], plus[2]);
        let g_minus = theta_gradient(minus[0], minus[1], minus[2]);
        assert_eq!(
            g_plus.len(),
            dim,
            "the link-ext gradient must be theta-length (rho plus the two SAS \
             coordinates); a shorter one means the fixture fell back to standard REML \
             and this gate is measuring a different objective"
        );
        for j in 0..dim {
            out[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * step);
        }
    }
    // Symmetrize: the analytic block is symmetric by construction and the FD
    // one is only symmetric up to its own truncation error, so comparing the
    // symmetric parts keeps the residual from being read as an asymmetry.
    0.5 * (&out + &out.t())
}

fn block_frobenius(matrix: &Array2<f64>, rows: (usize, usize), cols: (usize, usize)) -> f64 {
    let mut sum = 0.0;
    for i in rows.0..rows.1 {
        for j in cols.0..cols.1 {
            sum += matrix[[i, j]] * matrix[[i, j]];
        }
    }
    sum.sqrt()
}

fn block_relative_error(
    analytic: &Array2<f64>,
    reference: &Array2<f64>,
    rows: (usize, usize),
    cols: (usize, usize),
) -> f64 {
    let diff = reference - analytic;
    let scale = block_frobenius(reference, rows, cols);
    assert!(
        scale > 0.0,
        "the reference block at rows {rows:?} cols {cols:?} is identically zero, so it \
         cannot separate a correct block from an absent one"
    );
    block_frobenius(&diff, rows, cols) / scale
}

/// The link–link block of the analytic outer Hessian must reproduce a central
/// difference of the analytic θ-gradient.
///
/// Pre-#2665 this block was a structural zero plus first-order drifts, and the
/// measured relative error on the reproducer was **1.09** — larger than the
/// block's own magnitude. The bar here is `1e-2`, two orders inside that, and
/// the ρρ / ρψ control arms carry the same bar so a collapse of the whole
/// Hessian cannot satisfy it.
#[test]
fn link_link_outer_hessian_block_matches_finite_difference() {
    const BAR: f64 = 1.0e-2;
    let analytic = analytic_hessian();
    assert_eq!(
        analytic.dim(),
        (3, 3),
        "the probe is one rho plus the two SAS link coordinates"
    );

    // Two steps, an order apart. A finite-difference truncation error falls
    // with the step; a missing analytic term does not. Reporting both makes
    // "close but not exact" separable instead of arguable.
    for step in [1.0e-3_f64, 1.0e-4_f64] {
        let reference = fd_hessian(step);

        let psi_psi = block_relative_error(&analytic, &reference, (1, 3), (1, 3));
        let rho_rho = block_relative_error(&analytic, &reference, (0, 1), (0, 1));
        let rho_psi = block_relative_error(&analytic, &reference, (0, 1), (1, 3));

        assert!(
            psi_psi < BAR,
            "link-link (psi-psi) block relative error {psi_psi:.3e} at step {step:.1e} \
             exceeds {BAR:.1e}. Controls at the same step: rho_rho={rho_rho:.3e}, \
             rho_psi={rho_psi:.3e}. Analytic block:\n{:?}\nFinite difference:\n{:?}",
            analytic.slice(ndarray::s![1..3, 1..3]),
            reference.slice(ndarray::s![1..3, 1..3]),
        );
        assert!(
            rho_rho < BAR,
            "control arm rho_rho relative error {rho_rho:.3e} at step {step:.1e} exceeds \
             {BAR:.1e}; the instrument is not reading the criterion, so the psi_psi \
             verdict beside it means nothing"
        );
        assert!(
            rho_psi < BAR,
            "control arm rho_psi relative error {rho_psi:.3e} at step {step:.1e} exceeds \
             {BAR:.1e}; the instrument is not reading the criterion, so the psi_psi \
             verdict beside it means nothing"
        );
    }
}

/// The link–link block must carry the likelihood curvature, not just the
/// first-order drifts.
///
/// This is the falsifiability arm of the gate above: a comparison against a
/// finite difference passes trivially if both sides are near zero, so assert
/// separately that the block is **large** — the missing term measured
/// `‖·‖ ≈ 2063` against a total analytic block norm of `1894`, i.e. it is the
/// dominant contribution and not a correction.
#[test]
fn link_link_outer_hessian_block_is_not_a_first_order_residue() {
    let analytic = analytic_hessian();
    let psi_psi = block_frobenius(&analytic, (1, 3), (1, 3));
    let rho_rho = block_frobenius(&analytic, (0, 1), (0, 1));
    assert!(
        psi_psi.is_finite() && psi_psi > 0.0,
        "the link-link block is {psi_psi:e}; a zero or non-finite block is the #2665 \
         stub, not a measurement"
    );
    // The SAS likelihood curvature in (epsilon, log delta) is an O(n) sum; on
    // this 60-row fixture it cannot be a rounding-scale residue of the rho
    // block. Stated as a ratio so the fixture shrinking cannot satisfy it.
    assert!(
        psi_psi > 1.0e-3 * rho_rho,
        "the link-link block norm {psi_psi:e} is negligible against the rho-rho block \
         {rho_rho:e}; that is the signature of the block being assembled from \
         first-order drifts alone"
    );
}

/// The link-ext criterion value and its analytic θ-gradient at one point.
fn criterion_and_gradient(rho: f64, epsilon: f64, log_delta: f64) -> (f64, Array1<f64>) {
    let n = 60usize;
    let x = tiny_design(n);
    let y = binomial_response(n);
    let w = Array1::<f64>::ones(n);
    let offset = Array1::<f64>::zeros(n);
    let cfg = sas_config_at(epsilon, log_delta);
    let state = sas_state(&y, &w, &offset, &x, &cfg);
    let rho_vec = Array1::from_elem(1, rho);
    let evaluation = state
        .evaluate_unified_with_link_ext(
            &rho_vec,
            crate::estimate::reml::reml_outer_engine::EvalMode::ValueAndGradient,
        )
        .expect("the SAS link-ext criterion must evaluate at the probe point");
    let gradient = evaluation
        .gradient
        .expect("ValueAndGradient must return a gradient");
    (evaluation.cost, gradient)
}

/// The analytic θ-gradient is the derivative of the criterion it is shipped
/// with, at the probe point and at the heavy-tailed, small-δ region the
/// pyGAM-audit SAS reproducer (families.md F4) walks through. A gradient that
/// disagreed with its own cost there would stall the outer trust region on a
/// criterion it cannot descend.
#[test]
fn link_ext_outer_gradient_is_the_derivative_of_the_criterion() {
    // Central differences in the raw outer coordinates; the step balances the
    // O(h²) truncation against the O(ε·|V|/h) rounding of a cost of O(10).
    let step = 1e-5;
    for base in [
        [PROBE_RHO, PROBE_EPSILON, PROBE_LOG_DELTA],
        [4.8, 0.12, -2.1],
        [-2.6, 0.12, -2.1],
        [1.0, 0.0, 0.0],
        [1.0, -0.5, 1.0],
    ] {
        let (_, analytic) = criterion_and_gradient(base[0], base[1], base[2]);
        for coordinate in 0..3 {
            let mut plus = base;
            let mut minus = base;
            plus[coordinate] += step;
            minus[coordinate] -= step;
            let fd = (criterion_and_gradient(plus[0], plus[1], plus[2]).0
                - criterion_and_gradient(minus[0], minus[1], minus[2]).0)
                / (2.0 * step);
            let g = analytic[coordinate];
            assert!(
                (g - fd).abs() <= 1e-6 * (1.0 + g.abs()),
                "θ-gradient coordinate {coordinate} at {base:?}: analytic {g:.10e} vs \
                 central difference {fd:.10e}"
            );
        }
    }
}

/// The whole analytic θ-Hessian matches a central difference of the analytic
/// θ-gradient in the heavy-tailed, small-δ region the families.md F4 SAS
/// reproducer converges into, not only at the mild probe point. The outer
/// trust region reads its model curvature from this matrix, so a wrong block
/// there would make it propose steps the criterion rejects.
#[test]
fn link_ext_outer_hessian_matches_finite_difference_at_small_delta() {
    const BAR: f64 = 1.0e-2;
    for base in [[4.8, 0.12, -2.1], [-2.6, 0.12, -2.1]] {
        let analytic = analytic_hessian_at(base);
        let reference = fd_hessian_at(base, 1.0e-4);
        let scale = block_frobenius(&reference, (0, 3), (0, 3));
        let error = block_frobenius(&(&reference - &analytic), (0, 3), (0, 3)) / scale;
        assert!(
            error < BAR,
            "θ-Hessian relative error {error:.3e} at {base:?} exceeds {BAR:.1e}. \
             Analytic:\n{analytic:?}\nFinite difference:\n{reference:?}"
        );
    }
}
