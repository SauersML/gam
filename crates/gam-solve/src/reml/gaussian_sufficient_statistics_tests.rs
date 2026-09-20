//! Fixed-design Gaussian identity outer REML runs on sufficient statistics.
//!
//! For a Gaussian identity fit without Firth, box or linear constraints, the
//! REML value, its ρ-gradient and its ρ-Hessian are functions of `XᵀWX`,
//! `XᵀW(y−offset)`, `(y−offset)ᵀW(y−offset)`, `n` and the penalties. An outer
//! evaluation therefore needs no pass over the observation rows once the Gram
//! is built. These tests pin both halves of that contract:
//!
//! * every outer criterion request (value, value+gradient, value+gradient+
//!   Hessian) is served by a bundle marked [`BundleRows::SufficientStatistics`],
//!   and its value, gradient and Hessian agree with the same evaluation on a
//!   bundle whose rows were realised from the design;
//! * a consumer that reads rows (`obtain_eval_bundle`) never receives the
//!   sufficient-statistic carrier: it gets fitted rows at the same mode.

#![cfg(test)]

use super::tests::gaussian_identity_glm_spec;
use super::*;
use crate::rho_optimizer::OuterEvalOrder;
use gam_problem::HessianValue;
use gam_terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2};

const BASIS_PER_BLOCK: usize = 6;

/// Intercept plus two Gaussian-bump blocks on two covariates, with
/// non-uniform prior weights and a non-zero offset so every sufficient
/// statistic carries its weighted, offset-shifted form.
struct Fixture {
    y: Array1<f64>,
    w: Array1<f64>,
    offset: Array1<f64>,
    x: Array2<f64>,
    penalties: Vec<CanonicalPenalty>,
    cfg: RemlConfig,
}

impl Fixture {
    fn new(n: usize) -> Self {
        let p = 1 + 2 * BASIS_PER_BLOCK;
        let mut x = Array2::<f64>::zeros((n, p));
        let mut y = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        let mut offset = Array1::<f64>::zeros(n);
        let width = 1.0 / (BASIS_PER_BLOCK as f64 - 1.0);
        for i in 0..n {
            let t1 = (i as f64 + 0.5) / n as f64;
            // A second covariate that is a deterministic permutation of the
            // first, so the two blocks are not collinear.
            let t2 = ((i * 37) % n) as f64 / n as f64;
            x[[i, 0]] = 1.0;
            for c in 0..BASIS_PER_BLOCK {
                let centre = c as f64 * width;
                x[[i, 1 + c]] = (-((t1 - centre) / width).powi(2)).exp();
                x[[i, 1 + BASIS_PER_BLOCK + c]] = (-((t2 - centre) / width).powi(2)).exp();
            }
            let wiggle = [0.07, -0.03, -0.05, 0.04, 0.01][i % 5];
            offset[i] = 0.2 * (3.0 * t1).cos();
            y[i] = (std::f64::consts::TAU * t1).sin() + 0.5 * t2 * t2 + offset[i] + wiggle;
            w[i] = 0.5 + 0.4 * (i % 3) as f64;
        }
        let penalties = (0..2)
            .map(|block| {
                let first = 1 + block * BASIS_PER_BLOCK;
                let mut root = Array2::<f64>::zeros((BASIS_PER_BLOCK - 2, p));
                for r in 0..BASIS_PER_BLOCK - 2 {
                    root[[r, first + r]] = 1.0;
                    root[[r, first + r + 1]] = -2.0;
                    root[[r, first + r + 2]] = 1.0;
                }
                CanonicalPenalty::from_dense_root(root, p)
            })
            .collect();
        Self {
            y,
            w,
            offset,
            x,
            penalties,
            cfg: RemlConfig::external(gaussian_identity_glm_spec(), 1e-10, false),
        }
    }

    fn state(&self) -> RemlState<'_> {
        RemlState::newwith_offset(
            self.y.view(),
            self.x.clone(),
            self.w.view(),
            self.offset.view(),
            self.penalties.clone(),
            self.x.ncols(),
            &self.cfg,
            Some(vec![2, 2]),
            None,
            None,
        )
        .expect("Gaussian identity REML state")
    }
}

fn rho_points() -> Vec<Array1<f64>> {
    vec![
        Array1::from(vec![-1.0, 2.0]),
        Array1::from(vec![1.5, -0.5]),
        Array1::from(vec![4.0, 3.0]),
    ]
}

fn cached_rows(state: &RemlState<'_>, rho: &Array1<f64>) -> BundleRows {
    state
        .cache_manager
        .cached_eval_bundle(&state.rhokey_sanitized(rho))
        .expect("an outer evaluation leaves its bundle cached")
        .rows
}

fn dense_hessian(eval: &OuterEval) -> Array2<f64> {
    match &eval.hessian {
        HessianValue::Dense(h) => h.clone(),
        _ => panic!("a two-coordinate REML surface returns a dense analytic Hessian"),
    }
}

fn assert_close(what: &str, got: f64, want: f64) {
    let tol = 1e-8 * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tol,
        "{what}: sufficient-statistic {got:.15e} vs observed-row {want:.15e} (tol {tol:.1e})"
    );
}

/// One state evaluates from a fresh cache, so every request builds its own
/// bundle; the other realises fitted rows at ρ first, so the same requests
/// run on an `Observed` bundle. The criterion must not see the difference.
#[test]
fn outer_criterion_on_sufficient_statistics_matches_observed_rows() {
    let fixture = Fixture::new(400);
    for rho in rho_points() {
        let observed = fixture.state();
        let observed_bundle = observed.obtain_eval_bundle(&rho).expect("observed bundle");
        assert_eq!(observed_bundle.rows, BundleRows::Observed);
        let want = observed
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueGradientHessian)
            .expect("observed-row outer eval");
        assert_eq!(cached_rows(&observed, &rho), BundleRows::Observed);

        let suff = fixture.state();
        let got = suff
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueGradientHessian)
            .expect("sufficient-statistic outer eval");
        assert_eq!(
            cached_rows(&suff, &rho),
            BundleRows::SufficientStatistics,
            "a derivative-bearing outer eval at rho={rho} must not realise observation rows"
        );

        assert_close("REML value", got.cost, want.cost);
        assert_eq!(got.gradient.len(), rho.len());
        for k in 0..rho.len() {
            assert_close(&format!("dV/drho_{k}"), got.gradient[k], want.gradient[k]);
        }
        let (h_got, h_want) = (dense_hessian(&got), dense_hessian(&want));
        for k in 0..rho.len() {
            for l in 0..rho.len() {
                assert_close(&format!("d2V/drho_{k}drho_{l}"), h_got[[k, l]], h_want[[k, l]]);
            }
        }

        // The value-only and gradient-only entry points take the same route.
        let value_state = fixture.state();
        let value = value_state.compute_cost(&rho).expect("value-only cost");
        assert_eq!(cached_rows(&value_state, &rho), BundleRows::SufficientStatistics);
        assert_close("compute_cost", value, want.cost);

        let gradient_state = fixture.state();
        let gradient = gradient_state
            .compute_gradient(&rho)
            .expect("gradient-only eval");
        assert_eq!(
            cached_rows(&gradient_state, &rho),
            BundleRows::SufficientStatistics
        );
        for k in 0..rho.len() {
            assert_close(&format!("compute_gradient_{k}"), gradient[k], want.gradient[k]);
        }
    }
}

/// A row consumer after a sufficient-statistic outer evaluation gets fitted
/// rows at the same mode, never the ρ-invariant carrier.
#[test]
fn row_consumer_after_sufficient_statistic_eval_receives_fitted_rows() {
    let fixture = Fixture::new(400);
    let rho = Array1::from(vec![0.5, 1.0]);
    let state = fixture.state();
    state
        .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueAndGradient)
        .expect("outer eval");
    let suff = state
        .cache_manager
        .cached_eval_bundle(&state.rhokey_sanitized(&rho))
        .expect("cached outer bundle");
    assert_eq!(suff.rows, BundleRows::SufficientStatistics);

    let observed = state.obtain_eval_bundle(&rho).expect("row-bearing bundle");
    assert_eq!(observed.rows, BundleRows::Observed);

    let beta_suff = &suff.pirls_result.beta_transformed.0;
    let beta_obs = &observed.pirls_result.beta_transformed.0;
    let beta_scale = beta_obs.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    for (a, b) in beta_suff.iter().zip(beta_obs.iter()) {
        assert!(
            (a - b).abs() <= 1e-9 * beta_scale,
            "both bundles hold the same mode: {a:.15e} vs {b:.15e}"
        );
    }
    assert_close(
        "deviance",
        suff.pirls_result.deviance,
        observed.pirls_result.deviance,
    );

    // The fitted rows are the linear predictor of that mode.
    let qs = &observed.pirls_result.reparam_result.qs;
    let beta_original = qs.dot(beta_obs);
    let eta = fixture.x.dot(&beta_original) + &fixture.offset;
    let eta_scale = eta.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
    assert_eq!(observed.pirls_result.final_eta.len(), eta.len());
    for (got, want) in observed.pirls_result.final_eta.iter().zip(eta.iter()) {
        assert!(
            (got - want).abs() <= 1e-9 * eta_scale,
            "observed bundle rows are the fitted linear predictor: {got:.15e} vs {want:.15e}"
        );
    }
}
