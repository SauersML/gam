#![cfg(test)]
//! Does the ψ/ext block's `½ ∂_ψ log|H|` term differentiate the criterion's OWN
//! `logdet_h` scalar? (gam#2765 / gam#2767, the #979/#1040 lane.)
//!
//! The end-to-end audit on survival marginal-slope (the since-deleted
//! `examples/probe_2765_outer_gradient_fd.rs`) recorded the ψ gradient's
//! `fixed_beta` atom agreeing with its Ridders oracle to six digits while the
//! `logdet_h` atom disagrees by 130 % and 700 % of itself, with the oracle's own
//! uncertainty nine orders below the gap. Every existing gate sits on one side
//! of that seam or the other:
//!
//! * the family gates (`survival::marginal_slope::psi_terms_fd_tests`) certify
//!   `∂_ψ ℓ̄`, `∂_ψ∇_βℓ̄`, `∂_ψ∇²_βℓ̄` and `D_βH[v]` against finite differences of
//!   the family's own objective — they say nothing about what the outer engine
//!   does with them;
//! * the outer-engine gates in `tests.rs` check that the dense and operator
//!   ASSEMBLIES of the same formula agree with each other — a formula that is
//!   consistently wrong passes every one of them.
//!
//! Nothing differences the criterion. This module does, on a synthetic model
//! whose exact answer is known in closed form, at unit-test cost:
//!
//! ```text
//!   M(β, ψ) = M₀ + ψ·A + Σ_j β_j C_j        (unpenalized joint curvature)
//!   H(β, ψ, ρ) = M(β, ψ) + λ(ρ)·S
//!   β̂(ψ)      = b₀ + ψ·d + ½ψ²·e            (a prescribed smooth mode path)
//! ```
//!
//! The engine is handed the ext coordinate's `g_ψ = −H·β̂′(ψ)`, which makes the
//! implicit-function relation `β̂′ = −H⁻¹g_ψ` hold EXACTLY at every ψ by
//! construction rather than approximately at a converged fit. So
//!
//! ```text
//!   d/dψ ½log|H(β̂(ψ), ψ)| = ½ tr(H⁻¹ (A + Σ_j β̂′_j(ψ) C_j))
//!                             \_____/   \_______________/
//!                              frozen      mode response
//! ```
//! is available three independent ways — the engine's analytic ψ gradient, a
//! closed-form trace written out here, and a central difference of the
//! criterion's own published `logdet_h` component along the true mode path. All
//! three must agree, and each half is additionally isolated by zeroing the
//! other (`A = 0` ⇒ pure mode response; `C_j = 0` ⇒ pure frozen drift), because
//! a compensating pair of errors is exactly what a total-only comparison
//! cannot see.

use super::*;
use crate::model_types::ActiveLinearConstraintBlock;
use ndarray::{Array1, Array2, array};
use std::sync::{Arc, Mutex};

/// `D_β M[u] = Σ_j u_j C_j` for a curvature that is exactly linear in β.
///
/// The provider contract is stated on `HessianDerivativeProvider`: given the
/// mode response `v` (so that `∂β̂/∂θ = −v`), it returns the moving-curvature
/// part of `∂H/∂θ`, i.e. `D_βH[−v] = −D_βM[v]`. The sign lives here, once.
#[derive(Clone)]
struct LinearBetaCurvature {
    axes: Vec<Array2<f64>>,
}

impl LinearBetaCurvature {
    fn directional(&self, u: &Array1<f64>) -> Array2<f64> {
        let p = self.axes.len();
        let mut out = Array2::<f64>::zeros((p, p));
        for (j, axis) in self.axes.iter().enumerate() {
            out.scaled_add(u[j], axis);
        }
        out
    }
}

impl HessianDerivativeProvider for LinearBetaCurvature {
    fn hessian_derivative_correction(
        &self,
        v_k: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(-self.directional(v_k)))
    }

    fn hessian_second_derivative_correction(
        &self,
        v_k: &Array1<f64>,
        v_l: &Array1<f64>,
        u_kl: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // `D²_βM ≡ 0` for a curvature linear in β, so the whole second-order
        // correction is the second mode response `D_βH[−u_kl]` and the two
        // first-order responses enter only as a shape contract — checked here
        // rather than ignored, because a length mismatch would otherwise be
        // read as a legitimately different model.
        for (name, vector) in [("v_k", v_k), ("v_l", v_l), ("u_kl", u_kl)] {
            if vector.len() != self.axes.len() {
                return Err(format!(
                    "synthetic curvature: {name} has length {} against {} coefficient axes",
                    vector.len(),
                    self.axes.len()
                ));
            }
        }
        Ok(Some(-self.directional(u_kl)))
    }

    fn has_corrections(&self) -> bool {
        true
    }
}

/// The synthetic model of the module doc, evaluated at one ψ.
struct PsiModel {
    m0: Array2<f64>,
    a_psi: Array2<f64>,
    axes: Vec<Array2<f64>>,
    s_root: Array2<f64>,
    b0: Array1<f64>,
    d1: Array1<f64>,
    d2: Array1<f64>,
    rho: f64,
    /// Whether the mode sits on an active inequality face. The criterion's
    /// VALUE does not change with the face; only the mode response is
    /// tangent-restricted (`4c3c7f960`).
    face: bool,
}

impl PsiModel {
    /// The `A ≠ 0`, `C_j ≠ 0` model both halves of the chain are live in.
    fn full() -> Self {
        Self {
            m0: array![
                [2.0, 0.3, -0.1],
                [0.3, 1.7, 0.2],
                [-0.1, 0.2, 2.4],
            ],
            a_psi: array![
                [0.5, -0.2, 0.1],
                [-0.2, 0.35, 0.05],
                [0.1, 0.05, -0.4],
            ],
            axes: vec![
                array![[0.21, 0.07, -0.03], [0.07, -0.11, 0.05], [-0.03, 0.05, 0.17]],
                array![[-0.09, 0.13, 0.02], [0.13, 0.24, -0.06], [0.02, -0.06, -0.15]],
                array![[0.06, -0.04, 0.11], [-0.04, 0.08, 0.09], [0.11, 0.09, 0.19]],
            ],
            s_root: array![[1.2, 0.1, -0.2], [0.0, 0.8, 0.15], [0.0, 0.0, 0.6]],
            b0: array![0.4, -0.7, 0.25],
            d1: array![0.31, 0.52, -0.44],
            d2: array![-0.18, 0.27, 0.36],
            rho: 0.2,
            face: false,
        }
    }

    /// Only the mode response is live: the ψ drift `B = ∂_ψM|_β` vanishes.
    fn mode_response_only() -> Self {
        Self {
            a_psi: Array2::zeros((3, 3)),
            ..Self::full()
        }
    }

    /// Only the frozen drift is live: the curvature does not move with β.
    fn frozen_only() -> Self {
        Self {
            axes: vec![Array2::zeros((3, 3)); 3],
            ..Self::full()
        }
    }

    fn lambda(&self) -> f64 {
        self.rho.exp()
    }

    fn penalty(&self) -> Array2<f64> {
        self.s_root.t().dot(&self.s_root)
    }

    fn beta_at(&self, psi: f64) -> Array1<f64> {
        &self.b0 + &(psi * &self.d1) + &(0.5 * psi * psi * &self.d2)
    }

    fn beta_dot_at(&self, psi: f64) -> Array1<f64> {
        &self.d1 + &(psi * &self.d2)
    }

    /// `H = M₀ + ψA + Σ_j β_j C_j + λS`, symmetric by construction.
    fn hessian_at(&self, psi: f64) -> Array2<f64> {
        let beta = self.beta_at(psi);
        let mut h = self.m0.clone();
        h.scaled_add(psi, &self.a_psi);
        for (j, axis) in self.axes.iter().enumerate() {
            h.scaled_add(beta[j], axis);
        }
        h.scaled_add(self.lambda(), &self.penalty());
        h
    }

    /// `Ḣ = A + Σ_j β̂′_j C_j` — the total ψ derivative of the curvature ALONG
    /// the prescribed mode path.
    fn hessian_dot_at(&self, psi: f64) -> Array2<f64> {
        let beta_dot = self.beta_dot_at(psi);
        let mut drift = self.a_psi.clone();
        for (j, axis) in self.axes.iter().enumerate() {
            drift.scaled_add(beta_dot[j], axis);
        }
        drift
    }

    /// The closed-form answer: `½ tr(H⁻¹ Ḣ)`.
    fn closed_form_logdet_h_gradient(&self, psi: f64) -> f64 {
        let h = self.hessian_at(psi);
        let hop = DenseSpectralOperator::from_symmetric(&h).expect("synthetic H factorizes");
        let drift = self.hessian_dot_at(psi);
        0.5 * hop.trace_logdet_h_k(&drift, None)
    }

    /// The same model, pinned onto an active inequality face, with the
    /// pseudo-logdet route's kernel installed and carrying a ψ-VARYING
    /// correction — the exact configuration `try_tangent_projected_evaluate`
    /// has to survive (#2765).
    ///
    /// `A = e₀ᵀ` and the mode path is built with no `β₀` motion, so `β̂(ψ)`
    /// genuinely lives on the face and `dβ̂/dψ` lies in `range(Z)`; the tangent
    /// operator's own solve then reproduces it exactly, as the real constrained
    /// inner solve does.
    fn on_an_active_face() -> Self {
        let mut model = Self::full();
        model.d1[0] = 0.0;
        model.d2[0] = 0.0;
        model.face = true;
        model
    }

    fn active_constraints(&self) -> Option<Arc<ActiveLinearConstraintBlock>> {
        self.face.then(|| {
            Arc::new(ActiveLinearConstraintBlock {
                a: array![[1.0, 0.0, 0.0]],
            })
        })
    }

    /// The same model with the mode held on the SAME path but no face declared:
    /// the arm the face arm's criterion value is compared against.
    fn on_the_face_path_without_the_face() -> Self {
        let mut model = Self::on_an_active_face();
        model.face = false;
        model
    }

    fn solution_at(&self, psi: f64) -> InnerSolution<'static> {
        let h = self.hessian_at(psi);
        let hop = Arc::new(DenseSpectralOperator::from_symmetric(&h).expect("synthetic H"));
        // `g_ψ = −H·β̂′(ψ)` is what makes `β̂′ = −H⁻¹g_ψ` exact rather than
        // approximate, so a mismatch downstream cannot be charged to an
        // unconverged inner solve.
        let g_psi = -h.dot(&self.beta_dot_at(psi));
        InnerSolution {
            log_likelihood: -2.3,
            penalty_quadratic: 0.6,
            hessian_op: hop,
            mode_response_op: None,
            beta: self.beta_at(psi),
            penalty_coords: vec![PenaltyCoordinate::from_dense_root(self.s_root.clone())],
            penalty_logdet: PenaltyLogdetDerivs {
                value: 0.0,
                first: array![0.4],
                second: Some(array![[0.13]]),
            },
            deriv_provider: Box::new(LinearBetaCurvature {
                axes: self.axes.clone(),
            }),
            firth: None,
            hessian_logdet_correction: 0.0,
            penalty_subspace_trace: None,
            rho_curvature_scale: 1.0,
            rho_prior: gam_problem::RhoPrior::Flat,
            n_observations: 64,
            nullspace_dim: 0.0,
            gaussian_weight_log_sum_half: 0.0,
            dp_floor_scale: 1.0,
            dispersion: DispersionHandling::Fixed {
                phi: 1.0,
                include_logdet_h: true,
                include_logdet_s: true,
            },
            // `a = 0` and `ld_s = 0` leave `∂V/∂ψ = ½tr(H⁻¹Ḣ)` alone in the
            // published gradient entry, so the assertion is on the log-det
            // chain and nothing else.
            ext_coords: vec![HyperCoord {
                a: 0.0,
                g: g_psi,
                drift: HyperCoordDrift::from_dense(self.a_psi.clone()),
                ld_s: 0.0,
                b_depends_on_beta: false,
                is_penalty_like: false,
                firth_g: None,
                tk_eta_fixed: None,
                tk_x_fixed: None,
            }],
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            barrier_config: None,
            kkt_residual: None,
            active_constraints: self.active_constraints(),
            stochastic_trace_state: Arc::new(Mutex::new(StochasticTraceState::default())),
        }
    }

    /// The criterion's OWN `logdet_h` scalar at ψ, on the prescribed mode path.
    fn criterion_logdet_h(&self, psi: f64) -> f64 {
        let solution = self.solution_at(psi);
        let result = reml_laml_evaluate(&solution, &[self.rho], EvalMode::ValueOnly, None)
            .expect("synthetic criterion evaluates");
        result.criterion_components.logdet_h
    }

    /// The engine's analytic `∂V/∂ψ`, which under this construction IS its
    /// `½ ∂_ψ log|H|` term.
    fn analytic_psi_gradient(&self, psi: f64) -> f64 {
        let solution = self.solution_at(psi);
        let result = reml_laml_evaluate(&solution, &[self.rho], EvalMode::ValueAndGradient, None)
            .expect("synthetic criterion evaluates");
        let gradient = result.gradient.expect("ValueAndGradient publishes a gradient");
        assert_eq!(gradient.len(), 2, "one ρ coordinate and one ψ coordinate");
        gradient[1]
    }

    /// A Richardson-extrapolated central difference of the criterion's own
    /// `logdet_h` along the mode path. `log|H(β̂(ψ),ψ)|` is analytic in ψ, so a
    /// single Richardson step already lands far below the tolerances asserted.
    fn finite_difference_logdet_h(&self, psi: f64, step: f64) -> f64 {
        let central = |h: f64| {
            (self.criterion_logdet_h(psi + h) - self.criterion_logdet_h(psi - h)) / (2.0 * h)
        };
        let coarse = central(step);
        let fine = central(0.5 * step);
        (4.0 * fine - coarse) / 3.0
    }
}

/// Every leg of the chain, on the model where both halves are live.
#[test]
fn psi_logdet_gradient_differences_the_criterions_own_logdet_2765() {
    let model = PsiModel::full();
    let psi = 0.3_f64;

    let analytic = model.analytic_psi_gradient(psi);
    let closed_form = model.closed_form_logdet_h_gradient(psi);
    let finite_difference = model.finite_difference_logdet_h(psi, 1.0e-3);

    assert!(
        (analytic - closed_form).abs() <= 1.0e-9 * closed_form.abs().max(1.0),
        "the engine's ψ gradient is not ½tr(H⁻¹Ḣ): analytic={analytic:.12e} \
         closed_form={closed_form:.12e}"
    );
    assert!(
        (analytic - finite_difference).abs() <= 1.0e-7 * finite_difference.abs().max(1.0),
        "the engine's ψ gradient does not differentiate its own logdet_h: \
         analytic={analytic:.12e} fd={finite_difference:.12e} \
         gap={:.3e}",
        (analytic - finite_difference).abs()
    );
}

/// The mode-response half alone (`B = ∂_ψM|_β = 0`). A sign error or a dropped
/// `D_βH[·]` shows up here as the whole answer, not as a fraction of it.
#[test]
fn psi_logdet_mode_response_half_differences_the_criterion_2765() {
    let model = PsiModel::mode_response_only();
    let psi = 0.3_f64;

    let analytic = model.analytic_psi_gradient(psi);
    let finite_difference = model.finite_difference_logdet_h(psi, 1.0e-3);
    assert!(
        analytic.abs() > 1.0e-3,
        "the mode-response-only arm must not be a trivial zero (got {analytic:.3e})"
    );
    assert!(
        (analytic - finite_difference).abs() <= 1.0e-7 * finite_difference.abs().max(1.0),
        "mode-response half: analytic={analytic:.12e} fd={finite_difference:.12e}"
    );
}

/// The frozen half alone (`C_j = 0`, so `H` does not move with β).
#[test]
fn psi_logdet_frozen_half_differences_the_criterion_2765() {
    let model = PsiModel::frozen_only();
    let psi = 0.3_f64;

    let analytic = model.analytic_psi_gradient(psi);
    let finite_difference = model.finite_difference_logdet_h(psi, 1.0e-3);
    assert!(
        analytic.abs() > 1.0e-3,
        "the frozen-only arm must not be a trivial zero (got {analytic:.3e})"
    );
    assert!(
        (analytic - finite_difference).abs() <= 1.0e-7 * finite_difference.abs().max(1.0),
        "frozen half: analytic={analytic:.12e} fd={finite_difference:.12e}"
    );
}

/// The active-face lane: the mode sits on an active inequality face, the
/// criterion's VALUE is the same object it is off the face, and only the mode
/// response is tangent-restricted (#2765, `4c3c7f960`).
///
/// The earlier form of this lane switched the value to the tangent determinant
/// `½log|ZᵀHZ|` and dropped the pseudo-logdet kernel on the way in, which made
/// the criterion's value a function of active-set bookkeeping: a face change
/// during the outer walk jumped the value, and the mode-response gradient on
/// the face differentiated a different object than the value the line search
/// read. The contract now is the one this pins: (1) declaring the face leaves
/// the criterion's `logdet_h` bit-identical to the same solution without the
/// face, and (2) the analytic ψ gradient on the face differences that same
/// value along the constrained mode path.
#[test]
fn psi_logdet_gradient_survives_the_tangent_projection_2765() {
    let model = PsiModel::on_an_active_face();
    let same_path = PsiModel::on_the_face_path_without_the_face();
    let psi = 0.3_f64;

    // (1) The face is bookkeeping about the mode's motion, not a change of
    // criterion: the value is the full-space log-determinant either way.
    let published = model.criterion_logdet_h(psi);
    let without_face = same_path.criterion_logdet_h(psi);
    assert!(
        published.to_bits() == without_face.to_bits(),
        "declaring the active face changed the criterion's logdet_h: on the face \
         {published:.12e}, same solution without the face {without_face:.12e}"
    );
    let h = model.hessian_at(psi);
    let full = 0.5
        * gam_linalg::faer_ndarray::FaerEigh::eigh(&h, faer::Side::Lower)
            .expect("synthetic H is symmetric")
            .0
            .iter()
            .map(|value| value.ln())
            .sum::<f64>();
    assert!(
        (published - full).abs() <= 1.0e-9 * full.abs().max(1.0),
        "the criterion's logdet_h on the face is not ½log|H| of the full space: \
         published={published:.12e} expected={full:.12e}"
    );

    // (2) The face restricts the mode response, and the gradient the engine
    // publishes there must still be the derivative of the value it publishes.
    let analytic = model.analytic_psi_gradient(psi);
    let finite_difference = model.finite_difference_logdet_h(psi, 1.0e-3);
    assert!(
        analytic.abs() > 1.0e-3,
        "the active-face arm must not be a trivial zero (got {analytic:.3e})"
    );
    assert!(
        (analytic - finite_difference).abs() <= 1.0e-7 * finite_difference.abs().max(1.0),
        "on an active face the ψ gradient does not differentiate the criterion's own logdet_h: \
         analytic={analytic:.12e} fd={finite_difference:.12e} gap={:.3e}",
        (analytic - finite_difference).abs()
    );
}
