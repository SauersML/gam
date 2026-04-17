use self::cache::AtomicFlagGuard;
use self::inner_strategy::GeometryBackendKind;
use super::*;
use crate::linalg::sparse_exact::{
    SparseExactFactor, SparsePenaltyBlock, assemble_and_factor_sparse_penalized_system,
};
use crate::solver::outer_strategy::OuterEval;
use crate::types::SasLinkState;
use ndarray::s;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

pub(crate) mod assembly;
mod cache;
mod eval;
mod firth;
pub(crate) mod hyper;
mod inner_strategy;
pub(crate) mod penalty_logdet;
mod runtime;
mod trace;
pub(crate) mod unified;

const EXACT_OUTER_HESSIAN_LARGE_N_THRESHOLD: usize = 50_000;
const EXACT_OUTER_HESSIAN_LARGE_N_MIN_DIM: usize = 32;
const EXACT_OUTER_HESSIAN_MAX_LINEAR_WORK: usize = 4_000_000;
const EXACT_OUTER_HESSIAN_MAX_QUADRATIC_WORK: usize = 50_000_000;
const EXACT_TAU_TAU_HESSIAN_DENSE_CACHE_BUDGET_BYTES: usize = 512 * 1024 * 1024;
const FIRTH_MAX_OBSERVATIONS: usize = 20_000;
const FIRTH_MAX_COEFFICIENTS: usize = 256;
const FIRTH_MAX_LINEAR_WORK: usize = 2_000_000;
const FIRTH_MAX_QUADRATIC_WORK: usize = 100_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TauTauPlanEstimate {
    pub(crate) dense_x_bytes: usize,
    pub(crate) first_order_tau_bytes: usize,
    pub(crate) second_order_tau_bytes: usize,
    pub(crate) penalty_first_bytes: usize,
    pub(crate) penalty_pair_bytes: usize,
    pub(crate) rho_tau_penalty_bytes: usize,
    pub(crate) vector_cache_bytes: usize,
    pub(crate) weighted_scratch_bytes: usize,
}

impl TauTauPlanEstimate {
    pub(crate) fn total_bytes(self) -> usize {
        self.dense_x_bytes
            .saturating_add(self.first_order_tau_bytes)
            .saturating_add(self.second_order_tau_bytes)
            .saturating_add(self.penalty_first_bytes)
            .saturating_add(self.penalty_pair_bytes)
            .saturating_add(self.rho_tau_penalty_bytes)
            .saturating_add(self.vector_cache_bytes)
            .saturating_add(self.weighted_scratch_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TauTauHessianPolicy {
    pub(crate) any_has_implicit: bool,
    pub(crate) implicit_multidim_duchon: bool,
    pub(crate) estimated_dense_tau_cache_bytes: usize,
    pub(crate) gradient_plan: TauTauPlanEstimate,
    pub(crate) hessian_plan: TauTauPlanEstimate,
    pub(crate) budget_bytes: usize,
    pub(crate) firth_pair_terms_unavailable: bool,
}

impl TauTauHessianPolicy {
    pub(crate) fn prefer_gradient_only(self) -> bool {
        self.implicit_multidim_duchon
            || self.estimated_dense_tau_cache_bytes > self.budget_bytes
            || self.hessian_plan.total_bytes() > self.budget_bytes
            || self.firth_pair_terms_unavailable
    }
}

pub(crate) fn exact_outer_hessian_problem_scale_allows(n_obs: usize, p_coeff: usize) -> bool {
    let linear_work = n_obs.saturating_mul(p_coeff);
    let quadratic_work = linear_work.saturating_mul(p_coeff);
    !((n_obs >= EXACT_OUTER_HESSIAN_LARGE_N_THRESHOLD
        && p_coeff >= EXACT_OUTER_HESSIAN_LARGE_N_MIN_DIM)
        || linear_work > EXACT_OUTER_HESSIAN_MAX_LINEAR_WORK
        || quadratic_work > EXACT_OUTER_HESSIAN_MAX_QUADRATIC_WORK)
}

pub(crate) fn exact_tau_tau_hessian_policy(
    n_obs: usize,
    p_coeff: usize,
    hyper_dirs: &[DirectionalHyperParam],
) -> TauTauHessianPolicy {
    exact_tau_tau_hessian_policy_with_firth(n_obs, p_coeff, hyper_dirs, false)
}

pub(crate) fn exact_tau_tau_hessian_policy_with_firth(
    n_obs: usize,
    p_coeff: usize,
    hyper_dirs: &[DirectionalHyperParam],
    firth_pair_terms_unavailable: bool,
) -> TauTauHessianPolicy {
    let f64_bytes = std::mem::size_of::<f64>();
    let dense_matrix_bytes =
        |rows: usize, cols: usize| -> usize { rows.saturating_mul(cols).saturating_mul(f64_bytes) };
    let dense_design_bytes = dense_matrix_bytes(n_obs, p_coeff);
    let dense_penalty_bytes = dense_matrix_bytes(p_coeff, p_coeff);
    let psi_dim = hyper_dirs.len();
    let implicit_n_axes = hyper_dirs
        .iter()
        .find_map(DirectionalHyperParam::implicit_first_axis_info)
        .map(|(op, _)| op.n_axes())
        .unwrap_or(0);
    let gradient_uses_implicit_design = hyper_dirs
        .iter()
        .any(DirectionalHyperParam::has_implicit_operator)
        && crate::terms::basis::should_use_implicit_operators(n_obs, p_coeff, implicit_n_axes);
    let dense_first_order_count = hyper_dirs
        .iter()
        .filter(|dir| !dir.has_implicit_operator())
        .count();
    let first_penalty_component_count = hyper_dirs
        .iter()
        .map(DirectionalHyperParam::penalty_first_component_count)
        .sum::<usize>();

    let mut dense_second_order_count = 0usize;
    let mut penalty_pair_count = 0usize;
    for i in 0..psi_dim {
        for j in i..psi_dim {
            if hyper_dirs[i]
                .x_tau_tau_entry_at(j)
                .or_else(|| hyper_dirs[j].x_tau_tau_entry_at(i))
                .is_some_and(|entry| !entry.uses_implicit_storage())
            {
                dense_second_order_count += if i == j { 1 } else { 2 };
            }
            if hyper_dirs[i].has_penaltysecond_pair_at(j)
                || hyper_dirs[j].has_penaltysecond_pair_at(i)
            {
                penalty_pair_count += if i == j { 1 } else { 2 };
            }
        }
    }

    let gradient_dense_first_order_count = if gradient_uses_implicit_design {
        dense_first_order_count
    } else {
        psi_dim
    };
    let gradient_needs_dense_x =
        firth_pair_terms_unavailable || gradient_dense_first_order_count > 0;
    let gradient_plan = TauTauPlanEstimate {
        dense_x_bytes: if gradient_needs_dense_x {
            dense_design_bytes
        } else {
            0
        },
        first_order_tau_bytes: if gradient_dense_first_order_count > 0 {
            dense_design_bytes
        } else {
            0
        },
        second_order_tau_bytes: 0,
        penalty_first_bytes: psi_dim.saturating_mul(dense_penalty_bytes),
        penalty_pair_bytes: 0,
        rho_tau_penalty_bytes: 0,
        vector_cache_bytes: n_obs.saturating_mul(f64_bytes),
        weighted_scratch_bytes: dense_penalty_bytes,
    };
    let hessian_plan = TauTauPlanEstimate {
        dense_x_bytes: if psi_dim > 0 { dense_design_bytes } else { 0 },
        first_order_tau_bytes: dense_first_order_count.saturating_mul(dense_design_bytes),
        second_order_tau_bytes: dense_second_order_count.saturating_mul(dense_design_bytes),
        penalty_first_bytes: psi_dim.saturating_mul(dense_penalty_bytes),
        penalty_pair_bytes: penalty_pair_count.saturating_mul(dense_penalty_bytes),
        rho_tau_penalty_bytes: first_penalty_component_count
            .saturating_mul(2)
            .saturating_mul(dense_penalty_bytes),
        vector_cache_bytes: psi_dim.saturating_mul(n_obs).saturating_mul(f64_bytes),
        weighted_scratch_bytes: dense_penalty_bytes,
    };
    let any_has_implicit = hyper_dirs
        .iter()
        .any(DirectionalHyperParam::has_implicit_operator);
    let implicit_multidim_duchon = hyper_dirs
        .iter()
        .any(DirectionalHyperParam::has_implicit_multidim_duchon);
    let estimated_dense_tau_cache_bytes = hessian_plan
        .first_order_tau_bytes
        .saturating_add(hessian_plan.second_order_tau_bytes);
    TauTauHessianPolicy {
        any_has_implicit,
        implicit_multidim_duchon,
        estimated_dense_tau_cache_bytes,
        gradient_plan,
        hessian_plan,
        budget_bytes: EXACT_TAU_TAU_HESSIAN_DENSE_CACHE_BUDGET_BYTES,
        firth_pair_terms_unavailable: firth_pair_terms_unavailable && !hyper_dirs.is_empty(),
    }
}

pub(crate) fn firth_problem_scale_allows(n_obs: usize, p_coeff: usize) -> bool {
    let linear_work = n_obs.saturating_mul(p_coeff);
    let quadratic_work = linear_work.saturating_mul(p_coeff);
    n_obs <= FIRTH_MAX_OBSERVATIONS
        && p_coeff <= FIRTH_MAX_COEFFICIENTS
        && linear_work <= FIRTH_MAX_LINEAR_WORK
        && quadratic_work <= FIRTH_MAX_QUADRATIC_WORK
}

#[cfg(test)]
mod tests {
    use super::{
        DirectionalHyperParam, EvalCacheManager, EvalShared, FirthDenseOperator,
        GlmLikelihoodFamily, HyperDesignDerivative, ImplicitDerivLevel, RemlConfig, RemlState,
    };
    use crate::estimate::EstimationError;
    use crate::faer_ndarray::{FaerCholesky, FaerEigh};
    use crate::pirls::PirlsCoordinateFrame;
    use crate::solver::outer_strategy::{HessianResult, OuterEval, OuterEvalOrder};
    use crate::terms::basis::{ImplicitDesignPsiDerivative, RadialScalarKind};
    use crate::types::GlmLikelihoodSpec;
    use faer::Side;
    use ndarray::{Array1, Array2, array, s};
    use std::sync::Arc;

    #[test]
    fn firth_problem_scale_gate_blocks_large_quadratic_work() {
        assert!(super::firth_problem_scale_allows(2_000, 200));
        assert!(!super::firth_problem_scale_allows(4_800, 241));
        assert!(!super::firth_problem_scale_allows(4_800, 433));
    }

    #[test]
    fn tau_tau_hessian_policy_prefers_gradient_only_for_implicit_tau() {
        let operator = ImplicitDesignPsiDerivative::new(
            array![1.0, 2.0, 3.0, 4.0],
            array![0.5, -1.0, 1.5, 2.0],
            array![0.1, 0.2, 0.3, 0.4],
            array![[1.0, 0.2], [0.5, 0.1], [1.5, 0.3], [2.0, 0.4]],
            None,
            None,
            2,
            2,
            1,
            2,
        );
        let dir = DirectionalHyperParam::new_compact(
            HyperDesignDerivative::from_implicit(
                Arc::new(operator),
                ImplicitDerivLevel::First(0),
                1..4,
                5,
            ),
            Vec::new(),
            None,
            None,
        )
        .expect("implicit directional hyperparam");
        let policy = super::exact_tau_tau_hessian_policy(10, 5, &[dir]);
        assert!(policy.any_has_implicit);
        assert_eq!(
            policy.gradient_plan.dense_x_bytes,
            10 * 5 * std::mem::size_of::<f64>()
        );
        assert!(!policy.prefer_gradient_only());
    }

    #[test]
    fn tau_tau_hessian_policy_prefers_gradient_only_for_implicit_multidim_duchon() {
        let operator = ImplicitDesignPsiDerivative::new_streaming(
            array![[0.0, 0.0], [1.0, 0.2]],
            array![[0.0, 0.0], [1.0, 1.0]],
            vec![0.0, 0.0],
            RadialScalarKind::PureDuchon {
                block_order: 1,
                p_order: 0,
                s_order: 0,
                dim: 2,
            },
            None,
            None,
            0,
        );
        let dir = DirectionalHyperParam::new_compact(
            HyperDesignDerivative::from_implicit(
                Arc::new(operator),
                ImplicitDerivLevel::First(0),
                0..2,
                2,
            ),
            Vec::new(),
            None,
            None,
        )
        .expect("implicit duchon directional hyperparam");
        let policy = super::exact_tau_tau_hessian_policy(10, 5, &[dir]);
        assert!(policy.any_has_implicit);
        assert!(policy.implicit_multidim_duchon);
        assert!(policy.prefer_gradient_only());
    }

    #[test]
    fn tau_tau_hessian_policy_prefers_gradient_only_when_cache_budget_is_exceeded() {
        let dirs = (0..16)
            .map(|_| {
                DirectionalHyperParam::new_compact(
                    HyperDesignDerivative::from(Array2::<f64>::zeros((2, 2))),
                    Vec::new(),
                    None,
                    None,
                )
                .expect("dense directional hyperparam")
            })
            .collect::<Vec<_>>();
        let policy = super::exact_tau_tau_hessian_policy(320_000, 71, &dirs);
        assert!(!policy.any_has_implicit);
        assert!(policy.hessian_plan.total_bytes() > policy.budget_bytes);
        assert!(policy.hessian_plan.total_bytes() > policy.gradient_plan.total_bytes());
        assert!(policy.prefer_gradient_only());
    }

    #[test]
    fn tau_tau_hessian_policy_prefers_gradient_only_for_firth_pair_gap() {
        let dir = DirectionalHyperParam::new_compact(
            HyperDesignDerivative::from(Array2::<f64>::zeros((2, 2))),
            Vec::new(),
            None,
            None,
        )
        .expect("dense directional hyperparam");
        let policy = super::exact_tau_tau_hessian_policy_with_firth(10, 5, &[dir], true);
        assert!(policy.firth_pair_terms_unavailable);
        assert!(policy.prefer_gradient_only());
    }

    fn build_logit_state<'a>(
        y: &'a Array1<f64>,
        w: &'a Array1<f64>,
        x: &Array2<f64>,
        s: &Array2<f64>,
        cfg: &'a RemlConfig,
    ) -> RemlState<'a> {
        use crate::estimate::PenaltySpec;
        let p = x.ncols();
        let offset = Array1::<f64>::zeros(y.len());
        let spec = PenaltySpec::Dense(s.clone());
        let canonical = crate::construction::canonicalize_penalty_specs(&[spec], &[1], p, "test")
            .map(|(canonical, _)| canonical)
            .expect("canonicalize");
        RemlState::newwith_offset(
            y.view(),
            x.clone(),
            w.view(),
            offset.view(),
            canonical,
            p,
            cfg,
            Some(vec![1]),
            None,
            None,
        )
        .expect("state")
    }

    fn beta_original_from_bundle(bundle: &EvalShared) -> Array1<f64> {
        let pr = bundle.pirls_result.as_ref();
        match pr.coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => pr.beta_transformed.as_ref().clone(),
            PirlsCoordinateFrame::TransformedQs => {
                pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
            }
        }
    }

    fn compute_joint_hypercostgradienthessian(
        state: &RemlState<'_>,
        theta: &Array1<f64>,
        rho_dim: usize,
        hyper_dirs: &[DirectionalHyperParam],
    ) -> Result<(f64, Array1<f64>, Array2<f64>), EstimationError> {
        let (cost, gradient, hessian) = state.compute_joint_hyper_eval_with_order(
            theta,
            rho_dim,
            hyper_dirs,
            crate::solver::outer_strategy::OuterEvalOrder::ValueGradientHessian,
        )?;
        Ok((
            cost,
            gradient,
            hessian
                .materialize_dense()
                .map_err(EstimationError::RemlOptimizationFailed)?
                .ok_or_else(|| {
                    EstimationError::RemlOptimizationFailed(
                        "joint hyper Hessian requested but unavailable".to_string(),
                    )
                })?,
        ))
    }

    fn h_original_from_bundle(bundle: &EvalShared) -> Array2<f64> {
        let pr = bundle.pirls_result.as_ref();
        match pr.coordinate_frame {
            PirlsCoordinateFrame::OriginalSparseNative => bundle.h_eff.as_ref().clone(),
            PirlsCoordinateFrame::TransformedQs => {
                let qs = &pr.reparam_result.qs;
                let tmp = qs.dot(bundle.h_eff.as_ref());
                tmp.dot(&qs.t())
            }
        }
    }

    fn single_directional_tau_gradient(
        state: &RemlState<'_>,
        rho: &Array1<f64>,
        hyper: DirectionalHyperParam,
    ) -> Result<f64, EstimationError> {
        let mut theta = Array1::<f64>::zeros(rho.len() + 1);
        theta.slice_mut(s![..rho.len()]).assign(rho);
        let (_, gradient, _) =
            compute_joint_hypercostgradienthessian(state, &theta, rho.len(), &[hyper])?;
        Ok(gradient[rho.len()])
    }

    #[test]
    fn eval_cache_manager_stores_first_order_outer_eval() {
        let cache = EvalCacheManager::new();
        let rho = array![0.25, -0.0];
        let rho_key = EvalCacheManager::sanitized_rhokey(&rho);
        let eval = OuterEval {
            cost: 3.5,
            gradient: array![1.0, -2.0],
            hessian: HessianResult::Unavailable,
        };

        cache.store_outer_eval(&rho_key, &eval);

        let cached = cache
            .cached_outer_eval(&rho_key)
            .expect("first-order outer eval should be cached");
        assert_eq!(cached.cost, eval.cost);
        assert_eq!(cached.gradient, eval.gradient);
        assert!(matches!(cached.hessian, HessianResult::Unavailable));

        cache.invalidate_eval_bundle();
        assert!(
            cache.cached_outer_eval(&rho_key).is_none(),
            "invalidating the bundle should clear the outer-eval cache too"
        );
    }

    #[test]
    fn implicit_hyper_design_derivative_respects_full_model_embedding() {
        let operator = ImplicitDesignPsiDerivative::new(
            array![1.0, 2.0, 3.0, 4.0],
            array![0.5, -1.0, 1.5, 2.0],
            array![0.1, 0.2, 0.3, 0.4],
            array![[1.0, 0.2], [0.5, 0.1], [1.5, 0.3], [2.0, 0.4]],
            None,
            None,
            2,
            2,
            1,
            2,
        );
        let local = operator
            .materialize_first(0)
            .expect("materialized first derivative");
        assert_eq!(
            local.ncols(),
            3,
            "operator-local derivative should stay smooth-local"
        );

        let implicit = HyperDesignDerivative::from_implicit(
            Arc::new(operator),
            ImplicitDerivLevel::First(0),
            1..4,
            5,
        );
        let embedded = HyperDesignDerivative::from_embedded(local.clone(), 1..4, 5);

        assert_eq!(implicit.nrows(), embedded.nrows());
        assert_eq!(implicit.ncols(), 5);
        assert_eq!(implicit.materialize(), embedded.materialize());

        let u = array![7.0, 1.5, -2.0, 0.25, -3.0];
        let v = array![0.75, -1.25];
        assert_eq!(
            implicit.forward_mul_original(&u).expect("implicit forward"),
            embedded.forward_mul_original(&u).expect("embedded forward")
        );
        assert_eq!(
            implicit
                .transpose_mul_original(&v)
                .expect("implicit transpose"),
            embedded
                .transpose_mul_original(&v)
                .expect("embedded transpose")
        );

        let qs = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ];
        assert_eq!(
            implicit
                .transformed(&qs, None)
                .expect("implicit transformed"),
            embedded
                .transformed(&qs, None)
                .expect("embedded transformed")
        );
        let u_transformed = array![1.0, -0.5, 2.0];
        assert_eq!(
            implicit
                .transformed_forward_mul(&qs, None, &u_transformed)
                .expect("implicit transformed forward"),
            embedded
                .transformed_forward_mul(&qs, None, &u_transformed)
                .expect("embedded transformed forward")
        );
        assert_eq!(
            implicit
                .transformed_transpose_mul(&qs, None, &v)
                .expect("implicit transformed transpose"),
            embedded
                .transformed_transpose_mul(&qs, None, &v)
                .expect("embedded transformed transpose")
        );
    }

    #[test]
    fn directional_hyper_identities_match_finite_differences_logit() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];

        // Use one directional hyperparameter τ with a penalty perturbation:
        // S(τ) = S + τ S_τ.
        // Keep X_τ = 0 so this identity test remains valid in both non-Firth
        // and Firth-logit modes.
        let x_tau = Array2::<f64>::zeros(x.raw_dim());
        let s_tau = array![[0.0, 0.0, 0.0], [0.0, 0.25, 0.04], [0.0, 0.04, 0.15],];
        let hyper =
            DirectionalHyperParam::single_penalty(0, x_tau.clone(), s_tau.clone(), None, None)
                .expect("single-penalty hyper direction");
        let rho = array![0.0];

        // Tight inner tolerance: the envelope theorem requires an exact inner
        // P-IRLS optimum; 1e-10 leaves enough residual gradient to cause ~12%
        // V_tau mismatch on this small (n=8) logistic problem.
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-14,
            false,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let bundle = state.obtain_eval_bundle(&rho).expect("bundle");
        let pr = bundle.pirls_result.as_ref();

        let beta = beta_original_from_bundle(&bundle);
        let h_orig = h_original_from_bundle(&bundle);
        let u = &pr.solveweights * &(&pr.solveworking_response - &pr.final_eta);

        // B from implicit solve:
        //   H B = X_τ^T g - X^T W(X_τ β̂) - S_τ β̂.
        let x_tau_beta = x_tau.dot(&beta);
        let weighted_x_tau_beta = &pr.finalweights * &x_tau_beta;
        let rhs = x_tau.t().dot(&u) - x.t().dot(&weighted_x_tau_beta) - s_tau.dot(&beta);
        let chol = h_orig.cholesky(Side::Lower).expect("chol(H)");
        let b_analytic = chol.solvevec(&rhs);

        // H_τ from exact total derivative:
        //   H_τ = X_τ^T W X + X^T W X_τ + X^T W_τ X + S_τ,
        // with W_τ provided by the family directional curvature callback.
        let eta_dot = &x_tau_beta + &x.dot(&b_analytic);
        let w_direction = crate::pirls::directionalworking_curvature_from_c_array(
            &pr.solve_c_array,
            &pr.finalweights,
            &eta_dot,
        );
        let wx = RemlState::row_scale(&x, &pr.finalweights);
        let wx_tau = RemlState::row_scale(&x_tau, &pr.finalweights);
        let mut xwtau_x = x.clone();
        match w_direction {
            crate::pirls::DirectionalWorkingCurvature::Diagonal(diag) => {
                xwtau_x = RemlState::row_scale(&xwtau_x, &diag);
            }
        }
        let mut h_tau_analytic = x_tau.t().dot(&wx);
        h_tau_analytic += &x.t().dot(&wx_tau);
        h_tau_analytic += &x.t().dot(&xwtau_x);
        h_tau_analytic += &s_tau;

        // Fit-block stationarity cancellation:
        //   -ℓ_β^T B + β̂^T S B = 0.
        // Here S is the effective penalty in the inner Hessian surface:
        //   S = H - X^T W X.
        let ell_beta = x.t().dot(&u);
        let s_eff = &h_orig - &x.t().dot(&wx);
        let cancellation = -ell_beta.dot(&b_analytic) + beta.dot(&s_eff.dot(&b_analytic));

        // Finite differences in τ against re-fit objective and mode.
        let h = 2e-5;
        let x_plus = &x + &(x_tau.mapv(|v| h * v));
        let x_minus = &x - &(x_tau.mapv(|v| h * v));
        let s_plus = &s0 + &(s_tau.mapv(|v| h * v));
        let s_minus = &s0 - &(s_tau.mapv(|v| h * v));

        let state_plus = build_logit_state(&y, &w, &x_plus, &s_plus, &cfg);
        let state_minus = build_logit_state(&y, &w, &x_minus, &s_minus, &cfg);
        let bundle_plus = state_plus.obtain_eval_bundle(&rho).expect("bundle+");
        let bundle_minus = state_minus.obtain_eval_bundle(&rho).expect("bundle-");
        let beta_plus = beta_original_from_bundle(&bundle_plus);
        let beta_minus = beta_original_from_bundle(&bundle_minus);
        let bfd = (&beta_plus - &beta_minus).mapv(|v| v / (2.0 * h));

        let h_plus = h_original_from_bundle(&bundle_plus);
        let h_minus = h_original_from_bundle(&bundle_minus);
        let h_taufd = (&h_plus - &h_minus).mapv(|v| v / (2.0 * h));

        let v_plus = state_plus.compute_cost(&rho).expect("cost+");
        let v_minus = state_minus.compute_cost(&rho).expect("cost-");
        let v_taufd = (v_plus - v_minus) / (2.0 * h);

        let v_tau_analytic = single_directional_tau_gradient(&state, &rho, hyper.clone())
            .expect("analytic directional gradient");

        let b_num = (&b_analytic - &bfd).mapv(|v| v * v).sum().sqrt();
        let b_den = bfd.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let b_rel = b_num / b_den;
        for i in 0..b_analytic.len() {
            assert_eq!(
                b_analytic[i].signum(),
                bfd[i].signum(),
                "B sign mismatch at i={i}: analytic={} fd={}",
                b_analytic[i],
                bfd[i]
            );
        }
        assert!(
            b_rel < 2e-2,
            "B implicit solve mismatch vs FD: rel={b_rel:.3e}, num={b_num:.3e}, den={b_den:.3e}"
        );

        let dh_num = (&h_tau_analytic - &h_taufd).mapv(|v| v * v).sum().sqrt();
        let dh_den = h_taufd.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let dh_rel = dh_num / dh_den;
        for i in 0..h_tau_analytic.nrows() {
            for j in 0..h_tau_analytic.ncols() {
                assert_eq!(
                    h_tau_analytic[[i, j]].signum(),
                    h_taufd[[i, j]].signum(),
                    "H_tau sign mismatch at ({i},{j}): analytic={} fd={}",
                    h_tau_analytic[[i, j]],
                    h_taufd[[i, j]]
                );
            }
        }
        assert!(
            dh_rel < 3e-2,
            "H_tau mismatch vs FD: rel={dh_rel:.3e}, num={dh_num:.3e}, den={dh_den:.3e}"
        );

        let v_abs = (v_tau_analytic - v_taufd).abs();
        let v_rel = v_abs / v_taufd.abs().max(1e-10);
        assert_eq!(
            v_tau_analytic.signum(),
            v_taufd.signum(),
            "V_tau sign mismatch: analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
        assert!(
            v_rel < 1e-3,
            "V_tau mismatch vs FD: rel={v_rel:.3e}, abs={v_abs:.3e}, analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );

        assert!(
            cancellation.abs() < 1e-10,
            "stationarity cancellation failed: | -ell_beta^T B + beta^T S B | = {:.3e}",
            cancellation.abs()
        );
    }

    #[test]
    fn large_non_gaussian_models_disable_analytic_outer_hessian() {
        let n = 5_001usize;
        let p = 100usize;
        let y = Array1::<f64>::zeros(n);
        let w = Array1::<f64>::ones(n);
        let x = Array2::<f64>::zeros((n, p));
        let s0 = Array2::<f64>::eye(p);
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-6,
            false,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);

        assert!(!state.analytic_outer_hessian_enabled());
    }

    #[test]
    fn downgraded_second_order_request_reuses_cached_first_order_outer_eval() {
        let n = 5_001usize;
        let p = 100usize;
        let y = Array1::<f64>::zeros(n);
        let w = Array1::<f64>::ones(n);
        let x = Array2::<f64>::zeros((n, p));
        let s0 = Array2::<f64>::eye(p);
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-6,
            false,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];
        let expected = OuterEval {
            cost: 7.5,
            gradient: array![1.25],
            hessian: HessianResult::Unavailable,
        };

        let rho_key = EvalCacheManager::sanitized_rhokey(&rho);
        state.cache_manager.store_outer_eval(&rho_key, &expected);
        state
            .cache_manager
            .current_eval_bundle
            .write()
            .unwrap()
            .take();
        assert!(
            state.cache_manager.cached_eval_bundle(&rho_key).is_none(),
            "test precondition: force the second request to rely on outer-eval cache only"
        );

        let cached_eval = state
            .compute_outer_eval_with_order(&rho, OuterEvalOrder::ValueGradientHessian)
            .expect("cached downgraded outer eval");
        assert_eq!(cached_eval.cost, expected.cost);
        assert_eq!(cached_eval.gradient, expected.gradient);
        assert!(matches!(cached_eval.hessian, HessianResult::Unavailable));
        assert!(
            state.cache_manager.cached_eval_bundle(&rho_key).is_none(),
            "downgraded second-order request should return from cached first-order outer eval"
        );
    }

    #[test]
    fn firth_exacthessian_matchesfd_on_rank_deficient_design() {
        // Rank-deficient X: the 4th column is 2x the 2nd column.
        let y = array![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.4, -2.4],
            [1.0, -0.9, -0.1, -1.8],
            [1.0, -0.6, 0.3, -1.2],
            [1.0, -0.2, -0.4, -0.4],
            [1.0, 0.1, 0.5, 0.2],
            [1.0, 0.4, -0.6, 0.8],
            [1.0, 0.8, 0.2, 1.6],
            [1.0, 1.1, -0.3, 2.2],
            [1.0, 1.4, 0.7, 2.8],
            [1.0, 1.7, -0.2, 3.4],
        ];
        let s0 = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.2, 0.0],
            [0.0, 0.2, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ];
        let s1 = array![
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, -0.1, 0.0],
            [0.0, -0.1, 0.6, 0.0],
            [0.0, 0.0, 0.0, 0.3],
        ];
        let offset = Array1::<f64>::zeros(y.len());
        // Rank-deficient Firth logit needs more inner iterations to converge
        // tightly enough for the envelope-theorem derivative tests.
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-9,
            true,
        )
        .with_max_iterations(500);
        let p = x.ncols();
        use crate::estimate::PenaltySpec;
        let specs = vec![PenaltySpec::Dense(s0), PenaltySpec::Dense(s1)];
        let canonical = crate::construction::canonicalize_penalty_specs(&specs, &[1, 1], p, "test")
            .map(|(canonical, _)| canonical)
            .expect("canonicalize");
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            w.view(),
            offset.view(),
            canonical,
            p,
            &cfg,
            Some(vec![1, 1]),
            None,
            None,
        )
        .expect("state");
        let rho = array![0.1, -0.2];
        let bundle = state.obtain_eval_bundle(&rho).expect("exact firth bundle");
        let h_exact = state
            .compute_lamlhessian_exact_from_bundle(&rho, &bundle)
            .expect("exact firth hessian");
        let h_fallback = state
            .compute_lamlhessian_diagnostic_numeric(&rho)
            .expect("diagnostic numeric hessian");

        assert!(h_exact.iter().all(|v: &f64| v.is_finite()));
        assert!(h_fallback.iter().all(|v: &f64| v.is_finite()));
        for i in 0..h_exact.nrows() {
            for j in 0..i {
                assert!(
                    (h_exact[[i, j]] - h_exact[[j, i]]).abs() < 1e-8,
                    "exact Hessian asymmetry at ({i},{j})"
                );
            }
        }

        let diff = &h_exact - &h_fallback;
        let num = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        let den = h_fallback
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-8);
        let rel = num / den;
        assert!(
            rel < 8.0e-1,
            "Firth exact-vs-diagnostic-numeric Hessian mismatch too large: rel={rel:.3e}, exact={h_exact:?}, fallback={h_fallback:?}"
        );
    }

    #[test]
    fn firthgradient_lives_in_design_column_space_under_rank_deficiency() {
        // Rank-deficient design: col4 = 2*col2.
        let x = array![
            [1.0, -1.2, 0.4, -2.4],
            [1.0, -0.9, -0.1, -1.8],
            [1.0, -0.6, 0.3, -1.2],
            [1.0, -0.2, -0.4, -0.4],
            [1.0, 0.1, 0.5, 0.2],
            [1.0, 0.4, -0.6, 0.8],
            [1.0, 0.8, 0.2, 1.6],
            [1.0, 1.1, -0.3, 2.2],
        ];
        let beta = array![0.1, -0.2, 0.3, 0.05];
        let eta = x.dot(&beta);
        let op = FirthDenseOperator::build(&x, &eta).expect("firth operator");

        // Exact reduced-space Firth gradient:
        //   gradPhi = 0.5 Xᵀ (w' ⊙ h), with h = diag(X_r K_r X_rᵀ).
        let gradphi = 0.5 * x.t().dot(&(&op.w1 * &op.h_diag));

        // Check (I - QQᵀ) gradPhi ≈ 0.
        let q = &op.q_basis;
        let proj = q.dot(&q.t().dot(&gradphi));
        let resid = &gradphi - &proj;
        let rel =
            resid.mapv(|v| v * v).sum().sqrt() / gradphi.mapv(|v| v * v).sum().sqrt().max(1e-12);
        assert!(
            rel < 1e-10,
            "Firth gradient should lie in Col(Xᵀ): rel residual={rel:.3e}"
        );
    }

    #[test]
    fn fixed_subspace_penalty_trace_uses_positive_penalty_subspace_only() {
        // Use a non-separable pattern so this test exercises penalty-subspace
        // tracing logic rather than separation handling.
        let y = array![0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.5, -0.4],
            [1.0, 0.0, 0.7],
            [1.0, 0.4, -0.3],
            [1.0, 0.9, 0.1],
            [1.0, 1.3, -0.6],
        ];
        // Rank-deficient penalty with clear nullspace on first coordinate.
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.1, 0.15], [0.0, 0.15, 0.8],];
        let rho = array![0.0];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-10,
            false,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let bundle = state.obtain_eval_bundle(&rho).expect("bundle");
        let pr = bundle.pirls_result.as_ref();
        let e = &pr.reparam_result.e_transformed;
        let p = e.ncols();
        let mut s_lambda = e.t().dot(e);
        let ridge = pr.ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..p {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, evecs) = s_lambda.eigh(Side::Lower).expect("penalty eigh");
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let pos_idx = order[0];
        let null_idx = order[order.len() - 1];
        let u_pos = evecs.column(pos_idx).to_owned();
        let u_null = evecs.column(null_idx).to_owned();

        let s_dir_null = {
            let col = u_null.clone().insert_axis(ndarray::Axis(1));
            let row = u_null.insert_axis(ndarray::Axis(0));
            col.dot(&row)
        };
        let s_dir_pos = {
            let col = u_pos.clone().insert_axis(ndarray::Axis(1));
            let row = u_pos.clone().insert_axis(ndarray::Axis(0));
            col.dot(&row)
        };

        // With the exact pseudoinverse, the null-direction trace is zero:
        // tr(S⁺ S_null) = 0 because S⁺ projects onto the positive eigenspace
        // and u₀ is orthogonal to it.
        let tr_null = state
            .fixed_subspace_penalty_trace(e, &s_dir_null, pr.ridge_passport)
            .expect("trace-null");
        assert!(
            tr_null.abs() < 1e-10,
            "nullspace direction trace should be ~0 with exact pseudoinverse: got {tr_null:.3e}"
        );

        let tr_pos = state
            .fixed_subspace_penalty_trace(e, &s_dir_pos, pr.ridge_passport)
            .expect("trace-pos");
        // With exact pseudoinverse, expected is 1/σ_pos.
        let expected_pos = 1.0 / evals[pos_idx];
        let rel = (tr_pos - expected_pos).abs() / expected_pos.abs().max(1e-12);
        assert!(
            rel < 1e-6,
            "positive-subspace contraction mismatch: got={tr_pos:.6e}, expected={expected_pos:.6e}, rel={rel:.3e}"
        );
    }

    #[test]
    fn firth_logit_directional_hypergradient_supports_penalty_only_direction() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.1, 0.8],];
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            Array2::<f64>::zeros((x.nrows(), x.ncols())),
            array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.03], [0.0, 0.03, 0.12],],
            None,
            None,
        )
        .expect("single-penalty hyper direction");
        let rho = array![0.0];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-8,
            true,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let g = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("firth penalty-only directional gradient should evaluate");
        assert!(
            g.is_finite(),
            "non-finite Firth penalty-only directional gradient"
        );
    }

    #[test]
    fn firth_logit_directional_hypergradient_supports_design_moving_direction() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.1, 0.2],
            [1.0, -0.6, -0.3],
            [1.0, -0.1, 0.5],
            [1.0, 0.3, -0.7],
            [1.0, 0.8, 0.1],
            [1.0, 1.2, -0.4],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.1, 0.8],];
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            Array2::from_elem((x.nrows(), x.ncols()), 1e-3),
            Array2::<f64>::zeros((x.ncols(), x.ncols())),
            None,
            None,
        )
        .expect("single-penalty hyper direction");
        let rho = array![0.0];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-8,
            true,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let g = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("firth design-moving directional gradient should evaluate");
        assert!(
            g.is_finite(),
            "non-finite Firth design-moving directional gradient"
        );
    }

    #[test]
    fn joint_hyperhessianwires_mixed_blocks() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-10,
            true,
        )
        .with_max_iterations(500);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];
        let theta = array![0.0, 0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
        ];

        let (_, _, h) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint hyper cost+gradient+hessian");
        assert_eq!(h.nrows(), theta.len());
        assert_eq!(h.ncols(), theta.len());
        assert!(h.iter().all(|v| v.is_finite()));
        for i in 0..h.nrows() {
            for j in 0..i {
                let diff = (h[[i, j]] - h[[j, i]]).abs();
                assert!(
                    diff < 1e-6,
                    "joint hessian asymmetry at ({i},{j}): {diff:.3e}"
                );
            }
        }
        // Mixed block must be nontrivial for at least one supplied direction.
        let mixed_0 = h[[0, 1]];
        let mixed_1 = h[[0, 2]];
        assert!(
            mixed_0.is_finite() && mixed_1.is_finite(),
            "mixed blocks must be finite"
        );
    }

    #[test]
    fn joint_tau_tau_linear_dirs_matchfd_reference_away_fromzero_psi() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-10,
            true,
        )
        .with_max_iterations(500);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];
        let psi = array![0.7, -0.4];
        let theta = array![rho[0], psi[0], psi[1]];
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
                None,
                None,
            )
            .expect("linear tau direction"),
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                None,
                None,
            )
            .expect("linear tau direction"),
        ];

        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint hyper cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![rho.len().., rho.len()..]).to_owned();

        // FD via physical perturbation of design/penalty matrices (matching
        // the V_tau FD pattern).  For column j we perturb X and S₀ along
        // direction j, build fresh states, and evaluate the τ-gradient for
        // every direction i at those perturbed states.
        let x_tau_mats: Vec<Array2<f64>> = vec![
            Array2::<f64>::zeros((x.nrows(), x.ncols())),
            Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
        ];
        let s_tau_mats: Vec<Array2<f64>> = vec![
            array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15]],
            Array2::<f64>::zeros((x.ncols(), x.ncols())),
        ];

        let mut h_ttfd = Array2::<f64>::zeros((hyper_dirs.len(), hyper_dirs.len()));
        let h = 1e-5;
        for j in 0..hyper_dirs.len() {
            let x_plus = &x + &x_tau_mats[j].mapv(|v| h * v);
            let x_minus = &x - &x_tau_mats[j].mapv(|v| h * v);
            let s_plus = &s0 + &s_tau_mats[j].mapv(|v| h * v);
            let s_minus = &s0 - &s_tau_mats[j].mapv(|v| h * v);

            let state_plus = build_logit_state(&y, &w, &x_plus, &s_plus, &cfg);
            let state_minus = build_logit_state(&y, &w, &x_minus, &s_minus, &cfg);
            for i in 0..hyper_dirs.len() {
                let g_plus =
                    single_directional_tau_gradient(&state_plus, &rho, hyper_dirs[i].clone())
                        .expect("g+ for FD");
                let g_minus =
                    single_directional_tau_gradient(&state_minus, &rho, hyper_dirs[i].clone())
                        .expect("g- for FD");
                h_ttfd[[i, j]] = (g_plus - g_minus) / (2.0 * h);
            }
        }
        for i in 0..h_ttfd.nrows() {
            for j in 0..i {
                let avg = 0.5 * (h_ttfd[[i, j]] + h_ttfd[[j, i]]);
                h_ttfd[[i, j]] = avg;
                h_ttfd[[j, i]] = avg;
            }
        }

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "linear-dir joint tau-tau block deviates from FD reference away from zero psi: rel={rel:.3e}, analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    #[test]
    fn joint_hypervalidation_rejects_out_of_boundssecond_order_penalty_index() {
        let y = array![0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -0.5, 0.2],
            [1.0, -0.1, -0.3],
            [1.0, 0.4, 0.6],
            [1.0, 0.9, -0.2],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.1], [0.0, 0.1, 0.8],];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-10,
            true,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let theta = array![0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam::new(
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                vec![(0, Array2::<f64>::zeros((x.ncols(), x.ncols())))],
                None,
                Some(vec![Some(vec![(1, Array2::<f64>::eye(x.ncols()))])]),
            )
            .expect("hyper direction with invalid second-order penalty index"),
        ];

        let msg = match compute_joint_hypercostgradienthessian(&state, &theta, 1, &hyper_dirs) {
            Ok(_) => panic!("invalid second-order penalty index should be rejected"),
            Err(err) => err.to_string(),
        };
        assert!(
            msg.contains("out of bounds") || msg.contains("penalty_index"),
            "unexpected validation error: {msg}"
        );
    }

    #[test]
    fn joint_tau_tau_analytic_matchesfd_reference() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9],];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-10,
            true,
        )
        .with_max_iterations(500);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];
        let psi = array![0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                None,
                None,
            )
            .expect("single-penalty hyper direction"),
        ];

        let theta = {
            let mut t = Array1::<f64>::zeros(rho.len() + psi.len());
            t.slice_mut(s![..rho.len()]).assign(&rho);
            t.slice_mut(s![rho.len()..]).assign(&psi);
            t
        };
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint hyper cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![rho.len().., rho.len()..]).to_owned();
        assert_eq!(h_tt_analytic.nrows(), hyper_dirs.len());
        assert_eq!(h_tt_analytic.ncols(), hyper_dirs.len());

        // FD via physical perturbation of design/penalty matrices (matching
        // the V_tau FD pattern).  For column j we perturb X and S₀ along
        // direction j, build fresh states, and evaluate the τ-gradient for
        // every direction i at those perturbed states.
        let x_tau_mats: Vec<Array2<f64>> = vec![
            Array2::<f64>::zeros((x.nrows(), x.ncols())),
            Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
        ];
        let s_tau_mats: Vec<Array2<f64>> = vec![
            array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15]],
            Array2::<f64>::zeros((x.ncols(), x.ncols())),
        ];

        let mut h_ttfd = Array2::<f64>::zeros((hyper_dirs.len(), hyper_dirs.len()));
        let h = 1e-5;
        for j in 0..hyper_dirs.len() {
            let x_plus = &x + &x_tau_mats[j].mapv(|v| h * v);
            let x_minus = &x - &x_tau_mats[j].mapv(|v| h * v);
            let s_plus = &s0 + &s_tau_mats[j].mapv(|v| h * v);
            let s_minus = &s0 - &s_tau_mats[j].mapv(|v| h * v);

            let state_plus = build_logit_state(&y, &w, &x_plus, &s_plus, &cfg);
            let state_minus = build_logit_state(&y, &w, &x_minus, &s_minus, &cfg);
            for i in 0..hyper_dirs.len() {
                let g_plus =
                    single_directional_tau_gradient(&state_plus, &rho, hyper_dirs[i].clone())
                        .expect("g+ for FD");
                let g_minus =
                    single_directional_tau_gradient(&state_minus, &rho, hyper_dirs[i].clone())
                        .expect("g- for FD");
                h_ttfd[[i, j]] = (g_plus - g_minus) / (2.0 * h);
            }
        }
        for i in 0..h_ttfd.nrows() {
            for j in 0..i {
                let avg = 0.5 * (h_ttfd[[i, j]] + h_ttfd[[j, i]]);
                h_ttfd[[i, j]] = avg;
                h_ttfd[[j, i]] = avg;
            }
        }

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "analytic tau-tau block deviates from FD reference: rel={rel:.3e}, analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    // ── Profiled Gaussian REML coverage for design-moving τ-directions ──
    //
    // The existing directional-hyper tests all use BinomialLogit, which has
    // DispersionHandling::Fixed.  These tests validate the profiled Gaussian
    // path (DispersionHandling::ProfiledGaussian) with design-moving
    // τ-directions, where the profiled scale φ̂ = D_p/(n−M) depends on ρ
    // and the envelope-theorem rescaling by (n−M)/D_p must be correct.

    /// Shared test fixture for profiled Gaussian REML tests.
    struct GaussianRemlFixture {
        y: Array1<f64>,
        w: Array1<f64>,
        x: Array2<f64>,
        s0: Array2<f64>,
        cfg: RemlConfig,
        rho: Array1<f64>,
        /// Design-moving τ-direction (non-zero X_τ, zero S_τ).
        x_tau_design: Array2<f64>,
        /// Penalty-only τ-direction (zero X_τ, non-zero S_τ).
        s_tau_penalty: Array2<f64>,
    }

    impl GaussianRemlFixture {
        fn new() -> Self {
            let y = array![0.5, 1.2, -0.3, 0.8, 1.1, -0.6, 0.9, 0.1, -0.2, 0.7];
            let x = array![
                [1.0, -1.2, 0.3],
                [1.0, -0.8, -0.4],
                [1.0, -0.3, 0.7],
                [1.0, 0.1, -0.9],
                [1.0, 0.5, 0.2],
                [1.0, 0.9, -0.1],
                [1.0, 1.3, 0.8],
                [1.0, 1.7, -0.6],
                [1.0, -0.5, 0.5],
                [1.0, 0.3, -0.3],
            ];
            Self {
                w: Array1::<f64>::ones(y.len()),
                y,
                x: x.clone(),
                s0: array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9]],
                cfg: RemlConfig::external(
                    GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::GaussianIdentity),
                    1e-14,
                    false,
                ),
                rho: array![0.0],
                x_tau_design: array![
                    [0.0, 1e-3, -2e-3],
                    [0.0, -3e-3, 1e-3],
                    [0.0, 2e-3, 0.5e-3],
                    [0.0, -1e-3, 3e-3],
                    [0.0, 0.5e-3, -1e-3],
                    [0.0, 1.5e-3, 2e-3],
                    [0.0, -2e-3, -0.5e-3],
                    [0.0, 3e-3, 1e-3],
                    [0.0, -0.5e-3, 2e-3],
                    [0.0, 1e-3, -1.5e-3],
                ],
                s_tau_penalty: array![[0.0, 0.0, 0.0], [0.0, 0.25, 0.04], [0.0, 0.04, 0.15]],
            }
        }

        fn state(&self) -> RemlState<'_> {
            build_logit_state(&self.y, &self.w, &self.x, &self.s0, &self.cfg)
        }

        fn state_perturbed(
            &self,
            x_tau: &Array2<f64>,
            s_tau: &Array2<f64>,
            eps: f64,
        ) -> (RemlState<'_>, RemlState<'_>) {
            let x_plus = &self.x + &x_tau.mapv(|v| eps * v);
            let x_minus = &self.x - &x_tau.mapv(|v| eps * v);
            let s_plus = &self.s0 + &s_tau.mapv(|v| eps * v);
            let s_minus = &self.s0 - &s_tau.mapv(|v| eps * v);
            (
                build_logit_state(&self.y, &self.w, &x_plus, &s_plus, &self.cfg),
                build_logit_state(&self.y, &self.w, &x_minus, &s_minus, &self.cfg),
            )
        }

        /// Central FD approximation to the directional cost derivative.
        fn fd_directional_gradient(&self, x_tau: &Array2<f64>, s_tau: &Array2<f64>) -> f64 {
            let h = 2e-5;
            let (state_plus, state_minus) = self.state_perturbed(x_tau, s_tau, h);
            let v_plus = state_plus.compute_cost(&self.rho).expect("cost+");
            let v_minus = state_minus.compute_cost(&self.rho).expect("cost-");
            (v_plus - v_minus) / (2.0 * h)
        }
    }

    #[test]
    fn profiled_gaussian_design_moving_gradient_matches_fd() {
        let f = GaussianRemlFixture::new();
        let state = f.state();
        let s_tau = Array2::<f64>::zeros((3, 3));
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            s_tau.clone(),
            None,
            None,
        )
        .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_taufd = f.fd_directional_gradient(&f.x_tau_design, &s_tau);

        let v_rel = (v_tau_analytic - v_taufd).abs() / v_taufd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Gaussian REML design-moving V_tau mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
    }

    #[test]
    fn profiled_gaussian_penalty_only_gradient_matches_fd() {
        let f = GaussianRemlFixture::new();
        let state = f.state();
        let x_tau = Array2::<f64>::zeros(f.x.raw_dim());
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            x_tau.clone(),
            f.s_tau_penalty.clone(),
            None,
            None,
        )
        .expect("penalty-only hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_taufd = f.fd_directional_gradient(&x_tau, &f.s_tau_penalty);

        let v_rel = (v_tau_analytic - v_taufd).abs() / v_taufd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Gaussian REML penalty-only V_tau mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
    }

    #[test]
    fn profiled_gaussian_joint_hessian_matches_fd() {
        // Validate the ττ Hessian block under profiled Gaussian REML with
        // both a penalty-only and a design-moving direction.
        let f = GaussianRemlFixture::new();
        let x_tau_0 = Array2::<f64>::zeros(f.x.raw_dim());
        let s_tau_0 = f.s_tau_penalty.clone();
        let x_tau_1 = f.x_tau_design.clone();
        let s_tau_1 = Array2::<f64>::zeros((3, 3));

        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(0, x_tau_0.clone(), s_tau_0.clone(), None, None)
                .expect("penalty-only direction"),
            DirectionalHyperParam::single_penalty(0, x_tau_1.clone(), s_tau_1.clone(), None, None)
                .expect("design-moving direction"),
        ];

        let state = f.state();
        let mut theta = Array1::<f64>::zeros(f.rho.len() + hyper_dirs.len());
        theta.slice_mut(s![..f.rho.len()]).assign(&f.rho);
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, f.rho.len(), &hyper_dirs)
                .expect("joint cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![f.rho.len().., f.rho.len()..]).to_owned();

        // Finite-difference Hessian: perturb each direction, re-evaluate
        // gradient of all directions at perturbed states.
        let x_tau_mats = [&x_tau_0, &x_tau_1];
        let s_tau_mats = [&s_tau_0, &s_tau_1];
        let n_dirs = hyper_dirs.len();
        let mut h_ttfd = Array2::<f64>::zeros((n_dirs, n_dirs));
        let eps = 1e-5;
        for j in 0..n_dirs {
            let (state_plus, state_minus) = f.state_perturbed(x_tau_mats[j], s_tau_mats[j], eps);
            for i in 0..n_dirs {
                let g_plus =
                    single_directional_tau_gradient(&state_plus, &f.rho, hyper_dirs[i].clone())
                        .expect("g+ for FD");
                let g_minus =
                    single_directional_tau_gradient(&state_minus, &f.rho, hyper_dirs[i].clone())
                        .expect("g- for FD");
                h_ttfd[[i, j]] = (g_plus - g_minus) / (2.0 * eps);
            }
        }
        // Symmetrize FD Hessian.
        for i in 0..n_dirs {
            for j in 0..i {
                let avg = 0.5 * (h_ttfd[[i, j]] + h_ttfd[[j, i]]);
                h_ttfd[[i, j]] = avg;
                h_ttfd[[j, i]] = avg;
            }
        }

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "Gaussian REML tau-tau Hessian mismatch: rel={rel:.3e}, \
             analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    // ── Non-Gaussian + design-motion: IFT Hessian-drift coverage ────────
    //
    // For non-Gaussian links (logit, probit, cloglog, ...), H = X'W(η)X + S
    // depends on β̂ through η = Xβ̂.  When ψ moves the design, the total
    // Hessian drift dH/dψ includes an IFT contribution from dβ̂/dψ:
    //
    //   dH/dψ = [explicit at fixed β] + X' diag(c ⊙ X(-v_i)) X
    //
    // where v_i = H⁻¹ g_i.  The standard GLM path handles this via
    // `hessian_derivative_correction(v_i)`.  This test validates that the
    // gradient is correct for logit + design-moving ψ, which would fail if
    // the IFT correction were missing.

    #[test]
    fn logit_design_moving_gradient_matches_fd() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
            [1.0, -0.5, 0.5],
            [1.0, 0.3, -0.3],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9]];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-14,
            false,
        );
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let rho = array![0.0];

        // Design-moving direction with non-zero X_τ.
        let x_tau = array![
            [0.0, 1e-3, -2e-3],
            [0.0, -3e-3, 1e-3],
            [0.0, 2e-3, 0.5e-3],
            [0.0, -1e-3, 3e-3],
            [0.0, 0.5e-3, -1e-3],
            [0.0, 1.5e-3, 2e-3],
            [0.0, -2e-3, -0.5e-3],
            [0.0, 3e-3, 1e-3],
            [0.0, -0.5e-3, 2e-3],
            [0.0, 1e-3, -1.5e-3],
        ];
        let s_tau = Array2::<f64>::zeros((3, 3));
        let hyper =
            DirectionalHyperParam::single_penalty(0, x_tau.clone(), s_tau.clone(), None, None)
                .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("analytic directional gradient");

        let h = 2e-5;
        let x_plus = &x + &x_tau.mapv(|v| h * v);
        let x_minus = &x - &x_tau.mapv(|v| h * v);
        let state_plus = build_logit_state(&y, &w, &x_plus, &s0, &cfg);
        let state_minus = build_logit_state(&y, &w, &x_minus, &s0, &cfg);
        let v_plus = state_plus.compute_cost(&rho).expect("cost+");
        let v_minus = state_minus.compute_cost(&rho).expect("cost-");
        let v_taufd = (v_plus - v_minus) / (2.0 * h);

        let v_rel = (v_tau_analytic - v_taufd).abs() / v_taufd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Logit REML design-moving V_tau mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );
    }

    #[test]
    fn logit_design_moving_hessian_matches_fd() {
        // Hessian-level validation for logit + design-motion.
        // The IFT correction enters the trace term through
        // hessian_derivative_correction(v_i), so the Hessian is the most
        // sensitive test of whether the correction is applied correctly.
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = Array1::<f64>::ones(y.len());
        let x = array![
            [1.0, -1.2, 0.3],
            [1.0, -0.8, -0.4],
            [1.0, -0.3, 0.7],
            [1.0, 0.1, -0.9],
            [1.0, 0.5, 0.2],
            [1.0, 0.9, -0.1],
            [1.0, 1.3, 0.8],
            [1.0, 1.7, -0.6],
            [1.0, -0.5, 0.5],
            [1.0, 0.3, -0.3],
        ];
        let s0 = array![[0.0, 0.0, 0.0], [0.0, 1.2, 0.2], [0.0, 0.2, 0.9]];
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-14,
            false,
        );
        let rho = array![0.0];

        // Two directions: one penalty-only, one design-moving.
        let x_tau_0 = Array2::<f64>::zeros(x.raw_dim());
        let s_tau_0 = array![[0.0, 0.0, 0.0], [0.0, 0.25, 0.04], [0.0, 0.04, 0.15]];
        let x_tau_1 = array![
            [0.0, 1e-3, -2e-3],
            [0.0, -3e-3, 1e-3],
            [0.0, 2e-3, 0.5e-3],
            [0.0, -1e-3, 3e-3],
            [0.0, 0.5e-3, -1e-3],
            [0.0, 1.5e-3, 2e-3],
            [0.0, -2e-3, -0.5e-3],
            [0.0, 3e-3, 1e-3],
            [0.0, -0.5e-3, 2e-3],
            [0.0, 1e-3, -1.5e-3],
        ];
        let s_tau_1 = Array2::<f64>::zeros((3, 3));

        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(0, x_tau_0.clone(), s_tau_0.clone(), None, None)
                .expect("penalty-only direction"),
            DirectionalHyperParam::single_penalty(0, x_tau_1.clone(), s_tau_1.clone(), None, None)
                .expect("design-moving direction"),
        ];

        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        let mut theta = Array1::<f64>::zeros(rho.len() + hyper_dirs.len());
        theta.slice_mut(s![..rho.len()]).assign(&rho);
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, rho.len(), &hyper_dirs)
                .expect("joint cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![rho.len().., rho.len()..]).to_owned();

        let x_tau_mats = [&x_tau_0, &x_tau_1];
        let s_tau_mats = [&s_tau_0, &s_tau_1];
        let n_dirs = hyper_dirs.len();
        let mut h_ttfd = Array2::<f64>::zeros((n_dirs, n_dirs));
        let eps = 1e-5;
        for j in 0..n_dirs {
            let x_plus = &x + &x_tau_mats[j].mapv(|v| eps * v);
            let x_minus = &x - &x_tau_mats[j].mapv(|v| eps * v);
            let s_plus = &s0 + &s_tau_mats[j].mapv(|v| eps * v);
            let s_minus = &s0 - &s_tau_mats[j].mapv(|v| eps * v);
            let state_plus = build_logit_state(&y, &w, &x_plus, &s_plus, &cfg);
            let state_minus = build_logit_state(&y, &w, &x_minus, &s_minus, &cfg);
            for i in 0..n_dirs {
                let g_plus =
                    single_directional_tau_gradient(&state_plus, &rho, hyper_dirs[i].clone())
                        .expect("g+ for FD");
                let g_minus =
                    single_directional_tau_gradient(&state_minus, &rho, hyper_dirs[i].clone())
                        .expect("g- for FD");
                h_ttfd[[i, j]] = (g_plus - g_minus) / (2.0 * eps);
            }
        }
        for i in 0..n_dirs {
            for j in 0..i {
                let avg = 0.5 * (h_ttfd[[i, j]] + h_ttfd[[j, i]]);
                h_ttfd[[i, j]] = avg;
                h_ttfd[[j, i]] = avg;
            }
        }

        let num = (&h_tt_analytic - &h_ttfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_ttfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "Logit REML design-moving tau-tau Hessian mismatch: rel={rel:.3e}, \
             analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    // ── Larger non-Gaussian + design-motion fixture (n=30, p=5) ────────
    //
    // Validates the IFT correction (hessian_derivative_correction) at a
    // scale large enough that the correction is numerically non-trivial:
    // with n=30 and p=5, the logistic Hessian W(η) is far from identity
    // and the IFT term dβ̂/dψ contributes meaningfully.

    /// Shared test fixture for binomial-logit REML with design-moving
    /// ψ-coordinates, n=30, p=5.
    struct BinomialLogitDesignMotionFixture {
        y: Array1<f64>,
        w: Array1<f64>,
        x: Array2<f64>,
        s0: Array2<f64>,
        cfg: RemlConfig,
        rho: Array1<f64>,
        /// Design-moving τ-direction: non-zero X_τ, zero S_τ.
        x_tau_design: Array2<f64>,
        /// Penalty-only τ-direction: zero X_τ, non-zero S_τ.
        s_tau_penalty: Array2<f64>,
    }

    impl BinomialLogitDesignMotionFixture {
        fn new() -> Self {
            // Binary response with roughly balanced classes.
            let y = array![
                1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
            ];
            // Design matrix: intercept + 4 covariate columns with varied magnitudes.
            let x = array![
                [1.0, -1.50, 0.42, 0.88, -0.31],
                [1.0, -1.12, -0.65, 0.14, 1.23],
                [1.0, -0.80, 1.10, -0.53, 0.07],
                [1.0, -0.55, -0.22, 1.40, -0.90],
                [1.0, -0.30, 0.73, -1.05, 0.44],
                [1.0, -0.05, -1.33, 0.60, 0.81],
                [1.0, 0.18, 0.55, -0.27, -1.15],
                [1.0, 0.42, -0.90, 1.12, 0.33],
                [1.0, 0.70, 1.28, -0.78, -0.56],
                [1.0, 0.95, -0.18, 0.45, 1.40],
                [1.0, 1.20, 0.66, -1.30, -0.02],
                [1.0, 1.45, -1.05, 0.22, 0.68],
                [1.0, -1.35, 0.90, 0.55, -0.43],
                [1.0, -0.98, -0.40, -0.88, 1.05],
                [1.0, -0.62, 1.42, 0.30, -0.70],
                [1.0, -0.28, -0.77, -1.18, 0.52],
                [1.0, 0.05, 0.15, 0.95, -1.35],
                [1.0, 0.33, -1.20, -0.40, 0.18],
                [1.0, 0.60, 0.82, 1.25, -0.85],
                [1.0, 0.88, -0.50, -0.65, 1.10],
                [1.0, 1.15, 1.05, 0.10, -0.22],
                [1.0, -1.22, -0.95, 0.72, 0.90],
                [1.0, -0.75, 0.38, -1.42, 0.15],
                [1.0, -0.42, -1.15, 0.50, -1.08],
                [1.0, -0.10, 0.60, -0.15, 0.75],
                [1.0, 0.25, -0.28, 1.05, -0.48],
                [1.0, 0.52, 1.35, -0.92, 0.30],
                [1.0, 0.80, -0.70, 0.38, 1.20],
                [1.0, 1.08, 0.48, -0.60, -0.95],
                [1.0, 1.35, -0.55, 0.85, 0.42]
            ];
            // Penalty matrix: zero on intercept, SPD on remaining 4 columns.
            let s0 = array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.40, 0.15, 0.05, -0.10],
                [0.0, 0.15, 1.10, -0.20, 0.08],
                [0.0, 0.05, -0.20, 0.95, 0.12],
                [0.0, -0.10, 0.08, 0.12, 1.25]
            ];
            let cfg = RemlConfig::external(
                GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
                1e-14,
                false,
            );
            // Design-moving direction: perturb covariate columns, leave
            // intercept untouched.
            let x_tau_design = array![
                [0.0, 1.2e-3, -0.8e-3, 0.5e-3, -1.5e-3],
                [0.0, -2.0e-3, 1.4e-3, -0.3e-3, 0.9e-3],
                [0.0, 0.6e-3, -1.1e-3, 1.8e-3, -0.4e-3],
                [0.0, -1.3e-3, 0.7e-3, -1.0e-3, 2.1e-3],
                [0.0, 0.9e-3, -0.5e-3, 0.2e-3, -0.8e-3],
                [0.0, -0.4e-3, 1.8e-3, -1.5e-3, 0.3e-3],
                [0.0, 1.5e-3, -1.3e-3, 0.8e-3, -1.1e-3],
                [0.0, -0.7e-3, 0.4e-3, -2.0e-3, 1.6e-3],
                [0.0, 2.2e-3, -0.9e-3, 1.3e-3, -0.6e-3],
                [0.0, -1.0e-3, 1.6e-3, -0.7e-3, 0.5e-3],
                [0.0, 0.3e-3, -2.1e-3, 1.1e-3, -1.8e-3],
                [0.0, -1.8e-3, 0.2e-3, -0.4e-3, 1.3e-3],
                [0.0, 1.1e-3, -1.5e-3, 2.0e-3, -0.2e-3],
                [0.0, -0.5e-3, 0.9e-3, -1.2e-3, 0.7e-3],
                [0.0, 1.7e-3, -0.3e-3, 0.6e-3, -2.0e-3],
                [0.0, -1.4e-3, 1.1e-3, -0.9e-3, 0.4e-3],
                [0.0, 0.8e-3, -1.7e-3, 1.5e-3, -0.1e-3],
                [0.0, -0.2e-3, 0.6e-3, -1.8e-3, 1.0e-3],
                [0.0, 1.4e-3, -0.4e-3, 0.3e-3, -1.3e-3],
                [0.0, -0.9e-3, 2.0e-3, -0.5e-3, 0.8e-3],
                [0.0, 0.5e-3, -1.0e-3, 1.6e-3, -0.7e-3],
                [0.0, -2.1e-3, 0.3e-3, -0.8e-3, 1.5e-3],
                [0.0, 0.7e-3, -1.8e-3, 0.9e-3, -0.3e-3],
                [0.0, -0.6e-3, 1.3e-3, -2.2e-3, 1.1e-3],
                [0.0, 1.9e-3, -0.7e-3, 0.4e-3, -0.9e-3],
                [0.0, -1.1e-3, 0.5e-3, -1.4e-3, 2.2e-3],
                [0.0, 0.4e-3, -1.6e-3, 1.2e-3, -0.5e-3],
                [0.0, -1.6e-3, 0.8e-3, -0.1e-3, 0.6e-3],
                [0.0, 1.3e-3, -2.2e-3, 0.7e-3, -1.4e-3],
                [0.0, -0.3e-3, 1.0e-3, -1.6e-3, 1.8e-3]
            ];
            // Penalty-only direction: non-zero S_τ, symmetric, zero on intercept.
            let s_tau_penalty = array![
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.30, 0.05, -0.02, 0.04],
                [0.0, 0.05, 0.22, 0.03, -0.01],
                [0.0, -0.02, 0.03, 0.18, 0.06],
                [0.0, 0.04, -0.01, 0.06, 0.26]
            ];
            Self {
                w: Array1::<f64>::ones(y.len()),
                y,
                x,
                s0,
                cfg,
                rho: array![0.0],
                x_tau_design,
                s_tau_penalty,
            }
        }

        fn state(&self) -> RemlState<'_> {
            build_logit_state(&self.y, &self.w, &self.x, &self.s0, &self.cfg)
        }

        fn state_perturbed(
            &self,
            x_tau: &Array2<f64>,
            s_tau: &Array2<f64>,
            eps: f64,
        ) -> (RemlState<'_>, RemlState<'_>) {
            let x_plus = &self.x + &x_tau.mapv(|v| eps * v);
            let x_minus = &self.x - &x_tau.mapv(|v| eps * v);
            let s_plus = &self.s0 + &s_tau.mapv(|v| eps * v);
            let s_minus = &self.s0 - &s_tau.mapv(|v| eps * v);
            (
                build_logit_state(&self.y, &self.w, &x_plus, &s_plus, &self.cfg),
                build_logit_state(&self.y, &self.w, &x_minus, &s_minus, &self.cfg),
            )
        }

        /// Central FD approximation to the directional cost derivative.
        fn fd_directional_gradient(&self, x_tau: &Array2<f64>, s_tau: &Array2<f64>) -> f64 {
            let h = 2e-5;
            let (state_plus, state_minus) = self.state_perturbed(x_tau, s_tau, h);
            let v_plus = state_plus.compute_cost(&self.rho).expect("cost+");
            let v_minus = state_minus.compute_cost(&self.rho).expect("cost-");
            (v_plus - v_minus) / (2.0 * h)
        }
    }

    // ── n=30, p=5 binomial-logit design-motion gradient tests ────────

    #[test]
    fn binomial_logit_n30_design_moving_gradient_matches_fd() {
        // Pure design-motion: X_τ ≠ 0, S_τ = 0.
        // The IFT correction is essential here: because the family is
        // binomial-logit, the working weights W(η) depend on β̂, so
        // when X moves with ψ, the implicit derivative dβ̂/dψ enters
        // the total Hessian drift.  Without hessian_derivative_correction
        // the analytic gradient would disagree with FD.
        let f = BinomialLogitDesignMotionFixture::new();
        let state = f.state();
        let s_tau = Array2::<f64>::zeros((5, 5));
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            s_tau.clone(),
            None,
            None,
        )
        .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_tau_fd = f.fd_directional_gradient(&f.x_tau_design, &s_tau);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 design-moving gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }

    #[test]
    fn binomial_logit_n30_penalty_only_gradient_matches_fd() {
        // Penalty-only direction: X_τ = 0, S_τ ≠ 0.
        // Serves as a baseline: the IFT correction should still be
        // present (since H depends on β̂ through W(η)), but the
        // explicit X_τ contribution is zero.
        let f = BinomialLogitDesignMotionFixture::new();
        let state = f.state();
        let x_tau = Array2::<f64>::zeros(f.x.raw_dim());
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            x_tau.clone(),
            f.s_tau_penalty.clone(),
            None,
            None,
        )
        .expect("penalty-only hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_tau_fd = f.fd_directional_gradient(&x_tau, &f.s_tau_penalty);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 penalty-only gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }

    #[test]
    fn binomial_logit_n30_joint_design_penalty_gradient_matches_fd() {
        // Joint direction: both X_τ ≠ 0 and S_τ ≠ 0 simultaneously.
        // This is the hardest case: the analytic gradient must correctly
        // combine the explicit penalty drift, the explicit design drift,
        // and the IFT Hessian-drift correction.
        let f = BinomialLogitDesignMotionFixture::new();
        let state = f.state();
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            f.s_tau_penalty.clone(),
            None,
            None,
        )
        .expect("joint design+penalty hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &f.rho, hyper)
            .expect("analytic directional gradient");
        let v_tau_fd = f.fd_directional_gradient(&f.x_tau_design, &f.s_tau_penalty);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 joint design+penalty gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }

    #[test]
    fn binomial_logit_n30_design_moving_hessian_matches_fd() {
        // Hessian-level validation with two τ-directions: one
        // penalty-only and one design-moving.  The ττ Hessian block is
        // the most sensitive test of the IFT correction because errors
        // in the correction accumulate quadratically in the trace term.
        let f = BinomialLogitDesignMotionFixture::new();
        let x_tau_0 = Array2::<f64>::zeros(f.x.raw_dim());
        let s_tau_0 = f.s_tau_penalty.clone();
        let x_tau_1 = f.x_tau_design.clone();
        let s_tau_1 = Array2::<f64>::zeros((5, 5));

        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(0, x_tau_0.clone(), s_tau_0.clone(), None, None)
                .expect("penalty-only direction"),
            DirectionalHyperParam::single_penalty(0, x_tau_1.clone(), s_tau_1.clone(), None, None)
                .expect("design-moving direction"),
        ];

        let state = f.state();
        let mut theta = Array1::<f64>::zeros(f.rho.len() + hyper_dirs.len());
        theta.slice_mut(s![..f.rho.len()]).assign(&f.rho);
        let (_, _, h_full) =
            compute_joint_hypercostgradienthessian(&state, &theta, f.rho.len(), &hyper_dirs)
                .expect("joint cost+gradient+hessian");
        let h_tt_analytic = h_full.slice(s![f.rho.len().., f.rho.len()..]).to_owned();

        let x_tau_mats = [&x_tau_0, &x_tau_1];
        let s_tau_mats = [&s_tau_0, &s_tau_1];
        let n_dirs = hyper_dirs.len();
        let mut h_tt_fd = Array2::<f64>::zeros((n_dirs, n_dirs));
        let eps = 1e-5;
        for j in 0..n_dirs {
            let (state_plus, state_minus) = f.state_perturbed(x_tau_mats[j], s_tau_mats[j], eps);
            for i in 0..n_dirs {
                let g_plus =
                    single_directional_tau_gradient(&state_plus, &f.rho, hyper_dirs[i].clone())
                        .expect("g+ for FD");
                let g_minus =
                    single_directional_tau_gradient(&state_minus, &f.rho, hyper_dirs[i].clone())
                        .expect("g- for FD");
                h_tt_fd[[i, j]] = (g_plus - g_minus) / (2.0 * eps);
            }
        }
        // Symmetrize FD Hessian.
        for i in 0..n_dirs {
            for j in 0..i {
                let avg = 0.5 * (h_tt_fd[[i, j]] + h_tt_fd[[j, i]]);
                h_tt_fd[[i, j]] = avg;
                h_tt_fd[[j, i]] = avg;
            }
        }

        let num = (&h_tt_analytic - &h_tt_fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = h_tt_fd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 1e-4,
            "Binomial-logit n=30 tau-tau Hessian mismatch: rel={rel:.3e}, \
             analytic={h_tt_analytic:?}, fd={h_tt_fd:?}"
        );
    }

    #[test]
    fn binomial_logit_n30_nonzero_rho_design_moving_gradient_matches_fd() {
        // Validate at a non-trivial smoothing parameter ρ = log(λ) = 1.5,
        // so the penalty term λS is scaled up and the balance between
        // likelihood and penalty is different from ρ=0.
        let f = BinomialLogitDesignMotionFixture::new();
        let rho = array![1.5];
        let s_tau = Array2::<f64>::zeros((5, 5));

        let state = f.state();
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            f.x_tau_design.clone(),
            s_tau.clone(),
            None,
            None,
        )
        .expect("design-moving hyper direction");

        let v_tau_analytic = single_directional_tau_gradient(&state, &rho, hyper)
            .expect("analytic directional gradient");

        // FD at the shifted ρ: perturb X, re-solve inner, evaluate cost.
        let h = 2e-5;
        let (state_plus, state_minus) = f.state_perturbed(&f.x_tau_design, &s_tau, h);
        let v_plus = state_plus.compute_cost(&rho).expect("cost+");
        let v_minus = state_minus.compute_cost(&rho).expect("cost-");
        let v_tau_fd = (v_plus - v_minus) / (2.0 * h);

        let v_rel = (v_tau_analytic - v_tau_fd).abs() / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 1e-3,
            "Binomial-logit n=30 rho=1.5 design-moving gradient mismatch: rel={v_rel:.3e}, \
             analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum RemlGeometry {
    DenseSpectral,
    SparseExactSpd,
}

trait PenalizedGeometry {
    fn backend_kind(&self) -> GeometryBackendKind;
}

#[derive(Clone)]
enum DerivativeMatrixStorage {
    Dense(Array2<f64>),
    Embedded(EmbeddedDerivativeMatrix),
    Implicit(ImplicitDerivativeOp),
}

/// Which derivative level the implicit operator should compute.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ImplicitDerivLevel {
    /// ∂X/∂ψ_d
    First(usize),
    /// ∂²X/∂ψ_d²
    SecondDiag(usize),
    /// ∂²X/∂ψ_d∂ψ_e
    SecondCross(usize, usize),
}

/// Lazy implicit operator storage: delegates matvecs to the
/// `ImplicitDesignPsiDerivative` and materializes dense form only on demand.
#[derive(Clone)]
struct ImplicitDerivativeOp {
    operator: std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
    level: ImplicitDerivLevel,
    global_range: Range<usize>,
    total_dim: usize,
    /// Cached dense materialization (lazy, populated on first call to ops that need the full matrix).
    cached_dense: std::sync::Arc<std::sync::OnceLock<Array2<f64>>>,
}

impl ImplicitDerivativeOp {
    fn materialize_local(&self) -> Array2<f64> {
        match self.level {
            ImplicitDerivLevel::First(axis) => self.operator.materialize_first(axis).expect(
                "radial scalar evaluation failed during implicit derivative materialization",
            ),
            ImplicitDerivLevel::SecondDiag(axis) => {
                self.operator.materialize_second_diag(axis).expect(
                    "radial scalar evaluation failed during implicit derivative materialization",
                )
            }
            ImplicitDerivLevel::SecondCross(d, e) => {
                self.operator.materialize_second_cross(d, e).expect(
                    "radial scalar evaluation failed during implicit derivative materialization",
                )
            }
        }
    }

    fn materialize_dense(&self) -> &Array2<f64> {
        self.cached_dense.get_or_init(|| {
            let local = self.materialize_local();
            let mut out = Array2::<f64>::zeros((local.nrows(), self.total_dim));
            out.slice_mut(s![.., self.global_range.clone()])
                .assign(&local);
            out
        })
    }

    fn nrows(&self) -> usize {
        self.operator.n_data()
    }

    fn ncols(&self) -> usize {
        self.total_dim
    }

    fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        let local = match self.level {
            ImplicitDerivLevel::First(axis) => self
                .operator
                .transpose_mul(axis, &v.view())
                .expect("radial scalar evaluation failed during implicit derivative transpose_mul"),
            ImplicitDerivLevel::SecondDiag(axis) => self
                .operator
                .transpose_mul_second_diag(axis, &v.view())
                .expect("radial scalar evaluation failed during implicit derivative transpose_mul"),
            ImplicitDerivLevel::SecondCross(d, e) => self
                .operator
                .transpose_mul_second_cross(d, e, &v.view())
                .expect("radial scalar evaluation failed during implicit derivative transpose_mul"),
        };
        let mut out = Array1::<f64>::zeros(self.total_dim);
        out.slice_mut(s![self.global_range.clone()]).assign(&local);
        out
    }

    fn forward_mul(&self, u: &Array1<f64>) -> Array1<f64> {
        let u_local = u.slice(s![self.global_range.clone()]).to_owned();
        match self.level {
            ImplicitDerivLevel::First(axis) => self
                .operator
                .forward_mul(axis, &u_local.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
            ImplicitDerivLevel::SecondDiag(axis) => self
                .operator
                .forward_mul_second_diag(axis, &u_local.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
            ImplicitDerivLevel::SecondCross(d, e) => self
                .operator
                .forward_mul_second_cross(d, e, &u_local.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
        }
    }
}

#[derive(Clone)]
struct EmbeddedDerivativeMatrix {
    local: Array2<f64>,
    global_range: Range<usize>,
    total_dim: usize,
}

impl EmbeddedDerivativeMatrix {
    fn new(local: Array2<f64>, global_range: Range<usize>, total_dim: usize) -> Self {
        Self {
            local,
            global_range,
            total_dim,
        }
    }
}

#[derive(Clone)]
pub(crate) struct HyperDesignDerivative {
    storage: DerivativeMatrixStorage,
}

impl HyperDesignDerivative {
    pub(crate) fn from_embedded(
        local: Array2<f64>,
        global_range: Range<usize>,
        total_cols: usize,
    ) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Embedded(EmbeddedDerivativeMatrix::new(
                local,
                global_range,
                total_cols,
            )),
        }
    }

    pub(crate) fn from_implicit(
        operator: std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
        level: ImplicitDerivLevel,
        global_range: Range<usize>,
        total_cols: usize,
    ) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Implicit(ImplicitDerivativeOp {
                operator,
                level,
                global_range,
                total_dim: total_cols,
                cached_dense: std::sync::Arc::new(std::sync::OnceLock::new()),
            }),
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => dense.nrows(),
            DerivativeMatrixStorage::Embedded(embedded) => embedded.local.nrows(),
            DerivativeMatrixStorage::Implicit(op) => op.nrows(),
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => dense.ncols(),
            DerivativeMatrixStorage::Embedded(embedded) => embedded.total_dim,
            DerivativeMatrixStorage::Implicit(op) => op.ncols(),
        }
    }

    pub(crate) fn uses_implicit_storage(&self) -> bool {
        matches!(self.storage, DerivativeMatrixStorage::Implicit(..))
    }

    pub(crate) fn materialize(&self) -> Array2<f64> {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => dense.clone(),
            DerivativeMatrixStorage::Embedded(embedded) => {
                let mut dense = Array2::<f64>::zeros((embedded.local.nrows(), embedded.total_dim));
                dense
                    .slice_mut(s![.., embedded.global_range.clone()])
                    .assign(&embedded.local);
                dense
            }
            DerivativeMatrixStorage::Implicit(op) => op.materialize_dense().clone(),
        }
    }

    pub(crate) fn any_nonzero(&self) -> bool {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => dense.iter().any(|v| *v != 0.0),
            DerivativeMatrixStorage::Embedded(embedded) => embedded.local.iter().any(|v| *v != 0.0),
            DerivativeMatrixStorage::Implicit(..) => true,
        }
    }

    pub(crate) fn forward_mul_original(
        &self,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => {
                if dense.ncols() != u.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense hyper design derivative forward_mul_original width mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        u.len()
                    )));
                }
                Ok(dense.dot(u))
            }
            DerivativeMatrixStorage::Embedded(embedded) => {
                if embedded.total_dim != u.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "embedded hyper design derivative forward_mul_original width mismatch: total_dim={}, vector={}",
                        embedded.total_dim,
                        u.len()
                    )));
                }
                let u_local = u.slice(s![embedded.global_range.clone()]).to_owned();
                Ok(embedded.local.dot(&u_local))
            }
            DerivativeMatrixStorage::Implicit(op) => {
                if op.ncols() != u.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "implicit hyper design derivative forward_mul_original width mismatch: operator_cols={}, vector={}",
                        op.ncols(),
                        u.len()
                    )));
                }
                Ok(op.forward_mul(u))
            }
        }
    }

    pub(crate) fn transpose_mul_original(
        &self,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => {
                if dense.nrows() != v.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense hyper design derivative transpose_mul_original height mismatch: matrix={}x{}, vector={}",
                        dense.nrows(),
                        dense.ncols(),
                        v.len()
                    )));
                }
                Ok(dense.t().dot(v))
            }
            DerivativeMatrixStorage::Embedded(embedded) => {
                if embedded.local.nrows() != v.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "embedded hyper design derivative transpose_mul_original height mismatch: local_rows={}, vector={}",
                        embedded.local.nrows(),
                        v.len()
                    )));
                }
                let mut out = Array1::<f64>::zeros(embedded.total_dim);
                let pulled = embedded.local.t().dot(v);
                out.slice_mut(s![embedded.global_range.clone()])
                    .assign(&pulled);
                Ok(out)
            }
            DerivativeMatrixStorage::Implicit(op) => {
                if op.nrows() != v.len() {
                    return Err(EstimationError::InvalidInput(format!(
                        "implicit hyper design derivative transpose_mul_original height mismatch: operator_rows={}, vector={}",
                        op.nrows(),
                        v.len()
                    )));
                }
                Ok(op.transpose_mul(v))
            }
        }
    }

    pub(crate) fn transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        match &self.storage {
            DerivativeMatrixStorage::Embedded(embedded) => {
                if embedded.total_dim != qs.nrows() {
                    return Err(EstimationError::InvalidInput(format!(
                        "embedded design derivative width mismatch: total_cols={}, qs rows={}",
                        embedded.total_dim,
                        qs.nrows()
                    )));
                }
                let qs_local = qs.slice(s![embedded.global_range.clone(), ..]);
                let mut transformed = embedded.local.dot(&qs_local);
                if let Some(z) = free_basis_opt {
                    transformed = transformed.dot(z);
                }
                Ok(transformed)
            }
            DerivativeMatrixStorage::Dense(dense) => {
                Ok(crate::matrix::DenseRightProductView::new(dense)
                    .with_factor(qs)
                    .with_optional_factor(free_basis_opt)
                    .materialize())
            }
            DerivativeMatrixStorage::Implicit(op) => {
                let dense = op.materialize_dense();
                Ok(crate::matrix::DenseRightProductView::new(dense)
                    .with_factor(qs)
                    .with_optional_factor(free_basis_opt)
                    .materialize())
            }
        }
    }

    pub(crate) fn transformed_forward_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        u: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match &self.storage {
            DerivativeMatrixStorage::Implicit(op) => {
                let mut right = if let Some(z) = free_basis_opt {
                    z.dot(u)
                } else {
                    u.clone()
                };
                right = qs.dot(&right);
                Ok(op.forward_mul(&right))
            }
            _ => Ok(self.transformed(qs, free_basis_opt)?.dot(u)),
        }
    }

    pub(crate) fn transformed_transpose_mul(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
        v: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        match &self.storage {
            DerivativeMatrixStorage::Implicit(op) => {
                let mut pulled = qs.t().dot(&op.transpose_mul(v));
                if let Some(z) = free_basis_opt {
                    pulled = z.t().dot(&pulled);
                }
                Ok(pulled)
            }
            _ => Ok(self.transformed(qs, free_basis_opt)?.t().dot(v)),
        }
    }

    /// If this derivative uses implicit storage at the first-derivative level,
    /// return the shared implicit operator and the axis index.
    ///
    /// Returns `None` for dense/embedded storage or for second-derivative levels.
    pub(crate) fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        match &self.storage {
            DerivativeMatrixStorage::Implicit(op) => match op.level {
                ImplicitDerivLevel::First(axis) => Some((op.operator.clone(), axis)),
                _ => None,
            },
            _ => None,
        }
    }
}

impl From<Array2<f64>> for HyperDesignDerivative {
    fn from(value: Array2<f64>) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Dense(value),
        }
    }
}

#[derive(Clone)]
pub(crate) struct HyperPenaltyDerivative {
    storage: DerivativeMatrixStorage,
}

impl HyperPenaltyDerivative {
    pub(crate) fn from_embedded(
        local: Array2<f64>,
        global_range: Range<usize>,
        total_dim: usize,
    ) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Embedded(EmbeddedDerivativeMatrix::new(
                local,
                global_range,
                total_dim,
            )),
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => dense.nrows(),
            DerivativeMatrixStorage::Embedded(embedded) => embedded.total_dim,
            DerivativeMatrixStorage::Implicit(op) => op.nrows(),
        }
    }

    pub(crate) fn ncols(&self) -> usize {
        self.nrows()
    }

    pub(crate) fn scaled_materialize(&self, amp: f64) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.nrows(), self.ncols()));
        self.scaled_add_to(&mut out, amp)
            .expect("scaled materialize uses matching target shape");
        out
    }

    pub(crate) fn transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        match &self.storage {
            DerivativeMatrixStorage::Embedded(embedded) => {
                if embedded.total_dim != qs.nrows() {
                    return Err(EstimationError::InvalidInput(format!(
                        "embedded penalty derivative width mismatch: total_dim={}, qs rows={}",
                        embedded.total_dim,
                        qs.nrows()
                    )));
                }
                let qs_local = qs.slice(s![embedded.global_range.clone(), ..]);
                let mut transformed = qs_local.t().dot(&embedded.local).dot(&qs_local);
                if let Some(z) = free_basis_opt {
                    transformed = z.t().dot(&transformed).dot(z);
                }
                Ok(transformed)
            }
            DerivativeMatrixStorage::Dense(dense) => {
                let mut transformed = qs.t().dot(dense).dot(qs);
                if let Some(z) = free_basis_opt {
                    transformed = z.t().dot(&transformed).dot(z);
                }
                Ok(transformed)
            }
            DerivativeMatrixStorage::Implicit(op) => {
                let dense = op.materialize_dense();
                let mut transformed = qs.t().dot(dense).dot(qs);
                if let Some(z) = free_basis_opt {
                    transformed = z.t().dot(&transformed).dot(z);
                }
                Ok(transformed)
            }
        }
    }

    pub(crate) fn scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        match &self.storage {
            DerivativeMatrixStorage::Dense(dense) => {
                if target.raw_dim() != dense.raw_dim() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense hyper penalty derivative shape mismatch: target={}x{}, matrix={}x{}",
                        target.nrows(),
                        target.ncols(),
                        dense.nrows(),
                        dense.ncols()
                    )));
                }
                target.scaled_add(amp, dense);
            }
            DerivativeMatrixStorage::Embedded(embedded) => {
                if target.nrows() != embedded.total_dim || target.ncols() != embedded.total_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "embedded hyper penalty derivative shape mismatch: target={}x{}, expected {}x{}",
                        target.nrows(),
                        target.ncols(),
                        embedded.total_dim,
                        embedded.total_dim
                    )));
                }
                target
                    .slice_mut(s![
                        embedded.global_range.clone(),
                        embedded.global_range.clone()
                    ])
                    .scaled_add(amp, &embedded.local);
            }
            DerivativeMatrixStorage::Implicit(op) => {
                let dense = op.materialize_dense();
                if target.raw_dim() != dense.raw_dim() {
                    return Err(EstimationError::InvalidInput(format!(
                        "implicit hyper penalty derivative shape mismatch: target={}x{}, matrix={}x{}",
                        target.nrows(),
                        target.ncols(),
                        dense.nrows(),
                        dense.ncols()
                    )));
                }
                target.scaled_add(amp, dense);
            }
        }
        Ok(())
    }
}

impl From<Array2<f64>> for HyperPenaltyDerivative {
    fn from(value: Array2<f64>) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Dense(value),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PenaltyDerivativeComponent {
    pub(crate) penalty_index: usize,
    pub(crate) matrix: HyperPenaltyDerivative,
}

#[derive(Clone)]
pub(crate) struct DirectionalHyperParam {
    pub(crate) x_tau_original: HyperDesignDerivative,
    // Canonical penalty representation: every tau direction is decomposed into
    // base-penalty derivatives. There is no separate "assembled total" path.
    penalty_first_components: Vec<PenaltyDerivativeComponent>,
    // Optional pairwise second hyper-derivatives against all tau directions.
    // If provided, each vector must have length psi_dim and hold an optional
    // X_{tau_i,tau_j} entry in original coordinates.
    pub(crate) x_tau_tau_original: Option<Vec<Option<HyperDesignDerivative>>>,
    // Pairwise second derivatives are stored in the same canonical base-penalty
    // decomposition as the first derivatives.
    penaltysecond_components: Option<Vec<Option<Vec<PenaltyDerivativeComponent>>>>,
    penaltysecond_component_provider: Option<
        std::sync::Arc<
            dyn Fn(usize) -> Result<Option<Vec<PenaltyDerivativeComponent>>, EstimationError>
                + Send
                + Sync
                + 'static,
        >,
    >,
    penaltysecond_partner_indices: Option<std::sync::Arc<[usize]>>,
    /// Whether this coordinate is penalty-like (B_i = ∂H/∂τ_i is PSD).
    /// True for τ (penalty scaling) coordinates; false for ψ (design-moving,
    /// anisotropic length-scale) coordinates. Controls EFS eligibility.
    pub(crate) is_penalty_like: bool,
}

impl DirectionalHyperParam {
    fn canonicalize_penalty_components(
        components: Vec<(usize, HyperPenaltyDerivative)>,
    ) -> Result<Vec<PenaltyDerivativeComponent>, EstimationError> {
        let mut out: Vec<PenaltyDerivativeComponent> = Vec::with_capacity(components.len());
        for (penalty_index, matrix) in components {
            if out.iter().any(|c| c.penalty_index == penalty_index) {
                return Err(EstimationError::InvalidInput(format!(
                    "duplicate penalty derivative component for penalty {}",
                    penalty_index
                )));
            }
            out.push(PenaltyDerivativeComponent {
                penalty_index,
                matrix,
            });
        }
        Ok(out)
    }

    #[cfg(test)]
    pub(crate) fn new(
        x_tau_original: Array2<f64>,
        penalty_first_components: Vec<(usize, Array2<f64>)>,
        x_tau_tau_original: Option<Vec<Option<Array2<f64>>>>,
        penaltysecond_components: Option<Vec<Option<Vec<(usize, Array2<f64>)>>>>,
    ) -> Result<Self, EstimationError> {
        let x_tau_tau_original = x_tau_tau_original.map(|rows| {
            rows.into_iter()
                .map(|entry| entry.map(HyperDesignDerivative::from))
                .collect::<Vec<_>>()
        });
        let penalty_first_components = penalty_first_components
            .into_iter()
            .map(|(idx, matrix)| (idx, HyperPenaltyDerivative::from(matrix)))
            .collect();
        let penaltysecond_components = penaltysecond_components.map(|rows| {
            rows.into_iter()
                .map(|row| {
                    row.map(|components| {
                        components
                            .into_iter()
                            .map(|(idx, matrix)| (idx, HyperPenaltyDerivative::from(matrix)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
        });
        Self::new_compact(
            HyperDesignDerivative::from(x_tau_original),
            penalty_first_components,
            x_tau_tau_original,
            penaltysecond_components,
        )
    }

    pub(crate) fn new_compact(
        x_tau_original: HyperDesignDerivative,
        penalty_first_components: Vec<(usize, HyperPenaltyDerivative)>,
        x_tau_tau_original: Option<Vec<Option<HyperDesignDerivative>>>,
        penaltysecond_components: Option<Vec<Option<Vec<(usize, HyperPenaltyDerivative)>>>>,
    ) -> Result<Self, EstimationError> {
        let penalty_first_components =
            Self::canonicalize_penalty_components(penalty_first_components)?;
        let penaltysecond_components = match penaltysecond_components {
            Some(rows) => {
                let mut out = Vec::with_capacity(rows.len());
                for row in rows {
                    out.push(match row {
                        Some(components) => {
                            Some(Self::canonicalize_penalty_components(components)?)
                        }
                        None => None,
                    });
                }
                Some(out)
            }
            None => None,
        };
        Ok(Self {
            x_tau_original,
            penalty_first_components,
            x_tau_tau_original,
            penaltysecond_components,
            penaltysecond_component_provider: None,
            penaltysecond_partner_indices: None,
            is_penalty_like: true, // default: τ coords are penalty-like
        })
    }

    /// Mark this coordinate as non-penalty-like (design-moving).
    /// EFS will skip it; use Newton/BFGS for these coordinates.
    pub(crate) fn not_penalty_like(mut self) -> Self {
        self.is_penalty_like = false;
        self
    }

    pub(crate) fn with_penaltysecond_component_provider(
        mut self,
        provider: std::sync::Arc<
            dyn Fn(usize) -> Result<Option<Vec<PenaltyDerivativeComponent>>, EstimationError>
                + Send
                + Sync
                + 'static,
        >,
    ) -> Self {
        self.penaltysecond_component_provider = Some(provider);
        self
    }

    pub(crate) fn with_penaltysecond_partner_indices(mut self, partners: Vec<usize>) -> Self {
        self.penaltysecond_partner_indices = Some(std::sync::Arc::from(partners));
        self
    }

    pub(crate) fn x_tau_dense(&self) -> Array2<f64> {
        self.x_tau_original.materialize()
    }

    pub(crate) fn transformed_x_tau(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        self.x_tau_original.transformed(qs, free_basis_opt)
    }

    pub(crate) fn x_tau_tau_entry_at(&self, j: usize) -> Option<HyperDesignDerivative> {
        self.x_tau_tau_original
            .as_ref()
            .and_then(|rows| rows.get(j))
            .and_then(|entry| entry.clone())
    }

    #[cfg(test)]
    pub(crate) fn single_penalty(
        penalty_index: usize,
        x_tau_original: Array2<f64>,
        s_tau_original: Array2<f64>,
        x_tau_tau_original: Option<Vec<Option<Array2<f64>>>>,
        s_tau_tau_original: Option<Vec<Option<Array2<f64>>>>,
    ) -> Result<Self, EstimationError> {
        let penaltysecond_components = s_tau_tau_original.map(|rows| {
            rows.into_iter()
                .map(|mat| mat.map(|mat| vec![(penalty_index, mat)]))
                .collect::<Vec<_>>()
        });
        Self::new(
            x_tau_original,
            vec![(penalty_index, s_tau_original)],
            x_tau_tau_original,
            penaltysecond_components,
        )
    }

    /// Whether this coordinate's design derivative uses implicit storage at the
    /// first-derivative level.
    pub(crate) fn has_implicit_operator(&self) -> bool {
        self.x_tau_original.implicit_first_axis_info().is_some()
    }

    pub(crate) fn has_implicit_multidim_duchon(&self) -> bool {
        self.implicit_first_axis_info()
            .is_some_and(|(op, _)| op.n_axes() > 1 && op.is_duchon_family())
    }

    /// Extract the implicit design derivative operator and axis, if available.
    pub(crate) fn implicit_first_axis_info(
        &self,
    ) -> Option<(
        std::sync::Arc<crate::terms::basis::ImplicitDesignPsiDerivative>,
        usize,
    )> {
        self.x_tau_original.implicit_first_axis_info()
    }

    pub(crate) fn penalty_first_components(&self) -> &[PenaltyDerivativeComponent] {
        &self.penalty_first_components
    }

    pub(crate) fn penalty_total_at(
        &self,
        rho: &Array1<f64>,
        p: usize,
    ) -> Result<Array2<f64>, EstimationError> {
        let mut out = Array2::<f64>::zeros((p, p));
        for component in &self.penalty_first_components {
            if component.matrix.nrows() != p || component.matrix.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "S_tau shape mismatch for penalty {}: expected {}x{}, got {}x{}",
                    component.penalty_index,
                    p,
                    p,
                    component.matrix.nrows(),
                    component.matrix.ncols()
                )));
            }
            if component.penalty_index >= rho.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "penalty_index {} out of bounds for rho dimension {}",
                    component.penalty_index,
                    rho.len()
                )));
            }
            component
                .matrix
                .scaled_add_to(&mut out, rho[component.penalty_index].exp())?;
        }
        Ok(out)
    }

    pub(crate) fn penaltysecond_components_for(
        &self,
        j: usize,
    ) -> Result<Option<Vec<PenaltyDerivativeComponent>>, EstimationError> {
        if let Some(components) = self
            .penaltysecond_components
            .as_ref()
            .and_then(|rows| rows.get(j))
            .and_then(|row| row.clone())
        {
            return Ok(Some(components));
        }
        if let Some(provider) = self.penaltysecond_component_provider.as_ref() {
            return provider(j);
        }
        Ok(None)
    }

    pub(crate) fn penaltysecond_componentrows(
        &self,
    ) -> Option<&[Option<Vec<PenaltyDerivativeComponent>>]> {
        self.penaltysecond_components.as_deref()
    }

    pub(crate) fn penalty_first_component_count(&self) -> usize {
        self.penalty_first_components.len()
    }

    pub(crate) fn has_penaltysecond_pair_at(&self, j: usize) -> bool {
        self.penaltysecond_components
            .as_ref()
            .and_then(|rows| rows.get(j))
            .is_some_and(Option::is_some)
            || self
                .penaltysecond_partner_indices
                .as_ref()
                .is_some_and(|partners| partners.contains(&j))
    }
}

#[derive(Clone, Debug)]
struct SparseRemlDecision {
    geometry: RemlGeometry,
    reason: &'static str,
    p: usize,
    nnz_x: usize,
    nnz_h_upper_est: Option<usize>,
    density_h_upper_est: Option<f64>,
}

#[derive(Clone)]
struct SparseExactEvalData {
    factor: Arc<SparseExactFactor>,
    takahashi: Option<Arc<crate::linalg::sparse_exact::TakahashiInverse>>,
    logdet_h: f64,
    logdet_s_pos: f64,
    det1_values: Arc<Array1<f64>>,
}

#[derive(Clone)]
pub(crate) struct FirthDenseOperator {
    // Exact Firth/Jeffreys objects on the identifiable subspace.
    //
    // Let X in R^{n×p} potentially be rank-deficient with rank r.
    // With optional fixed observation weights a_i >= 0 we define A = diag(a),
    // choose an orthonormal coefficient-space basis Q for the identifiable
    // subspace of A^{1/2} X, and set:
    //   X_r := A^{1/2} X Q          (A = I when no fixed observation weights),
    //   W   := diag(w), with w_i = mu_i (1 - mu_i), 0 < w_i <= 1/4 for finite logit eta,
    //   I_r := X_rᵀ W X_r,
    //   S_r := X_rᵀ X_r.
    //
    // Firth term is represented as:
    //   Phi(beta) = 0.5 log |I_r(beta)| - 0.5 log |S_r|,
    // which is exactly
    //   0.5 log |Uᵀ W U|
    // for the canonical orthonormalized identifiable design
    //   U = X_r S_r^{-1/2}.
    // This removes the raw-basis term from explicit reduced designs while
    // keeping the same identifiable-subspace hat matrix and beta derivatives,
    // because S_r is fixed with respect to beta.
    //
    // Mapping back to the full p-space uses:
    //   I_+^dagger = Q I_r^{-1} Qᵀ.
    //
    // We store reduced-space factors so all derivatives can be evaluated exactly
    // without materializing dense n×n matrices M = X K Xᵀ or P = M⊙M.
    x_dense: Array2<f64>,
    x_dense_t: Array2<f64>,
    // Orthonormal coefficient-space basis for the identifiable subspace,
    // built from the retained eigenspace of (A^{1/2} X)ᵀ(A^{1/2} X).
    q_basis: Array2<f64>,
    // Reduced identifiable design. With fixed observation weights a_i this is
    // diag(sqrt(a_i)) X Q; otherwise it is X Q.
    x_reduced: Array2<f64>,
    // Reduced design used for M = Z K_r Zᵀ. Name kept for compatibility with
    // existing callsites; Z equals x_reduced for the current implementation.
    z_reduced: Array2<f64>,
    // Optional fixed case-weight square roots used when the Jeffreys/Firth
    // operator is formed from Xᵀ diag(case_weight ⊙ w(η)) X rather than
    // Xᵀ diag(w(η)) X. The exact directional tau derivatives must project and
    // row-scale with the same weights so the reduced Fisher, hat diagonals,
    // and tau kernels all live on one consistent identifiable subspace.
    observation_weight_sqrt: Option<Array1<f64>>,
    // I_r^{-1}
    k_reduced: Array2<f64>,
    // diag(S_r^{-1}) with S_r = X_rᵀ X_r. In the current canonical reduced
    // basis this completely characterizes the metric inverse, because Q
    // diagonalizes the design Gram. It is used to remove the reduced-coordinate
    // basis term from Phi_tau when the design moves.
    x_metric_reduced_inv_diag: Array1<f64>,
    // 0.5 (log|I_r| - log|S_r|) at the current eta.
    half_log_det: f64,
    // h = diag(M), M = X_r K_r X_r'
    h_diag: Array1<f64>,
    // Logistic Fisher-weight eta-derivatives: w', w'', w''', w'''' as n-vectors.
    w: Array1<f64>,
    w1: Array1<f64>,
    w2: Array1<f64>,
    w3: Array1<f64>,
    w4: Array1<f64>,
    // B = diag(w') X used in D Hphi and D^2 Hphi contractions.
    b_base: Array2<f64>,
    // Cached invariant contraction P*B where P = (X_r K_r X_r') ⊙ (X_r K_r X_r').
    // This avoids recomputing the same O(n r^2 p) block in every directional call.
    p_b_base: Array2<f64>,
}

#[derive(Clone)]
pub(crate) struct FirthDirection {
    deta: Array1<f64>,
    g_u_reduced: Array2<f64>,
    a_u_reduced: Array2<f64>,
    dh: Array1<f64>,
    // B_u = diag(w'' ⊙ δη_u) X is represented by the row-scaling vector only.
    b_uvec: Array1<f64>,
}

#[derive(Clone)]
pub(crate) struct FirthTauPartialKernel {
    dotw1: Array1<f64>,
    dotw2: Array1<f64>,
    dot_h_partial: Array1<f64>,
    // Reduced design drift X_{tau,r} = X_tau Q used in exact design-moving
    // Hadamard-Gram contractions.
    x_tau_reduced: Array2<f64>,
    // Reduced Fisher inverse drift:
    //   dot(K_r) = -K_r dot(I_r) K_r
    // where dot(I_r) includes explicit X_tau and weight drift at beta-fixed.
    dot_k_reduced: Array2<f64>,
}

#[derive(Clone)]
pub(crate) struct FirthTauExactKernel {
    gphi_tau: Array1<f64>,
    phi_tau_partial: f64,
    tau_kernel: Option<FirthTauPartialKernel>,
}

/// Pair-level (τ_i × τ_j) exact Firth bundle at fixed β.
///
/// Mirrors `FirthTauExactKernel` but for the 2nd-order cross
/// derivatives:
///   Phi_{τ_i τ_j}|β  (scalar, `phi_tau_tau_partial`)
///   (gphi)_{τ_i τ_j}|β (p-vector, `gphi_tau_tau`)
///
/// Carries an optional `tau_tau_kernel` so pair callbacks can chain
/// into Primitive A (`hphi_tau_tau_partial_apply`) for the operator-
/// valued Hessian 2nd drift without recomputing shared reduced Grams.
#[derive(Clone)]
pub(crate) struct FirthTauTauExactKernel {
    pub(super) phi_tau_tau_partial: f64,
    pub(super) gphi_tau_tau: Array1<f64>,
    pub(super) tau_tau_kernel: Option<FirthTauTauPartialKernel>,
}

/// Prepared state for `∂²H_φ/∂τ_i ∂τ_j |_β` (Primitive A).
///
/// Carries both τ-direction reduced designs, their η̇ vectors, and the
/// reduced-coordinate drifts (İ, K̇, ḣ) for i and j so the apply step can
/// form M̈_{ij}, K̈_{ij}, ḧ_{ij}, Γ̈_{ij}, and B̈_{ij} matrix-free.  Fields
/// are filled in by 13b; kept with a neutral internal shape so downstream
/// pair callbacks can hold the kernel across the pair dispatch.
///
/// Wired in by the Primitive A FD test and Task #20 pair-callback
/// integration.
#[derive(Clone, Default)]
pub(crate) struct FirthTauTauPartialKernel {
    pub(super) x_tau_i_reduced: Array2<f64>,
    pub(super) x_tau_j_reduced: Array2<f64>,
    pub(super) deta_i_partial: Array1<f64>,
    pub(super) deta_j_partial: Array1<f64>,
    pub(super) dot_h_i_partial: Array1<f64>,
    pub(super) dot_h_j_partial: Array1<f64>,
    pub(super) dot_k_i_reduced: Array2<f64>,
    pub(super) dot_k_j_reduced: Array2<f64>,
    pub(super) dot_i_i_partial: Array2<f64>,
    pub(super) dot_i_j_partial: Array2<f64>,
    pub(super) x_tau_tau_reduced: Option<Array2<f64>>,
    pub(super) deta_ij_partial: Option<Array1<f64>>,
}

/// Prepared state for `D_β((H_φ)_τ|_β)[v]` (Primitive B).
///
/// Carries the τ-kernel pieces (x_tau_reduced, İ, K̇, ḣ, w-chain), the
/// β-direction reduced quantities (δη_v, I'_v, A_v, dh_v, w-chain
/// derivatives), and the mixed β-τ pieces (D_β(İ_τ)[v], D_β(K̇_τ)[v],
/// D_β(ḣ_τ)[v], δη_{τ,v}) so the apply step collapses to the 9-term
/// β-τ expansion without recomputing shared reduced Grams.  Fields are
/// filled in by 13c.
///
/// Allow(dead_code): scaffold until 13c populates the fields inside
/// `d_beta_hphi_tau_partial_prepare_from_partials` and 13d wires the
/// kernel into the `fixed_drift_deriv` closure in
/// `build_tau_hyper_coords`.
#[allow(dead_code)]
#[derive(Clone, Default)]
pub(crate) struct FirthTauBetaPartialKernel {
    pub(super) x_tau_reduced: Array2<f64>,
    pub(super) deta_partial: Array1<f64>,
    pub(super) dot_h_partial: Array1<f64>,
    pub(super) dot_i_partial: Array2<f64>,
    pub(super) dot_k_reduced: Array2<f64>,
    pub(super) deta_v: Array1<f64>,
    pub(super) deta_tau_v: Array1<f64>,
    pub(super) g_v_reduced: Array2<f64>,
    pub(super) a_v_reduced: Array2<f64>,
    pub(super) dh_v: Array1<f64>,
    pub(super) b_vvec: Array1<f64>,
    pub(super) d_beta_dot_i: Array2<f64>,
    pub(super) d_beta_dot_k: Array2<f64>,
    pub(super) d_beta_dot_h: Array1<f64>,
}

/// Holds the state for the outer REML optimization and supplies cost and
/// gradient evaluations to the `opt` optimizer.
///
/// The `cache` field uses `RefCell` to enable interior mutability. This is a crucial
/// performance optimization. The `cost_andgrad` closure required by the BFGS
/// optimizer takes an immutable reference `&self`. However, we want to cache the
/// results of the expensive P-IRLS computation to avoid re-calculating the fit
/// for the same `rho` vector, which can happen during the line search.
/// `RefCell` allows us to mutate the cache through a `&self` reference,
/// making this optimization possible while adhering to the optimizer's API.

#[derive(Clone)]
pub(crate) struct EvalShared {
    key: Option<Vec<u64>>,
    pub(crate) pirls_result: Arc<PirlsResult>,
    ridge_passport: RidgePassport,
    geometry: RemlGeometry,
    h_eff: Arc<Array2<f64>>,
    /// The exact H_total matrix used for LAML cost computation.
    /// For Firth: h_eff - hphi. For non-Firth: h_eff.
    h_total: Arc<Array2<f64>>,
    sparse_exact: Option<Arc<SparseExactEvalData>>,
    firth_dense_operator: Option<Arc<FirthDenseOperator>>,
    /// Cached FirthDenseOperator built from the original (non-reparameterized)
    /// design matrix, for use by the sparse evaluation path.
    firth_dense_operator_original: Option<Arc<FirthDenseOperator>>,
}

impl EvalShared {
    fn matches(&self, key: &Option<Vec<u64>>) -> bool {
        match (&self.key, key) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

impl PenalizedGeometry for EvalShared {
    fn backend_kind(&self) -> GeometryBackendKind {
        match self.geometry {
            RemlGeometry::DenseSpectral => GeometryBackendKind::DenseSpectral,
            RemlGeometry::SparseExactSpd => GeometryBackendKind::SparseExactSpd,
        }
    }
}

struct PirlsLruCache {
    map: HashMap<Vec<u64>, (Arc<PirlsResult>, u64)>,
    capacity: usize,
    clock: u64,
}

impl PirlsLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            capacity: capacity.max(1),
            clock: 0,
        }
    }

    fn get(&mut self, key: &Vec<u64>) -> Option<Arc<PirlsResult>> {
        if let Some(entry) = self.map.get_mut(key) {
            self.clock += 1;
            entry.1 = self.clock;
            Some(entry.0.clone())
        } else {
            None
        }
    }

    fn insert(&mut self, key: Vec<u64>, value: Arc<PirlsResult>) {
        self.clock += 1;
        if self.map.contains_key(&key) {
            self.map.insert(key, (value, self.clock));
            return;
        }

        while self.map.len() >= self.capacity {
            // Evict least-recently-used entry (lowest timestamp)
            if let Some(evict_key) = self
                .map
                .iter()
                .min_by_key(|(_, (_, ts))| *ts)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&evict_key);
            } else {
                break;
            }
        }

        self.map.insert(key, (value, self.clock));
    }

    fn clear(&mut self) {
        self.map.clear();
    }
}

/// Centralized cache/memoization owner for REML evaluations.
///
/// This keeps cache-key identity, bundle reuse, and invalidation policy out of
/// the math kernels so objective/derivative routines can stay algebra-focused.
struct EvalCacheManager {
    pirls_cache: RwLock<PirlsLruCache>,
    current_eval_bundle: RwLock<Option<EvalShared>>,
    current_outer_eval: RwLock<Option<(Vec<u64>, OuterEval)>>,
    pirls_cache_enabled: AtomicBool,
}

impl EvalCacheManager {
    fn new() -> Self {
        Self {
            pirls_cache: RwLock::new(PirlsLruCache::new(MAX_PIRLS_CACHE_ENTRIES)),
            current_eval_bundle: RwLock::new(None),
            current_outer_eval: RwLock::new(None),
            pirls_cache_enabled: AtomicBool::new(true),
        }
    }

    /// Creates a sanitized cache key from rho values.
    /// Returns None if any component is NaN, in which case caching is skipped.
    /// Maps -0.0 to 0.0 to ensure key stability.
    fn sanitized_rhokey(rho: &Array1<f64>) -> Option<Vec<u64>> {
        self::cache::sanitized_rhokey(rho)
    }

    fn cached_eval_bundle(&self, key: &Option<Vec<u64>>) -> Option<EvalShared> {
        self.current_eval_bundle
            .read()
            .unwrap()
            .as_ref()
            .filter(|bundle| bundle.matches(key))
            .cloned()
    }

    fn store_eval_bundle(&self, bundle: EvalShared) {
        *self.current_eval_bundle.write().unwrap() = Some(bundle);
    }

    fn cached_outer_eval(&self, key: &Option<Vec<u64>>) -> Option<OuterEval> {
        let key = key.as_ref()?;
        self.current_outer_eval
            .read()
            .unwrap()
            .as_ref()
            .filter(|(cached_key, _)| cached_key == key)
            .map(|(_, eval)| eval.clone())
    }

    fn store_outer_eval(&self, key: &Option<Vec<u64>>, eval: &OuterEval) {
        if let Some(key) = key.clone() {
            *self.current_outer_eval.write().unwrap() = Some((key, eval.clone()));
        }
    }

    fn invalidate_eval_bundle(&self) {
        self.current_eval_bundle.write().unwrap().take();
        self.current_outer_eval.write().unwrap().take();
    }

    fn clear_eval_and_factor_caches(&self) {
        self.invalidate_eval_bundle();
    }
}

/// Reusable scratch/runtime memory that should not be part of mathematical
/// state invariants.
struct RemlArena {
    cost_eval_count: RwLock<u64>,
    lastgradient_used_stochastic_fallback: AtomicBool,
}

impl RemlArena {
    fn new() -> Self {
        Self {
            cost_eval_count: RwLock::new(0),
            lastgradient_used_stochastic_fallback: AtomicBool::new(false),
        }
    }
}

pub(crate) struct RemlState<'a> {
    y: ArrayView1<'a, f64>,
    x: DesignMatrix,
    weights: ArrayView1<'a, f64>,
    offset: Array1<f64>,
    /// Canonicalized block-local penalties with pre-computed roots.
    /// This is the single canonical penalty representation — no full-width
    /// `rank × p` roots are stored separately.
    canonical_penalties: Arc<Vec<crate::construction::CanonicalPenalty>>,
    balanced_penalty_root: Array2<f64>,
    reparam_invariant: ReparamInvariant,
    sparse_penalty_blocks: Option<Arc<Vec<SparsePenaltyBlock>>>,
    p: usize,
    config: Arc<RemlConfig>,
    runtime_mixture_link_state: Option<crate::types::MixtureLinkState>,
    runtime_sas_link_state: Option<SasLinkState>,
    nullspace_dims: Vec<usize>,
    coefficient_lower_bounds: Option<Array1<f64>>,
    linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    /// Relative shrinkage floor for penalized block eigenvalues (rho-independent).
    penalty_shrinkage_floor: Option<f64>,

    cache_manager: EvalCacheManager,
    arena: RemlArena,
    pub(crate) warm_start_beta: RwLock<Option<Coefficients>>,
    warm_start_enabled: AtomicBool,
    outer_hessian_downgrade_logged: AtomicBool,
    pub(crate) screening_max_inner_iterations: Arc<AtomicUsize>,

    /// When set, the penalties have Kronecker (tensor-product) structure and
    /// the REML evaluator can use O(∏q_j) logdet instead of O(p³) eigendecomposition.
    /// Populated via `set_kronecker_penalty_system` after construction.
    pub(crate) kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis (marginal designs + penalties + dims).
    /// Used by P-IRLS for factored reparameterization.
    pub(crate) kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
}
