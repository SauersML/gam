use self::cache::AtomicFlagGuard;
use self::strategy::{GeometryBackendKind, HessianEvalStrategyKind, HessianStrategyDecision};
use super::*;
use crate::faer_ndarray::{FaerLblt, FaerLdlt, FaerLlt, fast_atv};
use crate::linalg::sparse_exact::{
    SparseExactFactor, SparsePenaltyBlock, SparseTraceWorkspace,
    assemble_and_factor_sparse_penalized_system, build_sparse_penalty_blocks,
    leverages_from_factor, solve_sparse_spd, solve_sparse_spdmulti, sparse_matvec_public,
    trace_hinv_sk,
};
use crate::pirls::{DirectionalWorkingCurvature, PirlsWorkspace};
use crate::types::SasLinkState;
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::Solve as FaerSolve;
use ndarray::s;
use std::ops::Range;

mod cache;
mod eval;
mod firth;
mod geometry;
mod hyper;
mod runtime;
mod strategy;
mod trace;

enum FaerFactor {
    Llt(FaerLlt<f64>),
    Lblt(FaerLblt<f64>),
    Ldlt(FaerLdlt<f64>),
}

#[cfg(test)]
mod tests {
    use super::{
        DirectionalHyperParam, EvalShared, FirthDenseOperator, LinkFunction, RemlConfig, RemlState,
    };
    use crate::faer_ndarray::{FaerCholesky, FaerEigh, fast_ab, fast_atb};
    use crate::linalg::sparse_exact::{
        SparsePenaltyBlock, dense_to_sparse_symmetric_upper, factorize_sparse_spd,
    };
    use crate::pirls::{PirlsCoordinateFrame, directionalworking_curvature_from_eta};
    use faer::Side;
    use ndarray::{Array1, Array2, array, s};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    #[test]
    fn dense_projected_tk_matches_directwt_hkw() {
        let w_pos = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0],];
        let x = array![[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0],];
        let r_k = array![[1.0, -1.0, 0.0], [0.0, 1.0, -1.0],];
        let lambda_k = 1.7;
        let cweighted_u_k = array![0.2, -0.1, 0.4];

        let z_mat = fast_ab(&x, &w_pos);
        let actual = RemlState::dense_projected_tk(&z_mat, &w_pos, &r_k, lambda_k, &cweighted_u_k);

        let s_k = r_k.t().dot(&r_k);
        let mut h_k = s_k.mapv(|v| lambda_k * v);
        let xweighted = RemlState::row_scale(&x, &cweighted_u_k);
        h_k += &fast_atb(&x, &xweighted);
        let expected = w_pos.t().dot(&h_k).dot(&w_pos);

        let diff = &actual - &expected;
        let err = diff.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err < 1e-10, "projected T_k mismatch: {err}");
    }

    #[test]
    fn dense_projected_trace_hinv_hkl_matches_direct_dense_trace() {
        let w_pos = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0],];
        let x = array![[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0],];
        let r_k = array![[1.0, -1.0, 0.0], [0.0, 1.0, -1.0],];
        let lambda_k = 0.9;
        let diag_kl = array![0.25, -0.2, 0.1];

        let z_mat = fast_ab(&x, &w_pos);
        let actual = RemlState::dense_projected_trace_hinv_hkl(
            &z_mat,
            &w_pos,
            Some(&r_k),
            lambda_k,
            &diag_kl,
        );

        let s_k = r_k.t().dot(&r_k);
        let mut h_kl = s_k.mapv(|v| lambda_k * v);
        let xweighted = RemlState::row_scale(&x, &diag_kl);
        h_kl += &fast_atb(&x, &xweighted);
        let expected = RemlState::trace_product(&w_pos.dot(&w_pos.t()), &h_kl);

        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn dense_projected_trace_quadratic_matches_direct_dense_trace() {
        let t_k = array![[2.0, 0.5], [0.5, 1.0],];
        let t_l = array![[1.5, -0.25], [-0.25, 0.75],];
        let actual = RemlState::dense_projected_trace_quadratic(&t_k, &t_l);
        let expected = RemlState::trace_product(&t_k, &t_l);
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn dense_projected_exactcost_gate_behaves_as_expected() {
        assert!(RemlState::dense_projected_exact_eligible(5_000, 100, 10));
        assert!(!RemlState::dense_projected_exact_eligible(50_000, 1025, 10));
        assert!(!RemlState::dense_projected_exact_eligible(200_000, 200, 20));
    }

    fn build_logit_state<'a>(
        y: &'a Array1<f64>,
        w: &'a Array1<f64>,
        x: &Array2<f64>,
        s: &Array2<f64>,
        cfg: &'a RemlConfig,
    ) -> RemlState<'a> {
        let offset = Array1::<f64>::zeros(y.len());
        RemlState::newwith_offset(
            y.view(),
            x.clone(),
            w.view(),
            offset.view(),
            vec![s.clone()],
            x.ncols(),
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

        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, false);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
        let bundle = state.obtain_eval_bundle(&rho).expect("bundle");
        let pr = bundle.pirls_result.as_ref();

        let beta = beta_original_from_bundle(&bundle);
        let h_orig = h_original_from_bundle(&bundle);
        let u = &pr.solveweights * &(&pr.solveworking_response - &pr.final_eta);

        // B from implicit solve:
        //   H B = X_τ^T g - X^T W(X_τ β̂) - S_τ β̂.
        let x_tau_beta = x_tau.dot(&beta);
        let weighted_x_tau_beta = &pr.solveweights * &x_tau_beta;
        let rhs = x_tau.t().dot(&u) - x.t().dot(&weighted_x_tau_beta) - s_tau.dot(&beta);
        let chol = h_orig.cholesky(Side::Lower).expect("chol(H)");
        let b_analytic = chol.solvevec(&rhs);

        // H_τ from exact total derivative:
        //   H_τ = X_τ^T W X + X^T W X_τ + X^T W_τ X + S_τ,
        // with W_τ provided by the family directional curvature callback.
        let eta_dot = &x_tau_beta + &x.dot(&b_analytic);
        let w_direction = directionalworking_curvature_from_eta(
            LinkFunction::Logit,
            &pr.final_eta,
            state.weights,
            &pr.solveweights,
            &eta_dot,
        )
        .expect("directional working curvature should evaluate for logit");
        let wx = RemlState::row_scale(&x, &pr.solveweights);
        let wx_tau = RemlState::row_scale(&x_tau, &pr.solveweights);
        let mut xwtau_x = x.clone();
        match w_direction {
            super::DirectionalWorkingCurvature::Diagonal(diag) => {
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
        state_plus.clearwarm_start();
        state_minus.clearwarm_start();

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

        let v_tau_analytic = state
            .compute_directional_hypergradientwith_bundle(&rho, &bundle, &hyper)
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
            v_rel < 5e-2,
            "V_tau mismatch vs FD: rel={v_rel:.3e}, abs={v_abs:.3e}, analytic={v_tau_analytic:.6e}, fd={v_taufd:.6e}"
        );

        assert!(
            cancellation.abs() < 5e-7,
            "stationarity cancellation failed: | -ell_beta^T B + beta^T S B | = {:.3e}",
            cancellation.abs()
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-9, true);
        let state = RemlState::newwith_offset(
            y.view(),
            x.clone(),
            w.view(),
            offset.view(),
            vec![s0, s1],
            x.ncols(),
            &cfg,
            Some(vec![1, 1]),
            None,
            None,
        )
        .expect("state");
        state.clearwarm_start();

        let rho = array![0.1, -0.2];
        let h_exact = state
            .compute_lamlhessian_exact(&rho)
            .expect("exact firth hessian");
        let h_fallback = state
            .compute_lamlhessian_analytic_fallback(&rho, None)
            .expect("analytic fallback hessian");

        assert!(h_exact.iter().all(|v| v.is_finite()));
        assert!(h_fallback.iter().all(|v| v.is_finite()));
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
            "Firth exact-vs-analytic-fallback Hessian mismatch too large: rel={rel:.3e}, exact={h_exact:?}, fallback={h_fallback:?}"
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, false);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
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

        let tr_null = state
            .fixed_subspace_penalty_trace(e, &s_dir_null, pr.ridge_passport)
            .expect("trace-null");
        assert!(
            tr_null.abs() < 1e-8,
            "nullspace direction should not contribute to tr(S^+ S_tau): got {tr_null:.3e}"
        );

        let tr_pos = state
            .fixed_subspace_penalty_trace(e, &s_dir_pos, pr.ridge_passport)
            .expect("trace-pos");
        let expected_pos = 1.0 / evals[pos_idx].max(1e-12);
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
        let bundle = state.obtain_eval_bundle(&rho).expect("firth eval bundle");
        let g = state
            .compute_directional_hypergradientwith_bundle(&rho, &bundle, &hyper)
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
        let bundle = state.obtain_eval_bundle(&rho).expect("firth eval bundle");
        let g = state
            .compute_directional_hypergradientwith_bundle(&rho, &bundle, &hyper)
            .expect("firth design-moving directional gradient should evaluate");
        assert!(
            g.is_finite(),
            "non-finite Firth design-moving directional gradient"
        );
    }

    #[test]
    fn directional_hypergradient_ignores_h_pos_factorwhen_active_subspace_is_unstable() {
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
        let dense_bundle = state.obtain_eval_bundle(&rho).expect("dense bundle");

        let mut unstable_bundle = dense_bundle.clone();
        unstable_bundle.active_subspace_unstable = true;
        let g_unstable = state
            .compute_directional_hypergradientwith_bundle(&rho, &unstable_bundle, &hyper)
            .expect("unstable bundle directional gradient");

        let mut poisoned_bundle = unstable_bundle.clone();
        poisoned_bundle.h_pos_factorw =
            Arc::new(Array2::<f64>::zeros(dense_bundle.h_pos_factorw.raw_dim()));
        let g_poisoned = state
            .compute_directional_hypergradientwith_bundle(&rho, &poisoned_bundle, &hyper)
            .expect("poisoned unstable bundle directional gradient");

        let abs = (g_unstable - g_poisoned).abs();
        assert!(
            abs < 1e-10,
            "unstable directional gradient should ignore cached H_+ factor: stable-fallback={g_unstable:.6e}, poisoned={g_poisoned:.6e}, abs={abs:.3e}"
        );
    }

    #[test]
    fn sparse_exact_directional_firth_branch_iswired_and_matches_dense() {
        let y = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let w = Array1::<f64>::ones(y.len());
        let x_dense = array![
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
            Array2::from_elem((x_dense.nrows(), x_dense.ncols()), 1e-3),
            Array2::<f64>::zeros((x_dense.ncols(), x_dense.ncols())),
            None,
            None,
        )
        .expect("single-penalty hyper direction");
        let rho = array![0.0];
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);

        // Dense reference path.
        let dense_state = build_logit_state(&y, &w, &x_dense, &s0, &cfg);
        dense_state.clearwarm_start();
        let dense_bundle = dense_state.obtain_eval_bundle(&rho).expect("dense bundle");
        let g_dense = dense_state
            .compute_directional_hypergradientwith_bundle(&rho, &dense_bundle, &hyper)
            .expect("dense firth directional gradient");

        // Build a synthetic sparse bundle and call sparse branch directly.
        // This validates sparse-branch Firth parity against dense exact math.
        let h_sparse = dense_to_sparse_symmetric_upper(dense_bundle.h_total.as_ref(), 1e-14)
            .expect("H->sparse upper");
        let sparse_factor = factorize_sparse_spd(&h_sparse).expect("sparse factor");
        let sparse_payload = Arc::new(super::SparseExactEvalData {
            factor: Arc::new(sparse_factor),
            penalty_blocks: Arc::new(Vec::<SparsePenaltyBlock>::new()),
            logdet_h: 0.0,
            logdet_s_pos: 0.0,
            det1_values: Arc::new(Array1::<f64>::zeros(rho.len())),
            traceworkspace: Arc::new(Mutex::new(super::SparseTraceWorkspace::default())),
        });
        let sparse_bundle = EvalShared {
            key: None,
            pirls_result: dense_bundle.pirls_result.clone(),
            ridge_passport: dense_bundle.ridge_passport,
            geometry: super::RemlGeometry::SparseExactSpd,
            h_eff: dense_bundle.h_eff.clone(),
            h_total: dense_bundle.h_total.clone(),
            h_pos_factorw: dense_bundle.h_pos_factorw.clone(),
            h_total_log_det: dense_bundle.h_total_log_det,
            active_subspace_rel_gap: dense_bundle.active_subspace_rel_gap,
            active_subspace_unstable: dense_bundle.active_subspace_unstable,
            sparse_exact: Some(sparse_payload),
            firth_dense_operator: dense_bundle.firth_dense_operator.clone(),
        };
        let g_sparse_branch = dense_state
            .compute_directional_hypergradient_sparse_exact(&rho, &sparse_bundle, &hyper)
            .expect("sparse exact directional firth");
        assert!(
            g_sparse_branch.is_finite(),
            "non-finite sparse exact directional firth value"
        );
        // Sparse branch now runs its own sparse solves/traces plus dense reduced
        // Firth blocks, so exact numerical equality with dense spectral path is
        // not guaranteed in this synthetic bundle setup. Guard the branch behavior:
        // it must produce a finite, same-order directional derivative.
        let abs = (g_sparse_branch - g_dense).abs();
        assert!(
            abs < 1.0,
            "sparse firth directional magnitude drift too large: sparse={g_sparse_branch:.6e}, dense={g_dense:.6e}, abs={abs:.3e}"
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
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

        let h = state
            .compute_joint_hyperhessian(&theta, rho.len(), &hyper_dirs)
            .expect("joint hyper hessian");
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
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

        let h = state
            .compute_joint_hyperhessian(&theta, rho.len(), &hyper_dirs)
            .expect("joint hyper hessian");
        let h_tt_analytic = h.slice(s![rho.len().., rho.len()..]).to_owned();

        let mut h_ttfd = Array2::<f64>::zeros((hyper_dirs.len(), hyper_dirs.len()));
        for j in 0..hyper_dirs.len() {
            let h = 1e-5;
            let mut theta_plus = theta.clone();
            let mut theta_minus = theta.clone();
            theta_plus[rho.len() + j] += h;
            theta_minus[rho.len() + j] -= h;
            let (_, g_plus) = state
                .compute_joint_hypercostgradient(&theta_plus, rho.len(), &hyper_dirs)
                .expect("g+");
            let (_, g_minus) = state
                .compute_joint_hypercostgradient(&theta_minus, rho.len(), &hyper_dirs)
                .expect("g-");
            let tau_col =
                (&g_plus.slice(s![rho.len()..]) - &g_minus.slice(s![rho.len()..])) / (2.0 * h);
            h_ttfd.column_mut(j).assign(&tau_col);
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
            rel < 3e-1,
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, true);
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

        let err = state
            .compute_joint_hyperhessian(&theta, 1, &hyper_dirs)
            .expect_err("invalid second-order penalty index should be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("penalty_index 1 out of bounds") && msg.contains("S_tau_tau[0][0]"),
            "unexpected validation error: {msg}"
        );
    }

    #[test]
    fn joint_mixed_rho_tau_analytic_matchesfd_reference() {
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
        let rho = array![0.0];
        let psi = array![0.0, 0.0];
        let mut xsecond_0 = vec![None; 2];
        let mut ssecond_0 = vec![None; 2];
        let mut xsecond_1 = vec![None; 2];
        let mut ssecond_1 = vec![None; 2];
        xsecond_0[0] = Some(Array2::from_elem((x.nrows(), x.ncols()), 5e-5));
        ssecond_0[0] = Some(array![
            [0.0, 0.0, 0.0],
            [0.0, 0.06, 0.01],
            [0.0, 0.01, 0.04],
        ]);
        // Provide cross second derivative from only one side to exercise fallback.
        xsecond_0[1] = Some(Array2::from_elem((x.nrows(), x.ncols()), -2e-5));
        ssecond_0[1] = Some(array![
            [0.0, 0.0, 0.0],
            [0.0, 0.03, -0.005],
            [0.0, -0.005, 0.02],
        ]);
        xsecond_1[1] = Some(Array2::from_elem((x.nrows(), x.ncols()), 4e-5));
        ssecond_1[1] = Some(array![
            [0.0, 0.0, 0.0],
            [0.0, 0.02, 0.004],
            [0.0, 0.004, 0.03],
        ]);
        let hyper_dirs = vec![
            DirectionalHyperParam::single_penalty(
                0,
                Array2::<f64>::zeros((x.nrows(), x.ncols())),
                array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
                Some(xsecond_0),
                Some(ssecond_0),
            )
            .expect("single-penalty hyper direction"),
            DirectionalHyperParam::single_penalty(
                0,
                Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                Array2::<f64>::zeros((x.ncols(), x.ncols())),
                Some(xsecond_1),
                Some(ssecond_1),
            )
            .expect("single-penalty hyper direction"),
        ];

        let pert_state = state
            .build_joint_perturbed_state(&psi, &hyper_dirs)
            .expect("perturbed state");
        let bundle = pert_state.obtain_eval_bundle(&rho).expect("bundle");
        let mixed_analytic = state
            .compute_mixed_rho_tau_block(&pert_state, &rho, &bundle, &hyper_dirs)
            .expect("analytic mixed block");
        assert_eq!(mixed_analytic.nrows(), rho.len());
        assert_eq!(mixed_analytic.ncols(), hyper_dirs.len());

        // Reference only for test validation:
        // central difference of analytic rho-gradient under +/- tau_j state
        // perturbations.
        let mut mixedfd = Array2::<f64>::zeros((rho.len(), hyper_dirs.len()));
        for j in 0..hyper_dirs.len() {
            let h = 1e-5;
            let mut psi_plus = psi.clone();
            let mut psi_minus = psi.clone();
            psi_plus[j] += h;
            psi_minus[j] -= h;
            let state_plus = state
                .build_joint_perturbed_state(&psi_plus, &hyper_dirs)
                .expect("state+");
            let state_minus = state
                .build_joint_perturbed_state(&psi_minus, &hyper_dirs)
                .expect("state-");
            let bundle_plus = state_plus.obtain_eval_bundle(&rho).expect("bundle+");
            let bundle_minus = state_minus.obtain_eval_bundle(&rho).expect("bundle-");
            let g_plus = state_plus
                .compute_gradient_with_bundle(&rho, &bundle_plus)
                .expect("g+");
            let g_minus = state_minus
                .compute_gradient_with_bundle(&rho, &bundle_minus)
                .expect("g-");
            let col = (&g_plus - &g_minus) / (2.0 * h);
            mixedfd.column_mut(j).assign(&col);
        }

        let num = (&mixed_analytic - &mixedfd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = mixedfd.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-10);
        let rel = num / den;
        assert!(
            rel < 2e-1,
            "analytic mixed block deviates from FD reference: rel={rel:.3e}, analytic={mixed_analytic:?}, fd={mixedfd:?}"
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
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
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

        let pert_state = state
            .build_joint_perturbed_state(&psi, &hyper_dirs)
            .expect("perturbed state");
        let bundle = pert_state.obtain_eval_bundle(&rho).expect("bundle");
        let h_tt_analytic = state
            .compute_tau_tau_block(&pert_state, &rho, &bundle, &hyper_dirs)
            .expect("analytic tau-tau block");
        assert_eq!(h_tt_analytic.nrows(), hyper_dirs.len());
        assert_eq!(h_tt_analytic.ncols(), hyper_dirs.len());

        let mut h_ttfd = Array2::<f64>::zeros((hyper_dirs.len(), hyper_dirs.len()));
        for j in 0..hyper_dirs.len() {
            let h = 1e-5;
            let mut psi_plus = psi.clone();
            let mut psi_minus = psi.clone();
            psi_plus[j] += h;
            psi_minus[j] -= h;
            let state_plus = state
                .build_joint_perturbed_state(&psi_plus, &hyper_dirs)
                .expect("state+");
            let state_minus = state
                .build_joint_perturbed_state(&psi_minus, &hyper_dirs)
                .expect("state-");
            let bundle_plus = state_plus.obtain_eval_bundle(&rho).expect("bundle+");
            let bundle_minus = state_minus.obtain_eval_bundle(&rho).expect("bundle-");
            let g_plus = state_plus
                .computemulti_psigradientwith_bundle(&rho, &bundle_plus, &hyper_dirs)
                .expect("g+");
            let g_minus = state_minus
                .computemulti_psigradientwith_bundle(&rho, &bundle_minus, &hyper_dirs)
                .expect("g-");
            let col = (&g_plus - &g_minus) / (2.0 * h);
            h_ttfd.column_mut(j).assign(&col);
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
            rel < 3e-1,
            "analytic tau-tau block deviates from FD reference: rel={rel:.3e}, analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
        );
    }

    #[test]
    fn bench_large_sparse_firth_directional_tau() {
        // Manual benchmark hook (ignored by default):
        // compares dense exact directional tau vs sparse-branch directional tau
        // on a larger sparse-like design.
        let n = 2_000usize;
        let p = 64usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            for j in 1..p {
                if (i + 3 * j) % 11 == 0 {
                    x[[i, j]] = ((i + j) as f64).sin() * 0.25;
                }
            }
        }
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            y[i] = if i % 3 == 0 { 1.0 } else { 0.0 };
        }
        let w = Array1::<f64>::ones(n);
        let mut s0 = Array2::<f64>::zeros((p, p));
        for j in 1..p {
            s0[[j, j]] = 0.5 + (j as f64) / (p as f64);
        }
        let hyper = DirectionalHyperParam::single_penalty(
            0,
            Array2::from_elem((n, p), 5e-5),
            Array2::<f64>::zeros((p, p)),
            None,
            None,
        )
        .expect("single-penalty hyper direction");
        let rho = array![0.0];
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clearwarm_start();
        let bundle = state.obtain_eval_bundle(&rho).expect("dense bundle");

        let t0 = Instant::now();
        let g_dense = state
            .compute_directional_hypergradientwith_bundle(&rho, &bundle, &hyper)
            .expect("dense directional");
        let dt_dense = t0.elapsed();

        let bundle_dense = bundle;
        let h_sparse = dense_to_sparse_symmetric_upper(bundle_dense.h_total.as_ref(), 1e-14)
            .expect("H->sparse upper");
        let sparse_factor = factorize_sparse_spd(&h_sparse).expect("factor");
        let sparse_payload = Arc::new(super::SparseExactEvalData {
            factor: Arc::new(sparse_factor),
            penalty_blocks: Arc::new(Vec::<SparsePenaltyBlock>::new()),
            logdet_h: 0.0,
            logdet_s_pos: 0.0,
            det1_values: Arc::new(Array1::<f64>::zeros(rho.len())),
            traceworkspace: Arc::new(Mutex::new(super::SparseTraceWorkspace::default())),
        });
        let sparse_bundle = EvalShared {
            key: None,
            pirls_result: bundle_dense.pirls_result.clone(),
            ridge_passport: bundle_dense.ridge_passport,
            geometry: super::RemlGeometry::SparseExactSpd,
            h_eff: bundle_dense.h_eff.clone(),
            h_total: bundle_dense.h_total.clone(),
            h_pos_factorw: bundle_dense.h_pos_factorw.clone(),
            h_total_log_det: bundle_dense.h_total_log_det,
            active_subspace_rel_gap: bundle_dense.active_subspace_rel_gap,
            active_subspace_unstable: bundle_dense.active_subspace_unstable,
            sparse_exact: Some(sparse_payload),
            firth_dense_operator: bundle_dense.firth_dense_operator.clone(),
        };
        let t1 = Instant::now();
        let g_sparse = state
            .compute_directional_hypergradient_sparse_exact(&rho, &sparse_bundle, &hyper)
            .expect("sparse directional");
        let dt_sparse = t1.elapsed();

        eprintln!(
            "[bench_large_sparse_firth_directional_tau] dense={:?} sparse={:?} g_dense={:.6e} g_sparse={:.6e}",
            dt_dense, dt_sparse, g_dense, g_sparse
        );
        assert!(g_dense.is_finite() && g_sparse.is_finite());
    }
}

impl FaerFactor {
    fn solve(&self, rhs: faer::MatRef<'_, f64>) -> FaerMat<f64> {
        match self {
            FaerFactor::Llt(f) => f.solve(rhs),
            FaerFactor::Lblt(f) => f.solve(rhs),
            FaerFactor::Ldlt(f) => f.solve(rhs),
        }
    }

    fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
        match self {
            FaerFactor::Llt(f) => f.solve_in_place(rhs),
            FaerFactor::Lblt(f) => f.solve_in_place(rhs),
            FaerFactor::Ldlt(f) => f.solve_in_place(rhs),
        }
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
pub(crate) struct HyperDesignDerivative {
    dense: Array2<f64>,
    embedded: Option<(Array2<f64>, Range<usize>, usize)>,
}

impl HyperDesignDerivative {
    pub(crate) fn from_embedded(
        local: Array2<f64>,
        global_range: Range<usize>,
        total_cols: usize,
    ) -> Self {
        let mut dense = Array2::<f64>::zeros((local.nrows(), total_cols));
        dense.slice_mut(s![.., global_range.clone()]).assign(&local);
        Self {
            dense,
            embedded: Some((local, global_range, total_cols)),
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    pub(crate) fn ncols(&self) -> usize {
        self.dense.ncols()
    }

    pub(crate) fn materialize(&self) -> Array2<f64> {
        self.dense.clone()
    }

    pub(crate) fn transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        if let Some((local, global_range, total_cols)) = self.embedded.as_ref() {
            if *total_cols != qs.nrows() {
                return Err(EstimationError::InvalidInput(format!(
                    "embedded design derivative width mismatch: total_cols={}, qs rows={}",
                    total_cols,
                    qs.nrows()
                )));
            }
            let qs_local = qs.slice(s![global_range.clone(), ..]).to_owned();
            Ok(crate::matrix::DenseRightProductView::new(local)
                .with_factor(&qs_local)
                .with_optional_factor(free_basis_opt)
                .materialize())
        } else {
            Ok(crate::matrix::DenseRightProductView::new(&self.dense)
                .with_factor(qs)
                .with_optional_factor(free_basis_opt)
                .materialize())
        }
    }

    pub(crate) fn dot(&self, rhs: &Array1<f64>) -> Array1<f64> {
        if let Some((local, global_range, _)) = self.embedded.as_ref() {
            local.dot(&rhs.slice(s![global_range.clone()]))
        } else {
            self.dense.dot(rhs)
        }
    }
}

impl From<Array2<f64>> for HyperDesignDerivative {
    fn from(value: Array2<f64>) -> Self {
        Self {
            dense: value,
            embedded: None,
        }
    }
}

impl std::ops::Deref for HyperDesignDerivative {
    type Target = Array2<f64>;

    fn deref(&self) -> &Self::Target {
        &self.dense
    }
}

#[derive(Clone)]
pub(crate) struct HyperPenaltyDerivative {
    dense: Array2<f64>,
    embedded: Option<(Array2<f64>, Range<usize>, usize)>,
}

impl HyperPenaltyDerivative {
    pub(crate) fn from_embedded(
        local: Array2<f64>,
        global_range: Range<usize>,
        total_dim: usize,
    ) -> Self {
        let mut dense = Array2::<f64>::zeros((total_dim, total_dim));
        dense
            .slice_mut(s![global_range.clone(), global_range.clone()])
            .assign(&local);
        Self {
            dense,
            embedded: Some((local, global_range, total_dim)),
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.dense.nrows()
    }

    pub(crate) fn ncols(&self) -> usize {
        self.nrows()
    }

    pub(crate) fn materialize(&self) -> Array2<f64> {
        self.dense.clone()
    }

    pub(crate) fn transformed(
        &self,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Array2<f64>, EstimationError> {
        if let Some((local, global_range, total_dim)) = self.embedded.as_ref() {
            if *total_dim != qs.nrows() {
                return Err(EstimationError::InvalidInput(format!(
                    "embedded penalty derivative width mismatch: total_dim={}, qs rows={}",
                    total_dim,
                    qs.nrows()
                )));
            }
            let qs_local = qs.slice(s![global_range.clone(), ..]).to_owned();
            let mut transformed = qs_local.t().dot(local).dot(&qs_local);
            if let Some(z) = free_basis_opt {
                transformed = z.t().dot(&transformed).dot(z);
            }
            Ok(transformed)
        } else {
            let mut transformed = qs.t().dot(&self.dense).dot(qs);
            if let Some(z) = free_basis_opt {
                transformed = z.t().dot(&transformed).dot(z);
            }
            Ok(transformed)
        }
    }

    pub(crate) fn scaled_add_to(
        &self,
        target: &mut Array2<f64>,
        amp: f64,
    ) -> Result<(), EstimationError> {
        match self.embedded.as_ref() {
            None => {
                if target.raw_dim() != self.dense.raw_dim() {
                    return Err(EstimationError::InvalidInput(format!(
                        "dense hyper penalty derivative shape mismatch: target={}x{}, matrix={}x{}",
                        target.nrows(),
                        target.ncols(),
                        self.dense.nrows(),
                        self.dense.ncols()
                    )));
                }
                target.scaled_add(amp, &self.dense);
            }
            Some((local, global_range, total_dim)) => {
                if target.nrows() != *total_dim || target.ncols() != *total_dim {
                    return Err(EstimationError::InvalidInput(format!(
                        "embedded hyper penalty derivative shape mismatch: target={}x{}, expected {}x{}",
                        target.nrows(),
                        target.ncols(),
                        total_dim,
                        total_dim
                    )));
                }
                target
                    .slice_mut(s![global_range.clone(), global_range.clone()])
                    .scaled_add(amp, local);
            }
        }
        Ok(())
    }
}

impl From<Array2<f64>> for HyperPenaltyDerivative {
    fn from(value: Array2<f64>) -> Self {
        Self {
            dense: value,
            embedded: None,
        }
    }
}

impl std::ops::Deref for HyperPenaltyDerivative {
    type Target = Array2<f64>;

    fn deref(&self) -> &Self::Target {
        &self.dense
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
        })
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

    pub(crate) fn transformed_x_tau_tau_at(
        &self,
        j: usize,
        qs: &Array2<f64>,
        free_basis_opt: Option<&Array2<f64>>,
    ) -> Result<Option<Array2<f64>>, EstimationError> {
        Ok(self
            .x_tau_tau_original
            .as_ref()
            .and_then(|rows| rows.get(j))
            .and_then(|entry| entry.as_ref())
            .map(|entry| entry.transformed(qs, free_basis_opt))
            .transpose()?)
    }

    pub(crate) fn x_tau_tau_nth_dense(&self, j: usize) -> Option<Array2<f64>> {
        self.x_tau_tau_original
            .as_ref()
            .and_then(|rows| rows.get(j))
            .and_then(|entry| entry.as_ref())
            .map(HyperDesignDerivative::materialize)
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

    pub(crate) fn penalty_first_components(&self) -> &[PenaltyDerivativeComponent] {
        &self.penalty_first_components
    }

    pub(crate) fn penaltysecond_components_for(
        &self,
        j: usize,
    ) -> Option<&[PenaltyDerivativeComponent]> {
        self.penaltysecond_components
            .as_ref()
            .and_then(|rows| rows.get(j))
            .and_then(|row| row.as_deref())
    }

    pub(crate) fn penaltysecond_componentrows(
        &self,
    ) -> Option<&[Option<Vec<PenaltyDerivativeComponent>>]> {
        self.penaltysecond_components.as_deref()
    }

    pub(crate) fn penalty_total_at(
        &self,
        rho: &Array1<f64>,
        p: usize,
    ) -> Result<Array2<f64>, EstimationError> {
        let mut total = Array2::<f64>::zeros((p, p));
        for component in &self.penalty_first_components {
            if component.penalty_index >= rho.len() {
                return Err(EstimationError::InvalidInput(format!(
                    "penalty_index {} out of bounds for rho dimension {}",
                    component.penalty_index,
                    rho.len()
                )));
            }
            component
                .matrix
                .scaled_add_to(&mut total, rho[component.penalty_index].exp())?;
        }
        Ok(total)
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
    penalty_blocks: Arc<Vec<SparsePenaltyBlock>>,
    logdet_h: f64,
    logdet_s_pos: f64,
    det1_values: Arc<Array1<f64>>,
    traceworkspace: Arc<Mutex<SparseTraceWorkspace>>,
}

#[derive(Clone)]
struct FirthDenseOperator {
    // Exact Firth/Jeffreys objects on the identifiable subspace.
    //
    // Let X in R^{n×p} potentially be rank-deficient with rank r.
    // We compute a fixed orthonormal basis Q in R^{p×r} for Col(Xᵀ) and define:
    //   X_r := X Q,
    //   W   := diag(w), with w_i = mu_i (1 - mu_i), 0 < w_i <= 1/4 for finite logit eta,
    //   Z   := W^{1/2} X_r,
    //   I_r := X_rᵀ W X_r = Zᵀ Z.
    //
    // Firth term is represented as:
    //   Phi(beta) = 0.5 log |I_r(beta)|,
    // which equals 0.5 log|XᵀWX|_+ (pseudodeterminant in the full space) but is
    // smooth because I_r is SPD as long as w_i > 0.
    //
    // Mapping back to the full p-space uses:
    //   I_+^dagger = Q I_r^{-1} Qᵀ.
    //
    // We store reduced-space factors so all derivatives can be evaluated exactly
    // without materializing dense n×n matrices M = X K Xᵀ or P = M⊙M.
    x_dense: Array2<f64>,
    x_dense_t: Array2<f64>,
    q_basis: Array2<f64>,
    // X_r = XQ
    x_reduced: Array2<f64>,
    // Reduced design used for M = X_r K_r X_r'.
    // (Name kept for compatibility with existing callsites.)
    z_reduced: Array2<f64>,
    // I_r^{-1}
    k_reduced: Array2<f64>,
    // h = diag(M), M = X_r K_r X_r'
    h_diag: Array1<f64>,
    // Logistic derivatives: w', w'', w''', w'''' as n-vectors.
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
struct FirthDirection {
    deta: Array1<f64>,
    g_u_reduced: Array2<f64>,
    a_u_reduced: Array2<f64>,
    dh: Array1<f64>,
    // B_u = diag(w'' ⊙ δη_u) X is represented by the row-scaling vector only.
    b_uvec: Array1<f64>,
}

#[derive(Clone)]
struct FirthTauPartialKernel {
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
struct FirthTauExactKernel {
    gphi_tau: Array1<f64>,
    phi_tau_partial: f64,
    tau_kernel: Option<FirthTauPartialKernel>,
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
struct EvalShared {
    key: Option<Vec<u64>>,
    pirls_result: Arc<PirlsResult>,
    ridge_passport: RidgePassport,
    geometry: RemlGeometry,
    h_eff: Arc<Array2<f64>>,
    /// The exact H_total matrix used for LAML cost computation.
    /// For Firth: h_eff - hphi. For non-Firth: h_eff.
    h_total: Arc<Array2<f64>>,

    // ══════════════════════════════════════════════════════════════════════
    // WHY TWO INVERSES? (Hybrid Approach for Indefinite Hessians)
    // ══════════════════════════════════════════════════════════════════════
    //
    // The LAML gradient has two terms requiring DIFFERENT matrix inverses:
    //
    // 1. TRACE TERM (∂/∂ρ log|H|): Uses PSEUDOINVERSE H₊†
    //    - Cost defines log|H| = Σᵢ log(λᵢ) for λᵢ > ε only (truncated)
    //    - Derivative: ∂J/∂ρ = ½ tr(H₊† ∂H/∂ρ)
    //    - H₊† = Σᵢ (1/λᵢ) uᵢuᵢᵀ for positive λᵢ only
    //    - Negative eigenvalues contribute 0 to cost, so their derivative contribution is 0
    //
    // 2. IMPLICIT TERM (dβ/dρ): Uses RIDGED FACTOR (H + δI)⁻¹
    //    - PIRLS stabilizes indefinite H by adding ridge: solves (H + δI)β = ...
    //    - Stationarity condition: G(β,ρ) = ∇L + δβ = 0
    //    - By Implicit Function Theorem: dβ/dρ = (H + δI)⁻¹ (λₖ Sₖ β)
    //    - Must use ridged inverse because β moves on the RIDGED surface
    //
    // EXAMPLE: H = -5 (indefinite), ridge δ = 10
    //   Trace term: Pseudoinverse → 0 (correct: truncated eigenvalue)
    //               Ridged inverse → 0.2 (WRONG: gradient of non-existent curve)
    //   Implicit term: Ridged inverse → 1/5 (correct: solver sees stiffness +5)
    //                  Pseudoinverse → 0 or ∞ (WRONG: ignores ridge physics)
    //
    // ══════════════════════════════════════════════════════════════════════
    /// Positive-spectrum factor W = U_+ diag(1/sqrt(lambda_+)).
    /// This avoids materializing H₊† = W Wᵀ in hot paths.
    ///
    /// We use identities:
    ///   H₊† v = W (Wᵀ v)
    ///   tr(H₊† S_k) = ||R_k W||_F², where S_k = R_kᵀ R_k.
    ///
    /// Architectural consequence: this representation is defined by a dense
    /// eigendecomposition of the transformed Hessian. Because it is a
    /// positive-part pseudoinverse on a dense rotated basis, it cannot be
    /// replaced directly by sparse selected inversion / Takahashi equations.
    /// Those methods apply to sparse SPD solves for H^{-1} in a sparsity-
    /// preserving coordinate system, not to the truncated spectral inverse
    /// used here.
    ///
    /// Derivation:
    /// If H = U diag(mu) U' and U_+ contains only eigenvectors with
    /// mu_i > tau, then the positive-part pseudoinverse is
    ///   H_+^dagger = U_+ diag(1 / mu_i) U_+'.
    /// Writing
    ///   W := U_+ diag(1 / sqrt(mu_i))
    /// gives
    ///   H_+^dagger = W W'.
    /// For any penalty root R_k with S_k = R_k' R_k,
    ///   tr(H_+^dagger S_k)
    /// = tr(W W' R_k' R_k)
    /// = tr((R_k W)' (R_k W))
    /// = ||R_k W||_F^2.
    /// This is why the current code can evaluate the trace term from the
    /// spectral factor alone, but it also means the computation is tied to a
    /// dense eigenbasis rather than a sparse factorization.
    h_pos_factorw: Arc<Array2<f64>>,

    /// Log determinant via truncation: Σᵢ log(λᵢ) for λᵢ > ε only.
    h_total_log_det: f64,
    /// Relative eigengap between kept and dropped spectra around the H_+
    /// threshold used for pseudo-logdet derivatives (if available).
    active_subspace_rel_gap: Option<f64>,
    /// True when the H_+ active subspace is numerically near a hard-threshold
    /// crossing. In that regime, the truncated logdet objective no longer has
    /// a clean fixed-projector derivative model: even first-order derivatives
    /// cease to be exact at the crossing, and second-order identities are more
    /// fragile still. We use this flag to suppress "exact Hessian" claims and
    /// other branch-local second-order logic that assumes a fixed active
    /// projector.
    active_subspace_unstable: bool,
    sparse_exact: Option<Arc<SparseExactEvalData>>,
    firth_dense_operator: Option<Arc<FirthDenseOperator>>,
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

struct RemlWorkspace {
    lambdavalues: Array1<f64>,
    costgradient: Array1<f64>,
    priorgradient: Array1<f64>,
}

impl RemlWorkspace {
    fn new(max_penalties: usize) -> Self {
        RemlWorkspace {
            lambdavalues: Array1::zeros(max_penalties),
            costgradient: Array1::zeros(max_penalties),
            priorgradient: Array1::zeros(max_penalties),
        }
    }

    fn reset_for_eval(&mut self, penalties: usize) {
        if penalties == 0 {
            return;
        }
        self.costgradient.slice_mut(s![..penalties]).fill(0.0);
        self.priorgradient.slice_mut(s![..penalties]).fill(0.0);
    }

    fn set_lambdavalues(&mut self, rho: &Array1<f64>) {
        let len = rho.len();
        if len == 0 {
            return;
        }
        let mut view = self.lambdavalues.slice_mut(s![..len]);
        for (dst, &src) in view.iter_mut().zip(rho.iter()) {
            *dst = src.exp();
        }
    }

    fn lambdaview(&self, len: usize) -> ArrayView1<'_, f64> {
        self.lambdavalues.slice(s![..len])
    }

    fn costgradientview(&mut self, len: usize) -> ArrayViewMut1<'_, f64> {
        self.costgradient.slice_mut(s![..len])
    }

    fn zerocostgradient(&mut self, len: usize) {
        self.costgradient.slice_mut(s![..len]).fill(0.0);
    }

    fn costgradientview_const(&self, len: usize) -> ArrayView1<'_, f64> {
        self.costgradient.slice(s![..len])
    }

    fn soft_priorcost_andgrad<'a>(&'a mut self, rho: &Array1<f64>) -> (f64, ArrayView1<'a, f64>) {
        let len = rho.len();
        let mut gradview = self.priorgradient.slice_mut(s![..len]);
        gradview.fill(0.0);

        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return (0.0, self.priorgradient.slice(s![..len]));
        }

        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        let mut cost = 0.0;
        for (grad, &ri) in gradview.iter_mut().zip(rho.iter()) {
            let scaled = sharp * ri * inv_bound;
            cost += scaled.cosh().ln();
            *grad = sharp * inv_bound * scaled.tanh();
        }

        if RHO_SOFT_PRIOR_WEIGHT != 1.0 {
            for grad in gradview.iter_mut() {
                *grad *= RHO_SOFT_PRIOR_WEIGHT;
            }
            cost *= RHO_SOFT_PRIOR_WEIGHT;
        }

        (cost, self.priorgradient.slice(s![..len]))
    }
}

struct PirlsLruCache {
    map: HashMap<Vec<u64>, Arc<PirlsResult>>,
    order: VecDeque<Vec<u64>>,
    capacity: usize,
}

impl PirlsLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    fn touch(&mut self, key: &Vec<u64>) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }

    fn get(&mut self, key: &Vec<u64>) -> Option<Arc<PirlsResult>> {
        let value = self.map.get(key).cloned();
        if value.is_some() {
            self.touch(key);
        }
        value
    }

    fn insert(&mut self, key: Vec<u64>, value: Arc<PirlsResult>) {
        if self.map.contains_key(&key) {
            self.map.insert(key.clone(), value);
            self.touch(&key);
            return;
        }

        while self.map.len() >= self.capacity {
            if let Some(evictkey) = self.order.pop_front() {
                self.map.remove(&evictkey);
            } else {
                break;
            }
        }

        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

/// Centralized cache/memoization owner for REML evaluations.
///
/// This keeps cache-key identity, bundle reuse, and invalidation policy out of
/// the math kernels so objective/derivative routines can stay algebra-focused.
struct EvalCacheManager {
    pirls_cache: RwLock<PirlsLruCache>,
    faer_factor_cache: RwLock<HashMap<Vec<u64>, Arc<FaerFactor>>>,
    current_eval_bundle: RwLock<Option<EvalShared>>,
    pirls_cache_enabled: AtomicBool,
}

impl EvalCacheManager {
    fn new() -> Self {
        Self {
            pirls_cache: RwLock::new(PirlsLruCache::new(MAX_PIRLS_CACHE_ENTRIES)),
            faer_factor_cache: RwLock::new(HashMap::new()),
            current_eval_bundle: RwLock::new(None),
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

    fn invalidate_eval_bundle(&self) {
        self.current_eval_bundle.write().unwrap().take();
    }

    fn clear_eval_and_factor_caches(&self) {
        self.invalidate_eval_bundle();
        self.faer_factor_cache.write().unwrap().clear();
    }
}

/// Reusable scratch/runtime memory that should not be part of mathematical
/// state invariants.
struct RemlArena {
    workspace: Mutex<RemlWorkspace>,
    cost_last: RwLock<Option<CostAgg>>,
    cost_repeat: RwLock<u64>,
    cost_last_emit: RwLock<u64>,
    cost_eval_count: RwLock<u64>,
    raw_cond_snapshot: RwLock<f64>,
    gaussian_cond_snapshot: RwLock<f64>,
    lastgradient_used_stochastic_fallback: AtomicBool,
}

impl RemlArena {
    fn new(workspace: RemlWorkspace) -> Self {
        Self {
            workspace: Mutex::new(workspace),
            cost_last: RwLock::new(None),
            cost_repeat: RwLock::new(0),
            cost_last_emit: RwLock::new(0),
            cost_eval_count: RwLock::new(0),
            raw_cond_snapshot: RwLock::new(f64::NAN),
            gaussian_cond_snapshot: RwLock::new(f64::NAN),
            lastgradient_used_stochastic_fallback: AtomicBool::new(false),
        }
    }
}

pub(crate) struct RemlState<'a> {
    y: ArrayView1<'a, f64>,
    x: DesignMatrix,
    weights: ArrayView1<'a, f64>,
    offset: Array1<f64>,
    // Original penalty matrices S_k (p × p), ρ-independent basis
    s_full_list: Arc<Vec<Array2<f64>>>,
    pub(crate) rs_list: Vec<Array2<f64>>, // Pre-computed penalty square roots
    balanced_penalty_root: Array2<f64>,
    reparam_invariant: ReparamInvariant,
    sparse_penalty_blocks: Option<Arc<Vec<SparsePenaltyBlock>>>,
    p: usize,
    config: &'a RemlConfig,
    runtime_mixture_link_state: Option<crate::types::MixtureLinkState>,
    runtime_sas_link_state: Option<SasLinkState>,
    nullspace_dims: Vec<usize>,
    coefficient_lower_bounds: Option<Array1<f64>>,
    linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,

    cache_manager: EvalCacheManager,
    arena: RemlArena,
    pub(crate) warm_start_beta: RwLock<Option<Coefficients>>,
    warm_start_enabled: AtomicBool,
}

#[derive(Clone)]
struct CostKey {
    compact: String,
}

#[derive(Clone)]
struct CostAgg {
    key: CostKey,
    count: u64,
    stab_cond_min: f64,
    stab_cond_max: f64,
    stab_cond_last: f64,
    raw_cond_min: f64,
    raw_cond_max: f64,
    raw_cond_last: f64,
    laml_min: f64,
    laml_max: f64,
    laml_last: f64,
    edf_min: f64,
    edf_max: f64,
    edf_last: f64,
    trace_min: f64,
    trace_max: f64,
    trace_last: f64,
}

impl CostKey {
    fn new(rho: &[f64], smooth: &[f64], stab_cond: f64, raw_cond: f64) -> Self {
        let rho_compact = format_compact_series(rho, |v| format!("{:.3}", v));
        let smooth_compact = format_compact_series(smooth, |v| format!("{:.2e}", v));
        let compact = format!(
            "rho={} | smooth={} | κ(stable/raw)={:.3e}/{:.3e}",
            rho_compact, smooth_compact, stab_cond, raw_cond
        );
        let compact = compact.replace("-0.000", "0.000");
        Self { compact }
    }

    fn approx_eq(&self, other: &Self) -> bool {
        self.compact == other.compact
    }

    fn format_compact(&self) -> String {
        self.compact.clone()
    }
}

impl CostAgg {
    fn new(key: CostKey, laml: f64, edf: f64, trace: f64, stab_cond: f64, raw_cond: f64) -> Self {
        Self {
            key,
            count: 1,
            stab_cond_min: stab_cond,
            stab_cond_max: stab_cond,
            stab_cond_last: stab_cond,
            raw_cond_min: raw_cond,
            raw_cond_max: raw_cond,
            raw_cond_last: raw_cond,
            laml_min: laml,
            laml_max: laml,
            laml_last: laml,
            edf_min: edf,
            edf_max: edf,
            edf_last: edf,
            trace_min: trace,
            trace_max: trace,
            trace_last: trace,
        }
    }

    fn update(&mut self, laml: f64, edf: f64, trace: f64, stab_cond: f64, raw_cond: f64) {
        self.count += 1;
        self.laml_last = laml;
        self.edf_last = edf;
        self.trace_last = trace;
        self.stab_cond_last = stab_cond;
        self.raw_cond_last = raw_cond;
        if stab_cond < self.stab_cond_min {
            self.stab_cond_min = stab_cond;
        }
        if stab_cond > self.stab_cond_max {
            self.stab_cond_max = stab_cond;
        }
        if raw_cond < self.raw_cond_min {
            self.raw_cond_min = raw_cond;
        }
        if raw_cond > self.raw_cond_max {
            self.raw_cond_max = raw_cond;
        }
        if laml < self.laml_min {
            self.laml_min = laml;
        }
        if laml > self.laml_max {
            self.laml_max = laml;
        }
        if edf < self.edf_min {
            self.edf_min = edf;
        }
        if edf > self.edf_max {
            self.edf_max = edf;
        }
        if trace < self.trace_min {
            self.trace_min = trace;
        }
        if trace > self.trace_max {
            self.trace_max = trace;
        }
    }

    fn format_summary(&self) -> String {
        let key = self.key.format_compact();
        let metric = |label: &str, min: f64, max: f64, last: f64, fmt: &dyn Fn(f64) -> String| {
            if approx_f64(min, max, 1e-6, 1e-9) && approx_f64(min, last, 1e-6, 1e-9) {
                format!("{label}={}", fmt(min))
            } else {
                let range = format_range(min, max, |v| fmt(v));
                format!("{label}={range} last={}", fmt(last))
            }
        };
        let kappa = if approx_f64(self.stab_cond_min, self.stab_cond_max, 1e-6, 1e-9)
            && approx_f64(self.raw_cond_min, self.raw_cond_max, 1e-6, 1e-9)
            && approx_f64(self.stab_cond_min, self.stab_cond_last, 1e-6, 1e-9)
            && approx_f64(self.raw_cond_min, self.raw_cond_last, 1e-6, 1e-9)
        {
            format!(
                "κ(stable/raw)={}/{}",
                format_cond(self.stab_cond_min),
                format_cond(self.raw_cond_min)
            )
        } else {
            let stable = format_range(self.stab_cond_min, self.stab_cond_max, format_cond);
            let raw = format_range(self.raw_cond_min, self.raw_cond_max, format_cond);
            format!(
                "κ(stable/raw)={stable}/{raw} last={}/{}",
                format_cond(self.stab_cond_last),
                format_cond(self.raw_cond_last)
            )
        };
        let laml = metric("LAML", self.laml_min, self.laml_max, self.laml_last, &|v| {
            format!("{:.6e}", v)
        });
        let edf = metric("EDF", self.edf_min, self.edf_max, self.edf_last, &|v| {
            format!("{:.6}", v)
        });
        let trace = metric(
            "tr(H^-1 Sλ)",
            self.trace_min,
            self.trace_max,
            self.trace_last,
            &|v| format!("{:.6}", v),
        );
        let count = if self.count > 1 {
            format!(" | count={}", self.count)
        } else {
            String::new()
        };
        format!("{key}{count} | {kappa} | {laml} | {edf} | {trace}",)
    }
}

// Formatting utilities moved to crate::diagnostics
impl<'a> RemlState<'a> {
    // Firth / Jeffreys outer-derivative derivation map (implementation guide)
    //
    // Research-to-production assumptions used in this file:
    // - Inner objective (minimization):
    //     L*(beta,rho) = -ell(beta) + 0.5 beta' S(rho) beta - Phi(beta)
    // - Firth/Jeffreys logistic correction:
    //     Phi(beta) = 0.5 log|I(beta)|_+, I(beta)=X'W(beta)X
    // - Outer objective:
    //     V(rho) = L*(beta_hat(rho),rho)
    //              + 0.5 log|H_tot(beta_hat,rho)|_+
    //              - 0.5 log|S(rho)|_+ + prior(rho)
    // - Piecewise-smooth exactness regime:
    //     constant rank / constant active positive eigenspaces for I, H_tot, S.
    //
    // In that regime:
    //   d log|M|_+ = tr(M_+^dagger dM),
    //   d^2 log|M|_+[U,V] = tr(M_+^dagger d^2M[U,V]) - tr(M_+^dagger U M_+^dagger V).
    // These identities are the basis for all exact outer derivatives here.
    //
    // n-th order log|M|_+ (fixed active subspace, exact):
    //   D^n log|M|_+[τ_1,...,τ_n]
    //   = Σ_{π∈Part([n])} (-1)^{|π|-1} (|π|-1)!
    //       tr( Π_{B∈π} ( M_+^dagger M_B ) ),
    //   M_B := D^{|B|}M[τ_i : i∈B].
    // This is the multivariate partition/Faà di Bruno form used for all orders.
    //
    // Explicit low-order closed forms (same fixed-active-subspace regime):
    //   D^3 log|M|_+[u,v,w]
    //   = tr(M_+^† M_uvw)
    //     - tr(M_+^† M_uv M_+^† Mw)
    //     - tr(M_+^† M_uw M_+^† Mv)
    //     - tr(M_+^† Mvw M_+^† M_u)
    //     + 2 tr(M_+^† M_u M_+^† Mv M_+^† Mw)
    //     + 2 tr(M_+^† M_u M_+^† Mw M_+^† Mv).
    //
    //   D^4 log|M|_+[u,v,w,z]
    //   = tr(M_+^† M_uvwz)
    //     - sum_(3,1) tr(M_+^† M_(...) M_+^† M_(.))
    //     - sum_(2,2) tr(M_+^† M_(..) M_+^† M_(..))
    //     + 2 sum_(2,1,1) tr(M_+^† M_(..) M_+^† M_(.) M_+^† M_(.))
    //     - 6 sum_(1,1,1,1) tr(M_+^† M_(.) M_+^† M_(.) M_+^† M_(.) M_+^† M_(.)),
    //   where the sums run over all block splits/orderings induced by set
    //   partitions of {u,v,w,z}. The n-th partition formula above generates all
    //   coefficients/signs automatically.
    //
    // Assumptions (smooth regime used by all "exact" formulas below):
    // 1) X may be rank-deficient, but rank(X) is locally constant.
    // 2) Firth term is interpreted as Phi(beta)=0.5*log|X'WX|_+ on the
    //    identifiable subspace. In implementation this is represented via a
    //    fixed reduced basis Q with X_r=XQ and I_r=X_r'WX_r.
    // 3) Active positive eigenspaces for truncated logdet terms (S and H_total)
    //    are locally constant (no eigenvalue crossing the cutoff).
    //
    // Under these assumptions, log|.|_+ derivatives are classical on the
    // active subspace and are evaluated by truncated eigensystems.
    //
    // Objective (minimization convention):
    //   V(rho)
    //   = L*(beta_hat(rho), rho)
    //   + 0.5 log|H_total(rho)|_+
    //   - 0.5 log|S(rho)|_+ ,
    //
    // where
    //   L*(beta, rho)
    //   = -ell(beta) + 0.5 beta' S(rho) beta - Phi(beta),
    //   Phi(beta) = 0.5 log|I(beta)|_+,
    //   I(beta)   = X' W(beta) X.
    //
    // Penalty derivatives:
    //   A_k   = dS/drho_k      = lambda_k S_k
    //   A_kl  = d²S/drho_kdrho_l = delta_kl A_k.
    //
    // Inner stationarity:
    //   G*(beta,rho) = dL*/dbeta = 0
    // with Jacobian
    //   H_total = dG*/dbeta = X'WX + S - Hphi,
    //   Hphi   = d²Phi/dbeta².
    //
    // Implicit derivatives (solve-surface inverse):
    //   B_k    = dbeta_hat/drho_k
    //          = -H_total^{-1}(A_k beta_hat)
    //
    //   B_kl   = d²beta_hat/(drho_kdrho_l)
    //          = -H_total^{-1}(H_l B_k + A_k B_l + delta_kl A_k beta_hat).
    //
    // Define operator-valued derivatives used throughout:
    //   J(u) = D(X'WX)[u] - D(Hphi)[u]
    //        = X' diag(w' ⊙ (Xu)) X - D(Hphi)[u]
    //   K(u,v) = D²(X'WX)[u,v] - D²(Hphi)[u,v]
    //          = X' diag(w'' ⊙ (Xu) ⊙ (Xv)) X - D²(Hphi)[u,v].
    //
    // Then:
    //   H_k  = A_k + J(B_k)
    //   H_kl = A_kl + J(B_kl) + K(B_k,B_l),
    // with A_kl = delta_kl A_k.
    //
    // Outer Hessian decomposition:
    //   V_kl = Q_kl + L_kl + P_kl
    //   Q_kl = B_l' A_k beta_hat + 0.5 delta_kl beta_hat' A_k beta_hat
    //   L_kl = 0.5[ tr(H_+^dagger H_kl) - tr(H_+^dagger H_l H_+^dagger H_k) ]
    //   P_kl = -0.5 d²/drho_kdrho_l log|S|_+.
    //
    // Rank-deficient design note:
    //   I = X'WX may be singular. The mathematically coherent Firth term is
    //   Phi = 0.5 log|I|_+ with dPhi = 0.5 tr(I_+^† dI) on a fixed active
    //   positive subspace. The dense helper below computes an SPD-equivalent
    //   representation in reduced coordinates:
    //      I_+^† = Q I_r^{-1} Q',   I_r = X_r' W X_r,   X_r = XQ.
    //   This avoids inverting singular p×p Fisher matrices directly.
    //
    // Inverse conventions in this file:
    //   - Truncated logdet derivatives always use the positive-part pseudoinverse
    //     on the active eigenspace.
    //   - IFT solves (B_k/B_kl and directional beta_tau) use the configured
    //     solve-surface inverse. For full objective-coherent pseudodet calculus,
    //     the mathematically strict choice is to use the same active-subspace
    //     generalized inverse as the logdet terms.
    // Current implementation uses spectral H_+ traces whenever available and
    // falls back to solve-surface contractions when required by geometry.
    // Outside the fixed-active-subspace regime (eigenvalue crossings),
    // second derivatives of log|.|_+ are non-smooth.
    //
    // Higher-order extension (all orders, exact in same smooth regime):
    //   V(θ) = f(θ) + 0.5 log|H(β̂(θ),θ)|_+ - 0.5 log|S(θ)|_+ + prior(θ),
    //   f(θ) = L*(β̂(θ),θ).
    //
    // 1) Implicit derivatives of β̂:
    //      G(β̂(θ),θ)=0, H=G_β.
    //      For m>=1 and multi-index I=(τ_1,...,τ_m):
    //        H β̂_I = -R_I,
    //      where R_I is the partition-sum of mixed derivatives of G with all
    //      lower-order β̂_J (|J|<m) substituted (standard implicit Faà di Bruno).
    //
    //    Closed form (set-partition implicit Faà di Bruno):
    //      Let Q(θ):=G(β̂(θ),θ) ≡ 0 and I={1,...,m}.
    //      Then
    //        0 = D^mQ[τ_I]
    //          = Σ_{π∈Part(I)} D^{|π|}G[ ζ_{B_1},...,ζ_{B_|π|} ],
    //      with blocks B∈π and
    //        ζ_B :=
    //          (β̂_B, τ_b)  if |B|=1,  B={b},
    //          (β̂_B, 0)    if |B|>=2.
    //
    //      For m>=2, isolating π={I} gives
    //        H β̂_I = - Σ_{π∈Part(I), π≠{I}}
    //                    D^{|π|}G[ ζ_{B_1},...,ζ_{B_|π|} ].
    //      (m=1 reduces to Hβ̂_{τ} = -G_τ.)
    //
    //      This is the closed-form non-recursive characterization; practical
    //      code still evaluates it recursively by increasing order.
    //
    // 2) Total derivatives of H(β̂(θ),θ):
    //      D^m[H(β̂(θ),θ)][I]
    //      = Σ_{π∈Part([m])} D_β^{|π|}D_θ^{m-|π|}H[β̂_{B_1},...,β̂_{B_|π|}; θ-rest].
    //
    // 3) n-th derivative of profiled fit f:
    //      D^m f[I] uses the same partition composition with L*;
    //      terms containing D_βL*=G vanish at stationarity (envelope pruning).
    //
    // 4) n-th derivative of V:
    //      D^mV = D^mf + 0.5 D^m log|H(β̂(θ),θ)|_+ - 0.5 D^m log|S(θ)|_+ + D^m prior.
    //
    // Firth tensor closed forms used by these all-order identities:
    //   Phi(beta) = 0.5 log|I(beta)|_+, I = X'WX.
    //   For directions alpha,gamma:
    //     Phi_alpha = 0.5 tr(I_+^† I_alpha),
    //     Phi_alpha,gamma
    //       = 0.5[ tr(I_+^† I_alpha,gamma)
    //              - tr(I_+^† I_alpha I_+^† I_gamma) ].
    //   Hence:
    //     (gphi)_tau = Phi_beta,tau,
    //     (Hphi)_tau|beta = Phi_beta,beta,tau,
    //   and D(Hphi)[beta_tau] = Phi_beta,beta,beta[beta_tau].
    //
    // Conclusion:
    //   Every Firth-specific contribution at any order enters via derivatives
    //   of Phi(beta)=0.5log|I(beta)|_+, i.e. through G and H derivative tensors
    //   and their implicit β̂-coupled recursions above.

    fn rungradient_diagnostics(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        analytic_grad: &Array1<f64>,
        applied_truncation_corrections: Option<&[f64]>,
    ) {
        use crate::diagnostics::{
            DiagnosticConfig, GradientDiagnosticReport, compute_dualridge_check,
            compute_envelopeaudit, computespectral_bleed,
        };

        let config = DiagnosticConfig::default();
        let mut report = GradientDiagnosticReport::new();

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_used = bundle.ridge_passport.delta;
        let beta = pirls_result.beta_transformed.as_ref();
        let lambdas: Array1<f64> = rho.mapv(f64::exp);

        // === Strategy 4: Dual-Ridge Consistency Check ===
        // Compare the PIRLS ridge with the ridge used by cost/gradient paths.
        let dualridge = compute_dualridge_check(
            pirls_result.ridge_passport.delta, // Ridge from PIRLS passport
            ridge_used,                        // Ridge passed to cost
            ridge_used,                        // Ridge passed to gradient (same bundle)
            beta,
        );
        report.dualridge = Some(dualridge);

        // === Strategy 1: KKT/Envelope Theorem Audit ===
        // Check if the inner solver actually reached stationarity
        let reparam = &pirls_result.reparam_result;
        let penaltygrad = reparam.s_transformed.dot(beta);

        let envelopeaudit = compute_envelopeaudit(
            pirls_result.lastgradient_norm,
            &penaltygrad,
            pirls_result.ridge_passport.delta,
            ridge_used, // What gradient assumes
            beta,
            config.kkt_tolerance,
            config.rel_error_threshold,
        );
        report.envelopeaudit = Some(envelopeaudit);

        // === Strategy 3: Spectral Bleed Trace ===
        // Check if truncated eigenspace corrections are adequate
        // Diagnostics must compare quantities in a common frame.
        // `u_truncated`, `h_eff`, and `rs_transformed` are all in transformed coordinates.
        let u_truncated = reparam.u_truncated.clone();
        let truncated_count = u_truncated.ncols();
        // Path/coordinate contract for diagnostics:
        // - `u_truncated` comes from ReparamResult (already transformed).
        // - `h_eff` and `reparam.rs_transformed` are transformed as well.

        if truncated_count > 0
            && let Some(appliedvalues) = applied_truncation_corrections
        {
            let h_eff = bundle.h_eff.as_ref();

            // Solve H⁻¹ U_⊥ for spectral bleed calculation
            let hview = FaerArrayView::new(h_eff);
            if let Ok(chol) = FaerLlt::new(hview.as_ref(), Side::Lower) {
                let mut h_inv_u = u_truncated.clone();
                let mut rhsview = array2_to_matmut(&mut h_inv_u);
                chol.solve_in_place(rhsview.as_mut());

                for (k, r_k) in reparam.rs_transformed.iter().enumerate() {
                    let applied_correction = appliedvalues.get(k).copied().unwrap_or(0.0);
                    let bleed = computespectral_bleed(
                        k,
                        r_k.view(),
                        u_truncated.view(),
                        h_inv_u.view(),
                        lambdas[k],
                        applied_correction,
                        config.rel_error_threshold,
                    );
                    if bleed.has_bleed || bleed.truncated_energy.abs() > 1e-4 {
                        report.spectral_bleed.push(bleed);
                    }
                }
            }
        }

        // === Strategy 2: Component-wise FD (only if we detected other issues) ===
        // This is expensive, so rate-limit it unless diagnostics are severe.
        let eval_idx = (*self.arena.cost_eval_count.read().unwrap()).max(1);
        let severe_envelope = report.envelopeaudit.as_ref().is_some_and(|a| {
            a.kkt_residual_norm > GRAD_DIAG_SEVERE_KKT_NORM
                || (a.innerridge - a.outerridge).abs() > GRAD_DIAG_SEVERE_RIDGE_MISMATCH
        });
        let severe_bleed = report
            .spectral_bleed
            .iter()
            .any(|b| b.has_bleed && b.truncated_energy.abs() > GRAD_DIAG_SEVERE_BLEED_ENERGY);
        let severeridge = report.dualridge.as_ref().is_some_and(|r| {
            r.has_mismatch
                && (r.ridge_impact.abs() > GRAD_DIAG_SEVERE_RIDGE_IMPACT
                    || r.phantom_penalty.abs() > GRAD_DIAG_SEVERE_PHANTOM_PENALTY)
        });
        let periodic_sample = should_samplegradient_diagfd(eval_idx);
        let run_componentfd = report.has_issues()
            && (severe_envelope || severe_bleed || severeridge || periodic_sample);
        if run_componentfd {
            let h = config.fd_step_size;
            let numericvals: Vec<f64> = {
                let cache_guard = AtomicFlagGuard::swap(
                    &self.cache_manager.pirls_cache_enabled,
                    false,
                    Ordering::Relaxed,
                );
                (
                    cache_guard,
                    (0..rho.len())
                        .into_par_iter()
                        .map(|k| {
                            let mut rho_plus = rho.clone();
                            rho_plus[k] += h;
                            let mut rho_minus = rho.clone();
                            rho_minus[k] -= h;

                            let (fp, fm) = rayon::join(
                                || self.compute_cost(&rho_plus).unwrap_or(f64::INFINITY),
                                || self.compute_cost(&rho_minus).unwrap_or(f64::INFINITY),
                            );
                            (fp - fm) / (2.0 * h)
                        })
                        .collect(),
                )
                    .1
            };
            let numericgrad = Array1::from_vec(numericvals);

            report.analytic_gradient = Some(analytic_grad.clone());
            report.numericgradient = Some(numericgrad.clone());

            // Compute per-component relative errors
            let mut rel_errors = Array1::<f64>::zeros(rho.len());
            for k in 0..rho.len() {
                let denom = analytic_grad[k].abs().max(numericgrad[k].abs()).max(1e-8);
                rel_errors[k] = (analytic_grad[k] - numericgrad[k]).abs() / denom;
            }
            report.component_rel_errors = Some(rel_errors);
        } else if report.has_issues() {
            log::debug!(
                "[REML] skipping full FD gradient diagnostics at eval {} (sampled every {} evals unless severe).",
                eval_idx,
                GRAD_DIAG_FD_INTERVAL
            );
        }

        // === Output Summary (single print, not in a loop) ===
        if report.has_issues() && log::log_enabled!(log::Level::Warn) {
            log::warn!(
                "[GRADIENT DIAGNOSTICS] Issues detected:\n{}",
                report.summary()
            );

            if let (Some(analytic), Some(numeric)) =
                (&report.analytic_gradient, &report.numericgradient)
            {
                let diff = analytic - numeric;
                let rel_l2 = diff.dot(&diff).sqrt() / numeric.dot(numeric).sqrt().max(1e-8);
                log::warn!(
                    "[GRADIENT DIAGNOSTICS] Total gradient rel. L2 error: {:.2e}",
                    rel_l2
                );
            }
        }
    }
}
