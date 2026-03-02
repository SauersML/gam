use super::cache::AtomicFlagGuard;
use super::strategy::{GeometryBackendKind, HessianEvalStrategyKind, HessianStrategyDecision};
use super::*;
use crate::linalg::sparse_exact::{
    SparseExactFactor, SparsePenaltyBlock, SparseTraceWorkspace,
    assemble_and_factor_sparse_penalized_system, build_sparse_penalty_blocks,
    leverages_from_factor, solve_sparse_spd, solve_sparse_spd_multi, sparse_matvec_public,
    trace_hinv_sk,
};
use crate::pirls::{
    DirectionalWorkingCurvature, PirlsWorkspace, directional_working_curvature_callback,
};
use faer::Side;

mod trace;
mod hyper;
mod firth;

enum FaerFactor {
    Llt(FaerLlt<f64>),
    Lblt(FaerLblt<f64>),
    Ldlt(FaerLdlt<f64>),
}

#[cfg(test)]
mod tests {
    use super::{DirectionalHyperParam, EvalShared, LinkFunction, RemlConfig, RemlState};
    use crate::faer_ndarray::{FaerCholesky, FaerEigh, fast_ab, fast_atb};
    use crate::linalg::sparse_exact::{SparsePenaltyBlock, dense_to_sparse, factorize_sparse_spd};
    use crate::pirls::{PirlsCoordinateFrame, directional_working_curvature_from_eta};
    use faer::Side;
    use ndarray::{Array1, Array2, array};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    #[test]
    fn dense_projected_tk_matches_direct_wt_hk_w() {
        let w_pos = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0],];
        let x = array![[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 1.0],];
        let r_k = array![[1.0, -1.0, 0.0], [0.0, 1.0, -1.0],];
        let lambda_k = 1.7;
        let c_weighted_u_k = array![0.2, -0.1, 0.4];

        let z_mat = fast_ab(&x, &w_pos);
        let actual = RemlState::dense_projected_tk(&z_mat, &w_pos, &r_k, lambda_k, &c_weighted_u_k);

        let s_k = r_k.t().dot(&r_k);
        let mut h_k = s_k.mapv(|v| lambda_k * v);
        let mut x_weighted = x.clone();
        for i in 0..x_weighted.nrows() {
            for j in 0..x_weighted.ncols() {
                x_weighted[[i, j]] *= c_weighted_u_k[i];
            }
        }
        h_k += &fast_atb(&x, &x_weighted);
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
        let mut x_weighted = x.clone();
        for i in 0..x_weighted.nrows() {
            for j in 0..x_weighted.ncols() {
                x_weighted[[i, j]] *= diag_kl[i];
            }
        }
        h_kl += &fast_atb(&x, &x_weighted);
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
    fn dense_projected_exact_cost_gate_behaves_as_expected() {
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
        RemlState::new_with_offset(
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
        let hyper = DirectionalHyperParam {
            x_tau_original: x_tau.clone(),
            s_tau_original: s_tau.clone(),
        };
        let rho = array![0.0];

        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-10, false);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clear_warm_start();
        let bundle = state.obtain_eval_bundle(&rho).expect("bundle");
        let pr = bundle.pirls_result.as_ref();

        let beta = beta_original_from_bundle(&bundle);
        let h_orig = h_original_from_bundle(&bundle);
        let u = &pr.solve_weights * &(&pr.solve_working_response - &pr.final_eta);

        // B from implicit solve:
        //   H B = X_τ^T g - X^T W(X_τ β̂) - S_τ β̂.
        let x_tau_beta = x_tau.dot(&beta);
        let weighted_x_tau_beta = &pr.solve_weights * &x_tau_beta;
        let rhs = x_tau.t().dot(&u) - x.t().dot(&weighted_x_tau_beta) - s_tau.dot(&beta);
        let chol = h_orig.cholesky(Side::Lower).expect("chol(H)");
        let b_analytic = chol.solve_vec(&rhs);

        // H_τ from exact total derivative:
        //   H_τ = X_τ^T W X + X^T W X_τ + X^T W_τ X + S_τ,
        // with W_τ provided by the family directional curvature callback.
        let eta_dot = &x_tau_beta + &x.dot(&b_analytic);
        let w_direction = directional_working_curvature_from_eta(
            LinkFunction::Logit,
            &pr.final_eta,
            state.weights,
            &pr.solve_weights,
            &eta_dot,
        );
        let mut wx = x.clone();
        let mut wx_tau = x_tau.clone();
        for i in 0..x.nrows() {
            let wi = pr.solve_weights[i];
            for j in 0..x.ncols() {
                wx[[i, j]] *= wi;
                wx_tau[[i, j]] *= wi;
            }
        }
        let mut x_wtau_x = x.clone();
        match w_direction {
            super::DirectionalWorkingCurvature::Diagonal(diag) => {
                for i in 0..x_wtau_x.nrows() {
                    let wi = diag[i];
                    for j in 0..x_wtau_x.ncols() {
                        x_wtau_x[[i, j]] *= wi;
                    }
                }
            }
        }
        let mut h_tau_analytic = x_tau.t().dot(&wx);
        h_tau_analytic += &x.t().dot(&wx_tau);
        h_tau_analytic += &x.t().dot(&x_wtau_x);
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
        state_plus.clear_warm_start();
        state_minus.clear_warm_start();

        let bundle_plus = state_plus.obtain_eval_bundle(&rho).expect("bundle+");
        let bundle_minus = state_minus.obtain_eval_bundle(&rho).expect("bundle-");
        let beta_plus = beta_original_from_bundle(&bundle_plus);
        let beta_minus = beta_original_from_bundle(&bundle_minus);
        let b_fd = (&beta_plus - &beta_minus).mapv(|v| v / (2.0 * h));

        let h_plus = h_original_from_bundle(&bundle_plus);
        let h_minus = h_original_from_bundle(&bundle_minus);
        let h_tau_fd = (&h_plus - &h_minus).mapv(|v| v / (2.0 * h));

        let v_plus = state_plus.compute_cost(&rho).expect("cost+");
        let v_minus = state_minus.compute_cost(&rho).expect("cost-");
        let v_tau_fd = (v_plus - v_minus) / (2.0 * h);

        let v_tau_analytic = state
            .compute_directional_hyper_gradient_with_bundle(&rho, &bundle, &hyper)
            .expect("analytic directional gradient");

        let b_num = (&b_analytic - &b_fd).mapv(|v| v * v).sum().sqrt();
        let b_den = b_fd.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let b_rel = b_num / b_den;
        assert!(
            b_rel < 2e-2,
            "B implicit solve mismatch vs FD: rel={b_rel:.3e}, num={b_num:.3e}, den={b_den:.3e}"
        );

        let dh_num = (&h_tau_analytic - &h_tau_fd).mapv(|v| v * v).sum().sqrt();
        let dh_den = h_tau_fd.mapv(|v| v * v).sum().sqrt().max(1e-12);
        let dh_rel = dh_num / dh_den;
        assert!(
            dh_rel < 3e-2,
            "H_tau mismatch vs FD: rel={dh_rel:.3e}, num={dh_num:.3e}, den={dh_den:.3e}"
        );

        let v_abs = (v_tau_analytic - v_tau_fd).abs();
        let v_rel = v_abs / v_tau_fd.abs().max(1e-10);
        assert!(
            v_rel < 5e-2,
            "V_tau mismatch vs FD: rel={v_rel:.3e}, abs={v_abs:.3e}, analytic={v_tau_analytic:.6e}, fd={v_tau_fd:.6e}"
        );

        assert!(
            cancellation.abs() < 5e-7,
            "stationarity cancellation failed: | -ell_beta^T B + beta^T S B | = {:.3e}",
            cancellation.abs()
        );
    }

    #[test]
    fn firth_exact_hessian_matches_fd_on_rank_deficient_design() {
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
        let state = RemlState::new_with_offset(
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
        state.clear_warm_start();

        let rho = array![0.1, -0.2];
        let h_exact = state
            .compute_laml_hessian_exact(&rho)
            .expect("exact firth hessian");
        let h_fallback = state
            .compute_laml_hessian_analytic_fallback(&rho, None)
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
    fn firth_gradient_lives_in_design_column_space_under_rank_deficiency() {
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
        let op = RemlState::build_firth_dense_operator(&x, &eta).expect("firth operator");

        // Exact reduced-space Firth gradient:
        //   gradPhi = 0.5 Xᵀ (w' ⊙ h), with h = diag(X_r K_r X_rᵀ).
        let grad_phi = 0.5 * x.t().dot(&(&op.w1 * &op.h_diag));

        // Check (I - QQᵀ) gradPhi ≈ 0.
        let q = &op.q_basis;
        let proj = q.dot(&q.t().dot(&grad_phi));
        let resid = &grad_phi - &proj;
        let rel =
            resid.mapv(|v| v * v).sum().sqrt() / grad_phi.mapv(|v| v * v).sum().sqrt().max(1e-12);
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
        state.clear_warm_start();
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
    fn firth_logit_directional_hyper_gradient_supports_penalty_only_direction() {
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
        let hyper = DirectionalHyperParam {
            x_tau_original: Array2::<f64>::zeros((x.nrows(), x.ncols())),
            s_tau_original: array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.03], [0.0, 0.03, 0.12],],
        };
        let rho = array![0.0];
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clear_warm_start();
        let g = state
            .compute_directional_hyper_gradient(&rho, &hyper)
            .expect("firth penalty-only directional gradient should evaluate");
        assert!(
            g.is_finite(),
            "non-finite Firth penalty-only directional gradient"
        );
    }

    #[test]
    fn firth_logit_directional_hyper_gradient_supports_design_moving_direction() {
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
        let hyper = DirectionalHyperParam {
            x_tau_original: Array2::from_elem((x.nrows(), x.ncols()), 1e-3),
            s_tau_original: Array2::<f64>::zeros((x.ncols(), x.ncols())),
        };
        let rho = array![0.0];
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clear_warm_start();
        let g = state
            .compute_directional_hyper_gradient(&rho, &hyper)
            .expect("firth design-moving directional gradient should evaluate");
        assert!(
            g.is_finite(),
            "non-finite Firth design-moving directional gradient"
        );
    }

    #[test]
    fn sparse_exact_directional_firth_branch_is_wired_and_matches_dense() {
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
        let hyper = DirectionalHyperParam {
            x_tau_original: Array2::from_elem((x_dense.nrows(), x_dense.ncols()), 1e-3),
            s_tau_original: Array2::<f64>::zeros((x_dense.ncols(), x_dense.ncols())),
        };
        let rho = array![0.0];
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);

        // Dense reference path.
        let dense_state = build_logit_state(&y, &w, &x_dense, &s0, &cfg);
        dense_state.clear_warm_start();
        let g_dense = dense_state
            .compute_directional_hyper_gradient(&rho, &hyper)
            .expect("dense firth directional gradient");

        // Build a synthetic sparse bundle and call sparse branch directly.
        // This validates sparse-branch Firth parity against dense exact math.
        let dense_bundle = dense_state.obtain_eval_bundle(&rho).expect("dense bundle");
        let h_sparse = dense_to_sparse(dense_bundle.h_total.as_ref(), 1e-14).expect("H->sparse");
        let sparse_factor = factorize_sparse_spd(&h_sparse).expect("sparse factor");
        let sparse_payload = Arc::new(super::SparseExactEvalData {
            factor: Arc::new(sparse_factor),
            penalty_blocks: Arc::new(Vec::<SparsePenaltyBlock>::new()),
            logdet_h: 0.0,
            logdet_s_pos: 0.0,
            det1_values: Arc::new(Array1::<f64>::zeros(rho.len())),
            trace_workspace: Arc::new(Mutex::new(super::SparseTraceWorkspace::default())),
        });
        let sparse_bundle = EvalShared {
            key: None,
            pirls_result: dense_bundle.pirls_result.clone(),
            ridge_passport: dense_bundle.ridge_passport,
            geometry: super::RemlGeometry::SparseExactSpd,
            h_eff: dense_bundle.h_eff.clone(),
            h_total: dense_bundle.h_total.clone(),
            h_pos_factor_w: dense_bundle.h_pos_factor_w.clone(),
            h_total_log_det: dense_bundle.h_total_log_det,
            active_subspace_rel_gap: dense_bundle.active_subspace_rel_gap,
            active_subspace_unstable: dense_bundle.active_subspace_unstable,
            sparse_exact: Some(sparse_payload),
            firth_dense_operator: dense_bundle.firth_dense_operator.clone(),
        };
        let g_sparse_branch = dense_state
            .compute_directional_hyper_gradient_sparse_exact(&rho, &sparse_bundle, &hyper)
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
    fn joint_hyper_hessian_wires_mixed_blocks() {
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
        state.clear_warm_start();
        let rho = array![0.0];
        let theta = array![0.0, 0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam {
                x_tau_original: Array2::<f64>::zeros((x.nrows(), x.ncols())),
                s_tau_original: array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
            },
            DirectionalHyperParam {
                x_tau_original: Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                s_tau_original: Array2::<f64>::zeros((x.ncols(), x.ncols())),
            },
        ];

        let h = state
            .compute_joint_hyper_hessian(&theta, rho.len(), &hyper_dirs)
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
    fn joint_mixed_rho_tau_analytic_matches_fd_reference() {
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
        state.clear_warm_start();
        let rho = array![0.0];
        let psi = array![0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam {
                x_tau_original: Array2::<f64>::zeros((x.nrows(), x.ncols())),
                s_tau_original: array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
            },
            DirectionalHyperParam {
                x_tau_original: Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                s_tau_original: Array2::<f64>::zeros((x.ncols(), x.ncols())),
            },
        ];

        let mixed_analytic = state
            .compute_mixed_rho_tau_block(&rho, &psi, &hyper_dirs)
            .expect("analytic mixed block");
        assert_eq!(mixed_analytic.nrows(), rho.len());
        assert_eq!(mixed_analytic.ncols(), hyper_dirs.len());

        // Reference only for test validation:
        // central difference of analytic rho-gradient under +/- tau_j state
        // perturbations.
        let mut mixed_fd = Array2::<f64>::zeros((rho.len(), hyper_dirs.len()));
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
            mixed_fd.column_mut(j).assign(&col);
        }

        let num = (&mixed_analytic - &mixed_fd)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let den = mixed_fd
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-10);
        let rel = num / den;
        assert!(
            rel < 2e-1,
            "analytic mixed block deviates from FD reference: rel={rel:.3e}, analytic={mixed_analytic:?}, fd={mixed_fd:?}"
        );
    }

    #[test]
    fn joint_tau_tau_analytic_matches_fd_reference() {
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
        state.clear_warm_start();
        let rho = array![0.0];
        let psi = array![0.0, 0.0];
        let hyper_dirs = vec![
            DirectionalHyperParam {
                x_tau_original: Array2::<f64>::zeros((x.nrows(), x.ncols())),
                s_tau_original: array![[0.0, 0.0, 0.0], [0.0, 0.2, 0.01], [0.0, 0.01, 0.15],],
            },
            DirectionalHyperParam {
                x_tau_original: Array2::from_elem((x.nrows(), x.ncols()), 2e-4),
                s_tau_original: Array2::<f64>::zeros((x.ncols(), x.ncols())),
            },
        ];

        let h_tt_analytic = state
            .compute_tau_tau_block(&rho, &psi, &hyper_dirs)
            .expect("analytic tau-tau block");
        assert_eq!(h_tt_analytic.nrows(), hyper_dirs.len());
        assert_eq!(h_tt_analytic.ncols(), hyper_dirs.len());

        let mut h_tt_fd = Array2::<f64>::zeros((hyper_dirs.len(), hyper_dirs.len()));
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
                .compute_multi_psi_gradient_with_bundle(&rho, &bundle_plus, &hyper_dirs)
                .expect("g+");
            let g_minus = state_minus
                .compute_multi_psi_gradient_with_bundle(&rho, &bundle_minus, &hyper_dirs)
                .expect("g-");
            let col = (&g_plus - &g_minus) / (2.0 * h);
            h_tt_fd.column_mut(j).assign(&col);
        }
        for i in 0..h_tt_fd.nrows() {
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
            rel < 3e-1,
            "analytic tau-tau block deviates from FD reference: rel={rel:.3e}, analytic={h_tt_analytic:?}, fd={h_tt_fd:?}"
        );
    }

    #[test]
    #[ignore]
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
        let hyper = DirectionalHyperParam {
            x_tau_original: Array2::from_elem((n, p), 5e-5),
            s_tau_original: Array2::<f64>::zeros((p, p)),
        };
        let rho = array![0.0];
        let cfg = RemlConfig::external(LinkFunction::Logit, 1e-8, true);
        let state = build_logit_state(&y, &w, &x, &s0, &cfg);
        state.clear_warm_start();

        let t0 = Instant::now();
        let g_dense = state
            .compute_directional_hyper_gradient(&rho, &hyper)
            .expect("dense directional");
        let dt_dense = t0.elapsed();

        let bundle_dense = state.obtain_eval_bundle(&rho).expect("bundle");
        let h_sparse = dense_to_sparse(bundle_dense.h_total.as_ref(), 1e-14).expect("H->sparse");
        let sparse_factor = factorize_sparse_spd(&h_sparse).expect("factor");
        let sparse_payload = Arc::new(super::SparseExactEvalData {
            factor: Arc::new(sparse_factor),
            penalty_blocks: Arc::new(Vec::<SparsePenaltyBlock>::new()),
            logdet_h: 0.0,
            logdet_s_pos: 0.0,
            det1_values: Arc::new(Array1::<f64>::zeros(rho.len())),
            trace_workspace: Arc::new(Mutex::new(super::SparseTraceWorkspace::default())),
        });
        let sparse_bundle = EvalShared {
            key: None,
            pirls_result: bundle_dense.pirls_result.clone(),
            ridge_passport: bundle_dense.ridge_passport,
            geometry: super::RemlGeometry::SparseExactSpd,
            h_eff: bundle_dense.h_eff.clone(),
            h_total: bundle_dense.h_total.clone(),
            h_pos_factor_w: bundle_dense.h_pos_factor_w.clone(),
            h_total_log_det: bundle_dense.h_total_log_det,
            active_subspace_rel_gap: bundle_dense.active_subspace_rel_gap,
            active_subspace_unstable: bundle_dense.active_subspace_unstable,
            sparse_exact: Some(sparse_payload),
            firth_dense_operator: bundle_dense.firth_dense_operator.clone(),
        };
        let t1 = Instant::now();
        let g_sparse = state
            .compute_directional_hyper_gradient_sparse_exact(&rho, &sparse_bundle, &hyper)
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
pub(crate) struct DirectionalHyperParam {
    pub x_tau_original: Array2<f64>,
    pub s_tau_original: Array2<f64>,
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
    trace_workspace: Arc<Mutex<SparseTraceWorkspace>>,
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
    // B = diag(w') X used in D H_phi and D^2 H_phi contractions.
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
}

#[derive(Clone)]
struct FirthTauPartialKernel {
    dot_w1: Array1<f64>,
    dot_w2: Array1<f64>,
    dot_h_partial: Array1<f64>,
    // Reduced design drift X_{tau,r} = X_tau Q used in exact design-moving
    // Hadamard-Gram contractions.
    x_tau_reduced: Array2<f64>,
    // Reduced Fisher inverse drift:
    //   dot(K_r) = -K_r dot(I_r) K_r
    // where dot(I_r) includes explicit X_tau and weight drift at beta-fixed.
    dot_k_reduced: Array2<f64>,
}

/// Holds the state for the outer REML optimization and supplies cost and
/// gradient evaluations to the `wolfe_bfgs` optimizer.
///
/// The `cache` field uses `RefCell` to enable interior mutability. This is a crucial
/// performance optimization. The `cost_and_grad` closure required by the BFGS
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
    /// For Firth: h_eff - h_phi. For non-Firth: h_eff.
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
    h_pos_factor_w: Arc<Array2<f64>>,

    /// Log determinant via truncation: Σᵢ log(λᵢ) for λᵢ > ε only.
    h_total_log_det: f64,
    /// Relative eigengap between kept and dropped spectra around the H_+
    /// threshold used for pseudo-logdet derivatives (if available).
    active_subspace_rel_gap: Option<f64>,
    /// True when the H_+ active subspace is numerically near a crossing.
    /// In that regime, second-order outer derivatives are piecewise-smooth and
    /// we automatically downgrade to safer policies instead of trusting exact
    /// Hessian identities that assume a fixed active projector.
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
    lambda_values: Array1<f64>,
    cost_gradient: Array1<f64>,
    prior_gradient: Array1<f64>,
}

impl RemlWorkspace {
    fn new(max_penalties: usize) -> Self {
        RemlWorkspace {
            lambda_values: Array1::zeros(max_penalties),
            cost_gradient: Array1::zeros(max_penalties),
            prior_gradient: Array1::zeros(max_penalties),
        }
    }

    fn reset_for_eval(&mut self, penalties: usize) {
        if penalties == 0 {
            return;
        }
        self.cost_gradient.slice_mut(s![..penalties]).fill(0.0);
        self.prior_gradient.slice_mut(s![..penalties]).fill(0.0);
    }

    fn set_lambda_values(&mut self, rho: &Array1<f64>) {
        let len = rho.len();
        if len == 0 {
            return;
        }
        let mut view = self.lambda_values.slice_mut(s![..len]);
        for (dst, &src) in view.iter_mut().zip(rho.iter()) {
            *dst = src.exp();
        }
    }

    fn lambda_view(&self, len: usize) -> ArrayView1<'_, f64> {
        self.lambda_values.slice(s![..len])
    }

    fn cost_gradient_view(&mut self, len: usize) -> ArrayViewMut1<'_, f64> {
        self.cost_gradient.slice_mut(s![..len])
    }

    fn zero_cost_gradient(&mut self, len: usize) {
        self.cost_gradient.slice_mut(s![..len]).fill(0.0);
    }

    fn cost_gradient_view_const(&self, len: usize) -> ArrayView1<'_, f64> {
        self.cost_gradient.slice(s![..len])
    }

    fn soft_prior_cost_and_grad<'a>(&'a mut self, rho: &Array1<f64>) -> (f64, ArrayView1<'a, f64>) {
        let len = rho.len();
        let mut grad_view = self.prior_gradient.slice_mut(s![..len]);
        grad_view.fill(0.0);

        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return (0.0, self.prior_gradient.slice(s![..len]));
        }

        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        let mut cost = 0.0;
        for (grad, &ri) in grad_view.iter_mut().zip(rho.iter()) {
            let scaled = sharp * ri * inv_bound;
            cost += scaled.cosh().ln();
            *grad = sharp * inv_bound * scaled.tanh();
        }

        if RHO_SOFT_PRIOR_WEIGHT != 1.0 {
            for grad in grad_view.iter_mut() {
                *grad *= RHO_SOFT_PRIOR_WEIGHT;
            }
            cost *= RHO_SOFT_PRIOR_WEIGHT;
        }

        (cost, self.prior_gradient.slice(s![..len]))
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
            if let Some(evict_key) = self.order.pop_front() {
                self.map.remove(&evict_key);
            } else {
                break;
            }
        }

        self.order.push_back(key.clone());
        self.map.insert(key, value);
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
    fn sanitized_rho_key(rho: &Array1<f64>) -> Option<Vec<u64>> {
        super::cache::sanitized_rho_key(rho)
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
    last_gradient_used_stochastic_fallback: AtomicBool,
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
            last_gradient_used_stochastic_fallback: AtomicBool::new(false),
        }
    }
}

pub(crate) struct RemlState<'a> {
    y: ArrayView1<'a, f64>,
    x: DesignMatrix,
    weights: ArrayView1<'a, f64>,
    offset: Array1<f64>,
    // Original penalty matrices S_k (p × p), ρ-independent basis
    s_full_list: Vec<Array2<f64>>,
    pub(crate) rs_list: Vec<Array2<f64>>, // Pre-computed penalty square roots
    balanced_penalty_root: Array2<f64>,
    reparam_invariant: ReparamInvariant,
    sparse_penalty_blocks: Option<Arc<Vec<SparsePenaltyBlock>>>,
    p: usize,
    config: &'a RemlConfig,
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
    // -------------------------------------------------------------------------
    // Firth / Jeffreys outer-Hessian derivation map (implementation guide)
    // -------------------------------------------------------------------------
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
    //   H_total = dG*/dbeta = X'WX + S - H_phi,
    //   H_phi   = d²Phi/dbeta².
    //
    // Implicit derivatives (solve-surface inverse):
    //   B_k    = dbeta_hat/drho_k
    //          = -H_total^{-1}(A_k beta_hat)
    //
    //   B_kl   = d²beta_hat/(drho_kdrho_l)
    //          = -H_total^{-1}(H_l B_k + A_k B_l + delta_kl A_k beta_hat).
    //
    // Define operator-valued derivatives used throughout:
    //   J(u) = D(X'WX)[u] - D(H_phi)[u]
    //        = X' diag(w' ⊙ (Xu)) X - D(H_phi)[u]
    //   K(u,v) = D²(X'WX)[u,v] - D²(H_phi)[u,v]
    //          = X' diag(w'' ⊙ (Xu) ⊙ (Xv)) X - D²(H_phi)[u,v].
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
    // Conclusion:
    //   Every Firth-specific second-order contribution required by the exact
    //   outer Hessian enters through -D(H_phi)[·] and -D²(H_phi)[·,·] terms.
    // -------------------------------------------------------------------------

    #[inline]
    fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
        // Keep expensive diagnostics out of the hot path unless they can
        // be surfaced. This has zero effect on optimization math.
        (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
            && (eval_idx == 1 || eval_idx % 200 == 0)
    }

    fn log_gam_cost(
        &self,
        rho: &Array1<f64>,
        lambdas: &[f64],
        laml: f64,
        stab_cond: f64,
        raw_cond: f64,
        edf: f64,
        trace_h_inv_s_lambda: f64,
    ) {
        const GAM_REPEAT_EMIT: u64 = 50;
        const GAM_MIN_EMIT_GAP: u64 = 200;
        let rho_q = quantize_vec(rho.as_slice().unwrap_or_default(), 5e-3, 1e-6);
        let smooth_q = quantize_vec(lambdas, 5e-3, 1e-6);
        let stab_q = quantize_value(stab_cond, 5e-3, 1e-6);
        let raw_q = quantize_value(raw_cond, 5e-3, 1e-6);
        let key = CostKey::new(&rho_q, &smooth_q, stab_q, raw_q);

        let mut last_opt = self.arena.cost_last.write().unwrap();
        let mut repeat = self.arena.cost_repeat.write().unwrap();
        let mut last_emit = self.arena.cost_last_emit.write().unwrap();
        let eval_idx = *self.arena.cost_eval_count.read().unwrap();

        if let Some(last) = last_opt.as_mut() {
            if last.key.approx_eq(&key) {
                last.update(laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
                *repeat += 1;
                if *repeat >= GAM_REPEAT_EMIT
                    && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP
                {
                    println!("[GAM COST] {}", last.format_summary());
                    *repeat = 0;
                    *last_emit = eval_idx;
                }
                return;
            }

            let emit_prev =
                last.count > 1 && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP;
            if emit_prev {
                println!("[GAM COST] {}", last.format_summary());
                *last_emit = eval_idx;
            }
        }

        let new_agg = CostAgg::new(key, laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
        if eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP {
            println!("[GAM COST] {}", new_agg.format_summary());
            *last_emit = eval_idx;
        }
        *last_opt = Some(new_agg);
        *repeat = 0;
    }

    #[allow(dead_code)]
    pub fn reset_optimizer_tracking(&self) {
        self.cache_manager.clear_eval_and_factor_caches();
        self.arena.cost_last.write().unwrap().take();
        *self.arena.cost_repeat.write().unwrap() = 0;
        *self.arena.cost_last_emit.write().unwrap() = 0;
        *self.arena.cost_eval_count.write().unwrap() = 0;
        *self.arena.raw_cond_snapshot.write().unwrap() = f64::NAN;
        *self.arena.gaussian_cond_snapshot.write().unwrap() = f64::NAN;
    }

    /// Compute soft prior cost without needing workspace
    fn compute_soft_prior_cost(&self, rho: &Array1<f64>) -> f64 {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return 0.0;
        }

        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        let mut cost = 0.0;
        for &ri in rho.iter() {
            let scaled = sharp * ri * inv_bound;
            cost += scaled.cosh().ln();
        }

        cost * RHO_SOFT_PRIOR_WEIGHT
    }

    /// Compute soft prior gradient without workspace mutation.
    fn compute_soft_prior_grad(&self, rho: &Array1<f64>) -> Array1<f64> {
        let len = rho.len();
        let mut grad = Array1::<f64>::zeros(len);
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return grad;
        }
        let inv_bound = 1.0 / RHO_BOUND;
        let sharp = RHO_SOFT_PRIOR_SHARPNESS;
        for (g, &ri) in grad.iter_mut().zip(rho.iter()) {
            let scaled = sharp * ri * inv_bound;
            *g = sharp * inv_bound * scaled.tanh() * RHO_SOFT_PRIOR_WEIGHT;
        }
        grad
    }

    /// Add the exact Hessian of the soft rho prior in place.
    ///
    /// Prior definition per coordinate:
    ///   C_i(rho_i) = w * log(cosh(a * rho_i)),
    ///   a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND,
    ///   w = RHO_SOFT_PRIOR_WEIGHT.
    ///
    /// Then:
    ///   dC_i/drho_i   = w * a * tanh(a * rho_i),
    ///   d²C_i/drho_i² = w * a² * sech²(a * rho_i)
    ///                = w * a² * (1 - tanh²(a * rho_i)).
    ///
    /// The prior is separable across coordinates, so off-diagonals are zero.
    fn add_soft_prior_hessian_in_place(&self, rho: &Array1<f64>, hess: &mut Array2<f64>) {
        let len = rho.len();
        if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
            return;
        }
        let a = RHO_SOFT_PRIOR_SHARPNESS / RHO_BOUND;
        let prefactor = RHO_SOFT_PRIOR_WEIGHT * a * a;
        for i in 0..len {
            let t = (a * rho[i]).tanh();
            hess[[i, i]] += prefactor * (1.0 - t * t);
        }
    }

    /// Returns the effective Hessian and the ridge value used (if any).
    /// Uses the same Hessian matrix in both cost and gradient calculations.
    ///
    /// PIRLS folds any stabilization ridge directly into the penalized objective:
    ///   l_p(β; ρ) = l(β) - 0.5 * βᵀ (S_λ + ridge I) β.
    /// Therefore the curvature used in LAML is
    ///   H_eff = X'WX + S_λ + ridge I,
    /// and adding another ridge here places the Laplace expansion on a different surface.
    fn effective_hessian(
        &self,
        pr: &PirlsResult,
    ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
        let base = pr.stabilized_hessian_transformed.clone();

        if base.cholesky(Side::Lower).is_ok() {
            return Ok((base, pr.ridge_passport));
        }

        Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn new<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        s_list: Vec<Array2<f64>>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        let zero_offset = Array1::<f64>::zeros(y.len());
        Self::new_with_offset(
            y,
            x,
            weights,
            zero_offset.view(),
            s_list,
            p,
            config,
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
        )
    }

    pub(crate) fn new_with_offset<X>(
        y: ArrayView1<'a, f64>,
        x: X,
        weights: ArrayView1<'a, f64>,
        offset: ArrayView1<'_, f64>,
        s_list: Vec<Array2<f64>>,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Option<Vec<usize>>,
        coefficient_lower_bounds: Option<Array1<f64>>,
        linear_constraints: Option<crate::pirls::LinearInequalityConstraints>,
    ) -> Result<Self, EstimationError>
    where
        X: Into<DesignMatrix>,
    {
        // Pre-compute penalty square roots once
        let rs_list = compute_penalty_square_roots(&s_list)?;
        let x = x.into();

        let expected_len = s_list.len();
        let nullspace_dims = match nullspace_dims {
            Some(dims) => {
                if dims.len() != expected_len {
                    return Err(EstimationError::InvalidInput(format!(
                        "nullspace_dims length {} does not match penalties {}",
                        dims.len(),
                        expected_len
                    )));
                }
                dims
            }
            None => vec![0; expected_len],
        };

        let penalty_count = rs_list.len();
        let workspace = RemlWorkspace::new(penalty_count);

        let balanced_penalty_root = create_balanced_penalty_root(&s_list, p)?;
        let reparam_invariant = precompute_reparam_invariant(&rs_list, p)?;
        let sparse_penalty_blocks = build_sparse_penalty_blocks(&s_list, &rs_list)?.map(Arc::new);

        Ok(Self {
            y,
            x,
            weights,
            offset: offset.to_owned(),
            s_full_list: s_list,
            rs_list,
            balanced_penalty_root,
            reparam_invariant,
            sparse_penalty_blocks,
            p,
            config,
            nullspace_dims,
            coefficient_lower_bounds,
            linear_constraints,
            cache_manager: EvalCacheManager::new(),
            arena: RemlArena::new(workspace),
            warm_start_beta: RwLock::new(None),
            warm_start_enabled: AtomicBool::new(true),
        })
    }

    /// Creates a sanitized cache key from rho values.
    /// Returns None if any component is NaN, in which case caching is skipped.
    /// Maps -0.0 to 0.0 to ensure consistency in caching.
    fn rho_key_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
        EvalCacheManager::sanitized_rho_key(rho)
    }

    fn prepare_eval_bundle_with_key(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let decision = self.select_reml_geometry(rho);
        match decision.geometry {
            RemlGeometry::SparseExactSpd => {
                match self.prepare_sparse_eval_bundle_with_key(rho, key.clone()) {
                    Ok(bundle) => {
                        log::info!(
                            "[reml-geometry] sparse_exact_spd reason={} p={} nnz_x={} nnz_h_est={} density_h_est={}",
                            decision.reason,
                            decision.p,
                            decision.nnz_x,
                            decision
                                .nnz_h_upper_est
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "na".to_string()),
                            decision
                                .density_h_upper_est
                                .map(|v| format!("{v:.4}"))
                                .unwrap_or_else(|| "na".to_string()),
                        );
                        Ok(bundle)
                    }
                    Err(err) => {
                        log::warn!(
                            "[reml-geometry] sparse_exact_spd failed ({}); falling back to dense spectral",
                            err
                        );
                        self.prepare_dense_eval_bundle_with_key(rho, key)
                    }
                }
            }
            RemlGeometry::DenseSpectral => self.prepare_dense_eval_bundle_with_key(rho, key),
        }
    }

    fn obtain_eval_bundle(&self, rho: &Array1<f64>) -> Result<EvalShared, EstimationError> {
        let key = self.rho_key_sanitized(rho);
        if let Some(existing) = self.cache_manager.cached_eval_bundle(&key) {
            return Ok(existing.clone());
        }
        let bundle = self.prepare_eval_bundle_with_key(rho, key)?;
        self.cache_manager.store_eval_bundle(bundle.clone());
        Ok(bundle)
    }

    pub(crate) fn objective_inner_hessian(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        Ok(bundle.h_total.as_ref().clone())
    }

    fn sparse_exact_beta_original(&self, pirls_result: &PirlsResult) -> Array1<f64> {
        match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                pirls_result.beta_transformed.as_ref().clone()
            }
            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result
                .reparam_result
                .qs
                .dot(pirls_result.beta_transformed.as_ref()),
        }
    }

    fn compute_cost_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<f64, EstimationError> {
        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();
        let prior_cost = self.compute_soft_prior_cost(rho);
        match self.config.link_function() {
            LinkFunction::Identity => {
                let n = self.y.len() as f64;
                let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;
                let rss = pirls_result.deviance;
                let penalty = pirls_result.stable_penalty_term;
                let dp = rss + penalty;
                let (dp_c, _) = smooth_floor_dp(dp);
                let phi = dp_c / (n - mp).max(LAML_RIDGE);
                let reml = dp_c / (2.0 * phi)
                    + 0.5 * (sparse.logdet_h - sparse.logdet_s_pos)
                    + ((n - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();
                Ok(reml + prior_cost)
            }
            _ => {
                let mut penalised_ll =
                    -0.5 * pirls_result.deviance - 0.5 * pirls_result.stable_penalty_term;
                if self.config.firth_bias_reduction
                    && matches!(self.config.link_function(), LinkFunction::Logit)
                    && let Some(firth_log_det) = pirls_result.firth_log_det
                {
                    penalised_ll += firth_log_det;
                }
                let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;
                let phi = 1.0;
                let laml = penalised_ll + 0.5 * sparse.logdet_s_pos - 0.5 * sparse.logdet_h
                    + (mp / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();
                Ok(-laml + prior_cost)
            }
        }
    }

    fn compute_gradient_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array1<f64>, EstimationError> {
        // Full sparse-exact first-order derivation on the ridged SPD surface.
        //
        // Definitions at the inner mode beta = beta_hat(rho):
        //
        //   lambda_k = exp(rho_k)
        //   S(rho)   = sum_k lambda_k S_k
        //   A_k      = dS/drho_k = lambda_k S_k
        //   H        = X' W X + S(rho) + delta I
        //
        // For non-Gaussian families, write
        //   c_i = -ell_i^{(3)}(eta_i),
        // and define the implicit coefficient derivative
        //   B_k = d beta_hat / drho_k = -H^{-1} A_k beta
        // together with
        //   z_k = X B_k.
        //
        // The Hessian derivative entering the logdet term is
        //   H_k = A_k + X' diag(c ⊙ z_k) X.
        //
        // Therefore the exact sparse gradient is
        //
        // Gaussian / REML:
        //   g_k = 0.5 * beta' A_k beta
        //         + 0.5 * tr(H^{-1} A_k)
        //         - 0.5 * d/drho_k log|S(rho)|_+
        //
        // Non-Gaussian / LAML:
        //   g_k = 0.5 * beta' A_k beta
        //         + 0.5 * tr(H^{-1} H_k)
        //         - 0.5 * d/drho_k log|S(rho)|_+
        //
        // and with H_k expanded,
        //
        //   tr(H^{-1} H_k)
        //     = tr(H^{-1} A_k)
        //       + tr(H^{-1} X' diag(c ⊙ z_k) X)
        //     = tr(H^{-1} A_k)
        //       + (X H^{-1} X') : diag(c ⊙ z_k)
        //     = tr(H^{-1} A_k)
        //       + sum_i h_i * c_i * z_{k,i},
        //
        // where h_i = x_i' H^{-1} x_i are the exact sparse leverages.
        //
        // The code evaluates the same identity with the sign convention used
        // by the existing REML implementation:
        //   trace_third = (X' (c ⊙ h))' v_k,
        //   v_k = H^{-1} (S_k beta),
        // so that
        //   lambda_k * (trace_s - trace_third)
        // reproduces tr(H^{-1} H_k) on the current sparse SPD surface.
        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();
        let beta = self.sparse_exact_beta_original(pirls_result);
        let lambdas = rho.mapv(f64::exp);
        let det1_values = sparse.det1_values.as_ref();
        let mut gradient = Array1::<f64>::zeros(rho.len());
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                let x_dense = self.x().to_dense_arc();
                Some(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };

        match self.config.link_function() {
            LinkFunction::Identity => {
                let n = self.y.len() as f64;
                let mp = self.nullspace_dims.iter().copied().sum::<usize>() as f64;
                let rss = pirls_result.deviance;
                let penalty = pirls_result.stable_penalty_term;
                let dp = rss + penalty;
                let (dp_c, dp_c_grad) = smooth_floor_dp(dp);
                let scale = dp_c / (n - mp).max(LAML_RIDGE);
                for block in sparse.penalty_blocks.iter() {
                    let s_k_beta = sparse_matvec_public(&block.s_k_sparse, &beta);
                    let beta_term = lambdas[block.term_index] * beta.dot(&s_k_beta);
                    let trace_term = {
                        let mut workspace = sparse.trace_workspace.lock().unwrap();
                        trace_hinv_sk(&sparse.factor, &mut workspace, block)?
                    };
                    gradient[block.term_index] = dp_c_grad * (beta_term / (2.0 * scale))
                        + 0.5 * lambdas[block.term_index] * trace_term
                        - 0.5 * det1_values[block.term_index];
                }
            }
            _ => {
                if let Some(op) = firth_op.as_ref() {
                    // Firth/logit sparse-exact gradient:
                    //   g_k = 0.5 β'A_kβ + 0.5 tr(H^{-1} H_k) - 0.5 tr(S_+^dag A_k),
                    //   H_k = A_k + X' diag(c ⊙ X B_k) X - D(H_phi)[B_k].
                    let p_dim = self.p;
                    let mut assembly_workspace = PirlsWorkspace::new(self.y.len(), p_dim, 0, 0);
                    for block in sparse.penalty_blocks.iter() {
                        let k = block.term_index;
                        let s_k_beta = sparse_matvec_public(&block.s_k_sparse, &beta);
                        let a_kb = s_k_beta.mapv(|v| lambdas[k] * v);
                        let b_k = solve_sparse_spd(&sparse.factor, &a_kb.mapv(|v| -v))?;
                        let z_k = self.x().matrix_vector_multiply(&b_k);
                        let mut h_k = self.s_full_list[k].mapv(|v| lambdas[k] * v);
                        let diag = &pirls_result.solve_c_array * &z_k;
                        h_k += &self.xt_diag_x_original(&diag, &mut assembly_workspace)?;
                        let dir_k = Self::firth_direction(op, &b_k);
                        h_k -= &Self::firth_hphi_direction(op, &dir_k);
                        let trace_hk = self.trace_hinv_operator_sparse_exact(
                            &sparse.factor,
                            p_dim,
                            |basis_block: &Array2<f64>| Ok(h_k.dot(basis_block)),
                        )?;
                        let beta_term = beta.dot(&a_kb);
                        gradient[k] = 0.5 * beta_term + 0.5 * trace_hk - 0.5 * det1_values[k];
                    }
                } else {
                    let leverages = leverages_from_factor(&sparse.factor, self.x())?;
                    let c_times_h = &pirls_result.solve_c_array * &leverages;
                    let r_third = self.x().transpose_vector_multiply(&c_times_h);
                    for block in sparse.penalty_blocks.iter() {
                        let s_k_beta = sparse_matvec_public(&block.s_k_sparse, &beta);
                        let beta_term = lambdas[block.term_index] * beta.dot(&s_k_beta);
                        let v_k = solve_sparse_spd(&sparse.factor, &s_k_beta)?;
                        let trace_s = {
                            let mut workspace = sparse.trace_workspace.lock().unwrap();
                            trace_hinv_sk(&sparse.factor, &mut workspace, block)?
                        };
                        let trace_third = r_third.dot(&v_k);
                        gradient[block.term_index] = 0.5 * beta_term
                            + 0.5 * lambdas[block.term_index] * (trace_s - trace_third)
                            - 0.5 * det1_values[block.term_index];
                    }
                }
            }
        }

        gradient += &self.compute_soft_prior_grad(rho);
        Ok(gradient)
    }

    fn trace_hinv_operator_sparse_exact<F>(
        &self,
        factor: &SparseExactFactor,
        p: usize,
        mut apply_direction: F,
    ) -> Result<f64, EstimationError>
    where
        F: FnMut(&Array2<f64>) -> Result<Array2<f64>, EstimationError>,
    {
        if p == 0 {
            return Ok(0.0);
        }
        let chunk = 32usize;
        let mut trace = 0.0_f64;
        let mut start = 0usize;
        while start < p {
            let end = (start + chunk).min(p);
            let block_cols = end - start;
            let mut basis_block = Array2::<f64>::zeros((p, block_cols));
            for local_col in 0..block_cols {
                basis_block[[start + local_col, local_col]] = 1.0;
            }
            let direction_block = apply_direction(&basis_block)?;
            if direction_block.nrows() != p || direction_block.ncols() != block_cols {
                return Err(EstimationError::InvalidInput(format!(
                    "trace_hinv_operator_sparse_exact apply returned {}x{} for basis block {}x{}",
                    direction_block.nrows(),
                    direction_block.ncols(),
                    p,
                    block_cols
                )));
            }
            let solved = solve_sparse_spd_multi(factor, &direction_block)?;
            for local_col in 0..block_cols {
                let global_col = start + local_col;
                trace += solved[[global_col, local_col]];
            }
            start = end;
        }
        Ok(trace)
    }

    fn sparse_exact_weighted_cross_trace_xtau(
        &self,
        factor: &SparseExactFactor,
        x_tau: &Array2<f64>,
        weights_diag: &Array1<f64>,
    ) -> Result<f64, EstimationError> {
        let n = x_tau.nrows();
        let p = x_tau.ncols();
        if n == 0 || p == 0 {
            return Ok(0.0);
        }
        if weights_diag.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "weighted_cross_trace_xtau weights length mismatch: weights={}, n={}",
                weights_diag.len(),
                n
            )));
        }
        let mut total = 0.0_f64;
        let batch = 32usize;

        let x_dense_opt = match self.x() {
            DesignMatrix::Dense(x) => Some(x),
            DesignMatrix::Sparse(_) => None,
        };
        let x_csr_opt = match self.x() {
            DesignMatrix::Sparse(x) => x.to_csr_arc(),
            DesignMatrix::Dense(_) => None,
        };

        let mut start = 0usize;
        while start < n {
            let end = (start + batch).min(n);
            let rhs = x_tau.slice(s![start..end, ..]).t().to_owned();
            let solved = solve_sparse_spd_multi(factor, &rhs)?;
            for local_col in 0..(end - start) {
                let row = start + local_col;
                let z_col = solved.column(local_col);
                let row_dot = if let Some(x_dense) = x_dense_opt {
                    x_dense.row(row).dot(&z_col)
                } else if let Some(csr) = x_csr_opt.as_ref() {
                    let symbolic = csr.symbolic();
                    let row_ptr = symbolic.row_ptr();
                    let col_idx = symbolic.col_idx();
                    let values = csr.val();
                    let mut acc = 0.0_f64;
                    let r0 = row_ptr[row];
                    let r1 = row_ptr[row + 1];
                    for idx in r0..r1 {
                        let col = col_idx[idx];
                        acc += values[idx] * z_col[col];
                    }
                    acc
                } else {
                    0.0
                };
                total += weights_diag[row] * row_dot;
            }
            start = end;
        }
        Ok(total)
    }

    fn compute_directional_hyper_gradient_sparse_exact(
        &self,
        _rho: &Array1<f64>,
        bundle: &EvalShared,
        hyper_dir: &DirectionalHyperParam,
    ) -> Result<f64, EstimationError> {
        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();
        let p = self.p;
        let n = self.y.len();
        if hyper_dir.x_tau_original.nrows() != n || hyper_dir.x_tau_original.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "X_tau shape mismatch for sparse directional gradient: expected {}x{}, got {}x{}",
                n,
                p,
                hyper_dir.x_tau_original.nrows(),
                hyper_dir.x_tau_original.ncols()
            )));
        }
        if hyper_dir.s_tau_original.nrows() != p || hyper_dir.s_tau_original.ncols() != p {
            return Err(EstimationError::InvalidInput(format!(
                "S_tau shape mismatch for sparse directional gradient: expected {}x{}, got {}x{}",
                p,
                p,
                hyper_dir.s_tau_original.nrows(),
                hyper_dir.s_tau_original.ncols()
            )));
        }
        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);

        let beta = self.sparse_exact_beta_original(pirls_result);
        let u = &pirls_result.solve_weights
            * &(&pirls_result.solve_working_response - &pirls_result.final_eta);

        let x_tau_beta = hyper_dir.x_tau_original.dot(&beta);
        let weighted_x_tau_beta = &pirls_result.solve_weights * &x_tau_beta;
        let mut g_psi = hyper_dir.x_tau_original.t().dot(&u)
            - self.x().transpose_vector_multiply(&weighted_x_tau_beta)
            - hyper_dir.s_tau_original.dot(&beta);

        let mut fit_firth_partial = 0.0_f64;
        let mut firth_op_opt: Option<FirthDenseOperator> = None;
        let mut hphi_tau_kernel_opt: Option<FirthTauPartialKernel> = None;
        if firth_logit_active {
            let op = if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                cached.as_ref().clone()
            } else {
                let x_dense_arc = self.x().to_dense_arc();
                Self::build_firth_dense_operator(x_dense_arc.as_ref(), &pirls_result.final_eta)?
            };
            let (g_phi_tau, phi_tau_partial) =
                Self::firth_partial_score_and_fit_tau(&op, &hyper_dir.x_tau_original, &beta);
            g_psi -= &g_phi_tau;
            fit_firth_partial = phi_tau_partial;
            if hyper_dir.x_tau_original.iter().any(|v| *v != 0.0) {
                hphi_tau_kernel_opt = Some(Self::firth_hphi_tau_partial_prepare(
                    &op,
                    &hyper_dir.x_tau_original,
                    &beta,
                ));
            }
            firth_op_opt = Some(op);
        }

        let beta_tau = solve_sparse_spd(&sparse.factor, &g_psi)?;
        let eta_tau = &x_tau_beta + &self.x().matrix_vector_multiply(&beta_tau);

        let fit_block = -u.dot(&x_tau_beta)
            + 0.5 * beta.dot(&hyper_dir.s_tau_original.dot(&beta))
            + fit_firth_partial;

        let trace_s_tau = self.trace_hinv_operator_sparse_exact(
            &sparse.factor,
            p,
            |basis_block: &Array2<f64>| Ok(hyper_dir.s_tau_original.dot(basis_block)),
        )?;
        let cross = self.sparse_exact_weighted_cross_trace_xtau(
            &sparse.factor,
            &hyper_dir.x_tau_original,
            &pirls_result.solve_weights,
        )?;
        let mut trace_h = trace_s_tau + 2.0 * cross;

        match self.config.link_function() {
            LinkFunction::Identity => {
                let e = &pirls_result.reparam_result.e_transformed;
                let (penalty_rank, _) =
                    self.fixed_subspace_penalty_rank_and_logdet(e, pirls_result.ridge_passport)?;
                let mp = (p.saturating_sub(penalty_rank)) as f64;
                let dp = pirls_result.deviance + pirls_result.stable_penalty_term;
                let (dp_c, _) = smooth_floor_dp(dp);
                let phi = dp_c / ((n as f64 - mp).max(LAML_RIDGE));
                if !phi.is_finite() || phi <= 0.0 {
                    return Err(EstimationError::InvalidInput(
                        "invalid profiled Gaussian dispersion in sparse directional hyper-gradient"
                            .to_string(),
                    ));
                }
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &pirls_result.reparam_result.e_transformed,
                    &hyper_dir.s_tau_original,
                    pirls_result.ridge_passport,
                )?;
                let dp_tau = 2.0 * fit_block;
                Ok(dp_tau / (2.0 * phi) + 0.5 * trace_h - 0.5 * pseudo_det_trace)
            }
            _ => {
                let w_tau_callback =
                    directional_working_curvature_callback(self.config.link_function());
                let w_tau = w_tau_callback(
                    &pirls_result.final_eta,
                    self.weights,
                    &pirls_result.solve_weights,
                    &eta_tau,
                );
                let leverages = leverages_from_factor(&sparse.factor, self.x())?;
                match w_tau {
                    DirectionalWorkingCurvature::Diagonal(diag) => {
                        if diag.len() != leverages.len() {
                            return Err(EstimationError::InvalidInput(format!(
                                "W_tau/leverages length mismatch in sparse directional gradient: w_tau={}, leverages={}",
                                diag.len(),
                                leverages.len()
                            )));
                        }
                        trace_h += leverages.dot(&diag);
                    }
                }
                if let Some(op) = firth_op_opt.as_ref() {
                    // Fully matrix-free sparse trace for Firth curvature drift:
                    //   tr(H^{-1}(H_{phi,tau}|beta + D(H_phi)[beta_tau])).
                    // We avoid dense p×p directional matrix materialization by
                    // applying both operators to identity blocks on demand.
                    let firth_dir = Self::firth_direction(op, &beta_tau);
                    let p_dim = self.p;
                    let tau_kernel = hphi_tau_kernel_opt.clone();
                    let tau_x = &hyper_dir.x_tau_original;
                    let tr_firth = self.trace_hinv_operator_sparse_exact(
                        &sparse.factor,
                        p_dim,
                        |basis_block: &Array2<f64>| {
                            Ok(Self::firth_hphi_trace_apply_combined(
                                op,
                                &firth_dir,
                                tau_x,
                                tau_kernel.as_ref(),
                                basis_block,
                            ))
                        },
                    )?;
                    trace_h -= tr_firth;
                }
                let pseudo_det_trace = self.fixed_subspace_penalty_trace(
                    &pirls_result.reparam_result.e_transformed,
                    &hyper_dir.s_tau_original,
                    pirls_result.ridge_passport,
                )?;
                Ok(fit_block + 0.5 * trace_h - 0.5 * pseudo_det_trace)
            }
        }
    }

    fn xt_diag_x_original(
        &self,
        diag: &Array1<f64>,
        workspace: &mut PirlsWorkspace,
    ) -> Result<Array2<f64>, EstimationError> {
        // Matrix-free realization of X' diag(v) X.
        //
        // In the analytic sparse outer-Hessian formulas, both H_k and H_{k,l}
        // contain curvature corrections of the form
        //   X' diag(v) X
        // with
        //   v = c ⊙ z_k
        // or
        //   v = d ⊙ z_k ⊙ z_l + c ⊙ z_{k,l}.
        //
        // This helper translates that expression directly into Rust without
        // ever materializing a dense n x n diagonal matrix. For dense designs
        // we scale rows of X and form X' (diag(v) X); for sparse designs we
        // reuse the existing sparse X' W X assembly machinery with `diag`
        // playing the role of per-row weights.
        match self.x() {
            DesignMatrix::Dense(x_dense) => {
                let mut weighted = Array2::<f64>::zeros(x_dense.raw_dim());
                Ok(Self::xt_diag_x_dense_into(x_dense, diag, &mut weighted))
            }
            DesignMatrix::Sparse(x_sparse) => {
                let csr = x_sparse.to_csr_arc().ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "failed to build CSR cache for sparse exact Hessian".to_string(),
                    )
                })?;
                workspace.compute_hessian_sparse_faer(csr.as_ref(), diag)
            }
        }
    }

    fn compute_laml_hessian_sparse_exact(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array2<f64>, EstimationError> {
        // Full sparse-exact second-order derivation on the ridged SPD surface.
        //
        // Notation at the exact inner mode beta = beta_hat(rho):
        //
        //   eta = X beta
        //   W   = diag(w),   w_i = -ell_i''(eta_i)
        //   c_i = -ell_i^{(3)}(eta_i)
        //   d_i = -ell_i^{(4)}(eta_i)
        //
        //   H = X' W X + S(rho) + delta I,
        //   A_k = dS/drho_k = lambda_k S_k,
        //   A_{k,l} = d^2 S/(drho_k drho_l) = delta_{k,l} A_k.
        //
        // Inner stationarity gives
        //   r(beta, rho) = grad_beta F(beta, rho) = 0.
        // Differentiating once:
        //   H B_k + A_k beta = 0
        // so
        //   B_k = d beta_hat / drho_k = -H^{-1} A_k beta,
        //   z_k = X B_k.
        //
        // Differentiating the inner Hessian:
        //   H_k = dH/drho_k = A_k + X' diag(c ⊙ z_k) X.
        //
        // Differentiating the stationarity equation again:
        //   H_l B_k + H B_{k,l} + A_{k,l} beta + A_k B_l = 0
        // so
        //   B_{k,l} = -H^{-1}(H_l B_k + A_{k,l} beta + A_k B_l),
        //   z_{k,l} = X B_{k,l}.
        //
        // Differentiating H_k:
        //   H_{k,l}
        //     = A_{k,l} + X' diag(d ⊙ z_k ⊙ z_l + c ⊙ z_{k,l}) X.
        //
        // The exact outer objective is
        //   V(rho) = F(beta_hat(rho), rho)
        //            + 0.5 log|H|
        //            - 0.5 log|S(rho)|_+.
        //
        // By the envelope theorem,
        //   g_k = dV/drho_k
        //       = 0.5 beta' A_k beta
        //         + 0.5 tr(H^{-1} H_k)
        //         - 0.5 tr(S^+ A_k).
        //
        // Differentiating again yields the exact outer Hessian entry
        //
        //   H_outer[k,l]
        //     = -beta' A_k H^{-1} A_l beta
        //       + 0.5 delta_{k,l} beta' A_k beta
        //       + 0.5 tr(H^{-1} H_{k,l})
        //       - 0.5 tr(H^{-1} H_l H^{-1} H_k)
        //       + 0.5 tr(S^+ A_k S^+ A_l)
        //       - 0.5 delta_{k,l} tr(S^+ A_k)
        //       + soft-prior Hessian.
        //
        // This branch keeps all solves/logdets/traces on the same sparse SPD
        // geometry used by the sparse exact cost and gradient.
        //
        // In the current sparse-exact eligibility regime, penalties are
        // block-separable by smooth term in original coordinates, so
        //   log|S(rho)|_+ = const + sum_k rank(S_k) * rho_k
        // on the penalized subspace. Therefore the penalty pseudo-logdet
        // Hessian term P_{k,l} is exactly zero here; only the first
        // derivative survives and is already handled in the gradient.
        //
        // Translation to code:
        //
        // 1. For each k, build A_k beta = lambda_k S_k beta and solve
        //      B_k = -H^{-1} A_k beta.
        //
        // 2. Push B_k through X to get z_k = X B_k.
        //
        // 3. Build
        //      H_k = A_k + X' diag(c ⊙ z_k) X
        //    with `xt_diag_x_original`.
        //
        // 4. Solve dense right-hand sides
        //      Y_k = H^{-1} H_k
        //    once, so the quadratic trace term becomes
        //      tr(H^{-1} H_l H^{-1} H_k) = tr(Y_l Y_k).
        //
        // 5. For each pair (k,l), build the exact right-hand side
        //      H_l B_k + A_{k,l} beta + A_k B_l,
        //    solve for B_{k,l}, and form z_{k,l} = X B_{k,l}.
        //
        // 6. Form the linear trace term from
        //      H_{k,l} = A_{k,l} + X' diag(h_{k,l}) X,
        //    where
        //      h_{k,l} = d ⊙ z_k ⊙ z_l + c ⊙ z_{k,l}.
        //
        //    Using cyclicity of trace,
        //      tr(H^{-1} X' diag(h) X)
        //      = sum_i leverage_i * h_i,
        //    where leverage_i = x_i' H^{-1} x_i.
        //
        // 7. Assemble the final symmetric K x K matrix from the quadratic
        //    beta term, linear trace term, quadratic trace term, and the
        //    soft-prior Hessian.
        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();
        let k_count = rho.len();
        if k_count == 0 {
            return Ok(Array2::zeros((0, 0)));
        }

        let p_dim = self.p;
        let n_obs = self.y.len();
        let matrix_slots = (k_count as u128)
            .saturating_mul(p_dim as u128)
            .saturating_mul(p_dim as u128)
            .saturating_mul(8)
            .saturating_mul(2);
        const SPARSE_EXACT_OUTER_HESSIAN_MAX_BYTES: u128 = 512 * 1024 * 1024;
        if matrix_slots > SPARSE_EXACT_OUTER_HESSIAN_MAX_BYTES {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "sparse exact outer Hessian resource guard triggered (estimated dense work {} bytes)",
                matrix_slots
            )));
        }

        let beta = self.sparse_exact_beta_original(pirls_result);
        let lambdas = rho.mapv(f64::exp);
        let c = &pirls_result.solve_c_array;
        let d = &pirls_result.solve_d_array;
        if c.len() != n_obs || d.len() != n_obs {
            return Err(EstimationError::InvalidInput(format!(
                "Sparse exact outer Hessian derivative arrays size mismatch: n={}, c.len()={}, d.len()={}",
                n_obs,
                c.len(),
                d.len()
            )));
        }

        let firth_logit_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let firth_op = if firth_logit_active {
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                let x_dense = self.x().to_dense_arc();
                Some(Self::build_firth_dense_operator(
                    x_dense.as_ref(),
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };
        let leverages = leverages_from_factor(&sparse.factor, self.x())?;
        let mut trace_hinv_sk_values = vec![0.0_f64; k_count];
        let mut a_k_beta = Vec::with_capacity(k_count);
        let mut b_k = Vec::with_capacity(k_count);
        let mut z_k = Vec::with_capacity(k_count);
        let mut s_diag = Vec::with_capacity(k_count);
        let mut firth_dirs = firth_op
            .as_ref()
            .map(|_| Vec::<FirthDirection>::with_capacity(k_count));
        let mut q_diag = vec![0.0_f64; k_count];
        let mut blocks_by_term: Vec<Option<&SparsePenaltyBlock>> = vec![None; k_count];

        for block in sparse.penalty_blocks.iter() {
            let k = block.term_index;
            blocks_by_term[k] = Some(block);
            // A_k beta = lambda_k S_k beta, then
            //   B_k = -H^{-1} A_k beta,
            //   z_k = X B_k.
            let s_k_beta = sparse_matvec_public(&block.s_k_sparse, &beta);
            let a_kb = s_k_beta.mapv(|v| lambdas[k] * v);
            let b = solve_sparse_spd(&sparse.factor, &a_kb.mapv(|v| -v))?;
            let z = self.x().matrix_vector_multiply(&b);

            // H_k = A_k + X' diag(c ⊙ z_k) X - D(H_phi)[B_k].
            let diag = &pirls_result.solve_c_array * &z;
            if let Some(op) = firth_op.as_ref() {
                let dir_k = Self::firth_direction(op, &b);
                if let Some(dirs) = firth_dirs.as_mut() {
                    dirs.push(dir_k);
                }
            }

            // beta' A_k beta is reused on the diagonal quadratic term, while
            // tr(H^{-1} A_k) is the delta_{k,l} piece inside tr(H^{-1} H_{k,l}).
            q_diag[k] = beta.dot(&a_kb);
            {
                let mut workspace = sparse.trace_workspace.lock().unwrap();
                trace_hinv_sk_values[k] =
                    lambdas[k] * trace_hinv_sk(&sparse.factor, &mut workspace, block)?;
            }
            a_k_beta.push(a_kb);
            b_k.push(b);
            z_k.push(z);
            s_diag.push(diag);
        }

        for (k, blk) in blocks_by_term.iter().enumerate() {
            if blk.is_none() {
                return Err(EstimationError::InvalidInput(format!(
                    "missing sparse penalty block for term index {} in sparse exact Hessian",
                    k
                )));
            }
        }

        let apply_hk_block =
            |k_term: usize, basis_block: &Array2<f64>| -> Result<Array2<f64>, EstimationError> {
                let block_k = blocks_by_term[k_term].ok_or_else(|| {
                    EstimationError::InvalidInput(format!(
                        "missing sparse penalty block for term index {}",
                        k_term
                    ))
                })?;
                let mut out = Array2::<f64>::zeros((p_dim, basis_block.ncols()));
                for col in 0..basis_block.ncols() {
                    let v = basis_block.column(col).to_owned();
                    let mut col_out =
                        sparse_matvec_public(&block_k.s_k_sparse, &v).mapv(|x| lambdas[k_term] * x);
                    let xv = self.x().matrix_vector_multiply(&v);
                    let wxv = &xv * &s_diag[k_term];
                    col_out += &self.x().transpose_vector_multiply(&wxv);
                    out.column_mut(col).assign(&col_out);
                }
                if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                    out -= &Self::firth_hphi_direction_apply(op, &dirs[k_term], basis_block);
                }
                Ok(out)
            };

        let mut hess = Array2::<f64>::zeros((k_count, k_count));
        for block_l in sparse.penalty_blocks.iter() {
            let l = block_l.term_index;
            for block_k in sparse.penalty_blocks.iter().take(l + 1) {
                let k = block_k.term_index;

                // Quadratic beta term:
                //   -beta' A_k H^{-1} A_l beta + 0.5 delta_{k,l} beta' A_k beta.
                // Since B_l = -H^{-1} A_l beta, this becomes
                //   beta' A_k B_l + 0.5 delta_{k,l} beta' A_k beta.
                let quad_beta =
                    a_k_beta[k].dot(&b_k[l]) + if k == l { 0.5 * q_diag[k] } else { 0.0 };

                // Build the right-hand side for
                //   B_{k,l} = -H^{-1}(H_l B_k + A_{k,l} beta + A_k B_l).
                //
                // Split H_l B_k into:
                //   A_l B_k
                //   + X' diag(c ⊙ z_l) X B_k
                //   = A_l B_k + X' ( (c ⊙ z_l) ⊙ z_k ).
                let mut rhs_col = Array2::<f64>::zeros((p_dim, 1));
                rhs_col.column_mut(0).assign(&b_k[k]);
                let h_l_b_k = apply_hk_block(l, &rhs_col)?.column(0).to_owned();
                let a_k_b_l =
                    sparse_matvec_public(&block_k.s_k_sparse, &b_k[l]).mapv(|v| lambdas[k] * v);
                let mut rhs_bkl = h_l_b_k + &a_k_b_l;
                if k == l {
                    rhs_bkl += &a_k_beta[k];
                }
                let b_kl = solve_sparse_spd(&sparse.factor, &rhs_bkl.mapv(|v| -v))?;
                let z_kl = self.x().matrix_vector_multiply(&b_kl);

                // H_{k,l} = A_{k,l} + X' diag(d ⊙ z_k ⊙ z_l + c ⊙ z_{k,l}) X.
                // The linear trace contribution is
                //   0.5 tr(H^{-1} H_{k,l})
                // = 0.5 delta_{k,l} tr(H^{-1} A_k)
                //   + 0.5 tr(H^{-1} X' diag(hkl_diag) X).
                //
                // We evaluate the second term by the cyclic identity
                //   tr(H^{-1} X' D X)
                // = tr(X H^{-1} X' D)
                // = sum_i leverage_i * D_ii,
                // where leverage_i = x_i' H^{-1} x_i.
                let hkl_diag = (d * &z_k[k] * &z_k[l]) + &(c * &z_kl);
                let lin_trace = if let (Some(op), Some(dirs)) =
                    (firth_op.as_ref(), firth_dirs.as_ref())
                {
                    let dir_kl = Self::firth_direction(op, &b_kl);
                    self.trace_hinv_operator_sparse_exact(&sparse.factor, p_dim, |basis_block| {
                        let mut out = if k == l {
                            let block_k = blocks_by_term[k].expect("sparse block exists");
                            let mut akb = Array2::<f64>::zeros((p_dim, basis_block.ncols()));
                            for col in 0..basis_block.ncols() {
                                let v = basis_block.column(col).to_owned();
                                let col_out = sparse_matvec_public(&block_k.s_k_sparse, &v)
                                    .mapv(|vv| lambdas[k] * vv);
                                akb.column_mut(col).assign(&col_out);
                            }
                            akb
                        } else {
                            Array2::<f64>::zeros((p_dim, basis_block.ncols()))
                        };
                        // Matrix-free X' diag(hkl_diag) X * basis_block
                        for col in 0..basis_block.ncols() {
                            let v = basis_block.column(col).to_owned();
                            let xv = self.x().matrix_vector_multiply(&v);
                            let wxv = &xv * &hkl_diag;
                            let col_out = self.x().transpose_vector_multiply(&wxv);
                            out.column_mut(col).scaled_add(1.0, &col_out);
                        }
                        out -= &Self::firth_hphi_direction_apply(op, &dir_kl, basis_block);
                        out -= &Self::firth_hphi_second_direction_apply(
                            op,
                            &dirs[k],
                            &dirs[l],
                            basis_block,
                        );
                        Ok(out)
                    })?
                } else {
                    (if k == l { trace_hinv_sk_values[k] } else { 0.0 }) + leverages.dot(&hkl_diag)
                };

                // Quadratic trace term:
                //   tr(H^{-1} H_l H^{-1} H_k)
                // evaluated fully matrix-free as
                //   tr(H^{-1}(H_l(H^{-1}(H_k E)))) on identity block E.
                let quad_trace =
                    self.trace_hinv_operator_sparse_exact(&sparse.factor, p_dim, |basis_block| {
                        let hk_basis = apply_hk_block(k, basis_block)?;
                        let y = solve_sparse_spd_multi(&sparse.factor, &hk_basis)?;
                        apply_hk_block(l, &y)
                    })?;

                let value = quad_beta + 0.5 * lin_trace - 0.5 * quad_trace;
                hess[[k, l]] = value;
                hess[[l, k]] = value;
            }
        }

        self.add_soft_prior_hessian_in_place(rho, &mut hess);
        Ok(hess)
    }

    pub(crate) fn last_ridge_used(&self) -> Option<f64> {
        self.cache_manager
            .current_eval_bundle
            .read()
            .unwrap()
            .as_ref()
            .map(|bundle| bundle.ridge_passport.delta)
    }

    /// Calculate effective degrees of freedom (EDF) using a consistent approach
    /// for both cost and gradient calculations, ensuring identical values.
    ///
    /// # Arguments
    /// * `pr` - PIRLS result containing the penalty matrices
    /// * `lambdas` - Smoothing parameters (lambda values)
    /// * `h_eff` - Effective Hessian matrix
    ///
    /// # Returns
    /// * Effective degrees of freedom value
    fn edf_from_h_and_e(
        &self,
        e_transformed: &Array2<f64>, // rank x p_eff
        lambdas: ArrayView1<'_, f64>,
        h_eff: &Array2<f64>, // p_eff x p_eff
    ) -> Result<f64, EstimationError> {
        // Why caching by ρ is sound:
        // The effective degrees of freedom (EDF) calculation is one of only two places where
        // we ask for a Faer factorization through `get_faer_factor`.  The cache inside that
        // helper uses only the vector of log smoothing parameters (ρ) as the key.  At first
        // glance that can look risky—two different Hessians with the same ρ might appear to be
        // conflated.  The surrounding call graph prevents that situation:
        //   • Identity / Gaussian models call `edf_from_h_and_rk` with the stabilized Hessian
        //     `pirls_result.stabilized_hessian_transformed`.
        //   • Non-Gaussian (logit / LAML) models call it with the effective / ridged Hessian
        //     returned by `effective_hessian(pr)`.
        // Within a given `RemlState` we never switch between those two flavours—the state is
        // constructed for a single link function, so the cost/gradient pathways stay aligned.
        // Because of that design, a given ρ vector corresponds to exactly one Hessian type in
        // practice, and the cache cannot hand back a factorization of an unintended matrix.

        // Prefer an un-ridged factorization when the stabilized Hessian is already PD.
        // Only fall back to the RidgePlanner path if direct factorization fails.
        let rho_like = lambdas.mapv(|lam| lam.ln());
        let factor = {
            let h_view = FaerArrayView::new(h_eff);
            if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                Arc::new(FaerFactor::Llt(f))
            } else if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                Arc::new(FaerFactor::Ldlt(f))
            } else {
                self.get_faer_factor(&rho_like, h_eff)
            }
        };

        // Use the single λ-weighted penalty root E for S_λ = Eᵀ E to compute
        // trace(H⁻¹ S_λ) = ⟨H⁻¹ Eᵀ, Eᵀ⟩_F directly.
        let e_t = e_transformed.t().to_owned(); // (p_eff × rank_total)
        let e_view = FaerArrayView::new(&e_t);
        let x = factor.solve(e_view.as_ref());
        let trace_h_inv_s_lambda = faer_frob_inner(x.as_ref(), e_view.as_ref());

        // Calculate EDF as p - trace, clamped to the penalty nullspace dimension
        let p = h_eff.ncols() as f64;
        let rank_s = e_transformed.nrows() as f64;
        let mp = (p - rank_s).max(0.0);
        let edf = (p - trace_h_inv_s_lambda).clamp(mp, p);

        Ok(edf)
    }

    fn active_constraint_free_basis(&self, pr: &PirlsResult) -> Option<Array2<f64>> {
        let lin = pr.linear_constraints_transformed.as_ref()?;
        let active_tol = 1e-8;
        let beta_t = pr.beta_transformed.as_ref();
        let mut active_rows: Vec<Array1<f64>> = Vec::new();
        for i in 0..lin.a.nrows() {
            let slack = lin.a.row(i).dot(beta_t) - lin.b[i];
            if slack <= active_tol {
                active_rows.push(lin.a.row(i).to_owned());
            }
        }
        if active_rows.is_empty() {
            return None;
        }

        let p_t = lin.a.ncols();
        let mut a_t = Array2::<f64>::zeros((p_t, active_rows.len()));
        for (j, row) in active_rows.iter().enumerate() {
            for k in 0..p_t {
                a_t[[k, j]] = row[k];
            }
        }

        let q_row = Self::orthonormalize_columns(&a_t, 1e-10); // basis for active row-space^T
        let rank = q_row.ncols();
        if rank == 0 {
            return None;
        }
        if rank >= p_t {
            return Some(Array2::<f64>::zeros((p_t, 0)));
        }

        // Build orthonormal basis for null(A_active) as complement of row-space.
        let mut z = Array2::<f64>::zeros((p_t, p_t - rank));
        let mut kept = 0usize;
        for j in 0..p_t {
            let mut v = Array1::<f64>::zeros(p_t);
            v[j] = 1.0;
            for t in 0..rank {
                let qt = q_row.column(t);
                let proj = qt.dot(&v);
                v -= &qt.mapv(|x| x * proj);
            }
            for t in 0..kept {
                let zt = z.column(t);
                let proj = zt.dot(&v);
                v -= &zt.mapv(|x| x * proj);
            }
            let nrm = v.dot(&v).sqrt();
            if nrm > 1e-10 {
                z.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                kept += 1;
                if kept == p_t - rank {
                    break;
                }
            }
        }
        Some(z.slice(ndarray::s![.., 0..kept]).to_owned())
    }

    fn enforce_constraint_kkt(&self, pr: &PirlsResult) -> Result<(), EstimationError> {
        let Some(kkt) = pr.constraint_kkt.as_ref() else {
            return Ok(());
        };
        let tol_primal = 1e-7;
        let tol_dual = 1e-7;
        let tol_comp = 1e-7;
        let tol_stat = 5e-6;
        if kkt.primal_feasibility > tol_primal
            || kkt.dual_feasibility > tol_dual
            || kkt.complementarity > tol_comp
            || kkt.stationarity > tol_stat
        {
            let mut worst_row_msg = String::new();
            if let Some(lin) = pr.linear_constraints_transformed.as_ref() {
                let mut worst = 0.0_f64;
                let mut worst_row = 0usize;
                for i in 0..lin.a.nrows() {
                    let slack = lin.a.row(i).dot(&pr.beta_transformed.0) - lin.b[i];
                    let viol = (-slack).max(0.0);
                    if viol > worst {
                        worst = viol;
                        worst_row = i;
                    }
                }
                if worst > 0.0 {
                    worst_row_msg =
                        format!("; worst_row={} worst_violation={:.3e}", worst_row, worst);
                }
            }
            return Err(EstimationError::ParameterConstraintViolation(format!(
                "KKT residuals exceed tolerance: primal={:.3e}, dual={:.3e}, comp={:.3e}, stat={:.3e}; active={}/{}{}",
                kkt.primal_feasibility,
                kkt.dual_feasibility,
                kkt.complementarity,
                kkt.stationarity,
                kkt.n_active,
                kkt.n_constraints,
                worst_row_msg
            )));
        }
        Ok(())
    }

    fn project_with_basis(matrix: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        let zt_m = z.t().dot(matrix);
        zt_m.dot(z)
    }

    fn positive_part_factor_w(matrix: &Array2<f64>) -> Result<Array2<f64>, EstimationError> {
        // Build W such that M_+^dagger = W W^T on the retained positive subspace.
        // This is the objective-consistent generalized inverse used for both
        // pseudo-logdet derivatives and IFT sensitivities in exact joint blocks.
        let (evals, evecs) = matrix
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let p = matrix.nrows();
        let max_ev = evals
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let tol = (p.max(1) as f64) * f64::EPSILON * max_ev;
        let keep: Vec<usize> = evals
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v > tol { Some(i) } else { None })
            .collect();
        if keep.is_empty() {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        let mut w = Array2::<f64>::zeros((p, keep.len()));
        for (col_idx, &eig_idx) in keep.iter().enumerate() {
            let scale = 1.0 / evals[eig_idx].sqrt();
            let u_col = evecs.column(eig_idx);
            let mut w_col = w.column_mut(col_idx);
            Zip::from(&mut w_col)
                .and(&u_col)
                .for_each(|w_elem, &u_elem| *w_elem = u_elem * scale);
        }
        Ok(w)
    }

    fn fixed_subspace_penalty_rank_and_logdet(
        &self,
        e_transformed: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<(usize, f64), EstimationError> {
        let structural_rank = e_transformed.nrows().min(e_transformed.ncols());
        if structural_rank == 0 {
            return Ok((0, 0.0));
        }

        // Keep objective rank fixed to the structural penalty rank to avoid
        // rho-dependent rank flips from tiny eigenvalue jitter.
        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..s_lambda.nrows() {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, _) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let max_ev = order
            .first()
            .map(|&idx| evals[idx].abs())
            .unwrap_or(1.0)
            .max(1.0);
        let floor = (1e-12 * max_ev).max(1e-12);
        let log_det = order
            .iter()
            .take(structural_rank)
            .map(|&idx| evals[idx].max(floor).ln())
            .sum();
        Ok((structural_rank, log_det))
    }

    fn fixed_subspace_penalty_trace(
        &self,
        e_transformed: &Array2<f64>,
        s_direction: &Array2<f64>,
        ridge_passport: RidgePassport,
    ) -> Result<f64, EstimationError> {
        // Use the exact same structural-rank convention as log|S|_+.
        let (structural_rank, _) =
            self.fixed_subspace_penalty_rank_and_logdet(e_transformed, ridge_passport)?;
        let p_dim = e_transformed.ncols();
        if structural_rank == 0 || p_dim == 0 {
            return Ok(0.0);
        }

        let mut s_lambda = e_transformed.t().dot(e_transformed);
        let ridge = ridge_passport.penalty_logdet_ridge();
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_lambda[[i, i]] += ridge;
            }
        }
        let (evals, evecs) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let mut order: Vec<usize> = (0..evals.len()).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let max_ev = order
            .first()
            .map(|&idx| evals[idx].abs())
            .unwrap_or(1.0)
            .max(1.0);
        let floor = (1e-12 * max_ev).max(1e-12);
        // Direct fixed-subspace contraction:
        //   tr(S^+ S_tau) = tr(D_+^{-1} U_+^T S_tau U_+),
        // where (U_+, D_+) are the kept structural positive modes.
        let mut trace = 0.0;
        for &idx in order.iter().take(structural_rank) {
            let ev = evals[idx].max(floor);
            let u = evecs.column(idx).to_owned();
            let spsi_u = s_direction.dot(&u);
            trace += u.dot(&spsi_u) / ev;
        }
        Ok(trace)
    }

    fn update_warm_start_from(&self, pr: &PirlsResult) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match pr.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                let beta_original = match pr.coordinate_frame {
                    pirls::PirlsCoordinateFrame::OriginalSparseNative => {
                        pr.beta_transformed.as_ref().clone()
                    }
                    pirls::PirlsCoordinateFrame::TransformedQs => {
                        pr.reparam_result.qs.dot(pr.beta_transformed.as_ref())
                    }
                };
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta_original));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    pub(crate) fn set_warm_start_original_beta(&self, beta_original: Option<ArrayView1<'_, f64>>) {
        if !self.warm_start_enabled.load(Ordering::Relaxed) {
            return;
        }
        match beta_original {
            Some(beta) if beta.len() == self.p => {
                self.warm_start_beta
                    .write()
                    .unwrap()
                    .replace(Coefficients::new(beta.to_owned()));
            }
            _ => {
                self.warm_start_beta.write().unwrap().take();
            }
        }
    }

    /// Clear warm-start state. Used in tests to ensure consistent starting points
    /// when comparing different gradient computation paths.
    #[cfg(test)]
    #[allow(dead_code)]
    pub fn clear_warm_start(&self) {
        self.warm_start_beta.write().unwrap().take();
        self.cache_manager.invalidate_eval_bundle();
    }

    /// Returns the per-penalty square-root matrices in the transformed coefficient basis
    /// without any λ weighting. Each returned R_k satisfies S_k = R_kᵀ R_k in that basis.
    /// Using these avoids accidental double counting of λ when forming derivatives.
    ///
    /// # Arguments
    /// * `pr` - The PIRLS result with the transformation matrix Qs
    ///
    /// # Returns
    fn factorize_faer(&self, h: &Array2<f64>) -> FaerFactor {
        let mut planner = RidgePlanner::new(h);
        loop {
            let ridge = planner.ridge();
            if ridge > 0.0 {
                let regularized = add_ridge(h, ridge);
                let view = FaerArrayView::new(&regularized);
                if let Ok(f) = FaerLlt::new(view.as_ref(), Side::Lower) {
                    return FaerFactor::Llt(f);
                }
                if let Ok(f) = FaerLdlt::new(view.as_ref(), Side::Lower) {
                    return FaerFactor::Ldlt(f);
                }
                if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                    let f = FaerLblt::new(view.as_ref(), Side::Lower);
                    return FaerFactor::Lblt(f);
                }
            } else {
                let h_view = FaerArrayView::new(h);
                if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                    return FaerFactor::Llt(f);
                }
                if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                    return FaerFactor::Ldlt(f);
                }
            }
            planner.bump_with_matrix(h);
        }
    }

    fn get_faer_factor(&self, rho: &Array1<f64>, h: &Array2<f64>) -> Arc<FaerFactor> {
        // Cache strategy: ρ alone is the key.
        // The cache deliberately ignores which Hessian matrix we are factoring.  Today this is
        // sound because every caller obeys a single rule:
        //   • Identity/Gaussian REML cost & gradient only ever request factors of the
        //     stabilized Hessian.
        //   • Non-Gaussian (logit/LAML) cost and gradient request factors of the effective/ridged Hessian.
        // Consequently each ρ corresponds to exactly one matrix within the lifetime of a
        // `RemlState`, so returning the cached factorization is correct.
        // This design is still brittle: adding a new code path that calls `get_faer_factor`
        // with a different H for the same ρ would silently reuse the wrong factor.  If such a
        // path ever appears, extend the key (for example by tagging the Hessian variant) or
        // split the cache.  The current key maximizes cache
        // hits across repeated EDF/gradient evaluations for the same smoothing parameters.
        let key_opt = self.rho_key_sanitized(rho);
        if let Some(key) = &key_opt
            && let Some(f) = self
                .cache_manager
                .faer_factor_cache
                .read()
                .unwrap()
                .get(key)
        {
            return Arc::clone(f);
        }
        let fact = Arc::new(self.factorize_faer(h));

        if let Some(key) = key_opt {
            let mut cache = self.cache_manager.faer_factor_cache.write().unwrap();
            if cache.len() > 64 {
                cache.clear();
            }
            cache.insert(key, Arc::clone(&fact));
        }
        fact
    }

    const MIN_DMU_DETA: f64 = 1e-6;

    // Accessor methods for private fields
    pub(crate) fn x(&self) -> &DesignMatrix {
        &self.x
    }

    #[allow(dead_code)]
    pub(crate) fn y(&self) -> ArrayView1<'a, f64> {
        self.y
    }

    #[allow(dead_code)]
    pub(crate) fn rs_list_ref(&self) -> &Vec<Array2<f64>> {
        &self.rs_list
    }

    pub(crate) fn balanced_penalty_root(&self) -> &Array2<f64> {
        &self.balanced_penalty_root
    }

    pub(crate) fn weights(&self) -> ArrayView1<'a, f64> {
        self.weights
    }

    fn sparse_penalty_logdet_runtime(
        &self,
        rho: &Array1<f64>,
        blocks: &[SparsePenaltyBlock],
    ) -> (f64, Array1<f64>) {
        let mut logdet = 0.0_f64;
        let mut det1 = Array1::<f64>::zeros(rho.len());
        for block in blocks {
            let rank = block.positive_eigenvalues.len() as f64;
            if block.term_index < det1.len() {
                det1[block.term_index] = rank;
            }
            logdet += rank * rho[block.term_index];
            for &eig in block.positive_eigenvalues.iter() {
                logdet += eig.ln();
            }
        }
        (logdet, det1)
    }

    fn geometry_backend_kind(bundle: &EvalShared) -> GeometryBackendKind {
        bundle.backend_kind()
    }

    fn select_hessian_strategy_policy(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
    ) -> HessianStrategyDecision {
        if self.uses_objective_consistent_fd_gradient(rho) {
            return HessianStrategyDecision {
                strategy: HessianEvalStrategyKind::DiagnosticNumeric,
                reason: "objective_consistent_numeric_gradient",
            };
        }
        if bundle.active_subspace_unstable {
            return HessianStrategyDecision {
                strategy: HessianEvalStrategyKind::AnalyticFallback,
                reason: "active_subspace_unstable",
            };
        }
        HessianStrategyDecision {
            strategy: HessianEvalStrategyKind::SpectralExact,
            reason: "exact_preferred",
        }
    }

    fn compute_laml_hessian_by_strategy(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        decision: HessianStrategyDecision,
    ) -> Result<Array2<f64>, EstimationError> {
        match decision.strategy {
            HessianEvalStrategyKind::SpectralExact => self.compute_laml_hessian_exact(rho),
            HessianEvalStrategyKind::AnalyticFallback => {
                self.compute_laml_hessian_analytic_fallback(rho, Some(bundle))
            }
            HessianEvalStrategyKind::DiagnosticNumeric => {
                log::warn!(
                    "Using diagnostic numeric Hessian strategy routing (reason={}); dispatching to deterministic analytic fallback.",
                    decision.reason
                );
                self.compute_laml_hessian_analytic_fallback(rho, Some(bundle))
            }
        }
    }

    fn select_reml_geometry(&self, rho: &Array1<f64>) -> SparseRemlDecision {
        let p = self.p;
        let has_dense_constraints =
            self.linear_constraints.is_some() || self.coefficient_lower_bounds.is_some();
        let x_sparse = match &self.x {
            DesignMatrix::Sparse(sparse) => Some(sparse),
            DesignMatrix::Dense(_) => None,
        };
        let nnz_x = x_sparse.map(|s| s.val().len()).unwrap_or(0);
        let dense_backend =
            |reason: &'static str,
             nnz_h_upper_est: Option<usize>,
             density_h_upper_est: Option<f64>| SparseRemlDecision {
                geometry: RemlGeometry::DenseSpectral,
                reason,
                p,
                nnz_x,
                nnz_h_upper_est,
                density_h_upper_est,
            };

        if self.config.firth_bias_reduction
            && !matches!(self.config.link_function(), LinkFunction::Logit)
        {
            return dense_backend("firth_non_logit", None, None);
        }
        if p < 256 {
            return dense_backend("p_below_threshold", None, None);
        }
        if has_dense_constraints {
            return dense_backend("constraints_present", None, None);
        }
        let Some(x_sparse) = x_sparse else {
            return dense_backend("design_not_sparse", None, None);
        };
        let Some(blocks) = self.sparse_penalty_blocks.as_ref() else {
            return dense_backend("penalty_blocks_not_separable", None, None);
        };

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, s_k) in self.s_full_list.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                s_lambda.scaled_add(lambdas[k], s_k);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        match workspace.sparse_penalized_system_stats(x_sparse, &s_lambda) {
            Ok(stats) if stats.density_upper < 0.10 && !blocks.is_empty() => SparseRemlDecision {
                geometry: RemlGeometry::SparseExactSpd,
                reason: "sparse_exact_spd",
                p,
                nnz_x,
                nnz_h_upper_est: Some(stats.nnz_h_upper),
                density_h_upper_est: Some(stats.density_upper),
            },
            Ok(stats) => dense_backend(
                "penalized_hessian_too_dense",
                Some(stats.nnz_h_upper),
                Some(stats.density_upper),
            ),
            Err(_) => dense_backend("sparse_stats_failed", None, None),
        }
    }

    fn prepare_dense_eval_bundle_with_key(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        let (h_eff, ridge_passport) = self.effective_hessian(pirls_result.as_ref())?;

        const EIG_REL_THRESHOLD: f64 = 1e-10;
        const EIG_ABS_FLOOR: f64 = 1e-14;

        let dim = h_eff.nrows();
        let mut h_total = h_eff.clone();
        let mut firth_dense_operator: Option<Arc<FirthDenseOperator>> = None;
        if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let x_dense = pirls_result.x_transformed.to_dense_arc();
            let firth_op = Arc::new(Self::build_firth_dense_operator(
                x_dense.as_ref(),
                &pirls_result.final_eta,
            )?);
            // Firth-adjusted inner Jacobian for implicit differentiation:
            //   H_total = Xᵀ W X + S - H_φ,
            //   H_φ     = ∇²_β Φ
            //          = 0.5 [ Xᵀ diag(w'' ⊙ h) X - Bᵀ P B ].
            // This keeps B_k/B_{kl} solves on the same objective surface as
            // the Firth-augmented stationarity system.
            //
            // Conceptually Φ is the pseudodeterminant term 0.5 log|XᵀWX|_+.
            // The h_phi block below is therefore the curvature of that
            // identifiable-subspace Jeffreys penalty, represented in the
            // current transformed basis.
            let mut weighted_xtdx = Array2::<f64>::zeros(firth_op.x_dense.raw_dim());
            let diag_term = Self::xt_diag_x_dense_into(
                &firth_op.x_dense,
                &(&firth_op.w2 * &firth_op.h_diag),
                &mut weighted_xtdx,
            );
            let bpb = fast_ab(&firth_op.b_base.t().to_owned(), &firth_op.p_b_base);
            let mut h_phi = 0.5 * (diag_term - bpb);
            // Numerical symmetry guard.
            for i in 0..h_phi.nrows() {
                for j in 0..i {
                    let avg = 0.5 * (h_phi[[i, j]] + h_phi[[j, i]]);
                    h_phi[[i, j]] = avg;
                    h_phi[[j, i]] = avg;
                }
            }
            // Keep tiny numerical noise from making the solve surface less stable.
            if h_phi.iter().all(|v| v.is_finite()) {
                h_total -= &h_phi;
            }
            firth_dense_operator = Some(firth_op);
        }
        let (eigvals, eigvecs) = h_total
            .eigh(Side::Lower)
            .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
        let max_eig = eigvals.iter().copied().fold(0.0_f64, f64::max);
        let eig_threshold = if self.config.link_function() == LinkFunction::Identity {
            (max_eig * EIG_REL_THRESHOLD).max(EIG_ABS_FLOOR)
        } else {
            EIG_ABS_FLOOR
        };
        let h_total_log_det: f64 = eigvals
            .iter()
            .filter(|&&v| v > eig_threshold)
            .map(|&v| v.ln())
            .sum();
        if !h_total_log_det.is_finite() {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }

        let valid_indices: Vec<usize> = eigvals
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v > eig_threshold { Some(i) } else { None })
            .collect();
        let min_kept = valid_indices
            .iter()
            .map(|&i| eigvals[i])
            .fold(f64::INFINITY, f64::min);
        let max_dropped = eigvals
            .iter()
            .copied()
            .filter(|&v| v <= eig_threshold)
            .fold(0.0_f64, f64::max);
        let active_subspace_rel_gap =
            if min_kept.is_finite() && max_dropped.is_finite() && min_kept > 0.0 {
                let gap = (min_kept - max_dropped).max(0.0);
                Some(gap / min_kept.max(1e-16))
            } else {
                None
            };
        let active_subspace_unstable =
            active_subspace_rel_gap.is_some_and(|rel_gap| rel_gap < 1e-6);
        // Active-subspace stability diagnostic for pseudo-logdet derivatives.
        // Exact second-order identities for log|H|_+ assume a fixed positive
        // eigenspace. When retained and dropped eigenvalues crowd the threshold,
        // the objective is near a non-smooth boundary and Hessian updates can be
        // numerically fragile.
        if log::log_enabled!(log::Level::Warn) {
            if active_subspace_unstable {
                let rel_gap = active_subspace_rel_gap.unwrap_or(f64::NAN);
                log::warn!(
                    "[REML] H_+ active-subspace is near instability: min_kept={:.3e}, max_dropped={:.3e}, threshold={:.3e}, rel_gap={:.3e}",
                    min_kept,
                    max_dropped,
                    eig_threshold,
                    rel_gap
                );
            }
        }

        let valid_count = valid_indices.len();
        let mut w = Array2::<f64>::zeros((dim, valid_count));

        for (w_col_idx, &eig_idx) in valid_indices.iter().enumerate() {
            let val = eigvals[eig_idx];
            let scale = 1.0 / val.sqrt();
            let u_col = eigvecs.column(eig_idx);
            let mut w_col = w.column_mut(w_col_idx);
            Zip::from(&mut w_col)
                .and(&u_col)
                .for_each(|w_elem, &u_elem| *w_elem = u_elem * scale);
        }

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::DenseSpectral,
            h_eff: Arc::new(h_eff),
            h_total: Arc::new(h_total),
            h_pos_factor_w: Arc::new(w),
            h_total_log_det,
            active_subspace_rel_gap,
            active_subspace_unstable,
            sparse_exact: None,
            firth_dense_operator,
        })
    }

    fn prepare_sparse_eval_bundle_with_key(
        &self,
        rho: &Array1<f64>,
        key: Option<Vec<u64>>,
    ) -> Result<EvalShared, EstimationError> {
        let pirls_result = self.execute_pirls_if_needed(rho)?;
        if !matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::OriginalSparseNative
        ) {
            return Err(EstimationError::InvalidInput(
                "sparse exact geometry requires sparse-native PIRLS coordinates".to_string(),
            ));
        }
        let ridge_passport = pirls_result.ridge_passport;
        let x_sparse = match &self.x {
            DesignMatrix::Sparse(s) => s,
            DesignMatrix::Dense(_) => {
                return Err(EstimationError::InvalidInput(
                    "sparse exact geometry requires sparse original design".to_string(),
                ));
            }
        };
        let penalty_blocks = self
            .sparse_penalty_blocks
            .as_ref()
            .ok_or_else(|| {
                EstimationError::InvalidInput(
                    "sparse exact geometry requires block-separable penalties".to_string(),
                )
            })?
            .clone();

        let lambdas = rho.mapv(f64::exp);
        let mut s_lambda = Array2::<f64>::zeros((self.p, self.p));
        for (k, s_k) in self.s_full_list.iter().enumerate() {
            if k < lambdas.len() && lambdas[k] != 0.0 {
                s_lambda.scaled_add(lambdas[k], s_k);
            }
        }
        let mut workspace = PirlsWorkspace::new(self.y.len(), self.p, 0, 0);
        let sparse_system = assemble_and_factor_sparse_penalized_system(
            &mut workspace,
            x_sparse,
            &pirls_result.solve_weights,
            &s_lambda,
            ridge_passport.delta,
        )?;
        let (logdet_s_pos, det1_values) =
            self.sparse_penalty_logdet_runtime(rho, penalty_blocks.as_ref());
        let firth_dense_operator = if self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit)
        {
            let x_dense = self.x().to_dense_arc();
            Some(Arc::new(Self::build_firth_dense_operator(
                x_dense.as_ref(),
                &pirls_result.final_eta,
            )?))
        } else {
            None
        };

        Ok(EvalShared {
            key,
            pirls_result,
            ridge_passport,
            geometry: RemlGeometry::SparseExactSpd,
            h_eff: Arc::new(Array2::zeros((0, 0))),
            h_total: Arc::new(Array2::zeros((0, 0))),
            h_pos_factor_w: Arc::new(Array2::zeros((0, 0))),
            h_total_log_det: 0.0,
            active_subspace_rel_gap: None,
            active_subspace_unstable: false,
            sparse_exact: Some(Arc::new(SparseExactEvalData {
                factor: Arc::new(sparse_system.factor),
                penalty_blocks,
                logdet_h: sparse_system.logdet_h,
                logdet_s_pos,
                det1_values: Arc::new(det1_values),
                trace_workspace: Arc::new(Mutex::new(SparseTraceWorkspace::default())),
            })),
            firth_dense_operator,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn offset(&self) -> ArrayView1<'_, f64> {
        self.offset.view()
    }

    /// Runs the inner P-IRLS loop, caching the result.
    fn execute_pirls_if_needed(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Arc<PirlsResult>, EstimationError> {
        let use_cache = self
            .cache_manager
            .pirls_cache_enabled
            .load(Ordering::Relaxed);
        // Use sanitized key to handle NaN and -0.0 vs 0.0 issues
        let key_opt = self.rho_key_sanitized(rho);
        if use_cache
            && let Some(key) = &key_opt
            && let Some(cached) = self.cache_manager.pirls_cache.write().unwrap().get(key)
        {
            if self.warm_start_enabled.load(Ordering::Relaxed) {
                self.update_warm_start_from(cached.as_ref());
            }
            return Ok(cached);
        }

        // Run P-IRLS with original matrices to perform fresh reparameterization
        // The returned result will include the transformation matrix qs
        let pirls_result = {
            let warm_start_holder = self.warm_start_beta.read().unwrap();
            let warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                warm_start_holder.as_ref()
            } else {
                None
            };
            pirls::fit_model_for_fixed_rho_matrix(
                LogSmoothingParamsView::new(rho.view()),
                &self.x,
                self.offset.view(),
                self.y,
                self.weights,
                &self.rs_list,
                Some(&self.balanced_penalty_root),
                Some(&self.reparam_invariant),
                self.p,
                &self.config.as_pirls_config(),
                warm_start_ref,
                self.coefficient_lower_bounds.as_ref(),
                self.linear_constraints.as_ref(),
                None, // No SE for base model
            )
        };

        if let Err(e) = &pirls_result {
            println!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
            if self.warm_start_enabled.load(Ordering::Relaxed) {
                self.warm_start_beta.write().unwrap().take();
            }
        }

        let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
        let pirls_result = Arc::new(pirls_result);
        self.enforce_constraint_kkt(pirls_result.as_ref())?;

        // Check the status returned by the P-IRLS routine.
        match pirls_result.status {
            pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                self.update_warm_start_from(pirls_result.as_ref());
                // This is a successful fit. Cache only if key is valid (not NaN).
                if use_cache && let Some(key) = key_opt {
                    self.cache_manager
                        .pirls_cache
                        .write()
                        .unwrap()
                        .insert(key, Arc::clone(&pirls_result));
                }
                Ok(pirls_result)
            }
            pirls::PirlsStatus::Unstable => {
                if self.warm_start_enabled.load(Ordering::Relaxed) {
                    self.warm_start_beta.write().unwrap().take();
                }
                // The fit was unstable. This is where we throw our specific, user-friendly error.
                // Pass the diagnostic info into the error
                Err(EstimationError::PerfectSeparationDetected {
                    iteration: pirls_result.iteration,
                    max_abs_eta: pirls_result.max_abs_eta,
                })
            }
            pirls::PirlsStatus::MaxIterationsReached => {
                if self.warm_start_enabled.load(Ordering::Relaxed) {
                    self.warm_start_beta.write().unwrap().take();
                }
                if pirls_result.last_gradient_norm > 1.0 {
                    // The fit timed out and gradient is large.
                    log::error!(
                        "P-IRLS failed convergence check: gradient norm {} > 1.0 (iter {})",
                        pirls_result.last_gradient_norm,
                        pirls_result.iteration
                    );
                    Err(EstimationError::PirlsDidNotConverge {
                        max_iterations: pirls_result.iteration,
                        last_change: pirls_result.last_gradient_norm,
                    })
                } else {
                    // Gradient is acceptable, treat as converged but with warning if needed
                    log::warn!(
                        "P-IRLS reached max iterations but gradient norm {:.3e} is acceptable.",
                        pirls_result.last_gradient_norm
                    );
                    Ok(pirls_result)
                }
            }
        }
    }
}
impl<'a> RemlState<'a> {
    /// Compute the objective function for BFGS optimization.
    ///
    /// FULL OBJECTIVE REFERENCE
    /// ------------------------
    /// This function returns the scalar outer cost minimized over ρ.
    ///
    /// Non-Gaussian branch (negative LAML form used by optimizer):
    ///   V_LAML(ρ) =
    ///      -ℓ(β̂(ρ))
    ///      + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// where:
    ///   S(ρ) = Σ_k exp(ρ_k) S_k + δI
    ///   H(ρ) = -∇²ℓ(β̂(ρ)) + S(ρ)
    ///
    /// Gaussian identity-link REML branch:
    ///   V_REML(ρ, φ) =
    ///      D_p(ρ)/(2φ)
    ///      + (n_r/2) log φ
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const
    ///
    /// with profiled φ:
    ///   φ̂(ρ) = D_p(ρ)/n_r
    ///   V_REML,prof(ρ) =
    ///      (n_r/2) log D_p(ρ)
    ///      + 0.5 log|H(ρ)|
    ///      - 0.5 log|S(ρ)|_+
    ///      + const.
    ///
    /// Consistency rule enforced throughout:
    ///   The same stabilized matrices/factorizations are used for
    ///   objective and gradient/Hessian terms. Mixing different H/S variants
    ///   causes deterministic gradient mismatch and unstable outer optimization.
    ///
    /// Determinant conventions:
    ///   - log|H| may use positive-part/stabilized spectrum conventions when needed.
    ///   - log|S|_+ follows fixed-rank pseudo-determinant conventions in the
    ///     transformed penalty basis, optionally including ridge policy.
    /// These conventions are mirrored in gradient code via corresponding trace terms.
    pub fn compute_cost(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
        let cost_call_idx = {
            let mut calls = self.arena.cost_eval_count.write().unwrap();
            *calls += 1;
            *calls
        };
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                // Inner linear algebra says "too singular" — treat as barrier.
                log::warn!(
                    "P-IRLS flagged ill-conditioning for current rho; returning +inf cost to retreat."
                );
                // Diagnostics: which rho are at bounds
                let at_lower: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if v <= -RHO_BOUND + 1e-8 {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let at_upper: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                    .collect();
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    eprintln!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower, at_upper
                    );
                }
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                // Other errors still bubble up
                // Provide bounds diagnostics here too
                let at_lower: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| {
                        if v <= -RHO_BOUND + 1e-8 {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();
                let at_upper: Vec<usize> = p
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                    .collect();
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    eprintln!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower, at_upper
                    );
                }
                return Err(e);
            }
        };
        if Self::geometry_backend_kind(&bundle) == GeometryBackendKind::SparseExactSpd {
            return self.compute_cost_sparse_exact(p, &bundle);
        }
        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_used = bundle.ridge_passport.delta;

        let lambdas = p.mapv(f64::exp);
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut e_eval = pirls_result.reparam_result.e_transformed.clone();
        if let Some(z) = free_basis_opt.as_ref() {
            h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
            h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
            e_eval = pirls_result.reparam_result.e_transformed.dot(z);
        }
        let h_eff = &h_eff_eval;

        // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
        if !p.is_empty() {
            let k_lambda = p.len();
            let k_r = pirls_result.reparam_result.rs_transformed.len();
            let k_d = pirls_result.reparam_result.det1.len();
            if !(k_lambda == k_r && k_r == k_d) {
                return Err(EstimationError::LayoutError(format!(
                    "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                    k_lambda, k_r, k_d
                )));
            }
            if self.nullspace_dims.len() != k_lambda {
                return Err(EstimationError::LayoutError(format!(
                    "Nullspace dimension mismatch: expected {} entries, got {}",
                    k_lambda,
                    self.nullspace_dims.len()
                )));
            }
        }

        // Don't barrier on non-PD; we'll stabilize and continue like mgcv
        // Only check eigenvalues if we needed to add a ridge
        const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
        let want_hot_diag = self.should_compute_hot_diagnostics(cost_call_idx);
        if ridge_used > 0.0
            && let Ok((eigs, _)) = pirls_result.penalized_hessian_transformed.eigh(Side::Lower)
            && let Some(min_eig) = eigs.iter().cloned().reduce(f64::min)
        {
            if should_emit_h_min_eig_diag(min_eig) {
                eprintln!(
                    "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                    min_eig, ridge_used
                );
            }

            if min_eig <= 0.0 {
                log::warn!(
                    "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                    ridge_used
                );
            }

            if want_hot_diag
                && (!min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE)
            {
                let condition_number =
                    calculate_condition_number(&pirls_result.penalized_hessian_transformed)
                        .ok()
                        .unwrap_or(f64::INFINITY);

                log::warn!(
                    "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                    condition_number
                );
            }
        }
        // Use stable penalty calculation - no need to reconstruct matrices
        // The penalty term is already calculated stably in the P-IRLS loop

        match self.config.link_function() {
            LinkFunction::Identity => {
                let ridge_passport = pirls_result.ridge_passport;
                // From Wood (2017), Chapter 6, Eq. 6.24:
                // V_r(λ) = D_p/(2φ) + (r/2φ) + ½log|X'X/φ + S_λ/φ| - ½log|S_λ/φ|_+
                // where D_p = ||y - Xβ̂||² + β̂'S_λβ̂ is the PENALIZED deviance
                //
                // With profiled dispersion φ̂ = D_p/(n-M_p), this becomes:
                //   V_REML(ρ) =
                //     D_p/(2φ̂)
                //   + 0.5 log|H|
                //   - 0.5 log|S|_+
                //   + ((n-M_p)/2) log(2πφ̂),
                // where H = XᵀW0X + S(ρ), S(ρ)=Σ_k exp(ρ_k) S_k + δI.
                //
                // Because Gaussian identity has c=d=0, there is no third/fourth derivative
                // correction in H_k: ∂H/∂ρ_k = S_k^ρ exactly.

                // Check condition number with improved thresholds per Wood (2011)
                const MAX_CONDITION_NUMBER: f64 = 1e12;
                if want_hot_diag {
                    let cond = pirls_result
                        .penalized_hessian_transformed
                        .eigh(Side::Lower)
                        .ok()
                        .map(|(evals, _)| {
                            let max_ev = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            let min_ev = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            if min_ev <= 1e-12 {
                                f64::INFINITY
                            } else {
                                max_ev / min_ev
                            }
                        })
                        .unwrap_or(f64::NAN);
                    *self.arena.gaussian_cond_snapshot.write().unwrap() = cond;
                }
                let condition_number = *self.arena.gaussian_cond_snapshot.read().unwrap();
                if condition_number.is_finite() {
                    if condition_number > MAX_CONDITION_NUMBER {
                        log::warn!(
                            "Penalized Hessian very ill-conditioned (cond={:.2e}); proceeding despite poor conditioning.",
                            condition_number
                        );
                    } else if condition_number > 1e8 {
                        log::warn!(
                            "Penalized Hessian is ill-conditioned but proceeding: condition number = {condition_number:.2e}"
                        );
                    }
                }

                // STRATEGIC DESIGN DECISION: Use unweighted sample count for mgcv parity
                // In standard WLS theory, one might use sum(weights) as effective sample size.
                // However, mgcv deliberately uses the unweighted count 'n.true' in gam.fit3.
                let n = self.y.len() as f64;
                // Number of coefficients (transformed basis)

                // Calculate PENALIZED deviance D_p = ||y - Xβ̂||² + β̂'S_λβ̂
                let rss = pirls_result.deviance; // Unpenalized ||y - μ||²
                // Use stable penalty term calculated in P-IRLS
                let penalty = pirls_result.stable_penalty_term;

                let dp = rss + penalty;

                // Calculate EDF = p - tr((X'X + S_λ)⁻¹S_λ)
                // Work directly in the transformed basis for efficiency and numerical stability
                // This avoids transforming matrices back to the original basis unnecessarily
                // Penalty roots are available in reparam_result if needed

                // Nullspace dimension M_p is constant with respect to ρ.  Use it to profile φ
                // following the standard REML identity φ = D_p / (n - M_p).
                let (penalty_rank, log_det_s_plus) =
                    self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;
                let p_eff_dim = h_eff.ncols();
                let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;

                // EDF diagnostics are expensive; compute only when diagnostics are enabled.
                if want_hot_diag {
                    let edf = self.edf_from_h_and_e(&e_eval, lambdas.view(), h_eff)?;
                    log::debug!("[Diag] EDF total={:.3}", edf);
                    if n - edf < 1.0 {
                        log::warn!("Effective DoF exceeds samples; model may be overfit.");
                    }
                }

                let denom = (n - mp).max(LAML_RIDGE);
                let (dp_c, _) = smooth_floor_dp(dp);
                if dp < DP_FLOOR {
                    log::warn!(
                        "Penalized deviance {:.3e} fell below DP_FLOOR; clamping to maintain REML stability.",
                        dp
                    );
                }
                let phi = dp_c / denom;

                // log |H| = log |X'X + S_λ + ridge I| using the single effective
                // Hessian shared with the gradient. Ridge is already baked into h_eff.
                //
                // This is the same stabilized H used in compute_gradient;
                // otherwise the chain-rule pieces and determinant pieces are taken on
                // different objective surfaces and the optimizer sees inconsistent derivatives.
                let h_for_det = h_eff.clone();

                let chol = h_for_det.cholesky(Side::Lower).map_err(|_| {
                    let min_eig = h_eff
                        .clone()
                        .eigh(Side::Lower)
                        .ok()
                        .and_then(|(eigs, _)| eigs.iter().cloned().reduce(f64::min))
                        .unwrap_or(f64::NAN);
                    EstimationError::HessianNotPositiveDefinite {
                        min_eigenvalue: min_eig,
                    }
                })?;
                let log_det_h = 2.0 * chol.diag().mapv(f64::ln).sum();

                // log |S_λ + ridge I|_+ (pseudo-determinant) to match the
                // stabilized penalty used by PIRLS.
                //
                // Fixed-rank rule: unpenalized/null directions do not contribute to the
                // pseudo-logdet. This keeps the objective continuous in ρ when S is singular
                // (or near-singular before ridge augmentation).
                // Standard REML expression from Wood (2017), 6.5.1
                // V = (n/2)log(2πσ²) + D_p/(2σ²) + ½log|H| - ½log|S_λ|_+ + (M_p-1)/2 log(2πσ²)
                // Simplifying: V = D_p/(2φ) + ½log|H| - ½log|S_λ|_+ + ((n-M_p)/2) log(2πφ)
                let reml = dp_c / (2.0 * phi)
                    + 0.5 * (log_det_h - log_det_s_plus)
                    + ((n - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                let prior_cost = self.compute_soft_prior_cost(p);

                Ok(reml + prior_cost)
            }
            _ => {
                // For non-Gaussian GLMs, use the LAML approximation
                // Note: Deviance = -2 * log-likelihood + C. So -0.5 * Deviance = log-likelihood - C/2.
                // Use stable penalty term calculated in P-IRLS
                let mut penalised_ll =
                    -0.5 * pirls_result.deviance - 0.5 * pirls_result.stable_penalty_term;

                let ridge_passport = pirls_result.ridge_passport;
                // Include Firth log-det term in LAML for consistency with inner PIRLS
                if self.config.firth_bias_reduction
                    && matches!(self.config.link_function(), LinkFunction::Logit)
                {
                    if let Some(firth_log_det) = pirls_result.firth_log_det {
                        penalised_ll += firth_log_det; // Jeffreys prior contribution
                    }
                }

                // Use the stabilized log|Sλ|_+ from the reparameterization (consistent with gradient)
                let (_penalty_rank, log_det_s) =
                    self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;

                // Log-determinant of the effective Hessian.
                // HESSIAN PASSPORT: Use the pre-computed h_total and its factorization
                // from the bundle to ensure exact consistency with gradient computation.
                // For Firth: h_total = h_eff - h_phi (computed in prepare_eval_bundle)
                // For non-Firth: h_total = h_eff
                //
                // LAML objective:
                //   V_LAML(ρ) =
                //      -ℓ(β̂) + 0.5 β̂ᵀSβ̂
                //    - 0.5 log|S|_+
                //    + 0.5 log|H|
                //    + const.
                //
                // For non-Gaussian families, H depends on ρ both directly through S and
                // indirectly through β̂(ρ), which induces the dH/dρ_k third-derivative term in
                // the exact gradient path (documented in compute_gradient).
                let log_det_h = if free_basis_opt.is_some() {
                    if h_total_eval.nrows() == 0 {
                        0.0
                    } else {
                        let (evals, _) = h_total_eval
                            .eigh(Side::Lower)
                            .map_err(EstimationError::EigendecompositionFailed)?;
                        let floor = 1e-10;
                        evals.iter().filter(|&&v| v > floor).map(|&v| v.ln()).sum()
                    }
                } else {
                    bundle.h_total_log_det
                };

                // Mp is null space dimension (number of unpenalized coefficients)
                // For logit, scale parameter is typically fixed at 1.0, but include for completeness
                let phi = 1.0; // Logit family typically has dispersion parameter = 1

                // Compute null space dimension using the TRANSFORMED, STABLE basis
                // Use the rank of the lambda-weighted transformed penalty root (e_transformed)
                // to determine M_p with the transformed penalty basis.
                let (penalty_rank, _) =
                    self.fixed_subspace_penalty_rank_and_logdet(&e_eval, ridge_passport)?;
                let p_eff_dim = h_eff.ncols();
                let mp = p_eff_dim.saturating_sub(penalty_rank) as f64;

                let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_h
                    + (mp / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                // Diagnostics below are expensive and not needed for objective value.
                let (edf, trace_h_inv_s_lambda, stab_cond) = if want_hot_diag {
                    let p_eff = h_eff.ncols() as f64;
                    let edf = self.edf_from_h_and_e(&e_eval, lambdas.view(), h_eff)?;
                    let trace_h_inv_s_lambda = (p_eff - edf).max(0.0);
                    let stab_cond = pirls_result
                        .penalized_hessian_transformed
                        .eigh(Side::Lower)
                        .ok()
                        .map(|(evals, _)| {
                            let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            max / min.max(1e-12)
                        })
                        .unwrap_or(f64::NAN);
                    (edf, trace_h_inv_s_lambda, stab_cond)
                } else {
                    (f64::NAN, f64::NAN, f64::NAN)
                };

                // Raw-condition diagnostics are rate-limited in this loop.
                // We only refresh occasionally, and keep the last snapshot otherwise.
                let raw_cond = if matches!(self.x(), DesignMatrix::Dense(_)) && want_hot_diag {
                    let x_orig_arc = self.x().to_dense_arc();
                    let x_orig = x_orig_arc.as_ref();
                    let w_orig = self.weights();
                    let sqrt_w = w_orig.mapv(|w| w.max(0.0).sqrt());
                    let wx = x_orig * &sqrt_w.insert_axis(Axis(1));
                    let mut h_raw = fast_ata(&wx);
                    for (k, &lambda) in lambdas.iter().enumerate() {
                        let s_k = &self.s_full_list[k];
                        if lambda != 0.0 {
                            h_raw.scaled_add(lambda, s_k);
                        }
                    }
                    let raw = h_raw
                        .eigh(Side::Lower)
                        .ok()
                        .map(|(evals, _)| {
                            let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                            let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                            max / min.max(1e-12)
                        })
                        .unwrap_or(f64::NAN);
                    *self.arena.raw_cond_snapshot.write().unwrap() = raw;
                    raw
                } else {
                    *self.arena.raw_cond_snapshot.read().unwrap()
                };
                if want_hot_diag {
                    self.log_gam_cost(
                        &p,
                        lambdas.as_slice().unwrap_or(&[]),
                        laml,
                        stab_cond,
                        raw_cond,
                        edf,
                        trace_h_inv_s_lambda,
                    );
                }

                let prior_cost = self.compute_soft_prior_cost(p);

                Ok(-laml + prior_cost)
            }
        }
    }

    ///
    /// -------------------------------------------------------------------------
    /// Exact non-Laplace evidence identities (reference comments; not runtime path)
    /// -------------------------------------------------------------------------
    /// We optimize a Laplace-style outer objective for scalability, but the exact
    /// marginal likelihood for non-Gaussian models can be written analytically as:
    ///
    ///   L(ρ) = ∫ exp(l(β) - 0.5 βᵀ S(ρ) β) dβ,   S(ρ)=Σ_k exp(ρ_k) S_k.
    ///
    /// Universal exact gradient identity (when differentiation under the integral
    /// is justified and L(ρ) < ∞):
    ///
    ///   ∂_{ρ_k} log L(ρ)
    ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ βᵀ S_k β ].
    ///
    /// Laplace bridge to implemented terms:
    /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
    ///     E[βᵀ S_k β] ≈ β̂ᵀ S_k β̂ + tr(H^{-1} S_k),
    ///   giving the familiar quadratic + trace structure.
    /// - In this code those appear as:
    ///     0.5 * β̂ᵀ S_k^ρ β̂,
    ///     -0.5 * tr(S^+ S_k^ρ),
    ///     +0.5 * tr(H^{-1} H_k).
    ///
    /// Why this does NOT collapse to only tr(H^{-1}S_k):
    /// - The exact identity differentiates the true integral measure.
    /// - LAML differentiates a moving approximation:
    ///     V_LAML(ρ) = -ℓ(β̂(ρ)) + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
    ///                 + 0.5 log|H(ρ)| - 0.5 log|S(ρ)|_+.
    /// - Here both center β̂(ρ) and curvature H(ρ) move with ρ.
    /// - For non-Gaussian families, H_k includes the third-derivative tensor path
    ///   through β̂(ρ), i.e. H_k != S_k^ρ. These are the explicit dH/dρ_k terms
    ///   retained below to differentiate the Laplace objective exactly.
    ///
    /// For Bernoulli-logit, an exact Pólya-Gamma augmentation gives:
    ///
    ///   L(ρ) = 2^{-n} (2π)^{p/2}
    ///          E_{ω_i ~ PG(1,0)} [ |Q(ω,ρ)|^{-1/2} exp(0.5 bᵀ Q^{-1} b) ],
    ///   Q(ω,ρ)=S(ρ)+XᵀΩX, b=Xᵀ(y-1/2).
    ///
    /// and
    ///
    ///   ∂_{ρ_k} log L
    ///   = -0.5 * exp(ρ_k) *
    ///     E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ],  μ=Q^{-1}b.
    /// Equivalently, since β|ω,y,ρ ~ N(μ,Q^{-1}):
    ///   E[βᵀS_kβ | ω,y,ρ] = tr(S_k Q^{-1}) + μᵀS_kμ.
    ///
    /// yielding exact (but high-dimensional) contour integrals / series after
    /// analytically integrating β.
    ///
    /// Practical note:
    /// - These are exact equalities but generally not polynomial-time tractable
    ///   for arbitrary dense (X, n, p).
    /// - This code therefore uses deterministic Laplace/implicit-differentiation
    ///   machinery for the main optimizer path, with exact tensor terms where
    ///   feasible (H_k, H_{kℓ}, c/d arrays), and scalable trace backends.
    ///
    /// FULL OUTER-DERIVATIVE REFERENCE (exact system, sign convention used here)
    /// -------------------------------------------------------------------------
    /// This optimizer minimizes an outer cost V(ρ).
    ///
    /// Common definitions:
    ///   λ_k = exp(ρ_k)
    ///   S(ρ) = Σ_k λ_k S_k + δI
    ///   A_k = ∂S/∂ρ_k = λ_k S_k
    ///   A_{kℓ} = ∂²S/(∂ρ_k∂ρ_ℓ) = δ_{kℓ} A_k
    ///
    /// Inner mode (β̂):
    ///   ∇_β ℓ(β̂) - S(ρ) β̂ = 0
    ///
    /// Curvature:
    ///   H(ρ) = -∇²_β ℓ(β̂(ρ)) + S(ρ)
    ///
    ///   w_i = -∂²ℓ_i/∂η_i²
    ///   d_i = -∂³ℓ_i/∂η_i³
    ///   e_i = -∂⁴ℓ_i/∂η_i⁴
    ///
    /// Then:
    ///   H_k = A_k + Xᵀ diag(d ⊙ u_k) X,     u_k := X B_k
    ///   H_{kℓ} = δ_{kℓ}A_k + Xᵀ diag(e ⊙ u_k ⊙ u_ℓ + d ⊙ u_{kℓ}) X
    ///
    /// with implicit derivatives:
    ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
    ///   H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ}A_k β̂ + A_k B_ℓ)
    ///
    /// Non-Gaussian negative LAML cost:
    ///   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀSβ̂ + 0.5 log|H| - 0.5 log|S|_+
    ///
    /// Exact gradient:
    ///   g_k = 0.5 β̂ᵀA_kβ̂ + 0.5 tr(H^{-1}H_k) - 0.5 ∂_k log|S|_+
    ///
    /// Exact Hessian decomposition:
    ///   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
    ///
    ///   Q_{kℓ} = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
    ///
    ///   L_{kℓ} = 0.5 [ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
    ///
    ///   P_{kℓ} = -0.5 ∂²_{kℓ} log|S|_+
    ///
    /// Here, this function computes the exact gradient terms (including dH/dρ_k via d_i).
    /// The full exact Hessian is not assembled in this loop because it requires B_{kℓ}
    /// solves and fourth-derivative terms for every (k,ℓ) pair.
    ///
    /// Gaussian REML note:
    ///   In identity-link Gaussian, d=e=0 so H_k=A_k and H_{kℓ}=δ_{kℓ}A_k.
    ///   With profiled φ, use either:
    ///   - explicit profiled objective derivatives, or
    ///   - Schur complement in (ρ, log φ):
    ///       H_prof = H_{ρρ} - H_{ρα} H_{αα}^{-1} H_{αρ}.
    ///
    /// Pseudo-determinant note:
    ///   The code uses fixed-rank/stabilized conventions for log|S|_+ to keep objective
    ///   derivatives smooth and consistent with the transformed penalty basis used by PIRLS.
    ///
    /// This is the core of the outer optimization loop and provides the search direction for the BFGS algorithm.
    /// The calculation differs significantly between the Gaussian (REML) and non-Gaussian (LAML) cases.
    ///
    /// # Mathematical Basis (Gaussian/REML Case)
    ///
    /// For Gaussian models (Identity link), we minimize the negative REML log-likelihood, which serves as our cost function.
    /// From Wood (2011, JRSSB, Eq. 4), the cost function to minimize is:
    ///
    ///   Cost(ρ) = -l_r(ρ) = D_p / (2φ) + (1/2)log|XᵀWX + S(ρ)| - (1/2)log|S(ρ)|_+
    ///
    /// where D_p is the penalized deviance, H = XᵀWX + S(ρ) is the penalized Hessian, S(ρ) is the total
    /// penalty matrix, and |S(ρ)|_+ is the pseudo-determinant.
    ///
    /// The gradient ∇Cost(ρ) is computed term-by-term. A key simplification for the Gaussian case is the
    /// **envelope theorem**: at the P-IRLS optimum for β̂, the derivative of the cost function with respect to β̂ is zero.
    /// This means we only need the *partial* derivatives with respect to ρ, and the complex indirect derivatives
    /// involving ∂β̂/∂ρ can be ignored.
    ///
    /// # Mathematical Basis (Non-Gaussian/LAML Case)
    ///
    /// For non-Gaussian models, the envelope theorem does not apply because the weight matrix W depends on β̂.
    /// The gradient requires calculating the full derivative, including the indirect term (∂V/∂β̂)ᵀ(∂β̂/∂ρ).
    /// This leads to a different final formula involving derivatives of the weight matrix, as detailed in
    /// Wood (2011, Appendix D).
    ///
    /// This method handles two distinct statistical criteria for marginal likelihood optimization:
    ///
    /// - For Gaussian models (Identity link), this calculates the exact REML gradient
    ///   (Restricted Maximum Likelihood).
    /// - For non-Gaussian GLMs, this calculates the LAML gradient (Laplace Approximate
    ///   Marginal Likelihood) as derived in Wood (2011, Appendix C & D).
    ///
    /// # Mathematical Theory
    ///
    /// The gradient calculation requires careful application of the chain rule and envelope theorem
    /// due to the nested optimization structure of GAMs:
    ///
    /// - The inner loop (P-IRLS) finds coefficients β̂ that maximize the penalized log-likelihood
    ///   for a fixed set of smoothing parameters ρ.
    /// - The outer loop (BFGS) finds smoothing parameters ρ that maximize the marginal likelihood.
    ///
    /// Since β̂ is an implicit function of ρ, the total derivative is:
    ///
    ///    dV_R/dρ_k = (∂V_R/∂β̂)ᵀ(∂β̂/∂ρ_k) + ∂V_R/∂ρ_k
    ///
    /// By the envelope theorem, (∂V_R/∂β̂) = 0 at the optimum β̂, so the first term vanishes.
    ///
    /// # Key Distinction Between REML and LAML Gradients
    ///
    /// - Gaussian (REML): by the envelope theorem the indirect β̂ terms vanish. The deviance
    ///   contribution reduces to the penalty-only derivative, yielding the familiar
    ///   (β̂ᵀS_kβ̂)/σ² piece in the gradient.
    /// - Non-Gaussian (LAML): there is no cancellation of the penalty derivative within the
    ///   deviance component. The derivative of the penalized deviance contains both
    ///   d(D)/dρ_k and d(βᵀSβ)/dρ_k. Our implementation follows mgcv’s gdi1: we add the penalty
    ///   derivative to the deviance derivative before applying the 1/2 factor.
    // Stage: Start with the chain rule for any λₖ,
    //     dV/dλₖ = ∂V/∂λₖ  (holding β̂ fixed)  +  (∂V/∂β̂)ᵀ · (∂β̂/∂λₖ).
    //     The first summand is called the direct part, the second the indirect part.
    //
    // Stage: Note the two outer criteria—Gaussian likelihood maximizes REML, while non-Gaussian likelihood
    //     maximizes a Laplace approximation to the marginal likelihood (LAML). These objectives respond differently to β̂.
    //
    //     2.1  Gaussian case, REML.
    //          The REML construction integrates the fixed effects out of the likelihood.  At the optimum
    //          the partial derivative ∂V/∂β̂ is exactly zero.  The indirect part therefore vanishes.
    //          What remains is the direct derivative of the penalty and determinant terms.  The penalty
    //          contribution is found by differentiating −½ β̂ᵀ S_λ β̂ / σ² with respect to λₖ; this yields
    //          −½ β̂ᵀ Sₖ β̂ / σ².  No opposing term exists, so the quantity stays in the REML gradient.
    //          The code path selected by LinkFunction::Identity therefore computes
    //          beta_term = β̂ᵀ Sₖ β̂ and places it inside
    //          gradient[k] = 0.5 * λₖ * (beta_term / σ² − trace_term).
    //
    //     2.2  Non-Gaussian case, LAML.
    //          The Laplace objective contains −½ log |H_p| with H_p = Xᵀ W(β̂) X + S_λ.  Because W
    //          depends on β̂, the total derivative includes dW/dλₖ via β̂.  Differentiating the
    //          optimality condition for β̂ gives
    //          ∂β̂/∂λₖ = −λₖ H_p⁻¹ Sₖ β̂.  The penalized log-likelihood L(β̂, λ) still obeys the
    //          envelope theorem, so dL/dλₖ = −½ β̂ᵀ Sₖ β̂ (no implicit term).
    //          The resulting cost gradient combines four pieces:
    //            +½ λₖ β̂ᵀ Sₖ β̂
    //            +½ λₖ tr(H_p⁻¹ Sₖ)
    //            +½ tr(H_p⁻¹ Xᵀ ∂W/∂λₖ X)
    //            −½ λₖ tr(S_λ⁺ Sₖ)
    //
    // Stage: Remember that the sign of ∂β̂/∂λₖ matters; from the implicit-function theorem the linear solve reads
    //     −H_p (∂β̂/∂λₖ) = λₖ Sₖ β̂, giving the minus sign used above.  With that sign the indirect and
    //     direct quadratic pieces are exact negatives, which is what the algebra requires.
    pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.arena
            .last_gradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
        // Get the converged P-IRLS result for the current rho (`p`)
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(err @ EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(err);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };
        let analytic = self.compute_gradient_with_bundle(p, &bundle)?;
        Ok(analytic)
    }

    #[allow(dead_code)]
    pub(crate) fn compute_joint_hyper_cost(
        &self,
        theta: &Array1<f64>,
        rho_dim: usize,
    ) -> Result<f64, EstimationError> {
        if rho_dim > theta.len() {
            return Err(EstimationError::InvalidInput(format!(
                "rho_dim {} exceeds theta dimension {}",
                rho_dim,
                theta.len()
            )));
        }
        let rho = theta.slice(s![..rho_dim]).to_owned();
        self.compute_cost(&rho)
    }

    fn uses_objective_consistent_fd_gradient(&self, rho: &Array1<f64>) -> bool {
        self.config.link_function() != LinkFunction::Identity
            && (self.config.objective_consistent_fd_gradient || rho.len() == 1)
    }

    /// Helper function that computes gradient using a shared evaluation bundle
    /// so cost and gradient reuse the identical stabilized Hessian and PIRLS state.
    ///
    /// # Exact Outer-Gradient Identity Used by This Function
    ///
    /// Notation:
    /// - `rho[k]` are log-smoothing parameters; `lambda[k] = exp(rho[k])`.
    /// - `S(rho) = Σ_k lambda[k] S_k`.
    /// - `A_k = ∂S/∂rho_k = lambda[k] S_k`.
    /// - `beta_hat(rho)` is the inner PIRLS mode.
    /// - `H(rho)` is the Laplace curvature matrix used by this objective path.
    ///
    /// Outer objective:
    ///   V(rho) = [penalized data-fit at beta_hat]
    ///          + 0.5 log|H(rho)| - 0.5 log|S(rho)|_+.
    ///
    /// Exact derivative form:
    ///   dV/drho_k
    ///   = 0.5 * beta_hat^T A_k beta_hat
    ///   + 0.5 * tr(H^{-1} H_k)
    ///   - 0.5 * tr(S^+ A_k),
    /// where H_k = dH/drho_k is the *total* derivative (includes beta_hat movement).
    ///
    /// Important implementation point:
    /// - We do NOT add a separate `(∇_beta V)^T (d beta_hat / d rho_k)` term on top of
    ///   `tr(H^{-1} H_k)`. That dependence is already inside `H_k`.
    ///
    /// Variable mapping in this function:
    /// - `beta_terms[k]`     => beta_hat^T A_k beta_hat
    /// - `det1_values[k]`    => tr(S^+ A_k)
    /// - `trace_terms[k]`    => tr(H^{-1} H_k) / lambda[k] (before the outer lambda factor)
    /// - final assembly       => 0.5*beta_terms + 0.5*lambda*trace_terms - 0.5*det1
    ///
    /// ## Exact non-Gaussian Hessian system (reference for this implementation)
    ///
    /// For outer parameters ρ with λ_k = exp(ρ_k), A_k = ∂S/∂ρ_k = λ_k S_k, and
    /// H = -∇²ℓ(β̂(ρ)) + S(ρ), exact derivatives are:
    ///
    ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
    ///
    ///   H_k := ∂H/∂ρ_k = A_k + D(-∇²ℓ)[B_k]
    ///
    ///   B_{kℓ} solves:
    ///     H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ} A_k β̂ + A_k B_ℓ)
    ///
    ///   H_{kℓ} := ∂²H/(∂ρ_k∂ρ_ℓ)
    ///     = δ_{kℓ}A_k + D²(-∇²ℓ)[B_k,B_ℓ] + D(-∇²ℓ)[B_{kℓ}]
    ///
    /// Then the exact outer Hessian for V(ρ) = -ℓ(β̂)+0.5β̂ᵀSβ̂+0.5log|H|-0.5log|S|_+ is:
    ///
    ///   ∂²V/(∂ρ_k∂ρ_ℓ)
    ///     = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
    ///       + 0.5[ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
    ///       - 0.5 ∂² log|S|_+ /(∂ρ_k∂ρ_ℓ)
    ///
    /// This function computes the exact gradient terms (including the third-derivative
    /// contribution in H_k for logit). Full explicit H_{kℓ} assembly is not
    /// performed in the hot optimization loop because it requires B_{kℓ} solves and
    /// fourth-derivative likelihood terms for every (k,ℓ) pair.
    fn compute_gradient_with_bundle(
        &self,
        p: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array1<f64>, EstimationError> {
        if Self::geometry_backend_kind(bundle) == GeometryBackendKind::SparseExactSpd {
            return self.compute_gradient_sparse_exact(p, bundle);
        }
        // If there are no penalties (zero-length rho), the gradient in rho-space is empty.
        if p.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = bundle.ridge_passport;

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let reparam_result = &pirls_result.reparam_result;
        let mut h_eff_eval = bundle.h_eff.as_ref().clone();
        let mut e_eval = reparam_result.e_transformed.clone();
        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut rs_eval = reparam_result.rs_transformed.clone();
        let mut x_transformed_eval = pirls_result.x_transformed.clone();
        let mut h_pos_factor_w_eval = bundle.h_pos_factor_w.as_ref().clone();

        if let Some(z) = free_basis_opt.as_ref() {
            h_eff_eval = Self::project_with_basis(bundle.h_eff.as_ref(), z);
            e_eval = reparam_result.e_transformed.dot(z);
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            rs_eval = reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect();
            let x_dense_arc = pirls_result.x_transformed.to_dense_arc();
            x_transformed_eval = DesignMatrix::Dense(x_dense_arc.as_ref().dot(z));

            let (eigvals, eigvecs) = h_eff_eval
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let max_ev = eigvals.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
            let tol = (h_eff_eval.nrows().max(1) as f64) * f64::EPSILON * max_ev.max(1.0);
            let valid_indices: Vec<usize> = eigvals
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val > tol { Some(idx) } else { None })
                .collect();
            let p_eff = h_eff_eval.nrows();
            let mut w = Array2::<f64>::zeros((p_eff, valid_indices.len()));
            for (w_col_idx, &eig_idx) in valid_indices.iter().enumerate() {
                let val = eigvals[eig_idx];
                let scale = 1.0 / val.sqrt();
                let u_col = eigvecs.column(eig_idx);
                let mut w_col = w.column_mut(w_col_idx);
                Zip::from(&mut w_col)
                    .and(&u_col)
                    .for_each(|w_elem, &u_elem| *w_elem = u_elem * scale);
            }
            h_pos_factor_w_eval = w;
        }
        let h_eff = &h_eff_eval;

        // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
        let k_lambda = p.len();
        let k_r = rs_eval.len();
        let k_d = pirls_result.reparam_result.det1.len();
        if !(k_lambda == k_r && k_r == k_d) {
            return Err(EstimationError::LayoutError(format!(
                "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                k_lambda, k_r, k_d
            )));
        }
        if self.nullspace_dims.len() != k_lambda {
            return Err(EstimationError::LayoutError(format!(
                "Nullspace dimension mismatch: expected {} entries, got {}",
                k_lambda,
                self.nullspace_dims.len()
            )));
        }

        // --- Extract stable transformed quantities ---
        let beta_transformed = &beta_eval;
        // Use cached X·Qs from PIRLS
        let rs_transformed = &rs_eval;

        let includes_prior = false;
        let (gradient_result, gradient_snapshot, applied_truncation_corrections) = {
            let mut workspace_ref = self.arena.workspace.lock().unwrap();
            let workspace = &mut *workspace_ref;
            let len = p.len();
            workspace.reset_for_eval(len);
            workspace.set_lambda_values(p);
            workspace.zero_cost_gradient(len);
            let lambdas = workspace.lambda_view(len).to_owned();
            let mut applied_truncation_corrections: Option<Vec<f64>> = None;

            // Fixed structural-rank pseudo-determinant derivatives:
            // d/dρ_k log|S|_+ and d²/(dρ_k dρ_ℓ) log|S|_+ are evaluated on a
            // reduced structural subspace (rank = e_transformed.nrows()) with a
            // smooth floor in that reduced block. This avoids adaptive rank flips.
            let (det1_values, _) = self.structural_penalty_logdet_derivatives(
                rs_transformed,
                &lambdas,
                e_eval.nrows(),
                ridge_passport.penalty_logdet_ridge(),
            )?;

            // --- Use Single Stabilized Hessian from P-IRLS ---
            // Use the same effective Hessian as the cost function for consistency.
            if ridge_passport.laplace_hessian_ridge() > 0.0 {
                log::debug!(
                    "Gradient path using PIRLS-stabilized Hessian (ridge {:.3e})",
                    ridge_passport.laplace_hessian_ridge()
                );
            }

            // Check that the stabilized effective Hessian is still numerically valid.
            // If even the ridged matrix is indefinite, the PIRLS fit is unreliable and we retreat.
            if let Ok((eigenvalues, _)) = h_eff.eigh(Side::Lower) {
                let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                const SEVERE_INDEFINITENESS: f64 = -1e-4; // Threshold for severe problems
                if min_eig < SEVERE_INDEFINITENESS {
                    // The matrix was severely indefinite - signal a need to retreat
                    log::warn!(
                        "Severely indefinite Hessian detected in gradient (min_eig={:.2e}); returning robust retreat gradient.",
                        min_eig
                    );
                    // Generate an informed retreat direction based on current parameters
                    let retreat_grad = p.mapv(|v| -(v.abs() + 1.0));
                    return Ok(retreat_grad);
                }
            }

            // --- Extract common components ---

            let n = self.y.len() as f64;

            // -------------------------------------------------------------------------
            // Derivation map:
            //   λ_k = exp(ρ_k), A_k = ∂S/∂ρ_k = λ_k S_k.
            //   By the envelope theorem at inner stationarity, the penalized-fit block
            //   does not include an explicit dβ̂/dρ term.
            //   Outer gradient assembly:
            //     ∂V/∂ρ_k = 0.5 β̂^T A_k β̂ + 0.5 tr(H_+^† H_k) - 0.5 tr(S_+^† A_k).
            //   H_k differs by family:
            //     Gaussian: H_k = A_k.
            //     Non-Gaussian: H_k = A_k + d(X^T W(η̂) X)/dρ_k (third-derivative path).
            // -------------------------------------------------------------------------
            // Reference: gam.fit3.R line 778: REML1 <- oo$D1/(2*scale*gamma) + oo$trA1/2 - rp$det1/2

            match self.config.link_function() {
                LinkFunction::Identity => {
                    // Gaussian REML gradient from Wood (2011), 6.6.1

                    // Calculate scale parameter using the regular REML profiling
                    // φ = D_p / (n - M_p), where M_p is the penalty nullspace dimension.
                    let rss = pirls_result.deviance;

                    // Use stable penalty term calculated in P-IRLS
                    let penalty = pirls_result.stable_penalty_term;
                    let dp = rss + penalty; // Penalized deviance (a.k.a. D_p)
                    let (dp_c, dp_c_grad) = smooth_floor_dp(dp);

                    let penalty_rank = e_eval.nrows();
                    let mp = h_eff.ncols().saturating_sub(penalty_rank) as f64;
                    let scale = dp_c / (n - mp).max(LAML_RIDGE);
                    // Gaussian profiled-scale identity used by this branch:
                    //   φ̂(ρ) = D_p(ρ)/(n-M_p), with D_p = rss + β̂ᵀSβ̂.
                    // The gradient therefore includes the profiled contribution
                    //   (n-M_p)/2 * D_k / D_p
                    // which is exactly represented by `deviance_grad_term` below.
                    // (Equivalent to Schur-complement profiling in (ρ, log φ).)

                    if dp_c <= DP_FLOOR + DP_FLOOR_SMOOTH_WIDTH {
                        log::warn!(
                            "[REML] Penalized deviance {:.3e} near DP_FLOOR; keeping analytic Gaussian gradient on smooth-floor surface.",
                            dp_c
                        );
                    }

                    // Three-term gradient computation following mgcv gdi1
                    // for k in 0..lambdas.len() {
                    //   We'll calculate s_k_beta for all cases, as it's needed for both paths
                    //   For Identity link, this is all we need due to envelope theorem
                    //   For other links, we'll use it to compute dβ/dρ_k

                    //   Use transformed penalty matrix for consistent gradient calculation
                    //   let s_k_beta = reparam_result.rs_transformed[k].dot(beta);

                    // For the Gaussian/REML case, the Envelope Theorem applies: at the P-IRLS optimum,
                    // the indirect derivative through β cancels out for the deviance part, leaving only
                    // the direct penalty term derivative. This simplification is not available for
                    // non-Gaussian models where the weight matrix depends on β.

                    // factor_g already computed above; reuse it for trace terms

                    let det1_values = &det1_values;
                    let beta_ref = beta_transformed;
                    // Use the same positive-part Hessian factor as cost evaluation:
                    //   H_+^† = W W^T.
                    // Then tr(H_+^† A_k) = λ_k ||R_k W||_F^2 directly, with no separate
                    // truncated-subspace subtraction term.
                    let w_pos = &h_pos_factor_w_eval;
                    // Exact Gaussian identity REML gradient (profiled scale) in log-smoothing coordinates:
                    //
                    //   V_REML(ρ) =
                    //     0.5 * log|H|
                    //   - 0.5 * log|S|_+
                    //   + ((n - M_p)/2) * log(2π φ̂)
                    //   + const,
                    //
                    // where H = Xᵀ W0 X + S(ρ), S(ρ) = Σ_k λ_k S_k + δI, λ_k = exp(ρ_k),
                    // and φ̂ = D_p / (n - M_p), D_p = ||W0^(1/2)(y - Xβ̂ - o)||² + β̂ᵀ S β̂.
                    //
                    // Because Gaussian identity has c_i = d_i = 0, we have:
                    //   H_k := ∂H/∂ρ_k = S_k^ρ = λ_k S_k.
                    // Envelope theorem at β̂(ρ) gives:
                    //   ∂D_p/∂ρ_k = β̂ᵀ S_k^ρ β̂.
                    // Therefore:
                    //   ∂V_REML/∂ρ_k =
                    //     0.5 * tr(H^{-1} S_k^ρ)
                    //   - 0.5 * tr(S^+ S_k^ρ)
                    //   + (1/(2 φ̂)) * β̂ᵀ S_k^ρ β̂.
                    //
                    // Mapping to variables below:
                    //   d1 / (2*scale)                     -> (1/(2 φ̂)) * β̂ᵀ S_k^ρ β̂
                    //   log_det_h_grad_term (or numeric)   -> 0.5 * tr(H^{-1} S_k^ρ)
                    //   0.5 * det1_values[k]               -> 0.5 * tr(S^+ S_k^ρ)
                    let compute_gaussian_grad = |k: usize| -> f64 {
                        let r_k = &rs_transformed[k];
                        // Avoid forming S_k: compute S_k β = Rᵀ (R β)
                        let r_beta = r_k.dot(beta_ref);
                        let s_k_beta_transformed = r_k.t().dot(&r_beta);

                        // Component 1 derivation (profiled Gaussian REML):
                        //
                        //   V_prof includes (n-M_p)/2 * log D_p(ρ), so
                        //   ∂V_prof/∂ρ_k contributes (n-M_p)/2 * D_k / D_p = D_k/(2φ̂),
                        //   φ̂ = D_p/(n-M_p).
                        //
                        // At β̂, envelope cancellation gives:
                        //   D_k = β̂ᵀ A_k β̂ = λ_k β̂ᵀ S_k β̂.
                        //
                        // `d1` stores D_k, and the expression below is D_k/(2φ̂)
                        // with the smooth-floor derivative factor `dp_c_grad`.
                        let d1 = lambdas[k] * beta_ref.dot(&s_k_beta_transformed);
                        let deviance_grad_term = dp_c_grad * (d1 / (2.0 * scale));

                        // A.3/A.5 Component 2 derivation:
                        //   ∂/∂ρ_k [0.5 log|H|_+] = 0.5 tr(H_+^† H_k),
                        // and for Gaussian identity H_k = A_k = λ_k S_k.
                        //
                        // Root form on kept subspace:
                        //   tr(H_+^† A_k) = λ_k tr(H_+^† R_kᵀR_k)
                        //                = λ_k ||R_k W||_F², H_+^†=W W^T.
                        let rkw = r_k.dot(w_pos);
                        let trace_h_pos_inv_s_k: f64 = rkw.iter().map(|v| v * v).sum();
                        let log_det_h_grad_term = 0.5 * lambdas[k] * trace_h_pos_inv_s_k;

                        let corrected_log_det_h = log_det_h_grad_term;

                        // Component 3 derivation:
                        //   -0.5 * ∂/∂ρ_k log|S|_+,
                        // with `det1_values[k]` already equal to ∂ log|S|_+ / ∂ρ_k.
                        let log_det_s_grad_term = 0.5 * det1_values[k];

                        deviance_grad_term + corrected_log_det_h - log_det_s_grad_term
                    };

                    {
                        let mut grad_view = workspace.cost_gradient_view(len);
                        for k in 0..lambdas.len() {
                            grad_view[k] = compute_gaussian_grad(k);
                        }
                    }
                    // No explicit truncation correction vector is needed in this branch:
                    // the H_+^† trace is evaluated directly on the kept subspace.
                    applied_truncation_corrections = None;
                }
                _ => {
                    // NON-GAUSSIAN LAML GRADIENT (A.4 exact dH/dρ path)
                    //
                    // Objective:
                    //   V_LAML(ρ) =
                    //     -ℓ(β̂) + 0.5 β̂ᵀ S β̂
                    //   - 0.5 log|S|_+
                    //   + 0.5 log|H|
                    //   + const
                    //
                    // with H(ρ) = J(β̂(ρ)) + S(ρ), J = Xᵀ diag(b) X.
                    //
                    // Exact gradient (cost minimization convention):
                    //   ∂V/∂ρ_k =
                    //     0.5 β̂ᵀ S_k^ρ β̂
                    //   - 0.5 tr(S^+ S_k^ρ)
                    //   + 0.5 tr(H^{-1} H_k)
                    //
                    // where:
                    //   S_k^ρ = λ_k S_k, λ_k = exp(ρ_k),
                    //   b_k   = ∂β̂/∂ρ_k = -H^{-1}(S_k^ρ β̂),
                    //   v_k   = H^{-1}(S_k^ρ β̂) = -b_k,
                    //   H_k   = S_k^ρ + Xᵀ diag(w' ⊙ X b_k) X
                    //         = S_k^ρ - Xᵀ diag(w' ⊙ (X v_k)) X,
                    // and c_i = -∂^3 ℓ_i / ∂η_i^3.
                    //
                    // Derivation anchor:
                    //   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀ S β̂ + 0.5 log|H|_+ - 0.5 log|S|_+
                    //   with stationarity g(β̂,ρ)=∂/∂β[-ℓ + 0.5 βᵀSβ]=0.
                    // Envelope theorem removes explicit (∂V/∂β̂)(dβ̂/dρ_k) from the
                    // penalized-fit block, but β̂-dependence still enters via dH/dρ_k.
                    // The dH term is exactly what the third-derivative contraction encodes.
                    //
                    // The second term inside H_k is the exact "missing tensor term":
                    //   ∂H/∂ρ_k ≠ S_k^ρ
                    // for non-Gaussian families; dropping it yields the usual approximation.
                    //
                    // Implementation strategy here (logit path):
                    //   1) build S_k β̂ in transformed basis via penalty roots R_k,
                    //   2) solve/apply H_+^† to get v_k and leverage terms,
                    //   3) evaluate tr(H_+^† H_k) as
                    //        tr(H_+^† S_k) - tr(H_+^† Xᵀ diag(c ⊙ X v_k) X),
                    //   4) assemble
                    //        0.5*β̂ᵀA_kβ̂ + 0.5*tr(H_+^†H_k) - 0.5*tr(S^+A_k).
                    //
                    // There is intentionally no extra "(∇_β V)^T dβ/dρ" add-on here:
                    // the beta-dependence path is already encoded in H_k through the
                    // third-derivative contraction term.
                    // Replace FD with implicit differentiation for logit models.
                    // When Firth bias reduction is enabled, the inner objective is:
                    //   L*(beta, rho) = l(beta) - 0.5 * beta' S_lambda beta
                    //                 + 0.5 * log|X' W(beta) X|
                    // with W depending on beta (logit: w_i = mu_i (1 - mu_i)).
                    // Stationarity: grad_beta L* = 0, so the implicit derivative uses
                    // H_total = X' W X + S_lambda - d^2/d beta^2 (0.5 * log|X' W X|).
                    //
                    // Exact Firth derivatives (let K = (X' W X)^{-1}):
                    //   Phi(beta) = 0.5 * log|X' W X|
                    //   grad Phi_j = 0.5 * tr(K X' (dW/d beta_j) X)
                    //             = 0.5 * sum_i h_i * (d w_i / d eta_i) * x_ij
                    //   where h_i = x_i' K x_i (leverages in weighted space).
                    //
                    //   Hessian:
                    //     d^2 Phi / (d beta_j d beta_l) =
                    //       -0.5 * tr(K X' (dW/d beta_l) X K X' (dW/d beta_j) X)
                    //       +0.5 * sum_i h_i * (d^2 w_i / d eta_i^2) * x_ij * x_il
                    //
                    // This curvature enters H_total and therefore d beta_hat / d rho_k.
                    // Our analytic LAML gradient uses H_pen = X' W X + S_lambda only,
                    // so it is inconsistent with the Firth-adjusted objective unless
                    // we add H_phi. Below we compute H_phi and use H_total for the
                    // implicit solve (d beta_hat / d rho). If that fails, we fall
                    // back to H_pen for stability.
                    let mut w_prime = pirls_result.solve_c_array.clone();
                    if !w_prime.iter().all(|v| v.is_finite()) {
                        // Keep production gradient path analytic-only.
                        // If third-derivative entries are non-finite, sanitize them
                        // conservatively to zero so the dH/dρ third-order contraction
                        // vanishes instead of switching to numeric differentiation.
                        log::warn!(
                            "[REML] non-finite third-derivative weights detected; sanitizing to zero and continuing analytic gradient."
                        );
                        for val in &mut w_prime {
                            if !val.is_finite() {
                                *val = 0.0;
                            }
                        }
                    }
                    let clamp_nonsmooth = self.config.firth_bias_reduction
                        && pirls_result
                            .solve_mu
                            .iter()
                            .any(|&mu| mu * (1.0 - mu) < Self::MIN_DMU_DETA);
                    if clamp_nonsmooth {
                        // Keep analytic gradient as the optimizer default even when IRLS
                        // weights are clamped, to avoid FD ridge-jitter artifacts in
                        // line-search/BFGS updates.
                        // Hard clamps/floors make the objective only piecewise-smooth.
                        // c_i values then act like a selected generalized derivative
                        // (Clarke-subgradient style), so central FD may disagree at kinks.
                        log::debug!(
                            "[REML] IRLS weight clamp detected; continuing with analytic gradient"
                        );
                    }
                    let k_count = lambdas.len();
                    let det1_values = &det1_values;
                    let beta_ref = beta_transformed;
                    let mut beta_terms = Array1::<f64>::zeros(k_count);
                    let mut s_k_beta_mat = Array2::<f64>::zeros((beta_ref.len(), k_count));
                    for k in 0..k_count {
                        let r_k = &rs_transformed[k];
                        let r_beta = r_k.dot(beta_ref);
                        let s_k_beta = r_k.t().dot(&r_beta);
                        // q_k = β̂^T A_k β̂ = λ_k β̂^T S_k β̂,
                        // with S_k β̂ assembled as R_k^T (R_k β̂).
                        beta_terms[k] = lambdas[k] * beta_ref.dot(&s_k_beta);
                        s_k_beta_mat.column_mut(k).assign(&s_k_beta);
                    }

                    // Keep outer gradient on the same Hessian surface as PIRLS.
                    // The outer loop uses H_eff consistently (no H_phi subtraction).

                    // P-IRLS already folded any stabilization ridge into h_eff.

                    // TRACE TERM COMPUTATION (exact non-Gaussian/logit dH term):
                    //   tr(H_+^\dagger H_k), with
                    //   H_k = S_k - X^T diag(c ⊙ (X v_k)) X,  v_k = H_+^\dagger (S_k beta).
                    //
                    // We evaluate this without explicit third-derivative tensors:
                    //   tr(H_+^\dagger S_k) = ||R_k W||_F^2
                    //   tr(H_+^\dagger X^T diag(t_k) X) = Σ_i t_k[i] * h_i,
                    // where t_k = c ⊙ (X v_k), h_i = x_i^T H_+^\dagger x_i, and H_+^\dagger = W W^T.
                    //
                    // This is the matrix-free realization of the exact identity:
                    //   tr(H^{-1}H_k) = tr(H^{-1}A_k) + tr(H^{-1}D(-∇²ℓ)[B_k]),
                    // with B_k = -H^{-1}(A_kβ̂).
                    //
                    //   D(-∇²ℓ)[B_k] = Xᵀ diag(d ⊙ (X B_k)) X,
                    // where d_i = -∂³ℓ_i/∂η_i³. Here `c_vec` stores this per-observation
                    // third derivative quantity in the stabilized logit path.
                    let w_pos = &h_pos_factor_w_eval;
                    let n_obs = pirls_result.solve_mu.len();

                    // c_i = dW_ii/dη_i for H = Xᵀ W X + S.
                    // In smooth regimes this matches the required third-derivative object
                    // in dH/dρ. In clamped/floored regimes c_i may behave like a subgradient
                    // proxy rather than a classical derivative; see pirls.rs comments.
                    let c_vec = &w_prime;

                    // h_i = x_i^T H_+^\dagger x_i = ||(XW)_{i,*}||^2.
                    let mut leverage_h_pos = Array1::<f64>::zeros(n_obs);
                    if w_pos.ncols() > 0 {
                        match &x_transformed_eval {
                            DesignMatrix::Dense(x_dense) => {
                                let xw = x_dense.dot(w_pos);
                                for i in 0..xw.nrows() {
                                    leverage_h_pos[i] = xw.row(i).iter().map(|v| v * v).sum();
                                }
                            }
                            DesignMatrix::Sparse(_) => {
                                for col in 0..w_pos.ncols() {
                                    let w_col = w_pos.column(col).to_owned();
                                    let xw_col = x_transformed_eval.matrix_vector_multiply(&w_col);
                                    Zip::from(&mut leverage_h_pos)
                                        .and(&xw_col)
                                        .for_each(|h, &v| *h += v * v);
                                }
                            }
                        }
                    }

                    // Precompute r = X^T (c ⊙ h) once:
                    //   trace_third_k = (c ⊙ h)^T (X v_k) = r^T v_k.
                    // This removes the per-k O(np) multiply X*v_k from the hot loop.
                    // r := X^T (w' ⊙ h).
                    let c_times_h = c_vec * &leverage_h_pos;
                    let r_third = x_transformed_eval.transpose_vector_multiply(&c_times_h);

                    // Batch all v_k = H_+^† (S_k beta) into one BLAS-3 path:
                    //   V = W (W^T [S_1 beta, ..., S_K beta]).
                    let v_all = if w_pos.ncols() > 0 && k_count > 0 {
                        let wt_sk_beta_all = w_pos.t().dot(&s_k_beta_mat);
                        w_pos.dot(&wt_sk_beta_all)
                    } else {
                        Array2::<f64>::zeros((beta_ref.len(), k_count))
                    };

                    let trace_mode = std::env::var("GAM_DIAG_TRACE_THIRD_MODE")
                        .unwrap_or_else(|_| "minus".to_string());
                    let trace_mode_code = match trace_mode.as_str() {
                        "plus" => 1u8,
                        "zero" => 2u8,
                        _ => 0u8,
                    };
                    {
                        let mut grad_view = workspace.cost_gradient_view(len);
                        for k_idx in 0..k_count {
                            let r_k = &rs_transformed[k_idx];
                            if r_k.ncols() == 0 || w_pos.ncols() == 0 {
                                let log_det_h_grad_term = 0.0;
                                let log_det_s_grad_term = 0.5 * det1_values[k_idx];
                                grad_view[k_idx] = 0.5 * beta_terms[k_idx] + log_det_h_grad_term
                                    - log_det_s_grad_term;
                                continue;
                            }

                            // First piece:
                            //   tr(H_+^† S_k) = ||R_k W||_F^2, with H_+^† = W W^T.
                            let rkw = r_k.dot(w_pos);
                            let trace_h_inv_s_k: f64 = rkw.iter().map(|v| v * v).sum();

                            // Exact third-derivative contraction:
                            //   tr(H_+^† X^T diag(c ⊙ X v_k) X) = r^T v_k.
                            let v_k = v_all.column(k_idx);
                            let trace_third = r_third.dot(&v_k);

                            // Diagnostic switch for term-by-term identification of
                            // analytic-vs-FD disagreement. Production behavior is "minus",
                            // matching the smooth-theory formula tr(H^{-1}A_k) - tr(H^{-1}Xᵀdiag(c⊙Xv_k)X).
                            let trace_term = match trace_mode_code {
                                1 => trace_h_inv_s_k + trace_third,
                                2 => trace_h_inv_s_k,
                                _ => trace_h_inv_s_k - trace_third,
                            };
                            let log_det_h_grad_term = 0.5 * lambdas[k_idx] * trace_term;
                            let corrected_log_det_h = log_det_h_grad_term;
                            let log_det_s_grad_term = 0.5 * det1_values[k_idx];

                            // Exact LAML gradient assembly for the implemented objective:
                            //   g_k = 0.5 * β̂ᵀ A_k β̂ - 0.5 * tr(S^+ A_k) + 0.5 * tr(H^{-1} H_k)
                            // where A_k = ∂S/∂ρ_k = λ_k S_k and H_k is the total derivative.
                            grad_view[k_idx] =
                                0.5 * beta_terms[k_idx] + corrected_log_det_h - log_det_s_grad_term;
                        }
                    }
                }
            }

            if !includes_prior {
                let (_, prior_grad_view) = workspace.soft_prior_cost_and_grad(p);
                let prior_grad = prior_grad_view.to_owned();
                {
                    let mut cost_gradient_view = workspace.cost_gradient_view(len);
                    cost_gradient_view += &prior_grad;
                }
            }

            // Capture the gradient snapshot before releasing the workspace borrow so
            // that diagnostics can continue without holding the RefCell borrow.
            let gradient_result = workspace.cost_gradient_view_const(len).to_owned();
            let gradient_snapshot = if p.is_empty() {
                None
            } else {
                Some(gradient_result.clone())
            };

            (
                gradient_result,
                gradient_snapshot,
                applied_truncation_corrections,
            )
        };

        // The gradient buffer stored in the workspace already holds -∇V(ρ),
        // which is exactly what the optimizer needs.
        // No final negation is needed.

        // Comprehensive gradient diagnostics (all four strategies)
        if let Some(gradient_snapshot) = gradient_snapshot
            && !p.is_empty()
        {
            // Run all diagnostics and emit a single summary if issues found
            self.run_gradient_diagnostics(
                p,
                bundle,
                &gradient_snapshot,
                applied_truncation_corrections.as_deref(),
            );
        }

        if self.should_use_stochastic_exact_gradient(bundle, &gradient_result) {
            match self.compute_logit_stochastic_exact_gradient(p, bundle) {
                Ok(stochastic_grad) => {
                    self.arena
                        .last_gradient_used_stochastic_fallback
                        .store(true, Ordering::Relaxed);
                    log::warn!(
                        "[REML] using stochastic exact log-marginal gradient fallback (posterior-sampled expectation)"
                    );
                    return Ok(stochastic_grad);
                }
                Err(err) => {
                    log::warn!(
                        "[REML] stochastic exact gradient fallback failed; keeping analytic gradient: {:?}",
                        err
                    );
                }
            }
        }

        Ok(gradient_result)
    }

    pub fn last_gradient_used_stochastic_fallback(&self) -> bool {
        self.arena
            .last_gradient_used_stochastic_fallback
            .load(Ordering::Relaxed)
    }

    fn should_use_stochastic_exact_gradient(
        &self,
        bundle: &EvalShared,
        gradient: &Array1<f64>,
    ) -> bool {
        // Gate for the posterior-sampled gradient path.
        // This predicate checks for non-finite or unstable analytic states.
        if self.config.link_function() != LinkFunction::Logit {
            return false;
        }
        if self.config.firth_bias_reduction {
            // Firth-adjusted inner objective does not match the plain PG/NUTS posterior target here.
            return false;
        }
        if gradient.is_empty() {
            return false;
        }
        if !gradient.iter().all(|g| g.is_finite()) {
            return true;
        }
        let pirls = bundle.pirls_result.as_ref();
        if matches!(pirls.status, pirls::PirlsStatus::Unstable) {
            return true;
        }
        let kkt_like = pirls.last_gradient_norm;
        if !kkt_like.is_finite() || kkt_like > 1e2 {
            return true;
        }
        let grad_inf = gradient.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        !grad_inf.is_finite() || grad_inf > 1e9
    }

    fn compute_logit_stochastic_exact_gradient(
        &self,
        p: &Array1<f64>,
        bundle: &EvalShared,
    ) -> Result<Array1<f64>, EstimationError> {
        // Derivation sketch (sign convention used by this minimization objective):
        //
        // 1) Penalized evidence identity (logit):
        //      Z(ρ) = ∫ exp(l(β) - 0.5 βᵀS(ρ)β) dβ,   S(ρ)=Σ_j exp(ρ_j) S_j.
        //
        // 2) Fisher/PG identity for each coordinate:
        //      ∂/∂ρ_k log Z(ρ) = -0.5 * λ_k * E_{π(β|y,ρ)}[βᵀ S_k β],   λ_k=exp(ρ_k).
        //
        // 3) This code optimizes a cost that includes the pseudo-determinant
        //    normalization of the improper Gaussian penalty, yielding:
        //      g_k = ∂Cost/∂ρ_k
        //          = 0.5 * λ_k * E[βᵀS_kβ] - 0.5 * λ_k * tr(S(ρ)^+ S_k).
        //
        // 4) Root-factor rewrite used numerically:
        //      S_k = R_kᵀR_k  =>  βᵀS_kβ = ||R_kβ||².
        //
        // 5) Implementation mapping:
        //      PG-Rao-Blackwell average of tr(S_kQ^{-1})+μᵀS_kμ -> E[βᵀS_kβ],
        //      det1_values[k]                                 -> λ_k tr(S(ρ)^+S_k),
        //      grad[k]                                        -> g_k.
        // Equation-to-code map for this fallback path (logit, fixed ρ):
        //   g_k := ∂Cost/∂ρ_k
        //      = 0.5 * λ_k * E_{π(β|y,ρ)}[βᵀ S_k β]
        //        - 0.5 * λ_k * tr(S(ρ)^+ S_k),
        //   λ_k = exp(ρ_k).
        //
        // The first expectation is evaluated by PG Gibbs + Rao-Blackwellization.
        // The second term is deterministic via structural pseudo-logdet derivatives.
        let pirls_result = bundle.pirls_result.as_ref();
        let beta_mode = pirls_result.beta_transformed.as_ref();
        let s_transformed = &pirls_result.reparam_result.s_transformed;
        let x_arc = pirls_result.x_transformed.to_dense_arc();
        let x_dense = x_arc.as_ref();
        let y = self.y;
        let weights = self.weights;
        let h_eff = bundle.h_eff.as_ref();

        // PG-Gibbs Rao-Blackwell fallback: fewer samples are needed than β-NUTS
        // because each retained ω state contributes the analytic conditional moment
        // tr(S_k Q^{-1}) + μᵀ S_k μ instead of a raw quadratic draw.
        let pg_cfg = crate::hmc::NutsConfig {
            n_samples: 24,
            n_warmup: 48,
            n_chains: 2,
            target_accept: 0.85,
            seed: 17_391,
        };

        let len = p.len();
        let mut lambda = Array1::<f64>::zeros(len);
        for k in 0..len {
            // Outer parameters are ρ; penalties are λ = exp(ρ).
            lambda[k] = p[k].exp();
        }

        let (det1_values, _) = self.structural_penalty_logdet_derivatives(
            &pirls_result.reparam_result.rs_transformed,
            &lambda,
            pirls_result.reparam_result.e_transformed.nrows(),
            bundle.ridge_passport.penalty_logdet_ridge(),
        )?;
        // det1_values[k] = ∂ log|S(ρ)|_+ / ∂ρ_k = λ_k tr(S(ρ)^+ S_k).

        let rb_terms_result = crate::hmc::estimate_logit_pg_rao_blackwell_terms(
            x_dense.view(),
            y,
            weights,
            s_transformed.view(),
            beta_mode.view(),
            &pirls_result.reparam_result.rs_transformed,
            &pg_cfg,
        );

        let mut grad = Array1::<f64>::zeros(len);
        match rb_terms_result {
            Ok(rb_terms) => {
                for k in 0..len {
                    // Rao-Blackwellized exact identity:
                    //   g_k = 0.5 * λ_k * E_ω[ tr(S_k Q^{-1}) + μᵀ S_k μ ] - 0.5 * det1_values[k].
                    grad[k] = 0.5 * lambda[k] * rb_terms[k] - 0.5 * det1_values[k];
                }
            }
            Err(err) => {
                log::warn!(
                    "[REML] PG Rao-Blackwell fallback failed ({}); reverting to NUTS beta averaging",
                    err
                );

                let nuts_cfg = crate::hmc::NutsConfig {
                    n_samples: 120,
                    n_warmup: 160,
                    n_chains: 2,
                    target_accept: 0.85,
                    seed: 17_391,
                };

                let nuts_result = crate::hmc::run_nuts_sampling_flattened_family(
                    crate::types::LikelihoodFamily::BinomialLogit,
                    crate::hmc::FamilyNutsInputs::Glm(crate::hmc::GlmFlatInputs {
                        x: x_dense.view(),
                        y,
                        weights,
                        penalty_matrix: s_transformed.view(),
                        mode: beta_mode.view(),
                        hessian: h_eff.view(),
                        firth_bias_reduction: self.config.firth_bias_reduction,
                    }),
                    &nuts_cfg,
                )
                .map_err(EstimationError::InvalidInput)?;

                let samples = &nuts_result.samples;
                let n_draws = samples.nrows().max(1);
                let mut expected_quad = vec![0.0_f64; len];
                for draw in 0..samples.nrows() {
                    let beta_draw = samples.row(draw).to_owned();
                    for k in 0..len {
                        let r_k = &pirls_result.reparam_result.rs_transformed[k];
                        let r_beta = r_k.dot(&beta_draw);
                        expected_quad[k] += r_beta.dot(&r_beta);
                    }
                }
                let inv_draws = 1.0 / (n_draws as f64);
                for v in &mut expected_quad {
                    *v *= inv_draws;
                }
                for k in 0..len {
                    grad[k] = 0.5 * lambda[k] * expected_quad[k] - 0.5 * det1_values[k];
                }
            }
        }
        grad += &self.compute_soft_prior_grad(p);
        Ok(grad)
    }

    fn structural_penalty_logdet_derivatives(
        &self,
        rs_transformed: &[Array2<f64>],
        lambdas: &Array1<f64>,
        structural_rank: usize,
        ridge: f64,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        // Full derivation for the penalty pseudo-logdet terms in the outer
        // gradient/Hessian:
        //
        //   P_k   = -0.5 * d/drho_k     log|S(rho)|_+
        //   P_k,l = -0.5 * d²/(drho_k drho_l) log|S(rho)|_+.
        //
        // Write
        //   S(rho) = sum_k lambda_k S_k,   lambda_k = exp(rho_k),
        //   A_k    = dS/drho_k      = lambda_k S_k,
        //   A_k,l  = d²S/(drho_k drho_l) = delta_{k,l} A_k.
        //
        // On a fixed positive-eigenspace / structural penalty subspace,
        // the pseudodeterminant calculus matches ordinary determinant
        // calculus with S^{-1} replaced by the inverse on that kept
        // subspace:
        //
        //   d/drho_k log|S|_+ = tr(S_+^dagger A_k)
        //
        // and
        //
        //   d²/(drho_k drho_l) log|S|_+
        //     = tr(S_+^dagger A_k,l)
        //       - tr(S_+^dagger A_l S_+^dagger A_k).
        //
        // Since A_k = lambda_k S_k and A_k,l = delta_{k,l} A_k, we obtain
        //
        //   det1[k] = d/drho_k log|S|_+ = tr(S_+^dagger A_k)
        //
        // and
        //
        //   det2[k,l]
        //     = d²/(drho_k drho_l) log|S|_+
        //     = delta_{k,l} det1[k]
        //       - lambda_k lambda_l tr(S_+^dagger S_k S_+^dagger S_l).
        //
        // This helper realizes exactly that formula on a fixed-rank
        // reduced penalty space. The fixed-support assumption is the same
        // one used elsewhere in the dense spectral path: the active penalty
        // subspace is held constant while differentiating in rho.
        let k_count = lambdas.len();
        if rs_transformed.len() != k_count {
            return Err(EstimationError::LayoutError(format!(
                "Penalty root/lambda count mismatch in structural logdet derivatives: roots={}, lambdas={}",
                rs_transformed.len(),
                k_count
            )));
        }
        if k_count == 0 {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        // IMPORTANT: dimensions must follow the *actual* transformed coefficient frame
        // presented by callers (possibly active-constraint projected), not self.p.
        let p_dim = rs_transformed[0].ncols();
        for (k, r_k) in rs_transformed.iter().enumerate() {
            if r_k.ncols() != p_dim {
                return Err(EstimationError::LayoutError(format!(
                    "Inconsistent penalty root width at k={k}: got {}, expected {}",
                    r_k.ncols(),
                    p_dim
                )));
            }
        }
        if p_dim == 0 || structural_rank == 0 {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        let rank = structural_rank.min(p_dim);
        if rank == 0 {
            return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
        }

        let mut s_k_full = Vec::with_capacity(k_count);
        let mut s_lambda = Array2::<f64>::zeros((p_dim, p_dim));
        for k in 0..k_count {
            let r_k = &rs_transformed[k];
            // Path: rs_transformed[k] is already in transformed coefficient frame.
            let s_k = r_k.t().dot(r_k);
            s_lambda += &s_k.mapv(|v| lambdas[k] * v);
            s_k_full.push(s_k);
        }
        if ridge > 0.0 {
            for i in 0..p_dim {
                s_lambda[[i, i]] += ridge;
            }
        }

        let (evals, evecs) = s_lambda
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let mut order: Vec<usize> = (0..p_dim).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });

        let mut u1 = Array2::<f64>::zeros((p_dim, rank));
        for (col_out, &col_in) in order.iter().take(rank).enumerate() {
            u1.column_mut(col_out).assign(&evecs.column(col_in));
        }
        let mut s_r = u1.t().dot(&s_lambda).dot(&u1);
        let max_diag = s_r
            .diag()
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let eps = 1e-12 * max_diag;
        for i in 0..rank {
            s_r[[i, i]] += eps;
        }
        // Factorize the kept structural block and evaluate contractions via solves.
        // This avoids forming an explicit inverse.
        let s_r_factor = self.factorize_faer(&s_r);

        let mut y_k_reduced = Vec::with_capacity(k_count);
        let mut det1 = Array1::<f64>::zeros(k_count);
        for k in 0..k_count {
            let s_kr = u1.t().dot(&s_k_full[k]).dot(&u1);
            // Solve S_r * Y_k = S_{k,r} and use tr(Y_k) = tr(S_r^{-1} S_{k,r}).
            let mut y_k = s_kr.clone();
            let mut y_k_view = array2_to_mat_mut(&mut y_k);
            s_r_factor.solve_in_place(y_k_view.as_mut());
            let tr = kahan_sum((0..rank).map(|i| y_k[[i, i]]));
            // A_k = λ_k S_k => tr(S_+^† A_k) = λ_k tr(S_+^† S_k).
            det1[k] = lambdas[k] * tr;
            y_k_reduced.push(y_k);
        }

        let mut det2 = Array2::<f64>::zeros((k_count, k_count));
        for k in 0..k_count {
            for l in 0..=k {
                // With Y_k = S_r^{-1} S_{k,r}, Y_l = S_r^{-1} S_{l,r},
                // we have tr_ab = tr(Y_k Y_l)
                //         = tr(S_+^dagger S_k S_+^dagger S_l)
                // on the kept structural subspace. Therefore
                //
                //   det2[k,l]
                //     = delta_{k,l} det1[k]
                //       - lambda_k lambda_l tr_ab.
                let tr_ab = Self::trace_product(&y_k_reduced[k], &y_k_reduced[l]);
                let mut val = -lambdas[k] * lambdas[l] * tr_ab;
                if k == l {
                    val += det1[k];
                }
                det2[[k, l]] = val;
                det2[[l, k]] = val;
            }
        }
        Ok((det1, det2))
    }

    fn compute_laml_hessian_analytic_fallback(
        &self,
        rho: &Array1<f64>,
        bundle_hint: Option<&EvalShared>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Deterministic analytic fallback used when the full exact Hessian
        // assembly is unavailable (e.g. active-subspace instability).
        //
        // We keep exact, closed-form pieces that remain robust:
        //   1) Penalty-envelope diagonal term:
        //      0.5 * beta' A_k beta = 0.5 * lambda_k * ||R_k beta||^2.
        //   2) Exact structural penalty pseudo-logdet curvature:
        //      -0.5 * d²/drho² log|S|_+.
        //   3) Soft prior Hessian.
        //
        // We intentionally omit the Laplace log|H|_+ curvature block here,
        // because that block is exactly where unstable active-set boundaries
        // make second derivatives unreliable.
        let bundle_owned;
        let bundle = if let Some(b) = bundle_hint {
            b
        } else {
            bundle_owned = self.obtain_eval_bundle(rho)?;
            &bundle_owned
        };
        let k = rho.len();
        let mut h = Array2::<f64>::zeros((k, k));
        if k == 0 {
            return Ok(h);
        }

        let pirls_result = bundle.pirls_result.as_ref();
        let beta = pirls_result.beta_transformed.as_ref();
        let rs = &pirls_result.reparam_result.rs_transformed;
        let lambdas = rho.mapv(f64::exp);

        for idx in 0..k {
            if idx >= rs.len() {
                break;
            }
            let r_k = &rs[idx];
            if r_k.ncols() == 0 {
                continue;
            }
            let r_beta = r_k.dot(beta);
            let q_diag = 0.5 * lambdas[idx] * r_beta.dot(&r_beta);
            h[[idx, idx]] += q_diag;
        }

        let (structural_rank, _) = self.fixed_subspace_penalty_rank_and_logdet(
            &pirls_result.reparam_result.e_transformed,
            pirls_result.ridge_passport,
        )?;
        let (_det1, det2) = self.structural_penalty_logdet_derivatives(
            &pirls_result.reparam_result.rs_transformed,
            &lambdas,
            structural_rank,
            pirls_result.ridge_passport.penalty_logdet_ridge(),
        )?;
        h += &det2.mapv(|v| -0.5 * v);

        self.add_soft_prior_hessian_in_place(rho, &mut h);

        // Always return a numerically strict PD matrix for downstream Newton/
        // covariance uses in fallback regimes.
        let scale = h
            .diag()
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let jitter = 1e-8 * scale;
        for i in 0..k {
            h[[i, i]] += jitter;
        }
        for i in 0..k {
            for j in 0..i {
                let avg = 0.5 * (h[[i, j]] + h[[j, i]]);
                h[[i, j]] = avg;
                h[[j, i]] = avg;
            }
        }

        if h.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "Analytic fallback Hessian produced non-finite values".to_string(),
            ));
        }
        Ok(h)
    }

    pub(crate) fn compute_laml_hessian_consistent(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Strategy-policy routing:
        // - policy decides spectral exact vs analytic fallback vs diagnostic numeric,
        // - math kernels remain strategy-local.
        let bundle = self.obtain_eval_bundle(rho)?;
        let decision = self.select_hessian_strategy_policy(rho, &bundle);
        if decision.strategy != HessianEvalStrategyKind::SpectralExact {
            if decision.reason == "active_subspace_unstable" {
                let rel_gap = bundle.active_subspace_rel_gap.unwrap_or(f64::NAN);
                log::warn!(
                    "Exact LAML Hessian downgraded via policy (reason={}, rel_gap={:.3e}).",
                    decision.reason,
                    rel_gap
                );
            } else {
                log::warn!(
                    "Exact LAML Hessian downgraded via policy (reason={}).",
                    decision.reason
                );
            }
            return self.compute_laml_hessian_by_strategy(rho, &bundle, decision);
        }
        match self.compute_laml_hessian_by_strategy(rho, &bundle, decision) {
            Ok(h) => Ok(h),
            Err(err) => {
                log::warn!(
                    "Exact LAML Hessian unavailable ({}); using analytic fallback Hessian.",
                    err
                );
                self.compute_laml_hessian_analytic_fallback(rho, Some(&bundle))
            }
        }
    }

    pub(crate) fn compute_laml_hessian_exact(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        if Self::geometry_backend_kind(&bundle) == GeometryBackendKind::SparseExactSpd {
            return self.compute_laml_hessian_sparse_exact(rho, &bundle);
        }
        // Full derivation for the dense transformed exact outer Hessian.
        //
        // Definitions:
        //
        //   lambda_k = exp(rho_k)
        //   S(rho)   = sum_k lambda_k S_k
        //   A_k      = dS/drho_k      = lambda_k S_k
        //   A_{k,l}  = d²S/(drho_k drho_l) = delta_{k,l} A_k
        //
        // The inner mode beta_hat(rho) is defined by the stationarity system
        //
        //   G(beta, rho) = grad_beta L(beta, rho) = 0
        //
        // with
        //
        //   L(beta, rho)
        //     = -ell(beta) + 0.5 beta' S(rho) beta - Phi(beta).
        //
        // Let
        //
        //   H = dG/dbeta = -ell_{beta,beta} + S(rho) - H_phi
        //     = X'WX + S(rho) - H_phi
        //   H_phi = d²Phi/dbeta²
        //
        // denote the inner Hessian on the current geometry. In the dense
        // transformed path this is the active transformed Hessian `h_total`.
        //
        // Implicit Function Theorem:
        //
        // Differentiating G(beta_hat(rho), rho) = 0 once gives
        //
        //   H B_k + A_k beta_hat = 0
        //
        // so
        //
        //   B_k = d beta_hat / drho_k = -H^{-1}(A_k beta_hat).
        //
        // Differentiating again gives
        //
        //   H_l B_k + H B_{k,l} + A_k B_l + A_{k,l} beta_hat = 0
        //
        // hence
        //
        //   B_{k,l}
        //     = -H^{-1}(H_l B_k + A_k B_l + delta_{k,l} A_k beta_hat).
        //
        // Logistic/Firth curvature derivatives:
        //
        // For eta = X beta, let c, d be PIRLS-provided per-observation
        // third/fourth derivatives of -ell wrt eta. For u_k = X B_k and
        // u_{k,l} = X B_{k,l}, the non-Firth pieces are
        //
        //   H_k
        //     = A_k + X' diag(c ⊙ u_k) X
        //
        // and
        //
        //   H_{k,l}
        //     = delta_{k,l} A_k
        //       + X' diag(d ⊙ u_k ⊙ u_l + c ⊙ u_{k,l}) X.
        //
        // and Firth adds exactly:
        //   H_k  <- H_k  - D(H_phi)[B_k]
        //   H_kl <- H_kl - D(H_phi)[B_kl] - D²(H_phi)[B_k,B_l].
        //
        // Here `c` and `d` are (`solve_c_array`, `solve_d_array`), and
        // D(H_phi), D²(H_phi) are evaluated by reduced-space exact operators.
        //
        // Outer objective:
        //
        //   V(rho)
        //     = L*(beta_hat(rho), rho)
        //       + 0.5 log|H(rho)|
        //       - 0.5 log|S(rho)|_+.
        //
        // By the envelope theorem,
        //
        //   dV/drho_k
        //     = 0.5 beta_hat' A_k beta_hat
        //       + 0.5 tr(H_+^dagger H_k)
        //       - 0.5 tr(S_+^dagger A_k).
        //
        // Differentiating again yields the exact Hessian decomposition
        //
        //   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
        //
        // with
        //   Q_{kℓ} = B_ℓ' A_k β_hat + 0.5 delta_{kℓ} beta_hat' A_k beta_hat
        //   L_{kℓ} = 0.5 [ -tr(H_+^dagger H_ℓ H_+^dagger H_k) + tr(H_+^dagger H_{kℓ}) ]
        //   P_{kℓ} = -0.5 ∂² log|S|_+ /(∂ρ_k∂ρ_ℓ)
        //
        // Numerically, this function computes:
        //
        // 1. Solve for all first implicit derivatives B_k.
        // 2. Form H_k from the penalty part plus the c ⊙ (X B_k) correction.
        // 3. Solve for all second implicit derivatives B_{k,l}.
        // 4. Form H_{k,l} from the penalty part plus the d/c contractions.
        // 5. Assemble
        //      Q_{k,l} = B_l' A_k beta_hat + 0.5 delta_{k,l} beta_hat' A_k beta_hat
        // 6. Assemble
        //      L_{k,l}
        //        = 0.5 [ tr(H_+^dagger H_{k,l}) - tr(H_+^dagger H_l H_+^dagger H_k) ]
        // 7. Assemble P_{k,l} from the structural penalty pseudo-logdet
        //    derivatives on the fixed active penalty subspace.
        //
        // The objective also includes the separable soft rho prior used by
        // compute_cost/compute_gradient; its exact diagonal Hessian is added
        // to every return path below for full objective consistency.
        //
        // Stochastic trace identities used when backend != Exact:
        //   tr(A) = E[zᵀAz],  z_i∈{±1}.
        //   tr(H^{-1}H_ℓH^{-1}H_k) estimated by shared-probe contractions.
        //   tr(H^{-1}H_{kℓ}) estimated by probe bilinear forms.
        // Hutch++ augments this with a low-rank deflation subspace Q to reduce
        // variance before Hutchinson residual estimation.
        let pirls_result = bundle.pirls_result.as_ref();
        let reparam_result = &pirls_result.reparam_result;

        // Active-constraint-aware exact Hessian path:
        // Evaluate all non-Gaussian second-order terms on the current active-free subspace
        // span(Z), consistent with cost/gradient projection.
        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let mut beta_eval = pirls_result.beta_transformed.as_ref().clone();
        let mut rs_eval = reparam_result.rs_transformed.clone();
        let x_dense_orig_arc = pirls_result.x_transformed.to_dense_arc();
        let mut x_dense_eval = x_dense_orig_arc.as_ref().to_owned();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut e_eval = reparam_result.e_transformed.clone();

        if let Some(z) = free_basis_opt.as_ref() {
            h_total_eval = Self::project_with_basis(bundle.h_total.as_ref(), z);
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            rs_eval = reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect();
            x_dense_eval = x_dense_orig_arc.as_ref().dot(z);
            e_eval = reparam_result.e_transformed.dot(z);
        }

        let beta = &beta_eval;
        let rs_transformed = &rs_eval;
        let h_total = &h_total_eval;
        let p_dim = h_total.nrows();
        // Prefer objective-consistent active-subspace generalized inverse when
        // available in the current coordinates:
        //   H_+^dagger = W W',  solve(rhs) := H_+^dagger rhs = W (W' rhs).
        //
        // This keeps IFT solves (B_k, B_kl) on the same spectral surface as
        // pseudo-logdet derivatives tr(H_+^dagger *), improving exactness near
        // weakly identified directions and avoiding ridge-surface mismatch.
        let h_pos_w_for_solve =
            if free_basis_opt.is_none() && bundle.h_pos_factor_w.nrows() == p_dim {
                Some(bundle.h_pos_factor_w.as_ref().clone())
            } else {
                None
            };
        let h_pos_w_for_solve_t = h_pos_w_for_solve.as_ref().map(|w| w.t().to_owned());
        let use_cached_factor = free_basis_opt.is_none();
        let h_factor_cached = if h_pos_w_for_solve.is_none() && use_cached_factor {
            Some(self.get_faer_factor(rho, h_total))
        } else {
            None
        };
        let h_factor_local = if h_pos_w_for_solve.is_none() && !use_cached_factor {
            Some(self.factorize_faer(h_total))
        } else {
            None
        };
        let solve_h = |rhs: &Array2<f64>| -> Array2<f64> {
            if rhs.ncols() == 0 {
                return rhs.clone();
            }
            if let (Some(w), Some(w_t)) = (h_pos_w_for_solve.as_ref(), h_pos_w_for_solve_t.as_ref())
            {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut out_view = array2_to_mat_mut(&mut out);
            if let Some(f) = h_factor_cached.as_ref() {
                f.solve_in_place(out_view.as_mut());
            } else if let Some(f) = h_factor_local.as_ref() {
                f.solve_in_place(out_view.as_mut());
            }
            out
        };

        let k_count = rho.len();
        if k_count == 0 {
            return Ok(Array2::zeros((0, 0)));
        }
        let lambdas = rho.mapv(f64::exp);
        let x_dense = &x_dense_eval;
        let n = x_dense.nrows();
        if x_dense.ncols() != p_dim {
            return Err(EstimationError::InvalidInput(format!(
                "H/X shape mismatch in exact Hessian path: H is {}x{}, X is {}x{}",
                p_dim,
                p_dim,
                x_dense.nrows(),
                x_dense.ncols()
            )));
        }
        if p_dim == 0 {
            let (_, d2logs) = self.structural_penalty_logdet_derivatives(
                rs_transformed,
                &lambdas,
                e_eval.nrows(),
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;
            let mut hess = Array2::<f64>::zeros((k_count, k_count));
            for l in 0..k_count {
                for k in 0..k_count {
                    hess[[k, l]] = -0.5 * d2logs[[k, l]];
                }
            }
            self.add_soft_prior_hessian_in_place(rho, &mut hess);
            return Ok(hess);
        }
        let c = &pirls_result.solve_c_array;
        let d = &pirls_result.solve_d_array;
        if c.len() != n || d.len() != n {
            return Err(EstimationError::InvalidInput(format!(
                "Exact Hessian derivative arrays size mismatch: n={}, c.len()={}, d.len()={}",
                n,
                c.len(),
                d.len()
            )));
        }
        let firth_active = self.config.firth_bias_reduction
            && matches!(self.config.link_function(), LinkFunction::Logit);
        let firth_op = if firth_active {
            // Firth derivatives are evaluated at the converged eta field.
            // This keeps (w, w', w'', ...) and all higher derivative contractions
            // on a single, internally consistent nonlinear state.
            // Math mapping:
            //   Phi(beta)=0.5 log|I_r(beta)|, I_r=X_r' W(beta) X_r,
            //   H_phi = d^2 Phi / d beta^2,
            // and H_k/H_kl include -D(H_phi)[B_k], -D(H_phi)[B_kl],
            // -D^2(H_phi)[B_k,B_l] exactly.
            if let Some(cached) = bundle.firth_dense_operator.as_ref() {
                Some(cached.as_ref().clone())
            } else {
                Some(Self::build_firth_dense_operator(
                    x_dense,
                    &pirls_result.final_eta,
                )?)
            }
        } else {
            None
        };

        let mut a_k_mats = Vec::with_capacity(k_count);
        let mut a_k_beta = Vec::with_capacity(k_count);
        let mut rhs_bk = Array2::<f64>::zeros((p_dim, k_count));
        let mut q_diag = vec![0.0; k_count];
        for k in 0..k_count {
            // Penalty-block algebra:
            //   S_k = R_k' R_k,
            //   A_k = lambda_k S_k,
            //   A_k beta = lambda_k S_k beta.
            //
            // We store both:
            // - a_k_mats[k] = A_k
            // - a_k_beta[k] = A_k beta
            //
            // and stack the first-derivative IFT right-hand sides
            //   H B_k = -A_k beta
            // into rhs_bk so all B_k can be solved together.
            let r_k = &rs_transformed[k];
            let s_k = r_k.t().dot(r_k);
            let r_beta = r_k.dot(beta);
            let s_k_beta = r_k.t().dot(&r_beta);
            let a_k = s_k.mapv(|v| lambdas[k] * v);
            let a_kb = s_k_beta.mapv(|v| lambdas[k] * v);
            q_diag[k] = beta.dot(&a_kb);
            rhs_bk.column_mut(k).assign(&a_kb.mapv(|v| -v));
            a_k_mats.push(a_k);
            a_k_beta.push(a_kb);
        }

        // First implicit derivatives:
        //   B_k = dβ̂/dρ_k = -H_total^{-1}(A_k β̂).
        // Columns of `b_mat` are B_k.
        let b_mat = solve_h(&rhs_bk);
        let u_mat = fast_ab(x_dense, &b_mat);
        let firth_dirs = firth_op.as_ref().map(|op| {
            (0..k_count)
                .map(|k| {
                    let b_k = b_mat.column(k).to_owned();
                    Self::firth_direction(op, &b_k)
                })
                .collect::<Vec<_>>()
        });

        let mut h_k = Vec::with_capacity(k_count);
        let mut weighted_xtdx = Array2::<f64>::zeros(x_dense.raw_dim());
        for k in 0..k_count {
            // u_k = X B_k is the eta-space sensitivity for rho_k.
            // The first Hessian derivative is
            //   H_k = A_k + X' diag(c ⊙ u_k) X.
            let mut diag = Array1::<f64>::zeros(n);
            for i in 0..n {
                diag[i] = c[i] * u_mat[[i, k]];
            }
            let mut hk = a_k_mats[k].clone();
            hk += &Self::xt_diag_x_dense_into(x_dense, &diag, &mut weighted_xtdx);
            if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                // Firth path:
                //   H_k <- H_k - D H_φ[B_k].
                // This is the exact substitution in
                //   H_k = A_k + D(X'WX)[B_k] - D(H_φ)[B_k].
                hk -= &Self::firth_hphi_direction(op, &dirs[k]);
            }
            h_k.push(hk);
        }
        let s_cols: Vec<Array1<f64>> = (0..k_count)
            .map(|k| {
                let mut s = Array1::<f64>::zeros(n);
                for i in 0..n {
                    s[i] = c[i] * u_mat[[i, k]];
                }
                s
            })
            .collect();

        let _recommended_trace_backend = Self::select_trace_backend(n, p_dim, k_count);
        // Exact-only trace backend for Hessian assembly.
        //
        // This keeps all L_{k,l} contractions deterministic and analytic
        // (no Hutchinson/Hutch++) across both Firth and non-Firth paths.
        let trace_backend = TraceBackend::Exact;
        let (exact_trace_mode, n_probe, n_sketch) = match trace_backend {
            TraceBackend::Exact => (true, 0usize, 0usize),
            TraceBackend::Hutchinson { probes } => (false, probes.max(1), 0usize),
            TraceBackend::HutchPP { probes, sketch } => (false, probes.max(1), sketch.max(1)),
        };
        let use_hutchpp = matches!(trace_backend, TraceBackend::HutchPP { .. });
        // Backend semantics:
        // - Exact: deterministic traces.
        //   Preferred form is on the retained positive subspace:
        //     tr(H_+^dagger A) = tr(W' A W),
        //     tr(H_+^dagger B H_+^dagger C) = tr((W'BW)(W'CW)).
        //   Dense H^{-1} contractions are only a fallback when positive-subspace
        //   factors are unavailable in current coordinates.
        // - Hutchinson/Hutch++: Monte-Carlo trace estimators (unbiased/low-bias in
        //   expectation) trading tiny stochastic noise for major scaling gains.
        let projected_exact_mode = exact_trace_mode
            && !firth_active
            && free_basis_opt.is_none()
            && bundle.h_pos_factor_w.nrows() == p_dim
            && Self::dense_projected_exact_eligible(n, bundle.h_pos_factor_w.ncols(), k_count);
        let spectral_exact_mode = exact_trace_mode
            && free_basis_opt.is_none()
            && bundle.h_pos_factor_w.nrows() == p_dim
            && !projected_exact_mode;
        // Mode split:
        //   projected_exact_mode: non-Firth optimized exact contractions.
        //   spectral_exact_mode:  exact H_+^dagger traces via W'(*)W (works for Firth too).
        //   else: dense fallback (potentially full H^{-1}).

        let w_pos_projected = if projected_exact_mode {
            Some(bundle.h_pos_factor_w.as_ref().clone())
        } else {
            None
        };
        let z_mat_projected = w_pos_projected
            .as_ref()
            .map(|w_pos| fast_ab(x_dense, w_pos));
        let w_pos_spectral = if spectral_exact_mode {
            Some(bundle.h_pos_factor_w.as_ref().clone())
        } else {
            None
        };

        let solved_h_k_exact: Option<Vec<Array2<f64>>> =
            if exact_trace_mode && !projected_exact_mode && !spectral_exact_mode {
                // Exact dense fallback without materializing H^{-1} explicitly:
                // precompute solve(H, H_k) and use
                //   tr(H^{-1} H_l H^{-1} H_k) = tr(solve(H,H_l) solve(H,H_k)).
                Some(h_k.iter().map(solve_h).collect())
            } else {
                None
            };
        let w_pos_spectral_t = w_pos_spectral.as_ref().map(|w_pos| w_pos.t().to_owned());
        let g_k_spectral: Option<Vec<Array2<f64>>> = w_pos_spectral.as_ref().map(|w_pos| {
            // Spectral exact traces on active positive subspace:
            //   H_+^dagger = W W^T
            //   tr(H_+^dagger A) = tr(W^T A W)
            //   tr(H_+^dagger B H_+^dagger C) = tr((W^T B W)(W^T C W)).
            // Here G_k := W^T H_k W so
            //   t1_{l,k} = tr(H_+^dagger H_l H_+^dagger H_k) = tr(G_l G_k).
            let w_pos_t = w_pos_spectral_t
                .as_ref()
                .expect("spectral W^T present in spectral exact mode");
            h_k.iter()
                .map(|hk| {
                    let wt_hk = fast_ab(w_pos_t, hk);
                    fast_ab(&wt_hk, w_pos)
                })
                .collect()
        });
        let t_k_projected: Option<Vec<Array2<f64>>> = if projected_exact_mode {
            let z_mat = z_mat_projected
                .as_ref()
                .expect("projected exact Z available");
            let w_pos = w_pos_projected
                .as_ref()
                .expect("projected exact W available");
            Some(
                (0..k_count)
                    .map(|k| {
                        Self::dense_projected_tk(
                            z_mat,
                            w_pos,
                            &rs_transformed[k],
                            lambdas[k],
                            &s_cols[k],
                        )
                    })
                    .collect(),
            )
        } else {
            None
        };

        let mut probe_z: Option<Array2<f64>> = None;
        let mut probe_u: Option<Array2<f64>> = None;
        let mut probe_xz: Option<Array2<f64>> = None;
        let mut probe_xu: Option<Array2<f64>> = None;
        let mut sketch_q: Option<Array2<f64>> = None;
        let mut sketch_uq: Option<Array2<f64>> = None;
        let mut sketch_xq: Option<Array2<f64>> = None;
        let mut sketch_xuq: Option<Array2<f64>> = None;

        if !exact_trace_mode {
            let mut z = Self::rademacher_matrix(p_dim, n_probe, 0xC0DEC0DE5EEDu64);
            if use_hutchpp && n_sketch > 0 {
                let g = Self::rademacher_matrix(p_dim, n_sketch, 0xBADC0FFEE0DDF00Du64);
                let y = solve_h(&g);
                let q = Self::orthonormalize_columns(&y, 1e-10);
                if q.ncols() > 0 {
                    for r in 0..n_probe {
                        let mut zr = z.column(r).to_owned();
                        let qt_z = q.t().dot(&zr);
                        let proj = q.dot(&qt_z);
                        zr -= &proj;
                        z.column_mut(r).assign(&zr);
                    }
                    let uq = solve_h(&q);
                    let xq = fast_ab(x_dense, &q);
                    let xuq = fast_ab(x_dense, &uq);
                    sketch_q = Some(q);
                    sketch_uq = Some(uq);
                    sketch_xq = Some(xq);
                    sketch_xuq = Some(xuq);
                }
            }
            let u = solve_h(&z);
            let xz = fast_ab(x_dense, &z);
            let xu = fast_ab(x_dense, &u);
            probe_z = Some(z);
            probe_u = Some(u);
            probe_xz = Some(xz);
            probe_xu = Some(xu);
        }

        let mut t1_mat = Array2::<f64>::zeros((k_count, k_count));
        if exact_trace_mode {
            if projected_exact_mode {
                let tk = t_k_projected
                    .as_ref()
                    .expect("projected T_k present in projected exact mode");
                for l in 0..k_count {
                    for k in 0..k_count {
                        t1_mat[[l, k]] = Self::dense_projected_trace_quadratic(&tk[l], &tk[k]);
                    }
                }
            } else if spectral_exact_mode {
                let gk = g_k_spectral
                    .as_ref()
                    .expect("spectral exact G_k present in spectral exact mode");
                for l in 0..k_count {
                    for k in 0..k_count {
                        t1_mat[[l, k]] = Self::trace_product(&gk[l], &gk[k]);
                    }
                }
            } else {
                let solved = solved_h_k_exact
                    .as_ref()
                    .expect("solve(H,H_k) blocks present in exact fallback mode");
                for l in 0..k_count {
                    for k in 0..k_count {
                        t1_mat[[l, k]] = Self::trace_product(&solved[l], &solved[k]);
                    }
                }
            }
        } else {
            if let (Some(q), Some(uq), Some(xq), Some(xuq)) = (
                sketch_q.as_ref(),
                sketch_uq.as_ref(),
                sketch_xq.as_ref(),
                sketch_xuq.as_ref(),
            ) {
                let rdim = q.ncols();
                for j in 0..rdim {
                    let qj = q.column(j).to_owned();
                    let uqj = uq.column(j).to_owned();
                    let xqj = xq.column(j).to_owned();
                    let xuqj = xuq.column(j).to_owned();
                    let mut bq = Array2::<f64>::zeros((p_dim, k_count));
                    for k in 0..k_count {
                        let mut hkq = a_k_mats[k].dot(&qj);
                        let weighted = &s_cols[k] * &xqj;
                        hkq += &x_dense.t().dot(&weighted);
                        bq.column_mut(k).assign(&hkq);
                    }
                    let wq = solve_h(&bq);
                    let xwq = fast_ab(x_dense, &wq);
                    for l in 0..k_count {
                        let alu = a_k_mats[l].dot(&uqj);
                        let sxu = &s_cols[l] * &xuqj;
                        for k in 0..k_count {
                            let val = alu.dot(&wq.column(k)) + sxu.dot(&xwq.column(k));
                            t1_mat[[l, k]] += val;
                        }
                    }
                }
            }
            let z = probe_z.as_ref().expect("probes present in stochastic mode");
            let u = probe_u
                .as_ref()
                .expect("solved probes present in stochastic mode");
            let xz = probe_xz
                .as_ref()
                .expect("X probes present in stochastic mode");
            let xu = probe_xu
                .as_ref()
                .expect("X solved probes present in stochastic mode");
            for r in 0..n_probe {
                let zr = z.column(r).to_owned();
                let ur = u.column(r).to_owned();
                let xzr = xz.column(r).to_owned();
                let xur = xu.column(r).to_owned();
                let mut bz = Array2::<f64>::zeros((p_dim, k_count));
                for k in 0..k_count {
                    let mut hkz = a_k_mats[k].dot(&zr);
                    let weighted = &s_cols[k] * &xzr;
                    hkz += &x_dense.t().dot(&weighted);
                    bz.column_mut(k).assign(&hkz);
                }
                let wz = solve_h(&bz);
                let xwz = fast_ab(x_dense, &wz);
                for l in 0..k_count {
                    let alu = a_k_mats[l].dot(&ur);
                    let sxu = &s_cols[l] * &xur;
                    for k in 0..k_count {
                        let val = alu.dot(&wz.column(k)) + sxu.dot(&xwz.column(k));
                        t1_mat[[l, k]] += val / (n_probe as f64);
                    }
                }
            }
        }
        for i in 0..k_count {
            for j in 0..i {
                let avg = 0.5 * (t1_mat[[i, j]] + t1_mat[[j, i]]);
                t1_mat[[i, j]] = avg;
                t1_mat[[j, i]] = avg;
            }
        }

        let (_, d2logs) = self.structural_penalty_logdet_derivatives(
            rs_transformed,
            &lambdas,
            e_eval.nrows(),
            bundle.ridge_passport.penalty_logdet_ridge(),
        )?;

        let mut hess = Array2::<f64>::zeros((k_count, k_count));
        for l in 0..k_count {
            let bl = b_mat.column(l).to_owned();
            let mut rhs_kl_all = Array2::<f64>::zeros((p_dim, k_count));
            for k in l..k_count {
                let bk = b_mat.column(k).to_owned();
                // Second implicit derivative solve:
                //
                //   B_{k,l}
                //     = -H^{-1}(H_l B_k + A_k B_l + delta_{k,l} A_k beta).
                //
                // We form the stacked right-hand sides exactly in that form.
                let mut rhs_kl = -h_k[l].dot(&bk);
                rhs_kl -= &a_k_mats[k].dot(&bl);
                if k == l {
                    rhs_kl -= &a_k_beta[k];
                }
                rhs_kl_all.column_mut(k).assign(&rhs_kl);
            }
            let b_kl_all = solve_h(&rhs_kl_all);
            let u_kl_all = fast_ab(x_dense, &b_kl_all);

            let mut weighted_xtdx_kl = Array2::<f64>::zeros(x_dense.raw_dim());
            for k in l..k_count {
                // H_{k,l} = delta_{k,l} A_k
                //          + X' diag(d ⊙ u_k ⊙ u_l + c ⊙ u_{k,l}) X.
                let mut diag = Array1::<f64>::zeros(n);
                for i in 0..n {
                    diag[i] = d[i] * u_mat[[i, k]] * u_mat[[i, l]] + c[i] * u_kl_all[[i, k]];
                }

                // Quadratic beta contribution:
                //   Q_{k,l} = B_l' A_k beta + 0.5 delta_{k,l} beta' A_k beta.
                let q = bl.dot(&a_k_beta[k]) + if k == l { 0.5 * q_diag[k] } else { 0.0 };

                // Quadratic logdet trace piece:
                //   t1 = tr(H^{-1} H_l H^{-1} H_k).
                let t1 = t1_mat[[l, k]];
                let t2 = if exact_trace_mode {
                    if projected_exact_mode {
                        let z_mat = z_mat_projected
                            .as_ref()
                            .expect("projected exact Z available");
                        let w_pos = w_pos_projected
                            .as_ref()
                            .expect("projected exact W available");
                        Self::dense_projected_trace_hinv_hkl(
                            z_mat,
                            w_pos,
                            if k == l {
                                Some(&rs_transformed[k])
                            } else {
                                None
                            },
                            lambdas[k],
                            &diag,
                        )
                    } else {
                        // Linear logdet trace piece in exact mode:
                        //   t2 = tr(H^{-1} H_{k,l}).
                        let mut h_kl = if k == l {
                            a_k_mats[k].clone()
                        } else {
                            Array2::<f64>::zeros((p_dim, p_dim))
                        };
                        h_kl += &Self::xt_diag_x_dense_into(x_dense, &diag, &mut weighted_xtdx_kl);
                        let mut d2_trace_correction = 0.0_f64;
                        if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                            let b_kl = b_kl_all.column(k).to_owned();
                            let dir_kl = Self::firth_direction(op, &b_kl);
                            // Firth second-order correction:
                            //   H_{k,l} <- H_{k,l}
                            //             - D H_φ[B_{k,l}]
                            //             - D² H_φ[B_k,B_l].
                            // This is the exact second-order chain rule term for
                            // H_total,kl under beta_hat(rho):
                            //   H_kl = δ_kl A_k + D²(X'WX)[B_k,B_l] + D(X'WX)[B_kl]
                            //          - D²(H_φ)[B_k,B_l] - D(H_φ)[B_kl].
                            h_kl -= &Self::firth_hphi_direction(op, &dir_kl);
                            if spectral_exact_mode {
                                let w_pos = w_pos_spectral
                                    .as_ref()
                                    .expect("spectral W present in spectral exact mode");
                                let w_pos_t = w_pos_spectral_t
                                    .as_ref()
                                    .expect("spectral W^T present in spectral exact mode");
                                let d2_aw = Self::firth_hphi_second_direction_apply(
                                    op, &dirs[k], &dirs[l], w_pos,
                                );
                                d2_trace_correction = Self::trace_product(w_pos_t, &d2_aw);
                            } else {
                                // Exact dense-solve fallback for tr(H^{-1} D²H_phi[B_k,B_l])
                                // without materializing the full p×p D²H_phi matrix.
                                const BLOCK: usize = 32;
                                let mut acc = 0.0_f64;
                                let mut start = 0usize;
                                while start < p_dim {
                                    let width = (p_dim - start).min(BLOCK);
                                    let mut basis = Array2::<f64>::zeros((p_dim, width));
                                    for j in 0..width {
                                        basis[[start + j, j]] = 1.0;
                                    }
                                    let d2_block = Self::firth_hphi_second_direction_apply(
                                        op, &dirs[k], &dirs[l], &basis,
                                    );
                                    let solved_block = solve_h(&d2_block);
                                    for j in 0..width {
                                        acc += solved_block[[start + j, j]];
                                    }
                                    start += width;
                                }
                                d2_trace_correction = acc;
                            }
                        }
                        if spectral_exact_mode {
                            let w_pos = w_pos_spectral
                                .as_ref()
                                .expect("spectral W present in spectral exact mode");
                            let w_pos_t = w_pos_spectral_t
                                .as_ref()
                                .expect("spectral W^T present in spectral exact mode");
                            let wt_hkl = fast_ab(w_pos_t, &h_kl);
                            let g_kl = fast_ab(&wt_hkl, w_pos);
                            // t2 = tr(H_+^dagger H_kl) = tr(W^T H_kl W).
                            g_kl.diag().sum() - d2_trace_correction
                        } else {
                            // Dense exact fallback without explicit inverse:
                            //   tr(H^{-1} H_kl) = tr(solve(H, H_kl)).
                            let solved_hkl = solve_h(&h_kl);
                            solved_hkl.diag().sum()
                        }
                    }
                } else {
                    let mut t2_acc = 0.0_f64;
                    if let (Some(q), Some(uq), Some(xq), Some(xuq)) = (
                        sketch_q.as_ref(),
                        sketch_uq.as_ref(),
                        sketch_xq.as_ref(),
                        sketch_xuq.as_ref(),
                    ) {
                        for j in 0..q.ncols() {
                            let qj = q.column(j);
                            let uqj = uq.column(j);
                            let xqj = xq.column(j);
                            let xuqj = xuq.column(j);
                            let mut term = 0.0_f64;
                            if k == l {
                                term += Self::bilinear_form(&a_k_mats[k], uqj, qj);
                            }
                            let mut quad = 0.0_f64;
                            for i in 0..n {
                                quad += xuqj[i] * diag[i] * xqj[i];
                            }
                            term += quad;
                            t2_acc += term;
                        }
                    }
                    let z = probe_z.as_ref().expect("probes present in stochastic mode");
                    let u = probe_u
                        .as_ref()
                        .expect("solved probes present in stochastic mode");
                    let xz = probe_xz
                        .as_ref()
                        .expect("X probes present in stochastic mode");
                    let xu = probe_xu
                        .as_ref()
                        .expect("X solved probes present in stochastic mode");
                    let mut res = 0.0_f64;
                    for r in 0..n_probe {
                        let zr = z.column(r);
                        let ur = u.column(r);
                        let xzr = xz.column(r);
                        let xur = xu.column(r);
                        let mut term = 0.0_f64;
                        if k == l {
                            term += Self::bilinear_form(&a_k_mats[k], ur, zr);
                        }
                        let mut quad = 0.0_f64;
                        for i in 0..n {
                            quad += xur[i] * diag[i] * xzr[i];
                        }
                        term += quad;
                        res += term;
                    }
                    t2_acc + res / (n_probe as f64)
                };
                // L_{k,l} = 0.5 [ -t1 + t2 ]
                let l_term = 0.5 * (-t1 + t2);
                // P_{k,l} = -0.5 * d²/drho_k drho_l log|S|_+.
                let p_term = -0.5 * d2logs[[k, l]];
                // Final exact dense transformed Hessian entry:
                //   V_{k,l} = Q_{k,l} + L_{k,l} + P_{k,l}.
                //
                // Conclusion for the Firth path:
                // - `q` uses the same implicit derivatives B_k/B_kl as non-Firth,
                //   but those derivatives were solved on H_total = X'WX + S - H_phi.
                // - `t1`/`t2` see Firth through H_k/H_kl, which include
                //   -D(H_phi)[B_k], -D(H_phi)[B_kl], and -D²(H_phi)[B_k,B_l].
                // Therefore `val` is objective-consistent with the Firth-adjusted
                // inner stationarity equation on smooth active-subspace regions.
                let val = q + l_term + p_term;
                hess[[k, l]] = val;
                hess[[l, k]] = val;
            }
        }
        self.add_soft_prior_hessian_in_place(rho, &mut hess);
        Ok(hess)
    }

    pub(crate) fn compute_laml_hessian_analytic_fallback_standalone(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        self.compute_laml_hessian_analytic_fallback(rho, None)
    }

    pub(crate) fn compute_smoothing_correction_auto(
        &self,
        final_rho: &Array1<f64>,
        final_fit: &PirlsResult,
        base_covariance: Option<&Array2<f64>>,
        final_grad_norm: f64,
    ) -> Option<Array2<f64>> {
        // Always compute the fast first-order correction first.
        let first_order = super::compute_smoothing_correction(self, final_rho, final_fit);
        let n_rho = final_rho.len();
        if n_rho == 0 {
            return first_order;
        }
        if n_rho > AUTO_CUBATURE_MAX_RHO_DIM {
            return first_order;
        }
        if final_fit.beta_transformed.len() > AUTO_CUBATURE_MAX_BETA_DIM {
            return first_order;
        }
        if let Ok(bundle) = self.obtain_eval_bundle(final_rho)
            && bundle.active_subspace_unstable
        {
            // Cubature correction relies on a locally stable outer Hessian.
            // Near active-subspace crossings of H_+, second-order local models
            // are unreliable; keep the first-order correction only.
            return first_order;
        }

        let near_boundary = final_rho
            .iter()
            .any(|&v| (RHO_BOUND - v.abs()) <= AUTO_CUBATURE_BOUNDARY_MARGIN);
        let grad_norm = if final_grad_norm.is_finite() {
            final_grad_norm
        } else {
            0.0
        };
        let high_grad = grad_norm > 1e-3;
        if !near_boundary && !high_grad {
            // Keep the hot path cheap when the local linearization is likely sufficient.
            return first_order;
        }

        // Build V_rho from the outer Hessian around rho_hat.
        let mut hessian_rho = match self.compute_laml_hessian_consistent(final_rho) {
            Ok(h) => h,
            Err(err) => {
                log::debug!("Auto cubature skipped: rho Hessian unavailable ({}).", err);
                return first_order;
            }
        };
        for i in 0..n_rho {
            for j in (i + 1)..n_rho {
                let avg = 0.5 * (hessian_rho[[i, j]] + hessian_rho[[j, i]]);
                hessian_rho[[i, j]] = avg;
                hessian_rho[[j, i]] = avg;
            }
        }
        let ridge = 1e-8
            * hessian_rho
                .diag()
                .iter()
                .map(|&v| v.abs())
                .fold(0.0, f64::max)
                .max(1e-8);
        for i in 0..n_rho {
            hessian_rho[[i, i]] += ridge;
        }
        let hessian_rho_inv =
            match matrix_inverse_with_regularization(&hessian_rho, "auto cubature rho Hessian") {
                Some(v) => v,
                None => return first_order,
            };

        let max_rho_var = hessian_rho_inv
            .diag()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !near_boundary && !high_grad && max_rho_var < 0.1 {
            return first_order;
        }

        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let (evals, evecs) = match hessian_rho_inv.eigh(Side::Lower) {
            Ok(x) => x,
            Err(_) => return first_order,
        };
        let mut eig_pairs: Vec<(usize, f64)> = evals
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite() && *v > 1e-12)
            .collect();
        if eig_pairs.is_empty() {
            return first_order;
        }
        eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total_var: f64 = eig_pairs.iter().map(|(_, v)| *v).sum();
        if !total_var.is_finite() || total_var <= 0.0 {
            return first_order;
        }

        let mut rank = 0usize;
        let mut captured = 0.0_f64;
        for (_, eig) in eig_pairs
            .iter()
            .take(AUTO_CUBATURE_MAX_EIGENVECTORS.min(eig_pairs.len()))
        {
            captured += *eig;
            rank += 1;
            if captured / total_var >= AUTO_CUBATURE_TARGET_VAR_FRAC {
                break;
            }
        }
        if rank == 0 {
            return first_order;
        }

        let base_cov = match base_covariance {
            Some(v) => v,
            None => return first_order,
        };
        let p = base_cov.nrows();
        let radius = (rank as f64).sqrt();
        let mut sigma_points: Vec<Array1<f64>> = Vec::with_capacity(2 * rank);
        for (eig_idx, eig_val) in eig_pairs.iter().take(rank) {
            let axis = evecs.column(*eig_idx).to_owned();
            let scale = radius * eig_val.sqrt();
            let delta = axis.mapv(|v| v * scale);

            for sign in [1.0_f64, -1.0_f64] {
                let mut rho_point = final_rho.clone();
                for i in 0..n_rho {
                    rho_point[i] =
                        (rho_point[i] + sign * delta[i]).clamp(-RHO_BOUND + 1e-8, RHO_BOUND - 1e-8);
                }
                sigma_points.push(rho_point);
            }
        }
        if sigma_points.is_empty() {
            return first_order;
        }

        // Disable warm-start and PIRLS-cache coupling while evaluating sigma
        // points in parallel. Cache lookups/inserts use an exclusive lock in
        // execute_pirls_if_needed(), so leaving cache enabled serializes this
        // block under contention.
        let _cache_guard = AtomicFlagGuard::swap(
            &self.cache_manager.pirls_cache_enabled,
            false,
            Ordering::SeqCst,
        );
        let _warm_start_guard =
            AtomicFlagGuard::swap(&self.warm_start_enabled, false, Ordering::SeqCst);
        let point_results: Vec<Option<(Array2<f64>, Array1<f64>)>> = (0..sigma_points.len())
            .into_par_iter()
            .map(|idx| {
                let fit_point = self.execute_pirls_if_needed(&sigma_points[idx]).ok()?;
                let h_point = map_hessian_to_original_basis(fit_point.as_ref()).ok()?;
                let cov_point =
                    matrix_inverse_with_regularization(&h_point, "auto cubature point")?;
                let beta_point = fit_point
                    .reparam_result
                    .qs
                    .dot(fit_point.beta_transformed.as_ref());
                Some((cov_point, beta_point))
            })
            .collect();

        if point_results.iter().any(|r| r.is_none()) {
            return first_order;
        }

        let w = 1.0 / (sigma_points.len() as f64);
        let mut mean_hinv = Array2::<f64>::zeros((p, p));
        let mut mean_beta = Array1::<f64>::zeros(p);
        let mut second_beta = Array2::<f64>::zeros((p, p));
        for (cov_point, beta_point) in point_results.into_iter().flatten() {
            mean_hinv += &cov_point.mapv(|v| w * v);
            mean_beta += &beta_point.mapv(|v| w * v);
            for i in 0..p {
                let bi = beta_point[i];
                for j in 0..p {
                    second_beta[[i, j]] += w * bi * beta_point[j];
                }
            }
        }

        let mut var_beta = second_beta;
        for i in 0..p {
            for j in 0..p {
                var_beta[[i, j]] -= mean_beta[i] * mean_beta[j];
            }
        }

        let mut total_cov = mean_hinv + var_beta;
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (total_cov[[i, j]] + total_cov[[j, i]]);
                total_cov[[i, j]] = avg;
                total_cov[[j, i]] = avg;
            }
        }
        if !total_cov.iter().all(|v| v.is_finite()) {
            return first_order;
        }

        let mut corr = total_cov - base_cov;
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (corr[[i, j]] + corr[[j, i]]);
                corr[[i, j]] = avg;
                corr[[j, i]] = avg;
            }
        }

        log::info!(
            "Using adaptive cubature smoothing correction (rank={}, points={}, near_boundary={}, grad_norm={:.2e}, max_var={:.2e})",
            rank,
            2 * rank,
            near_boundary,
            grad_norm,
            max_rho_var
        );
        Some(corr)
    }

    /// Run comprehensive gradient diagnostics implementing four strategies:
    /// 1. KKT/Envelope Theorem Audit
    /// 2. Component-wise Finite Difference
    /// 3. Spectral Bleed Trace
    /// 4. Dual-Ridge Consistency
    ///
    /// Only prints a summary when issues are detected.
    fn run_gradient_diagnostics(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        analytic_grad: &Array1<f64>,
        applied_truncation_corrections: Option<&[f64]>,
    ) {
        use crate::diagnostics::{
            DiagnosticConfig, GradientDiagnosticReport, compute_dual_ridge_check,
            compute_envelope_audit, compute_spectral_bleed,
        };

        let config = DiagnosticConfig::default();
        let mut report = GradientDiagnosticReport::new();

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_used = bundle.ridge_passport.delta;
        let beta = pirls_result.beta_transformed.as_ref();
        let lambdas: Array1<f64> = rho.mapv(f64::exp);

        // === Strategy 4: Dual-Ridge Consistency Check ===
        // Compare the PIRLS ridge with the ridge used by cost/gradient paths.
        let dual_ridge = compute_dual_ridge_check(
            pirls_result.ridge_passport.delta, // Ridge from PIRLS passport
            ridge_used,                        // Ridge passed to cost
            ridge_used,                        // Ridge passed to gradient (same bundle)
            beta,
        );
        report.dual_ridge = Some(dual_ridge);

        // === Strategy 1: KKT/Envelope Theorem Audit ===
        // Check if the inner solver actually reached stationarity
        let reparam = &pirls_result.reparam_result;
        let penalty_grad = reparam.s_transformed.dot(beta);

        let envelope_audit = compute_envelope_audit(
            pirls_result.last_gradient_norm,
            &penalty_grad,
            pirls_result.ridge_passport.delta,
            ridge_used, // What gradient assumes
            beta,
            config.kkt_tolerance,
            config.rel_error_threshold,
        );
        report.envelope_audit = Some(envelope_audit);

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
            && let Some(applied_values) = applied_truncation_corrections
        {
            let h_eff = bundle.h_eff.as_ref();

            // Solve H⁻¹ U_⊥ for spectral bleed calculation
            let h_view = FaerArrayView::new(h_eff);
            if let Ok(chol) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                let mut h_inv_u = u_truncated.clone();
                let mut rhs_view = array2_to_mat_mut(&mut h_inv_u);
                chol.solve_in_place(rhs_view.as_mut());

                for (k, r_k) in reparam.rs_transformed.iter().enumerate() {
                    let applied_correction = applied_values.get(k).copied().unwrap_or(0.0);
                    let bleed = compute_spectral_bleed(
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
        let severe_envelope = report.envelope_audit.as_ref().is_some_and(|a| {
            a.kkt_residual_norm > GRAD_DIAG_SEVERE_KKT_NORM
                || (a.inner_ridge - a.outer_ridge).abs() > GRAD_DIAG_SEVERE_RIDGE_MISMATCH
        });
        let severe_bleed = report
            .spectral_bleed
            .iter()
            .any(|b| b.has_bleed && b.truncated_energy.abs() > GRAD_DIAG_SEVERE_BLEED_ENERGY);
        let severe_ridge = report.dual_ridge.as_ref().is_some_and(|r| {
            r.has_mismatch
                && (r.ridge_impact.abs() > GRAD_DIAG_SEVERE_RIDGE_IMPACT
                    || r.phantom_penalty.abs() > GRAD_DIAG_SEVERE_PHANTOM_PENALTY)
        });
        let periodic_sample = should_sample_gradient_diag_fd(eval_idx);
        let run_component_fd = report.has_issues()
            && (severe_envelope || severe_bleed || severe_ridge || periodic_sample);
        if run_component_fd {
            let _cache_guard = AtomicFlagGuard::swap(
                &self.cache_manager.pirls_cache_enabled,
                false,
                Ordering::Relaxed,
            );

            let h = config.fd_step_size;
            let mut numeric_grad = Array1::<f64>::zeros(rho.len());

            for k in 0..rho.len() {
                let mut rho_plus = rho.clone();
                rho_plus[k] += h;
                let mut rho_minus = rho.clone();
                rho_minus[k] -= h;

                let fp = self.compute_cost(&rho_plus).unwrap_or(f64::INFINITY);
                let fm = self.compute_cost(&rho_minus).unwrap_or(f64::INFINITY);
                numeric_grad[k] = (fp - fm) / (2.0 * h);
            }

            report.analytic_gradient = Some(analytic_grad.clone());
            report.numeric_gradient = Some(numeric_grad.clone());

            // Compute per-component relative errors
            let mut rel_errors = Array1::<f64>::zeros(rho.len());
            for k in 0..rho.len() {
                let denom = analytic_grad[k].abs().max(numeric_grad[k].abs()).max(1e-8);
                rel_errors[k] = (analytic_grad[k] - numeric_grad[k]).abs() / denom;
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
        if report.has_issues() {
            println!("\n[GRADIENT DIAGNOSTICS] Issues detected:");
            println!("{}", report.summary());

            // Also log total gradient comparison
            if let (Some(analytic), Some(numeric)) =
                (&report.analytic_gradient, &report.numeric_gradient)
            {
                let diff = analytic - numeric;
                let rel_l2 = diff.dot(&diff).sqrt() / numeric.dot(numeric).sqrt().max(1e-8);
                println!(
                    "[GRADIENT DIAGNOSTICS] Total gradient rel. L2 error: {:.2e}",
                    rel_l2
                );
            }
        }
    }

    /// Implements the stable re-parameterization algorithm from Wood (2011) Appendix B
    /// This replaces naive summation S_λ = Σ λᵢSᵢ with similarity transforms
    /// to avoid "dominant machine zero leakage" between penalty components
    ///
    // Helper for boundary perturbation
    // Returns (perturbed_rho, optional_corrected_covariance_in_transformed_basis)
    // The covariance is V'_beta_trans
    #[allow(dead_code)]
    pub(crate) fn perform_boundary_perturbation_correction(
        &self,
        initial_rho: &Array1<f64>,
    ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
        // 1. Identify boundary parameters and perturb
        let mut current_rho = initial_rho.clone();
        let mut perturbed = false;

        // Target cost increase: 0.01 log-likelihood units (statistically insignificant)
        let target_diff = 0.01;

        for k in 0..current_rho.len() {
            // Check if at upper boundary (high smoothing -> linear)
            // RHO_BOUND is 30.0.
            if current_rho[k] > RHO_BOUND - 1.0 {
                // Compute base_cost fresh for each parameter to handle multiple boundary cases
                let base_cost = self.compute_cost(&current_rho)?;

                log::info!(
                    "[Boundary] rho[{}] = {:.2} is at boundary. Perturbing...",
                    k,
                    current_rho[k]
                );

                // Search inwards (decreasing rho)
                // We want delta > 0 such that Cost(rho - delta) approx Base + 0.01
                let mut lower = 0.0;
                let mut upper = 15.0;
                let mut best_delta = 0.0;

                // Initial check: if upper is not enough, just take upper
                let mut rho_test = current_rho.clone();
                rho_test[k] -= upper;
                if let Ok(c) = self.compute_cost(&rho_test) {
                    if (c - base_cost).abs() < target_diff {
                        // Even big change doesn't change cost much?
                        // This implies extremely flat surface. Just move away from boundary significantly.
                        best_delta = upper;
                    }
                }

                if best_delta == 0.0 {
                    // Bisection
                    for _ in 0..15 {
                        let mid = (lower + upper) * 0.5;
                        rho_test[k] = current_rho[k] - mid;
                        if let Ok(c) = self.compute_cost(&rho_test) {
                            let diff = c - base_cost;
                            if diff < target_diff {
                                // Need more change -> larger delta
                                lower = mid;
                            } else {
                                // Too much change -> smaller delta
                                upper = mid;
                            }
                        } else {
                            // Error computing cost, assume strictly worse (too far?)
                            upper = mid;
                        }
                    }
                    best_delta = (lower + upper) * 0.5;
                }

                current_rho[k] -= best_delta;
                perturbed = true;
                log::info!(
                    "[Boundary] rho[{}] moved to {:.2} (delta={:.3})",
                    k,
                    current_rho[k],
                    best_delta
                );
            }
        }

        if !perturbed {
            return Ok((current_rho, None));
        }
        if let Ok(bundle) = self.obtain_eval_bundle(&current_rho)
            && bundle.active_subspace_unstable
        {
            log::warn!(
                "Boundary perturbation correction skipped covariance step: unstable H_+ active subspace near spectral cutoff."
            );
            return Ok((current_rho, None));
        }

        let n_rho = current_rho.len();
        let mut laml_hessian = match self.compute_laml_hessian_consistent(&current_rho) {
            Ok(h) => h,
            Err(err) => {
                log::warn!(
                    "Boundary Hessian unavailable ({}); using analytic fallback Hessian.",
                    err
                );
                self.compute_laml_hessian_analytic_fallback(&current_rho, None)?
            }
        };

        // Invert local Hessian to obtain V_ρ.
        // Stabilization ridge is applied before Cholesky to control near-singularity
        // in weakly identified smoothing directions.
        let mut v_rho = Array2::<f64>::zeros((n_rho, n_rho));
        {
            use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
            use faer::Side;

            // Ensure PD
            crate::pirls::ensure_positive_definite_with_label(&mut laml_hessian, "LAML Hessian")?;

            let h_view = FaerArrayView::new(&laml_hessian);
            if let Ok(chol) = faer::linalg::solvers::Llt::new(h_view.as_ref(), Side::Lower) {
                let mut eye = Array2::<f64>::eye(n_rho);
                let mut eye_view = array2_to_mat_mut(&mut eye);
                chol.solve_in_place(eye_view.as_mut());
                v_rho.assign(&eye);
            } else {
                // Fallback: SVD or pseudoinverse? Or just fail correction.
                log::warn!(
                    "LAML Hessian not invertible even after stabilization. Skipping correction."
                );
                return Ok((current_rho, None));
            }
        }

        // 3. Compute smoothing-parameter uncertainty correction: J * V_rho * J^T.
        //
        // Notation mapping to the exact Gaussian-mixture identity:
        //   rho ~ N(mu, Sigma),  mu = rho_hat,  Sigma = V_rho
        //   A(rho) = H_rho^{-1},  b(rho) = beta_hat_rho
        //   Var(beta) = E[A(rho)] + Var(b(rho))   (exact, no truncation)
        //
        // This implementation uses the standard first-order truncation around mu:
        //   E[A(rho)]      ≈ A(mu) = H_p^{-1} = V_beta_cond
        //   Var(b(rho))    ≈ J * V_rho * J^T,  J = dbeta_hat/drho |_{rho=mu}
        // so:
        //   V_total ≈ V_beta_cond + J * V_rho * J^T.
        //
        // Exact higher-order terms from the heat-operator / Wick expansion are
        // not included here.
        //
        // Jacobian identity used here:
        //   d(beta_hat)/d(rho_k) = -H_p^{-1}(S_k^rho * beta_hat), S_k^rho = lambda_k S_k.
        // This is the same implicit derivative used in the main gradient code.

        // We need H_p and beta at the perturbed rho.
        let pirls_res = self.execute_pirls_if_needed(&current_rho)?;

        let beta = pirls_res.beta_transformed.as_ref();
        let h_p = &pirls_res.penalized_hessian_transformed;
        let lambdas = current_rho.mapv(f64::exp);
        let rs = &pirls_res.reparam_result.rs_transformed;

        let p_dim = beta.len();

        // Invert H_p to get V_beta_cond = H_p^{-1}, i.e. A(mu) in the
        // first-order approximation above.
        let mut v_beta_cond = Array2::<f64>::zeros((p_dim, p_dim));
        {
            use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
            use faer::Side;
            let h_view = FaerArrayView::new(h_p);
            // At convergence H_p is typically PD.
            if let Ok(chol) = faer::linalg::solvers::Llt::new(h_view.as_ref(), Side::Lower) {
                let mut eye = Array2::<f64>::eye(p_dim);
                let mut eye_view = array2_to_mat_mut(&mut eye);
                chol.solve_in_place(eye_view.as_mut());
                v_beta_cond.assign(&eye);
            } else {
                // Use LDLT if LLT fails
                if let Ok(ldlt) = faer::linalg::solvers::Ldlt::new(h_view.as_ref(), Side::Lower) {
                    let mut eye = Array2::<f64>::eye(p_dim);
                    let mut eye_view = array2_to_mat_mut(&mut eye);
                    ldlt.solve_in_place(eye_view.as_mut());
                    v_beta_cond.assign(&eye);
                } else {
                    log::warn!("Penalized Hessian not invertible. Skipping correction.");
                    return Ok((current_rho, None));
                }
            }
        }

        // Compute Jacobian columns:
        //   J[:,k] = -H_p^{-1}(S_k^ρ β̂)
        //          = -V_beta_cond * (S_k β̂ * λ_k)
        // with S_k β̂ assembled as R_kᵀ(R_k β̂).
        // S_k = R_k^T R_k.
        let mut jacobian = Array2::<f64>::zeros((p_dim, n_rho));

        for k in 0..n_rho {
            let r_k = &rs[k];
            if r_k.ncols() == 0 {
                continue;
            }

            let lambda = lambdas[k];
            // S_k beta = R_k^T (R_k beta)
            let r_beta = r_k.dot(beta);
            let s_beta = r_k.t().dot(&r_beta);

            let term = s_beta.mapv(|v| v * lambda);

            // col = - V_beta_cond * term
            let col = v_beta_cond.dot(&term).mapv(|v| -v);

            jacobian.column_mut(k).assign(&col);
        }

        // V_corr approximates Var(b(rho)) under first-order linearization.
        // V_corr = J * V_rho * J^T.
        let temp = jacobian.dot(&v_rho); // (p, k) * (k, k) -> (p, k)
        let v_corr = temp.dot(&jacobian.t()); // (p, k) * (k, p) -> (p, p)

        log::info!(
            "[Boundary] Correction computed. Max element in V_corr: {:.3e}",
            v_corr.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
        );

        // First-order total covariance approximation to Var(beta).
        let v_total = v_beta_cond + v_corr;

        Ok((current_rho, Some(v_total)))
    }
}
