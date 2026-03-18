use self::cache::AtomicFlagGuard;
use self::inner_strategy::GeometryBackendKind;
use super::*;
use crate::linalg::sparse_exact::{
    SparseExactFactor, SparsePenaltyBlock, assemble_and_factor_sparse_penalized_system,
};
use crate::types::SasLinkState;
use ndarray::s;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

mod cache;
mod eval;
mod firth;
pub(crate) mod hyper;
mod inner_strategy;
pub(crate) mod penalty_logdet;
mod runtime;
mod trace;
pub(crate) mod unified;

#[cfg(test)]
mod tests {
    use super::{
        DirectionalHyperParam, EvalShared, FirthDenseOperator, GlmLikelihoodFamily, RemlConfig,
        RemlState,
    };
    use crate::estimate::EstimationError;
    use crate::faer_ndarray::{FaerCholesky, FaerEigh};
    use crate::pirls::PirlsCoordinateFrame;
    use crate::types::GlmLikelihoodSpec;
    use faer::Side;
    use ndarray::{Array1, Array2, array, s};

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
            state.compute_joint_hypercostgradienthessian(&theta, rho.len(), &[hyper])?;
        Ok(gradient[rho.len()])
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
        let cfg = RemlConfig::external(
            GlmLikelihoodSpec::canonical(GlmLikelihoodFamily::BinomialLogit),
            1e-9,
            true,
        );
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
        );
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

        let (_, _, h) = state
            .compute_joint_hypercostgradienthessian(&theta, rho.len(), &hyper_dirs)
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
        );
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

        let (_, _, h_full) = state
            .compute_joint_hypercostgradienthessian(&theta, rho.len(), &hyper_dirs)
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

        let msg = match state.compute_joint_hypercostgradienthessian(&theta, 1, &hyper_dirs) {
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
        );
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
        let (_, _, h_full) = state
            .compute_joint_hypercostgradienthessian(&theta, rho.len(), &hyper_dirs)
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
            rel < 3e-1,
            "analytic tau-tau block deviates from FD reference: rel={rel:.3e}, analytic={h_tt_analytic:?}, fd={h_ttfd:?}"
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
    /// Cached dense materialization (lazy, populated on first call to ops that need the full matrix).
    cached_dense: std::sync::Arc<std::sync::OnceLock<Array2<f64>>>,
}

impl ImplicitDerivativeOp {
    fn materialize_dense(&self) -> &Array2<f64> {
        self.cached_dense.get_or_init(|| match self.level {
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
        })
    }

    fn nrows(&self) -> usize {
        self.operator.n_data()
    }

    fn ncols(&self) -> usize {
        self.operator.p_out()
    }

    fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        match self.level {
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
        }
    }

    fn forward_mul(&self, u: &Array1<f64>) -> Array1<f64> {
        match self.level {
            ImplicitDerivLevel::First(axis) => self
                .operator
                .forward_mul(axis, &u.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
            ImplicitDerivLevel::SecondDiag(axis) => self
                .operator
                .forward_mul_second_diag(axis, &u.view())
                .expect("radial scalar evaluation failed during implicit derivative forward_mul"),
            ImplicitDerivLevel::SecondCross(d, e) => self
                .operator
                .forward_mul_second_cross(d, e, &u.view())
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
    ) -> Self {
        Self {
            storage: DerivativeMatrixStorage::Implicit(ImplicitDerivativeOp {
                operator,
                level,
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
            is_penalty_like: true, // default: τ coords are penalty-like
        })
    }

    /// Mark this coordinate as non-penalty-like (design-moving).
    /// EFS will skip it; use Newton/BFGS for these coordinates.
    pub(crate) fn not_penalty_like(mut self) -> Self {
        self.is_penalty_like = false;
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
    // choose Q for the identifiable subspace of A^{1/2} X, and set:
    //   X_r := A^{1/2} X Q          (A = I when no fixed observation weights),
    //   W   := diag(w), with w_i = mu_i (1 - mu_i), 0 < w_i <= 1/4 for finite logit eta,
    //   I_r := X_rᵀ W X_r.
    //
    // Firth term is represented as:
    //   Phi(beta) = 0.5 log |I_r(beta)|,
    // which equals 0.5 log|Xᵀ A W X|_+ (pseudodeterminant in the full space)
    // but is smooth because I_r is SPD as long as the identifiable weighted
    // design has full column rank and w_i > 0.
    //
    // Mapping back to the full p-space uses:
    //   I_+^dagger = Q I_r^{-1} Qᵀ.
    //
    // We store reduced-space factors so all derivatives can be evaluated exactly
    // without materializing dense n×n matrices M = X K Xᵀ or P = M⊙M.
    x_dense: Array2<f64>,
    x_dense_t: Array2<f64>,
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
    // 0.5 log|I_r| at the current eta.
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
    pirls_cache_enabled: AtomicBool,
}

impl EvalCacheManager {
    fn new() -> Self {
        Self {
            pirls_cache: RwLock::new(PirlsLruCache::new(MAX_PIRLS_CACHE_ENTRIES)),
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
    config: &'a RemlConfig,
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
    /// When nonzero, caps PIRLS max_iterations for cheap seed screening.
    /// Shared with `OuterConfig.screening_cap` so the screening loop can
    /// set/clear it atomically without accessing `RemlState` directly.
    pub(crate) screening_max_inner_iterations: Arc<AtomicUsize>,

    /// When set, the penalties have Kronecker (tensor-product) structure and
    /// the REML evaluator can use O(∏q_j) logdet instead of O(p³) eigendecomposition.
    /// Populated via `set_kronecker_penalty_system` after construction.
    pub(crate) kronecker_penalty_system: Option<crate::smooth::KroneckerPenaltySystem>,
    /// Full Kronecker factored basis (marginal designs + penalties + dims).
    /// Used by P-IRLS for factored reparameterization.
    pub(crate) kronecker_factored: Option<crate::basis::KroneckerFactoredBasis>,
}
