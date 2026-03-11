use super::*;
use crate::matrix::DenseRightProductView;
use ndarray::ShapeBuilder;

impl<'a> RemlState<'a> {
    pub(super) fn structural_penalty_logdet_derivatives(
        &self,
        rs_transformed: &[Array2<f64>],
        lambdas: &Array1<f64>,
        structural_rank: usize,
        ridge: f64,
    ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
        // Local derivation for the hard-truncated penalty logdet terms in the
        // outer gradient/Hessian:
        //
        //   P_k   = -0.5 * d/drho_k     log|S(rho)|_+
        //   P_k,l = -0.5 * d²/(drho_k drho_l) log|S(rho)|_+.
        //
        // Write
        //   S(rho) = sum_k lambda_k S_k,   lambda_k = exp(rho_k),
        //   A_k    = dS/drho_k      = lambda_k S_k,
        //   A_k,l  = d²S/(drho_k drho_l) = delta_{k,l} A_k.
        //
        // These identities are exact only on a neighborhood where the retained
        // structural eigenspace is fixed; equivalently, there must be a
        // nonzero spectral gap around the hard cutoff used by this helper. On
        // such a branch, the truncated logdet behaves like ordinary logdet on
        // the kept subspace, with S^{-1} replaced by the inverse on that
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
        // This helper realizes that branch-local formula on a fixed-rank
        // reduced penalty space. It should not be read as an exact derivative
        // statement at threshold crossings: for a hard truncation, even the
        // first derivative ceases to be exact when the active projector
        // changes. The caller is responsible for downgrading "exact" claims
        // when the retained spectrum loses a gap to the cutoff.
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
            let mut y_kview = array2_to_matmut(&mut y_k);
            s_r_factor.solve_in_place(y_kview.as_mut());
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

    pub(super) fn compute_lamlhessian_analytic_fallback(
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
        let (_, det2) = self.structural_penalty_logdet_derivatives(
            &pirls_result.reparam_result.rs_transformed,
            &lambdas,
            structural_rank,
            pirls_result.ridge_passport.penalty_logdet_ridge(),
        )?;
        h += &det2.mapv(|v| -0.5 * v);

        self.add_soft_priorhessian_in_place(rho, &mut h);

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

    pub(crate) fn compute_lamlhessian_consistent(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        // Strategy-policy routing:
        // - policy decides spectral exact vs analytic fallback vs diagnostic numeric,
        // - math kernels remain strategy-local.
        let bundle = self.obtain_eval_bundle(rho)?;
        let decision = self.selecthessian_strategy_policy(rho, &bundle);
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
            return self.compute_lamlhessian_by_strategy(rho, &bundle, decision);
        }
        match self.compute_lamlhessian_by_strategy(rho, &bundle, decision) {
            Ok(h) => Ok(h),
            Err(err) => {
                log::warn!(
                    "Exact LAML Hessian unavailable ({}); using analytic fallback Hessian.",
                    err
                );
                self.compute_lamlhessian_analytic_fallback(rho, Some(&bundle))
            }
        }
    }

    pub(crate) fn compute_lamlhessian_exact(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        let bundle = self.obtain_eval_bundle(rho)?;
        if Self::geometry_backend_kind(&bundle) == GeometryBackendKind::SparseExactSpd {
            return self.compute_lamlhessian_sparse_exact(rho, &bundle);
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
        // If one parameterizes directly as S(rho)=sum_k rho_k S_k (no exp map),
        // the same derivation holds with:
        //   A_k = S_k,  A_{k,l}=0.
        // The code path here uses log-smoothing coordinates rho with lambda=exp(rho).
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
        //   H = dG/dbeta = -ell_{beta,beta} + S(rho) - Hphi
        //     = X'WX + S(rho) - Hphi
        //   Hphi = d²Phi/dbeta²
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
        //   H_k  <- H_k  - D(Hphi)[B_k]
        //   H_kl <- H_kl - D(Hphi)[B_kl] - D²(Hphi)[B_k,B_l].
        //
        // Here `c` and `d` are (`solve_c_array`, `solve_d_array`), and
        // D(Hphi), D²(Hphi) are evaluated by reduced-space exact operators.
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
        // Equivalent notation often used in Laplace-REML derivations:
        //   u_k    := dβ̂/dρ_k,        u_{kℓ} := d²β̂/(dρ_k dρ_ℓ),
        //   J_k    := dJ/dρ_k,         J_{kℓ} := d²J/(dρ_k dρ_ℓ),
        // with
        //   u_k    = -J^{-1}(S_k β̂),
        //   u_{kℓ} = -J^{-1}(J_ℓ u_k + S_k u_ℓ),
        //   J_k    = S_k + dH[u_k],
        //   J_{kℓ} = dH[u_{kℓ}] + d²H[u_ℓ,u_k].
        //
        // Under this notation:
        //   V_{kℓ}
        //   = β̂' S_k u_ℓ
        //   + 0.5[-tr(J^{-1}J_ℓJ^{-1}J_k) + tr(J^{-1}J_{kℓ})]
        //   + 0.5 tr(P^+ S_ℓ P^+ S_k),
        // which is algebraically the same decomposition as Q/L/P above
        // after mapping A_k = λ_k S_k and the objective sign convention.
        //
        // Pseudodeterminant second derivative on fixed active penalty subspace:
        //   ∂² log|S|_+ /(∂ρ_k∂ρ_ℓ)
        //   = tr(S^+ A_{kℓ}) - tr(S^+ A_ℓ S^+ A_k),
        // and with λ_k = exp(ρ_k), A_{kℓ}=delta_{kℓ}A_k.
        //
        // Equivalent fixed-basis (rank-stable) form used by structural path:
        // choose an orthonormal basis R for the penalized range (common nullspace
        // complement), define S_R = R' S R (SPD), then
        //   log|S|_+ = log|S_R|,
        //   S^+ = R S_R^{-1} R',
        //   ∂ log|S|_+ /∂ρ_k      = tr(S_R^{-1} (R' A_k R)),
        //   ∂² log|S|_+ /∂ρ_k∂ρ_ℓ = tr(S_R^{-1}(R' A_{kℓ} R))
        //                           - tr(S_R^{-1}(R' A_ℓ R) S_R^{-1}(R' A_k R)).
        // This is exactly what `structural_penalty_logdet_derivatives` computes
        // under a fixed active structural subspace.
        //
        // Trace implementation identity behind exact/projected/spectral paths:
        // if H = L L' and W(A)=L^{-1} A L^{-T}, then
        //   tr(H^{-1}A)=tr(W(A)),
        //   tr(H^{-1}A H^{-1}B)=<W(A),W(B)>_F.
        // We realize these without explicitly forming H^{-1}.
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
        let x_dense_eval = DenseRightProductView::new(x_dense_orig_arc.as_ref())
            .with_optional_factor(free_basis_opt.as_ref())
            .materialize();
        let mut h_total_eval = bundle.h_total.as_ref().clone();
        let mut e_eval = reparam_result.e_transformed.clone();

        if let Some(z) = free_basis_opt.as_ref() {
            h_total_eval = Self::projectwith_basis(bundle.h_total.as_ref(), z);
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            rs_eval = reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect();
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
        let h_posw_for_solve = if free_basis_opt.is_none() && bundle.h_pos_factorw.nrows() == p_dim
        {
            Some(bundle.h_pos_factorw.as_ref().clone())
        } else {
            None
        };
        let h_posw_for_solve_t = h_posw_for_solve.as_ref().map(|w| w.t().to_owned());
        let use_cached_factor = free_basis_opt.is_none();
        let h_factor_cached = if h_posw_for_solve.is_none() && use_cached_factor {
            Some(self.get_faer_factor(rho, h_total))
        } else {
            None
        };
        let h_factor_local = if h_posw_for_solve.is_none() && !use_cached_factor {
            Some(self.factorize_faer(h_total))
        } else {
            None
        };
        let solve_h = |rhs: &Array2<f64>| -> Array2<f64> {
            if rhs.ncols() == 0 {
                return rhs.clone();
            }
            if let (Some(w), Some(w_t)) = (h_posw_for_solve.as_ref(), h_posw_for_solve_t.as_ref()) {
                let wt_rhs = fast_ab(w_t, rhs);
                return fast_ab(w, &wt_rhs);
            }
            let mut out = rhs.clone();
            let mut outview = array2_to_matmut(&mut out);
            if let Some(f) = h_factor_cached.as_ref() {
                f.solve_in_place(outview.as_mut());
            } else if let Some(f) = h_factor_local.as_ref() {
                f.solve_in_place(outview.as_mut());
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
            self.add_soft_priorhessian_in_place(rho, &mut hess);
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
            //   Hphi = d^2 Phi / d beta^2,
            // and H_k/H_kl include -D(Hphi)[B_k], -D(Hphi)[B_kl],
            // -D^2(Hphi)[B_k,B_l] exactly.
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
                    // Reuse already-batched eta sensitivities u_k = X B_k
                    // from `u_mat` to avoid redundant X.dot(B_k) work.
                    let deta_k = u_mat.column(k).to_owned();
                    Self::firth_direction_from_deta(op, deta_k)
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

        // Exact-only trace backend for Hessian assembly.
        //
        // This keeps all L_{k,l} contractions deterministic and analytic
        // across both Firth and non-Firth paths.
        let exact_trace_mode = true;
        let n_probe = 0usize;
        let n_sketch = 0usize;
        let use_hutchpp = false;
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
            && bundle.h_pos_factorw.nrows() == p_dim
            && Self::dense_projected_exact_eligible(n, bundle.h_pos_factorw.ncols(), k_count);
        let spectral_exact_mode = exact_trace_mode
            && free_basis_opt.is_none()
            && bundle.h_pos_factorw.nrows() == p_dim
            && !projected_exact_mode;
        // Mode split:
        //   projected_exact_mode: non-Firth optimized exact contractions.
        //   spectral_exact_mode:  exact H_+^dagger traces via W'(*)W (works for Firth too).
        //   else: dense fallback (potentially full H^{-1}).

        let w_pos_projected = if projected_exact_mode {
            Some(bundle.h_pos_factorw.as_ref().clone())
        } else {
            None
        };
        let z_mat_projected = w_pos_projected
            .as_ref()
            .map(|w_pos| fast_ab(x_dense, w_pos));
        let w_posspectral = if spectral_exact_mode {
            Some(bundle.h_pos_factorw.as_ref().clone())
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
        let w_posspectral_t = w_posspectral.as_ref().map(|w_pos| w_pos.t().to_owned());
        let g_kspectral: Option<Vec<Array2<f64>>> = w_posspectral.as_ref().map(|w_pos| {
            // Spectral exact traces on active positive subspace:
            //   H_+^dagger = W W^T
            //   tr(H_+^dagger A) = tr(W^T A W)
            //   tr(H_+^dagger B H_+^dagger C) = tr((W^T B W)(W^T C W)).
            // Here G_k := W^T H_k W so
            //   t1_{l,k} = tr(H_+^dagger H_l H_+^dagger H_k) = tr(G_l G_k).
            let w_pos_t = w_posspectral_t
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

        let mut probez: Option<Array2<f64>> = None;
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
                        let qtz = q.t().dot(&zr);
                        let proj = q.dot(&qtz);
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
            probez = Some(z);
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
                let gk = g_kspectral
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
            let z = probez.as_ref().expect("probes present in stochastic mode");
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
        let dense_exact_inner_solve_mode =
            exact_trace_mode && !projected_exact_mode && !spectral_exact_mode;
        // Full per-pair exact Hessian recipe:
        //
        // For each (k,l), with B_k = d beta_hat / d rho_k and B_{k,l} = d² beta_hat /(d rho_k d rho_l):
        //
        //   (i)  Solve second sensitivity:
        //          B_{k,l} = -H^{-1}(H_l B_k + A_k B_l + delta_{k,l} A_k beta_hat).
        //
        //   (ii) Build curvature second derivative:
        //          H_{k,l} = delta_{k,l} A_k
        //                   + D²(-∇²l)[B_k,B_l]
        //                   + D(-∇²l)[B_{k,l}]
        //                = delta_{k,l} A_k + X' diag(d ⊙ u_k ⊙ u_l + c ⊙ u_{k,l}) X.
        //
        //   (iii) Assemble objective pieces:
        //          Q_{k,l} = B_l' A_k beta_hat + 0.5 delta_{k,l} beta_hat' A_k beta_hat,
        //          L_{k,l} = 0.5( -tr(H^{-1}H_l H^{-1}H_k) + tr(H^{-1}H_{k,l}) ),
        //          P_{k,l} = -0.5 ∂² log|S|_+ /(∂rho_k ∂rho_l).
        //
        //   (iv) Final:
        //          V_{k,l} = Q_{k,l} + L_{k,l} + P_{k,l},
        //        mirrored to enforce symmetry.
        //
        // Fixed-subspace pseudodeterminant term:
        //   with S_R = R' S R, rank-stable R,
        //   ∂² log|S|_+ = tr(S_R^{-1}A_{k,l,R}) - tr(S_R^{-1}A_{l,R} S_R^{-1}A_{k,R}),
        // exactly supplied by `d2logs`.
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
                //
                // If we denote w_{k,l} := dB_k/dρ_l, then B_{k,l}=w_{k,l};
                // this is the exact "second sensitivity" solve from IFT.
                //
                // Term-by-term mapping:
                //   -h_k[l].dot(&bk)      = -(H_l B_k)
                //   -a_k_mats[k].dot(&bl) = -(A_k B_l)
                //   -a_k_beta[k]          = -(delta_{k,l} A_k beta), only when k=l.
                let mut rhs_kl = -h_k[l].dot(&bk);
                rhs_kl -= &a_k_mats[k].dot(&bl);
                if k == l {
                    rhs_kl -= &a_k_beta[k];
                }
                rhs_kl_all.column_mut(k).assign(&rhs_kl);
            }
            let b_kl_all = solve_h(&rhs_kl_all);
            let u_kl_all = fast_ab(x_dense, &b_kl_all);
            let row_entries: Vec<(usize, f64)> = if dense_exact_inner_solve_mode {
                let mut weighted_xtdx_kl = Array2::<f64>::zeros((0, 0).f());
                (l..k_count)
                    .map(|k| {
                        let val = {
                            // Inline the only per-k mutable scratch so the sequential
                            // branch can reuse one bounded chunk buffer too.
                            let mut diag = Array1::<f64>::zeros(n);
                            for i in 0..n {
                                diag[i] =
                                    d[i] * u_mat[[i, k]] * u_mat[[i, l]] + c[i] * u_kl_all[[i, k]];
                            }

                            let q =
                                bl.dot(&a_k_beta[k]) + if k == l { 0.5 * q_diag[k] } else { 0.0 };
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
                                    let mut h_kl = if k == l {
                                        a_k_mats[k].clone()
                                    } else {
                                        Array2::<f64>::zeros((p_dim, p_dim))
                                    };
                                    h_kl += &Self::xt_diag_x_dense_chunked_into(
                                        x_dense,
                                        &diag,
                                        &mut weighted_xtdx_kl,
                                    );
                                    let mut d2_trace_correction = 0.0_f64;
                                    if let (Some(op), Some(dirs)) =
                                        (firth_op.as_ref(), firth_dirs.as_ref())
                                    {
                                        let deta_kl = u_kl_all.column(k).to_owned();
                                        let dir_kl = Self::firth_direction_from_deta(op, deta_kl);
                                        h_kl -= &Self::firth_hphi_direction(op, &dir_kl);
                                        if spectral_exact_mode {
                                            let w_pos = w_posspectral.as_ref().expect(
                                                "spectral W present in spectral exact mode",
                                            );
                                            let w_pos_t = w_posspectral_t.as_ref().expect(
                                                "spectral W^T present in spectral exact mode",
                                            );
                                            let d2_aw = Self::firth_hphisecond_direction_apply(
                                                op, &dirs[k], &dirs[l], w_pos,
                                            );
                                            d2_trace_correction =
                                                Self::trace_product(w_pos_t, &d2_aw);
                                        } else {
                                            const BLOCK: usize = 32;
                                            let mut acc = 0.0_f64;
                                            let mut start = 0usize;
                                            while start < p_dim {
                                                let width = (p_dim - start).min(BLOCK);
                                                let mut basis =
                                                    Array2::<f64>::zeros((p_dim, width));
                                                for j in 0..width {
                                                    basis[[start + j, j]] = 1.0;
                                                }
                                                let d2_block =
                                                    Self::firth_hphisecond_direction_apply(
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
                                        let w_pos = w_posspectral
                                            .as_ref()
                                            .expect("spectral W present in spectral exact mode");
                                        let w_pos_t = w_posspectral_t
                                            .as_ref()
                                            .expect("spectral W^T present in spectral exact mode");
                                        let wt_hkl = fast_ab(w_pos_t, &h_kl);
                                        let g_kl = fast_ab(&wt_hkl, w_pos);
                                        g_kl.diag().sum() - d2_trace_correction
                                    } else {
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
                                let z = probez.as_ref().expect("probes present in stochastic mode");
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
                            let l_term = 0.5 * (-t1 + t2);
                            let p_term = -0.5 * d2logs[[k, l]];
                            q + l_term + p_term
                        };
                        (k, val)
                    })
                    .collect()
            } else {
                (l..k_count)
                    .into_par_iter()
                    .map_init(
                        || Array2::<f64>::zeros((0, 0).f()),
                        |weighted_xtdx_kl, k| {
                            let mut diag = Array1::<f64>::zeros(n);
                            for i in 0..n {
                                diag[i] =
                                    d[i] * u_mat[[i, k]] * u_mat[[i, l]] + c[i] * u_kl_all[[i, k]];
                            }

                            let q =
                                bl.dot(&a_k_beta[k]) + if k == l { 0.5 * q_diag[k] } else { 0.0 };
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
                                    let mut h_kl = if k == l {
                                        a_k_mats[k].clone()
                                    } else {
                                        Array2::<f64>::zeros((p_dim, p_dim))
                                    };
                                    h_kl += &Self::xt_diag_x_dense_chunked_into(
                                        x_dense,
                                        &diag,
                                        weighted_xtdx_kl,
                                    );
                                    let mut d2_trace_correction = 0.0_f64;
                                    if let (Some(op), Some(dirs)) =
                                        (firth_op.as_ref(), firth_dirs.as_ref())
                                    {
                                        let deta_kl = u_kl_all.column(k).to_owned();
                                        let dir_kl = Self::firth_direction_from_deta(op, deta_kl);
                                        h_kl -= &Self::firth_hphi_direction(op, &dir_kl);
                                        if spectral_exact_mode {
                                            let w_pos = w_posspectral.as_ref().expect(
                                                "spectral W present in spectral exact mode",
                                            );
                                            let w_pos_t = w_posspectral_t.as_ref().expect(
                                                "spectral W^T present in spectral exact mode",
                                            );
                                            let d2_aw = Self::firth_hphisecond_direction_apply(
                                                op, &dirs[k], &dirs[l], w_pos,
                                            );
                                            d2_trace_correction =
                                                Self::trace_product(w_pos_t, &d2_aw);
                                        } else {
                                            const BLOCK: usize = 32;
                                            let mut acc = 0.0_f64;
                                            let mut start = 0usize;
                                            while start < p_dim {
                                                let width = (p_dim - start).min(BLOCK);
                                                let mut basis =
                                                    Array2::<f64>::zeros((p_dim, width));
                                                for j in 0..width {
                                                    basis[[start + j, j]] = 1.0;
                                                }
                                                let d2_block =
                                                    Self::firth_hphisecond_direction_apply(
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
                                        let w_pos = w_posspectral
                                            .as_ref()
                                            .expect("spectral W present in spectral exact mode");
                                        let w_pos_t = w_posspectral_t
                                            .as_ref()
                                            .expect("spectral W^T present in spectral exact mode");
                                        let wt_hkl = fast_ab(w_pos_t, &h_kl);
                                        let g_kl = fast_ab(&wt_hkl, w_pos);
                                        g_kl.diag().sum() - d2_trace_correction
                                    } else {
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
                                let z = probez.as_ref().expect("probes present in stochastic mode");
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
                            let l_term = 0.5 * (-t1 + t2);
                            let p_term = -0.5 * d2logs[[k, l]];
                            (k, q + l_term + p_term)
                        },
                    )
                    .collect()
            };
            for (k, val) in row_entries {
                // Exact Hessian symmetry expectation: V_{k,l}=V_{l,k}.
                // We mirror-write to keep numerical symmetry explicit even when
                // stochastic traces or floating-point order differ slightly.
                // Conclusion for the Firth path:
                // - `q` uses the same implicit derivatives B_k/B_kl as non-Firth,
                //   but those derivatives were solved on H_total = X'WX + S - Hphi.
                // - `t1`/`t2` see Firth through H_k/H_kl, which include
                //   -D(Hphi)[B_k], -D(Hphi)[B_kl], and -D²(Hphi)[B_k,B_l].
                // Therefore `val` is objective-consistent with the Firth-adjusted
                // inner stationarity equation on smooth active-subspace regions.
                hess[[k, l]] = val;
                hess[[l, k]] = val;
            }
        }
        self.add_soft_priorhessian_in_place(rho, &mut hess);
        Ok(hess)
    }

    pub(crate) fn compute_lamlhessian_analytic_fallback_standalone(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        self.compute_lamlhessian_analytic_fallback(rho, None)
    }

    pub(crate) fn compute_smoothing_correction_auto(
        &self,
        final_rho: &Array1<f64>,
        final_fit: &PirlsResult,
        base_covariance: Option<&Array2<f64>>,
        finalgrad_norm: f64,
    ) -> Option<Array2<f64>> {
        // Always compute the fast first-order correction first.
        let first_order = super::compute_smoothing_correction(self, final_rho, final_fit);
        let first_order_correction = first_order.correction.clone();
        let n_rho = final_rho.len();
        if n_rho == 0 {
            return first_order_correction;
        }
        if n_rho > AUTO_CUBATURE_MAX_RHO_DIM {
            return first_order_correction;
        }
        if final_fit.beta_transformed.len() > AUTO_CUBATURE_MAX_BETA_DIM {
            return first_order_correction;
        }
        if let Ok(bundle) = self.obtain_eval_bundle(final_rho)
            && bundle.active_subspace_unstable
        {
            // Cubature correction relies on a locally stable outer Hessian.
            // Near active-subspace crossings of H_+, the hard-truncated
            // spectral objective is not described by one smooth quadratic
            // model, so we keep only the first-order correction.
            return first_order_correction;
        }

        let near_boundary = final_rho
            .iter()
            .any(|&v| (RHO_BOUND - v.abs()) <= AUTO_CUBATURE_BOUNDARY_MARGIN);
        let grad_norm = if finalgrad_norm.is_finite() {
            finalgrad_norm
        } else {
            0.0
        };
        let highgrad = grad_norm > 1e-3;
        if !near_boundary && !highgrad {
            // Keep the hot path cheap when the local linearization is likely sufficient.
            return first_order_correction;
        }

        // Build V_rho from the outer Hessian around rho_hat.
        let mut hessian_rho = if let Some(h) = first_order.hessian_rho {
            h
        } else {
            match self.compute_lamlhessian_consistent(final_rho) {
                Ok(h) => h,
                Err(err) => {
                    log::debug!("Auto cubature skipped: rho Hessian unavailable ({}).", err);
                    return first_order_correction;
                }
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
            match matrix_inversewith_regularization(&hessian_rho, "auto cubature rho Hessian") {
                Some(v) => v,
                None => return first_order_correction,
            };

        let max_rhovar = hessian_rho_inv
            .diag()
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !near_boundary && !highgrad && max_rhovar < 0.1 {
            return first_order_correction;
        }

        use crate::faer_ndarray::FaerEigh;
        use faer::Side;
        let (evals, evecs) = match hessian_rho_inv.eigh(Side::Lower) {
            Ok(x) => x,
            Err(_) => return first_order_correction,
        };
        let mut eig_pairs: Vec<(usize, f64)> = evals
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| v.is_finite() && *v > 1e-12)
            .collect();
        if eig_pairs.is_empty() {
            return first_order_correction;
        }
        eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let totalvar: f64 = eig_pairs.iter().map(|(_, v)| *v).sum();
        if !totalvar.is_finite() || totalvar <= 0.0 {
            return first_order_correction;
        }

        let mut rank = 0usize;
        let mut captured = 0.0_f64;
        for (_, eig) in eig_pairs
            .iter()
            .take(AUTO_CUBATURE_MAX_EIGENVECTORS.min(eig_pairs.len()))
        {
            captured += *eig;
            rank += 1;
            if captured / totalvar >= AUTO_CUBATURE_TARGET_VAR_FRAC {
                break;
            }
        }
        if rank == 0 {
            return first_order_correction;
        }

        let base_cov = match base_covariance {
            Some(v) => v,
            None => return first_order_correction,
        };
        let p = base_cov.nrows();
        let radius = (rank as f64).sqrt();
        let mut sigma_points: Vec<Array1<f64>> = Vec::with_capacity(2 * rank);
        for (eig_idx, eigval) in eig_pairs.iter().take(rank) {
            let axis = evecs.column(*eig_idx).to_owned();
            let scale = radius * eigval.sqrt();
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
            return first_order_correction;
        }

        // Disable warm-start and PIRLS-cache coupling while evaluating sigma
        // points in parallel. Cache lookups/inserts use an exclusive lock in
        // execute_pirls_if_needed(), so leaving cache enabled serializes this
        // block under contention.
        let point_results: Vec<Option<(Array2<f64>, Array1<f64>)>> = {
            let guards = (
                AtomicFlagGuard::swap(
                    &self.cache_manager.pirls_cache_enabled,
                    false,
                    Ordering::SeqCst,
                ),
                AtomicFlagGuard::swap(&self.warm_start_enabled, false, Ordering::SeqCst),
            );
            (
                guards,
                (0..sigma_points.len())
                    .into_par_iter()
                    .map(|idx| {
                        let fit_point = self.execute_pirls_if_needed(&sigma_points[idx]).ok()?;
                        let h_point = map_hessian_to_original_basis(fit_point.as_ref()).ok()?;
                        let cov_point =
                            matrix_inversewith_regularization(&h_point, "auto cubature point")?;
                        let beta_point = fit_point
                            .reparam_result
                            .qs
                            .dot(fit_point.beta_transformed.as_ref());
                        Some((cov_point, beta_point))
                    })
                    .collect(),
            )
                .1
        };

        if point_results.iter().any(|r| r.is_none()) {
            return first_order_correction;
        }

        let w = 1.0 / (sigma_points.len() as f64);
        let mut mean_hinv = Array2::<f64>::zeros((p, p));
        let mut mean_beta = Array1::<f64>::zeros(p);
        let mut second_beta = Array2::<f64>::zeros((p, p));
        for (cov_point, beta_point) in point_results.into_iter().flatten() {
            mean_hinv += &cov_point.mapv(|v| w * v);
            mean_beta += &beta_point.mapv(|v| w * v);
            let outer = beta_point
                .view()
                .insert_axis(ndarray::Axis(1))
                .dot(&beta_point.view().insert_axis(ndarray::Axis(0)));
            second_beta += &outer.mapv(|v| w * v);
        }

        let mean_outer = mean_beta
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&mean_beta.view().insert_axis(ndarray::Axis(0)));
        let var_beta = second_beta - mean_outer;

        let mut total_cov = mean_hinv + var_beta;
        for i in 0..p {
            for j in (i + 1)..p {
                let avg = 0.5 * (total_cov[[i, j]] + total_cov[[j, i]]);
                total_cov[[i, j]] = avg;
                total_cov[[j, i]] = avg;
            }
        }
        if !total_cov.iter().all(|v| v.is_finite()) {
            return first_order_correction;
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
            "Using adaptive cubature smoothing correction (rank={}, points={}, near_boundary={}, grad_norm={:.2e}, maxvar={:.2e})",
            rank,
            2 * rank,
            near_boundary,
            grad_norm,
            max_rhovar
        );
        Some(corr)
    }
}

impl<'a> RemlState<'a> {
    pub(super) fn usesobjective_consistentfdgradient(&self, rho: &Array1<f64>) -> bool {
        let _ = rho;
        self.config.link_function() != LinkFunction::Identity
            && self.config.objective_consistentfdgradient
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
    pub(super) fn compute_gradient_with_bundle(
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
        let mut h_pos_factorw_eval = bundle.h_pos_factorw.as_ref().clone();

        if let Some(z) = free_basis_opt.as_ref() {
            h_eff_eval = Self::projectwith_basis(bundle.h_eff.as_ref(), z);
            e_eval = reparam_result.e_transformed.dot(z);
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            rs_eval = reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect();
            let x_dense_arc = pirls_result.x_transformed.to_dense_arc();
            x_transformed_eval = DesignMatrix::Dense(
                DenseRightProductView::new(x_dense_arc.as_ref())
                    .with_factor(z)
                    .materialize(),
            );

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
            h_pos_factorw_eval = w;
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
            workspace.set_lambdavalues(p);
            workspace.zerocostgradient(len);
            let lambdas = workspace.lambdaview(len).to_owned();
            let mut applied_truncation_corrections: Option<Vec<f64>> = None;

            // Branch-local structural truncated-logdet derivatives:
            // d/dρ_k log|S|_+ and d²/(dρ_k dρ_ℓ) log|S|_+ are evaluated on a
            // reduced structural subspace (rank = e_transformed.nrows()) with a
            // smooth floor in that reduced block. These formulas are exact only
            // while that kept structural subspace stays fixed; they intentionally
            // avoid adaptive rank flips, but they are not exact at a hard
            // threshold crossing.
            let (det1_values, _) = self.structural_penalty_logdet_derivatives(
                rs_transformed,
                &lambdas,
                e_eval.nrows(),
                ridge_passport.penalty_logdet_ridge(),
            )?;

            // --- Use Single Stabilized Hessian from P-IRLS ---
            // Use the same effective Hessian as the cost function for consistency.
            if ridge_passport.laplacehessianridge() > 0.0 {
                log::debug!(
                    "Gradient path using PIRLS-stabilized Hessian (ridge {:.3e})",
                    ridge_passport.laplacehessianridge()
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
                    let retreatgrad = p.mapv(|v| -(v.abs() + 1.0));
                    return Ok(retreatgrad);
                }
            }

            // --- Extract common components ---

            let n = self.y.len() as f64;

            // Derivation map:
            //   λ_k = exp(ρ_k), A_k = ∂S/∂ρ_k = λ_k S_k.
            //   By the envelope theorem at inner stationarity, the penalized-fit block
            //   does not include an explicit dβ̂/dρ term.
            //   Outer gradient assembly:
            //     ∂V/∂ρ_k = 0.5 β̂^T A_k β̂ + 0.5 tr(H_+^† H_k) - 0.5 tr(S_+^† A_k).
            //   H_k differs by family:
            //     Gaussian: H_k = A_k.
            //     Non-Gaussian: H_k = A_k + d(X^T W(η̂) X)/dρ_k (third-derivative path).
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
                    let (dp_c, dp_cgrad) = smooth_floor_dp(dp);

                    let penalty_rank = e_eval.nrows();
                    let mp = h_eff.ncols().saturating_sub(penalty_rank) as f64;
                    let scale = dp_c / (n - mp).max(LAML_RIDGE);
                    // Gaussian profiled-scale identity used by this branch:
                    //   φ̂(ρ) = D_p(ρ)/(n-M_p), with D_p = rss + β̂ᵀSβ̂.
                    // The gradient therefore includes the profiled contribution
                    //   (n-M_p)/2 * D_k / D_p
                    // which is exactly represented by `deviancegrad_term` below.
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
                    let w_pos = &h_pos_factorw_eval;
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
                    //   log_det_hgrad_term (or numeric)   -> 0.5 * tr(H^{-1} S_k^ρ)
                    //   0.5 * det1_values[k]               -> 0.5 * tr(S^+ S_k^ρ)
                    let compute_gaussiangrad = |k: usize| -> f64 {
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
                        // with the smooth-floor derivative factor `dp_cgrad`.
                        let d1 = lambdas[k] * beta_ref.dot(&s_k_beta_transformed);
                        let deviancegrad_term = dp_cgrad * (d1 / (2.0 * scale));

                        // A.3/A.5 Component 2 derivation:
                        //   ∂/∂ρ_k [0.5 log|H|_+] = 0.5 tr(H_+^† H_k),
                        // and for Gaussian identity H_k = A_k = λ_k S_k.
                        //
                        // Root form on kept subspace:
                        //   tr(H_+^† A_k) = λ_k tr(H_+^† R_kᵀR_k)
                        //                = λ_k ||R_k W||_F², H_+^†=W W^T.
                        let rkw = r_k.dot(w_pos);
                        let trace_h_pos_inv_s_k: f64 = rkw.iter().map(|v| v * v).sum();
                        let log_det_hgrad_term = 0.5 * lambdas[k] * trace_h_pos_inv_s_k;

                        let corrected_log_det_h = log_det_hgrad_term;

                        // Component 3 derivation:
                        //   -0.5 * ∂/∂ρ_k log|S|_+,
                        // with `det1_values[k]` already equal to ∂ log|S|_+ / ∂ρ_k.
                        let log_det_sgrad_term = 0.5 * det1_values[k];

                        deviancegrad_term + corrected_log_det_h - log_det_sgrad_term
                    };

                    {
                        let mut gradview = workspace.costgradientview(len);
                        for k in 0..lambdas.len() {
                            gradview[k] = compute_gaussiangrad(k);
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
                    // we add Hphi. Below we compute Hphi and use H_total for the
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
                            .solvemu
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
                    let s_k_stats: Vec<(f64, Array1<f64>)> = (0..k_count)
                        .into_par_iter()
                        .map(|k| {
                            let r_k = &rs_transformed[k];
                            let r_beta = r_k.dot(beta_ref);
                            let s_k_beta = r_k.t().dot(&r_beta);
                            // q_k = β̂^T A_k β̂ = λ_k β̂^T S_k β̂,
                            // with S_k β̂ assembled as R_k^T (R_k β̂).
                            let beta_term = lambdas[k] * beta_ref.dot(&s_k_beta);
                            (beta_term, s_k_beta)
                        })
                        .collect();
                    for (k, (beta_term, s_k_beta)) in s_k_stats.into_iter().enumerate() {
                        beta_terms[k] = beta_term;
                        s_k_beta_mat.column_mut(k).assign(&s_k_beta);
                    }

                    // Keep outer gradient on the same Hessian surface as PIRLS.
                    // The outer loop uses H_eff consistently (no Hphi subtraction).

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
                    // where d_i = -∂³ℓ_i/∂η_i³. Here `cvec` stores this per-observation
                    // third derivative quantity in the stabilized logit path.
                    let w_pos = &h_pos_factorw_eval;
                    let nobs = pirls_result.solvemu.len();

                    // c_i = dW_ii/dη_i for H = Xᵀ W X + S.
                    // In smooth regimes this matches the required third-derivative object
                    // in dH/dρ. In clamped/floored regimes c_i may behave like a subgradient
                    // proxy rather than a classical derivative; see pirls.rs comments.
                    let cvec = &w_prime;

                    // h_i = x_i^T H_+^\dagger x_i = ||(XW)_{i,*}||^2.
                    let mut leverage_h_pos = Array1::<f64>::zeros(nobs);
                    if w_pos.ncols() > 0 {
                        for col in 0..w_pos.ncols() {
                            let w_col = w_pos.column(col).to_owned();
                            let xw_col = x_transformed_eval.matrixvectormultiply(&w_col);
                            Zip::from(&mut leverage_h_pos)
                                .and(&xw_col)
                                .for_each(|h, &v| *h += v * v);
                        }
                    }

                    // Precompute r = X^T (c ⊙ h) once:
                    //   tracethird_k = (c ⊙ h)^T (X v_k) = r^T v_k.
                    // This removes the per-k O(np) multiply X*v_k from the hot loop.
                    // r := X^T (w' ⊙ h).
                    let c_times_h = cvec * &leverage_h_pos;
                    let rthird = x_transformed_eval.transpose_vector_multiply(&c_times_h);

                    // Batch all v_k = H_+^† (S_k beta) into one BLAS-3 path:
                    //   V = W (W^T [S_1 beta, ..., S_K beta]).
                    let v_all = if w_pos.ncols() > 0 && k_count > 0 {
                        let wt_sk_beta_all = w_pos.t().dot(&s_k_beta_mat);
                        w_pos.dot(&wt_sk_beta_all)
                    } else {
                        Array2::<f64>::zeros((beta_ref.len(), k_count))
                    };

                    let grad_terms: Vec<f64> = (0..k_count)
                        .into_par_iter()
                        .map(|k_idx| {
                            let r_k = &rs_transformed[k_idx];
                            if r_k.ncols() == 0 || w_pos.ncols() == 0 {
                                let log_det_hgrad_term = 0.0;
                                let log_det_sgrad_term = 0.5 * det1_values[k_idx];
                                return 0.5 * beta_terms[k_idx] + log_det_hgrad_term
                                    - log_det_sgrad_term;
                            }

                            // First piece:
                            //   tr(H_+^† S_k) = ||R_k W||_F^2, with H_+^† = W W^T.
                            let rkw = r_k.dot(w_pos);
                            let trace_h_inv_s_k: f64 = rkw.iter().map(|v| v * v).sum();

                            // Exact third-derivative contraction:
                            //   tr(H_+^† X^T diag(c ⊙ X v_k) X) = r^T v_k.
                            let v_k = v_all.column(k_idx);
                            let mut tracethird = rthird.dot(&v_k);
                            if !tracethird.is_finite() {
                                tracethird = 0.0;
                            }
                            // Auto-correct third-derivative contribution in numerically
                            // brittle regimes: cap its magnitude relative to the primary
                            // trace term so one noisy contraction cannot dominate the
                            // hyper-gradient.
                            let cap = (trace_h_inv_s_k.abs() + 1.0) * 10.0;
                            tracethird = tracethird.clamp(-cap, cap);
                            let trace_term = trace_h_inv_s_k - tracethird;
                            let log_det_hgrad_term = 0.5 * lambdas[k_idx] * trace_term;
                            let corrected_log_det_h = log_det_hgrad_term;
                            let log_det_sgrad_term = 0.5 * det1_values[k_idx];

                            // Exact LAML gradient assembly for the implemented objective:
                            //   g_k = 0.5 * β̂ᵀ A_k β̂ - 0.5 * tr(S^+ A_k) + 0.5 * tr(H^{-1} H_k)
                            // where A_k = ∂S/∂ρ_k = λ_k S_k and H_k is the total derivative.
                            0.5 * beta_terms[k_idx] + corrected_log_det_h - log_det_sgrad_term
                        })
                        .collect();
                    {
                        let mut gradview = workspace.costgradientview(len);
                        for (k_idx, &gk) in grad_terms.iter().enumerate() {
                            gradview[k_idx] = gk;
                        }
                    }
                }
            }

            if !includes_prior {
                let (_, priorgradview) = workspace.soft_priorcost_andgrad(p);
                let priorgrad = priorgradview.to_owned();
                {
                    let mut costgradientview = workspace.costgradientview(len);
                    costgradientview += &priorgrad;
                }
            }

            // Capture the gradient snapshot before releasing the workspace borrow so
            // that diagnostics can continue without holding the RefCell borrow.
            let gradient_result = workspace.costgradientview_const(len).to_owned();
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
            self.rungradient_diagnostics(
                p,
                bundle,
                &gradient_snapshot,
                applied_truncation_corrections.as_deref(),
            );
        }

        if self.should_use_stochastic_exactgradient(bundle, &gradient_result) {
            match self.compute_logit_stochastic_exactgradient(p, bundle) {
                Ok(stochasticgrad) => {
                    self.arena
                        .lastgradient_used_stochastic_fallback
                        .store(true, Ordering::Relaxed);
                    log::warn!(
                        "[REML] using stochastic exact log-marginal gradient fallback (posterior-sampled expectation)"
                    );
                    return Ok(stochasticgrad);
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

    pub(super) fn should_use_stochastic_exactgradient(
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
        let kkt_like = pirls.lastgradient_norm;
        if !kkt_like.is_finite() || kkt_like > 1e2 {
            return true;
        }
        let grad_inf = gradient.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        !grad_inf.is_finite() || grad_inf > 1e9
    }

    pub(super) fn compute_logit_stochastic_exactgradient(
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
            nwarmup: 48,
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
                    nwarmup: 160,
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
        grad += &self.compute_soft_priorgrad(p);
        Ok(grad)
    }
}
