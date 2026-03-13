use super::*;
use crate::linalg::utils::enforce_symmetry;
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
        let x_dense_eval_owned = free_basis_opt.as_ref().map(|z| {
            DenseRightProductView::new(x_dense_orig_arc.as_ref())
                .with_factor(z)
                .materialize()
        });
        let x_dense_eval = x_dense_eval_owned
            .as_ref()
            .unwrap_or_else(|| x_dense_orig_arc.as_ref());
        let (h_total_eval, e_eval) = if let Some(z) = free_basis_opt.as_ref() {
            beta_eval = z.t().dot(pirls_result.beta_transformed.as_ref());
            rs_eval = reparam_result
                .rs_transformed
                .iter()
                .map(|r| r.dot(z))
                .collect();
            (
                Self::projectwith_basis(bundle.h_total.as_ref(), z),
                reparam_result.e_transformed.dot(z),
            )
        } else {
            (
                bundle.h_total.as_ref().clone(),
                reparam_result.e_transformed.clone(),
            )
        };

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
        let x_dense = x_dense_eval;
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
            let mut s_k = fast_ata(r_k);
            let r_beta = r_k.dot(beta);
            let s_k_beta = fast_atv(r_k, &r_beta);
            let lam_k = lambdas[k];
            let a_kb = s_k_beta.mapv(|v| lam_k * v);
            q_diag[k] = beta.dot(&a_kb);
            rhs_bk
                .column_mut(k)
                .zip_mut_with(&a_kb, |dst, &src| *dst = -src);
            s_k.mapv_inplace(|v| lam_k * v);
            a_k_mats.push(s_k);
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

        // Pre-compute c ⊙ u_k for each penalty term k.
        // These are reused for both H_k assembly and second-derivative traces.
        let s_cols: Vec<Array1<f64>> = (0..k_count)
            .map(|k| {
                let mut s = Array1::<f64>::zeros(n);
                for i in 0..n {
                    s[i] = c[i] * u_mat[[i, k]];
                }
                s
            })
            .collect();

        let mut h_k = Vec::with_capacity(k_count);
        let mut weighted_xtdx = Array2::<f64>::zeros((0, 0));
        for k in 0..k_count {
            // u_k = X B_k is the eta-space sensitivity for rho_k.
            // The first Hessian derivative is
            //   H_k = A_k + X' diag(c ⊙ u_k) X.
            let mut hk = a_k_mats[k].clone();
            hk += &Self::xt_diag_x_dense_into(x_dense, &s_cols[k], &mut weighted_xtdx);
            if let (Some(op), Some(dirs)) = (firth_op.as_ref(), firth_dirs.as_ref()) {
                // Firth path:
                //   H_k <- H_k - D H_φ[B_k].
                // This is the exact substitution in
                //   H_k = A_k + D(X'WX)[B_k] - D(H_φ)[B_k].
                hk -= &Self::firth_hphi_direction(op, &dirs[k]);
            }
            h_k.push(hk);
        }

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
        enforce_symmetry(&mut hessian_rho);
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
}
