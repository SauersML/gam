use super::*;

impl<'a> RemlState<'a> {
    /// Compute the scalar outer objective value used by the planner-selected
    /// outer optimizer.
    ///
    /// Full objective reference.
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
        self.compute_cost_with_ext_count(p, 0)
    }

    /// Per-ρ-coordinate canonical keys in NATIVE (formula) order.
    ///
    /// `keys[i]` is a stable, formula-order-independent fingerprint of the
    /// penalty coordinate that ρ-coordinate `i` controls. The outer REML driver
    /// uses these to present an identical *canonical* coordinate layout to the
    /// smoothing-parameter optimizer no matter which order the user wrote the
    /// smooth terms / tensor margins — so the flat double-penalty REML valley is
    /// resolved order-invariantly (#1538/#1539).
    ///
    /// Each key combines two formula-order-independent signatures:
    ///
    ///  1. The penalty's STRUCTURAL key (see
    ///     [`super::reml_outer_engine::PenaltyCoordinate::canonical_structural_key`])
    ///     — distinguishes a range block from its null-space ridge, and one
    ///     tensor margin's penalty family from another.
    ///  2. A DATA-dependent block signature: the outer REML gradient evaluated
    ///     once at the symmetric reference ρ = 0 (all λ = 1). `g[i]` depends only
    ///     on block `i`'s data + penalty, so it carries the SAME value for, e.g.,
    ///     `s(x)`'s range coordinate whether `s(x)` was written first or second.
    ///     This is what breaks the x↔z symmetry when two smooths share an
    ///     identical spline basis (so their penalty matrices — and hence their
    ///     structural keys — are equal): without it the canonical order would be
    ///     ambiguous and the optimizer's order-dependence would survive.
    ///
    /// Returns `None` when the penalty-coordinate count does not match the
    /// expected ρ dimension, or when the reference gradient cannot be evaluated
    /// (so the caller falls back to the native layout untouched rather than risk
    /// a misaligned permutation). For Gaussian additive `s()` smooths the
    /// coordinates are the (range + null-space) double-penalty blocks; for `te()`
    /// they are the per-margin Kronecker coordinates plus the appended
    /// null-space ridge.
    pub(crate) fn canonical_rho_keys(&self, expected_rho_dim: usize) -> Option<Vec<u64>> {
        use std::hash::{Hash, Hasher};
        let coords = self.build_penalty_coords();
        if coords.len() != expected_rho_dim || expected_rho_dim == 0 {
            return None;
        }
        // Symmetric reference point ρ = 0 (every λ = 1): identical for every
        // term order, so the per-coordinate gradient there is a clean
        // block-attached, order-invariant data signature. A failure to evaluate
        // (degenerate inner solve) disables canonicalization rather than
        // permuting on an unreliable signature.
        let reference = ndarray::Array1::<f64>::zeros(expected_rho_dim);
        let grad = self.compute_gradient(&reference).ok()?;
        if grad.len() != expected_rho_dim || !grad.iter().all(|v| v.is_finite()) {
            return None;
        }
        // Quantize the data signature to a coarse relative grid so floating
        // round-off cannot split two coordinates that are genuinely identical.
        let quant = |v: f64| -> i64 {
            if !v.is_finite() || v.abs() <= 1e-300 {
                0
            } else {
                (v * 1.0e6).round() as i64
            }
        };
        Some(
            coords
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    c.canonical_structural_key().hash(&mut hasher);
                    quant(grad[i]).hash(&mut hasher);
                    hasher.finish()
                })
                .collect(),
        )
    }

    pub(crate) fn compute_cost_with_ext_count(
        &self,
        p: &Array1<f64>,
        synthetic_ext_count: usize,
    ) -> Result<f64, EstimationError> {
        let cost_call_idx = {
            let mut calls = self.arena.cost_eval_count.write().unwrap();
            *calls += 1;
            *calls
        };
        let t_eval_start = std::time::Instant::now();
        {
            let prefix: Vec<String> = p.iter().take(4).map(|v| format!("{:.3}", v)).collect();
            log::debug!(
                "[REML] eval#{} begin cost-only | rho[..4]=[{}] | k={}",
                cost_call_idx,
                prefix.join(","),
                p.len()
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            log::debug!(
                "[REML] eval#{} cache hit | cost {:.6e} | elapsed {:.1}ms",
                cost_call_idx,
                eval.cost,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(eval.cost);
        }
        // Cost-order short-circuit (#778). The configured/soft ρ-prior cost is
        // the *cheapest* additive term in the objective — `O(K)`, a function of
        // ρ alone, with no dependence on the inner P-IRLS solve. The unified
        // evaluator forms the final cost as
        //   V(ρ) = data_term(ρ) + prior_cost(ρ) + barrier_cost(β̂),
        // where `data_term` (the inner solve + the `O(K³)` penalty/Hessian
        // log-determinants) is the dominant cost and is always finite for a
        // feasible inner solve, while `prior_cost` saturates to `+∞` whenever ρ
        // leaves the prior's support (Saturate policy, see
        // `evaluate_configured_rho_prior`). When the cheap prior term is already
        // non-finite the sum is `+∞` regardless of the data term, so the outer
        // optimizer will reject this step — evaluate it first and return `+∞`
        // without paying for the inner P-IRLS solve or the log-determinant
        // assembly. This is exact, not an approximation: it reproduces the value
        // the full path would return (`build_prior` projects the identical
        // configured-prior cost from `ConfiguredRhoPriorAtom` and adds the soft
        // guard prior).
        let prior_cost =
            self.soft_rho_guard_prior_atom(p).cost() + self.configured_rho_prior_atom(p).cost();
        if !prior_cost.is_finite() {
            log::debug!(
                "[REML] eval#{} prior short-circuit | prior_cost {:.6e} | rejecting step \
                 without inner solve | elapsed {:.1}ms",
                cost_call_idx,
                prior_cost,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            // Out-of-support ρ saturates the prior to `+∞` (never `−∞`/`NaN`,
            // since the soft prior is finite and non-negative and the configured
            // prior's Saturate policy folds to `+∞`); return the `+∞` retreat
            // signal explicitly to match the `obtain_eval_bundle` failure paths
            // below that also return `f64::INFINITY` on an infeasible step.
            return Ok(f64::INFINITY);
        }
        let t_pirls = std::time::Instant::now();
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                // Inner linear algebra says "too singular" — treat as barrier.
                log::debug!(
                    "P-IRLS flagged ill-conditioning for current rho; returning +inf cost to retreat."
                );
                // Diagnostics: which rho are at bounds
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Ok(f64::INFINITY);
            }
            Err(EstimationError::PerfectSeparationDetected { .. })
            | Err(EstimationError::PirlsDidNotConverge { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                log::debug!(
                    "P-IRLS separation/non-convergence at current rho; returning +inf cost to retreat."
                );
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                let (at_lower, at_upper) = boundary_hit_indices(p.view(), RHO_BOUND, 1e-8);
                if !(at_lower.is_empty() && at_upper.is_empty()) {
                    log::debug!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower,
                        at_upper
                    );
                }
                return Err(e);
            }
        };
        let pirls_ms = t_pirls.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "[REML] eval#{} pirls done | elapsed {:.1}ms | backend {:?}",
            cost_call_idx,
            pirls_ms,
            bundle.backend_kind()
        );
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            let t_assemble = std::time::Instant::now();
            let result = if synthetic_ext_count == 0 {
                self.evaluate_unified_sparse(
                    p,
                    &bundle,
                    super::reml_outer_engine::EvalMode::ValueOnly,
                )?
            } else {
                self.evaluate_unified_value_only_with_synthetic_ext_count(
                    p,
                    &bundle,
                    synthetic_ext_count,
                    true,
                )?
            };
            let cost = screening_residual_penalty(result.cost, bundle.pirls_result.as_ref());
            log::debug!(
                "[REML] eval#{} sparse cost {:.6e} | assemble {:.1}ms | total {:.1}ms",
                cost_call_idx,
                cost,
                t_assemble.elapsed().as_secs_f64() * 1000.0,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(cost);
        }
        {
            // Validation and diagnostics (before delegating to unified evaluator).
            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_used = bundle.ridge_passport.delta;

            if !p.is_empty() {
                let k_lambda = p.len();
                let k_r = pirls_result.reparam_result.canonical_transformed.len();
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

            const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
            // Hot diagnostics walk the Hessian eigenspectrum and emit
            // ill-conditioning warnings. They are only meaningful for fully
            // converged inner modes — partial fits accepted from seed
            // screening (cap=3) routinely yield indefinite Hessians, and
            // surfacing those as `Penalized Hessian not PD` warnings would
            // confuse log readers and mask real production-fit issues.
            let want_hot_diag = !pirls_result.status.is_failed_max_iterations()
                && self.should_compute_hot_diagnostics(cost_call_idx);
            if ridge_used > 0.0 && want_hot_diag {
                // Eigenvalue diagnostics require dense; only pay the cost when
                // hot diagnostics are requested and ridge was applied.
                let pht_dense = pirls_result.penalized_hessian_transformed.to_dense();
                if let Ok((eigs, _)) = pht_dense.eigh(Side::Lower)
                    && let Some(min_eig) = eigs.iter().cloned().reduce(f64::min)
                {
                    if gam_problem::diagnostics::should_emit_h_min_eig_diag(min_eig) {
                        log::debug!(
                            "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                            min_eig,
                            ridge_used
                        );
                    }
                    if min_eig <= 0.0 {
                        log::warn!(
                            "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                            ridge_used
                        );
                    }
                    if !min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE {
                        let condition_number = symmetric_spectrum_condition_number(&pht_dense);
                        log::warn!(
                            "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                            condition_number
                        );
                    }
                }
            }
        }
        // Delegate to the unified evaluator for the actual formula computation.
        // This ensures cost and gradient share the exact same formula.
        let t_assemble = std::time::Instant::now();
        let result = if synthetic_ext_count == 0 {
            self.evaluate_unified(p, &bundle, super::reml_outer_engine::EvalMode::ValueOnly)?
        } else {
            self.evaluate_unified_value_only_with_synthetic_ext_count(
                p,
                &bundle,
                synthetic_ext_count,
                false,
            )?
        };
        let cost = screening_residual_penalty(result.cost, bundle.pirls_result.as_ref());
        log::debug!(
            "[REML] eval#{} dense cost {:.6e} | assemble {:.1}ms | total {:.1}ms",
            cost_call_idx,
            cost,
            t_assemble.elapsed().as_secs_f64() * 1000.0,
            t_eval_start.elapsed().as_secs_f64() * 1000.0
        );
        Ok(cost)
    }

    /// Seed-screening ranking proxy.
    ///
    /// In screening mode (`screening_max_inner_iterations > 0`), runs the
    /// inner P-IRLS solve under the active iteration cap and returns the
    /// minimum penalized deviance (`-2·log L + βᵀSβ`, plus the structural
    /// ridge contribution carried in `stable_penalty_term`) observed across
    /// all inner iterations. Penalized deviance descends monotonically along
    /// any inner descent path P-IRLS takes, so this minimum is a meaningful
    /// quality signal even when the inner solver was capped before reaching
    /// the mode — strictly better than ranking by the partial-fit V_LAML
    /// criterion, whose `0.5·log|H|` term is dominated by noise at a
    /// poorly-conditioned partial β̂.
    ///
    /// Outside of screening mode this delegates to [`Self::compute_cost`] so
    /// the optimization objective itself is never changed by this method's
    /// presence.
    pub fn compute_screening_proxy(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
        let in_screening = self.screening_max_inner_iterations.load(Ordering::Relaxed) > 0;
        if !in_screening {
            return self.compute_cost(p);
        }
        // Use the same bundle pipeline as `compute_cost`, but read the proxy
        // directly from `pirls_result.min_penalized_deviance` instead of
        // assembling the LAML criterion. Bundle assembly already runs
        // P-IRLS under the screening cap; reusing it keeps the screening
        // path's behavioural invariants (no warm-start update, no LRU write,
        // no KKT enforcement at partial fits).
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(err) if err.is_inner_solve_retreat() => {
                self.cache_manager.invalidate_eval_bundle();
                return Ok(f64::INFINITY);
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };
        let proxy = bundle.pirls_result.min_penalized_deviance;
        if proxy.is_finite() {
            Ok(proxy)
        } else {
            Ok(f64::INFINITY)
        }
    }

    ///
    /// Exact non-Laplace evidence identities (reference comments; not runtime path)
    /// We optimize a Laplace-style outer objective for scalability, but the exact
    /// marginal likelihood for non-Gaussian models can be written analytically as:
    ///
    ///   L(ρ) = ∫ exp(l(β) - 0.5 βᵀ S(ρ) β) dβ,   S(ρ)=Σ_k exp(ρ_k) S_k.
    ///
    /// Universal exact gradient identity (when differentiation under the integral
    /// is justified and L(ρ) < ∞):
    ///
    ///   ∂_{ρ_k} log L(ρ)
    ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ (β-μ_k)ᵀ S_k (β-μ_k) ].
    ///
    /// Laplace bridge to implemented terms:
    /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
    ///     E[(β-μ_k)ᵀ S_k(β-μ_k)] ≈ (β̂-μ_k)ᵀ S_k(β̂-μ_k) + tr(H^{-1} S_k),
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
    /// Full outer-derivative reference (exact system, sign convention used here).
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
    /// This is the core gradient evaluator for outer optimization. The
    /// centralized planner may feed it to ARC, BFGS, or an EFS variant,
    /// depending on available curvature information.
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
    /// The gradient ∇Cost(ρ) is computed term-by-term. For the penalized
    /// likelihood part of the objective, the **envelope theorem** removes
    /// the indirect beta-mode derivative at an exact inner optimum. This is
    /// an objective-gradient statement only: a generic downstream VJP that
    /// explicitly seeds β̂ or μ̂ still has to propagate those seeds through
    /// the inner KKT system.
    ///
    /// # Mathematical Basis (Non-Gaussian/LAML Case)
    ///
    /// For non-Gaussian models, the envelope theorem still removes the
    /// beta-mode derivative of the penalized likelihood part at stationarity,
    /// but it does not remove the beta-dependence of the Laplace determinant.
    /// Exact LAML therefore differentiates the observed IRLS Hessian
    /// H_obs(β̂,ρ), including its curvature derivatives. Gaussian/identity
    /// and frozen-quadratic PIRLS are the cases where those terms vanish.
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
    /// The gradient calculation requires careful application of the chain rule
    /// and envelope theorem due to the nested optimization structure of GAMs:
    ///
    /// - The inner loop (P-IRLS) finds coefficients β̂ that maximize the penalized log-likelihood
    ///   for a fixed set of smoothing parameters ρ.
    /// - The outer loop finds smoothing parameters ρ that maximize the
    ///   marginal likelihood using the centralized outer-strategy planner.
    ///
    /// Since β̂ is an implicit function of ρ, the total derivative is:
    ///
    ///    dV_R/dρ_k = (∂V_R/∂β̂)ᵀ(∂β̂/∂ρ_k) + ∂V_R/∂ρ_k
    ///
    /// For the profiled penalized-likelihood terms, (∂V_R/∂β̂) = 0 at the
    /// optimum β̂, so that part's indirect term vanishes. Determinant terms
    /// and explicit downstream β̂/μ̂ losses are handled separately.
    ///
    /// # Key Distinction Between REML and LAML Gradients
    ///
    /// - Gaussian (REML): the Hessian-side curvature is beta-independent, so
    ///   the determinant drift is just λ_k S_k and the indirect β̂ terms
    ///   vanish from the outer objective gradient.
    /// - Non-Gaussian (LAML): the determinant differentiates
    ///   H_obs(β̂,ρ). The implementation keeps the β̂-sensitivity term
    ///   through c/d/e curvature arrays rather than pretending the logdet
    ///   only sees λ_k S_k.
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
    //          The minimized Laplace cost contains +½ log |H_p| with
    //          H_p = Xᵀ W_obs(β̂) X + S_λ. Because W_obs depends on β̂,
    //          the total derivative includes dW_obs/dλₖ via β̂.
    //          Differentiating the optimality condition for β̂ gives
    //          ∂β̂/∂λₖ = −H_p⁻¹ Sₖ β̂. The penalized log-likelihood part
    //          still obeys the envelope theorem, so its direct λ derivative
    //          is −½ β̂ᵀ Sₖ β̂ with no implicit beta term.
    //          The resulting cost gradient combines four pieces:
    //            +½ λₖ β̂ᵀ Sₖ β̂
    //            +½ λₖ tr(H_p⁻¹ Sₖ)
    //            +½ tr(H_p⁻¹ Xᵀ ∂W/∂λₖ X)
    //            −½ λₖ tr(S_λ⁺ Sₖ)
    //
    // Stage: Remember that the sign of ∂β̂/∂λₖ matters; from the implicit-function theorem the linear solve reads
    //     −H_p (∂β̂/∂λₖ) = λₖ Sₖ β̂, giving the minus sign used above.  With that sign the indirect and
    //     direct quadratic pieces are exact negatives, which is what the algebra requires.
    /// Build the exact Jeffreys operator in the same coefficient basis used by
    /// the outer Hessian operator and derivative provider.
    ///
    /// If the outer solve is projected to a free subspace `β = Z β_free`, the
    /// Jeffreys term must also be evaluated on `X_eff = X Z`. Reusing the
    /// cached full-basis operator in that situation would recreate the original
    /// value/derivative mismatch this code is meant to eliminate.
    pub(crate) fn build_dense_firth_operator_for_outer_basis(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
        free_basis_opt: &Option<Array2<f64>>,
    ) -> Result<Option<std::sync::Arc<super::FirthDenseOperator>>, EstimationError> {
        let Some(jeffreys_link) = reml_robust_jeffreys_link(&self.config) else {
            return Ok(None);
        };

        if let Some(z) = free_basis_opt.as_ref() {
            let x_projected = pirls_result.x_transformed.to_dense().dot(z);
            if !super::firth_problem_scale_allows(x_projected.nrows(), x_projected.ncols()) {
                log::info!(
                    "disabling Firth bias reduction for projected outer basis (n={}, p={}, n*p={}, n*p^2={}): \
                     exact Firth operator is small-model-only",
                    x_projected.nrows(),
                    x_projected.ncols(),
                    x_projected.nrows().saturating_mul(x_projected.ncols()),
                    x_projected
                        .nrows()
                        .saturating_mul(x_projected.ncols())
                        .saturating_mul(x_projected.ncols()),
                );
                return Ok(None);
            }
            return Ok(Some(std::sync::Arc::new(
                Self::build_firth_dense_operator_for_link(
                    &jeffreys_link,
                    &x_projected,
                    &pirls_result.final_eta.to_owned(),
                    self.weights,
                )?,
            )));
        }

        if let Some(cached) = bundle.firth_dense_operator.clone() {
            return Ok(Some(cached));
        }

        let x_dense = pirls_result.x_transformed.to_dense();
        Ok(Some(std::sync::Arc::new(
            Self::build_firth_dense_operator_for_link(
                &jeffreys_link,
                &x_dense,
                &pirls_result.final_eta.to_owned(),
                self.weights,
            )?,
        )))
    }

    /// Build the derivative provider, dispersion handling, zeroed TK placeholders,
    /// exact Firth operator, log-likelihood, and barrier config
    /// that are common to all dense evaluation paths.
    ///
    /// # Arguments
    /// - `pirls_result`: the inner-loop solution
    /// - `bundle`: the evaluation bundle (carries h_total, firth operator, etc.)
    /// - `free_basis_opt`: optional constraint-projection basis
    /// - `include_firth_derivs`: whether to wrap the derivative provider with
    ///   Firth-aware derivatives
    ///
    /// The actual Tierney-Kadane correction is added outside the unified
    /// evaluator so value, gradient, and Hessian can all come from the same
    /// invariant scalar correction.
    pub(crate) fn build_dense_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
        free_basis_opt: &Option<Array2<f64>>,
        include_firth_derivs: bool,
    ) -> Result<DerivativeContext, EstimationError> {
        use super::reml_outer_engine::{
            DispersionHandling, GaussianDerivatives, SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = reml_is_gaussian_identity(&pirls_result.likelihood);
        let firth_op =
            self.build_dense_firth_operator_for_outer_basis(pirls_result, bundle, free_basis_opt)?;

        // Derivative provider.
        let firth_active_for_derivs = include_firth_derivs && firth_op.is_some();
        let deriv_provider: Box<dyn super::reml_outer_engine::HessianDerivativeProvider> =
            if is_gaussian_identity || pirls_result.derivatives_unsupported {
                Box::new(GaussianDerivatives)
            } else {
                let (hessian_weights, c_array, d_array) =
                    self.hessian_surface_arrays(pirls_result)?;
                let x_transformed = if let Some(z) = free_basis_opt.as_ref() {
                    // Project the design: X_proj = X Z
                    let x_dense = pirls_result.x_transformed.to_dense();
                    gam_linalg::matrix::DesignMatrix::Dense(
                        gam_linalg::matrix::DenseDesignMatrix::from(x_dense.dot(z)),
                    )
                } else {
                    pirls_result.x_transformed.clone()
                };
                let base = SinglePredictorGlmDerivatives {
                    c_array,
                    d_array: Some(d_array),
                    hessian_weights,
                    x_transformed,
                };
                if firth_active_for_derivs {
                    if let Some(firth_op) = firth_op.clone() {
                        Box::new(super::reml_outer_engine::FirthAwareGlmDerivatives {
                            base,
                            firth_op,
                        })
                    } else {
                        Box::new(base)
                    }
                } else {
                    Box::new(base)
                }
            };

        // Dispersion handling.
        let dispersion = if is_gaussian_identity {
            DispersionHandling::ProfiledGaussian
        } else {
            DispersionHandling::Fixed {
                phi: reml_fixed_glm_dispersion(&pirls_result.likelihood),
                include_logdet_h: true,
                include_logdet_s: true,
            }
        };

        // For the Gaussian family the omitting-constants log-likelihood is
        // exactly `-0.5 * deviance` for any phi (deviance = Σ w·(y-μ)²/phi,
        // log-lik = Σ -0.5·w·(y-μ)²/phi). PIRLS already carries the k-space
        // `deviance` on `pirls_result`, computed from the sufficient statistics
        // on the #1033 ψ-tensor fast path where `finalmu` is a STALE reference
        // surface (`= offset`). Recomputing the log-likelihood from `finalmu`
        // here would (a) cost an O(n) row reduction every outer trial and
        // (b) read the stale rows, yielding a β̂/ψ-independent wrong value.
        // Reuse the cached deviance to stay n-free AND correct; this is
        // bit-identical to the recompute whenever the rows are live. Non-
        // Gaussian families never arm the n-free cache, so they recompute.
        let log_likelihood = if is_gaussian_identity {
            -0.5 * pirls_result.deviance
        } else {
            crate::pirls::calculate_loglikelihood_omitting_constants(
                self.y,
                &pirls_result.finalmu.to_owned(),
                &pirls_result.likelihood,
                self.weights,
            )
        };

        // Construct barrier config for monotonicity constraints when no
        // active-set projection is in effect (barrier indices are in the
        // full transformed-coefficient space).
        let barrier_config = if free_basis_opt.is_none() {
            pirls_result
                .linear_constraints_transformed
                .as_ref()
                .and_then(Self::barrier_config_from_constraints)
        } else {
            None
        };

        Ok(DerivativeContext {
            deriv_provider,
            dispersion,
            log_likelihood,
            firth_op,
            barrier_config,
        })
    }

    /// Build the dispersion handling, derivative provider, log-likelihood,
    /// exact Firth operator, and barrier config that are common
    /// to both sparse evaluation paths (`evaluate_unified_sparse` and
    /// `evaluate_efs` when sparse-dispatched).
    ///
    /// TK correction is NOT included here because the sparse paths compute
    /// it via sparse Cholesky solves with completely different logic.
    pub(crate) fn build_sparse_derivative_context(
        &self,
        pirls_result: &PirlsResult,
        bundle: &EvalShared,
    ) -> Result<DerivativeContext, EstimationError> {
        use super::reml_outer_engine::{
            DispersionHandling, FirthAwareGlmDerivatives, GaussianDerivatives,
            SinglePredictorGlmDerivatives,
        };

        let is_gaussian_identity = reml_is_gaussian_identity(&pirls_result.likelihood);

        // Sparse exact still uses the same dense Jeffreys operator; only the
        // H^{-1} applications move to the sparse Cholesky operator.
        let firth_op = if let Some(jeffreys_link) = reml_robust_jeffreys_link(&self.config) {
            if let Some(cached) = bundle.firth_dense_operator_original.clone() {
                Some(cached)
            } else {
                let x_dense = self
                    .x()
                    .try_to_dense_arc(
                        "sparse exact REML runtime requires dense design for Firth operator",
                    )
                    .map_err(EstimationError::InvalidInput)?;
                Some(std::sync::Arc::new(
                    Self::build_firth_dense_operator_for_link(
                        &jeffreys_link,
                        x_dense.as_ref(),
                        &pirls_result.final_eta.to_owned(),
                        self.weights,
                    )?,
                ))
            }
        } else {
            None
        };

        // Dispersion and derivative provider depend on family.
        let (dispersion, deriv_provider): (
            _,
            Box<dyn super::reml_outer_engine::HessianDerivativeProvider>,
        ) = if is_gaussian_identity {
            (
                DispersionHandling::ProfiledGaussian,
                Box::new(GaussianDerivatives),
            )
        } else if pirls_result.derivatives_unsupported {
            (
                DispersionHandling::Fixed {
                    phi: reml_fixed_glm_dispersion(&pirls_result.likelihood),
                    include_logdet_h: true,
                    include_logdet_s: true,
                },
                Box::new(GaussianDerivatives),
            )
        } else {
            let (hessian_weights, c_array, d_array) = self.hessian_surface_arrays(pirls_result)?;
            (
                DispersionHandling::Fixed {
                    phi: reml_fixed_glm_dispersion(&pirls_result.likelihood),
                    include_logdet_h: true,
                    include_logdet_s: true,
                },
                {
                    let base = SinglePredictorGlmDerivatives {
                        c_array,
                        d_array: Some(d_array),
                        hessian_weights,
                        x_transformed: self.x().clone(),
                    };
                    // Match the dense exact path: when Firth-logit is
                    // active, the directional log|H_total| drift includes
                    // -D(H_phi)[B_k]. The sparse backend differs only in how
                    // B_k = H^{-1} rhs is solved.
                    if let Some(firth_op) = firth_op.clone() {
                        Box::new(FirthAwareGlmDerivatives { base, firth_op })
                    } else {
                        Box::new(base)
                    }
                },
            )
        };

        // Gaussian: omitting-constants log-likelihood ≡ -0.5·deviance for any
        // phi. PIRLS carries the k-space `deviance` on the result; on the #1033
        // ψ-tensor fast path `finalmu` is a STALE reference surface (= offset),
        // so recomputing from it would be an O(n) row reduction AND a wrong,
        // β̂/ψ-independent value. Reuse the cached deviance: n-free + correct,
        // bit-identical to the recompute when the rows are live. Non-Gaussian
        // families never arm the n-free cache, so they keep the row recompute.
        let log_likelihood = if is_gaussian_identity {
            -0.5 * pirls_result.deviance
        } else {
            crate::pirls::calculate_loglikelihood_omitting_constants(
                self.y,
                &pirls_result.finalmu.to_owned(),
                &pirls_result.likelihood,
                self.weights,
            )
        };

        // Construct barrier config for monotonicity constraints.
        // The sparse path operates in the original (non-reparameterized)
        // coordinate system, so use the original linear constraints.
        let barrier_config = self
            .linear_constraints
            .as_ref()
            .and_then(Self::barrier_config_from_constraints);

        Ok(DerivativeContext {
            deriv_provider,
            dispersion,
            log_likelihood,
            firth_op,
            barrier_config,
        })
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified inner-solution assembly
    // ═══════════════════════════════════════════════════════════════════════════

    /// Build penalty coordinates from canonical penalties, with Kronecker
    /// fast-path when available and active.
    pub(crate) fn build_penalty_coords(&self) -> Vec<super::reml_outer_engine::PenaltyCoordinate> {
        if let Some(ref kron) = self.kronecker_penalty_system
            && self.kronecker_factored.is_some()
        {
            let d = kron.ndim();
            let total_dim = kron.p_total();
            let eigenvalues: Vec<ndarray::Array1<f64>> = kron
                .marginal_eigensystems
                .iter()
                .map(|(evals, _)| evals.clone())
                .collect();
            let mut coords = Vec::with_capacity(kron.num_penalties());
            for k in 0..d {
                coords.push(
                    super::reml_outer_engine::PenaltyCoordinate::KroneckerMarginal {
                        eigenvalues: eigenvalues.clone(),
                        dim_index: k,
                        marginal_dims: kron.marginal_dims.clone(),
                        total_dim,
                    },
                );
            }
            if kron.has_double_penalty {
                let identity_root = ndarray::Array2::<f64>::eye(total_dim);
                coords.push(
                    super::reml_outer_engine::PenaltyCoordinate::from_dense_root(identity_root),
                );
            }
            return coords;
        }
        self.canonical_penalties
            .iter()
            .map(|cp| cp.to_penalty_coordinate())
            .collect()
    }

    /// The penalized inner KKT residual `r = ∇_β L_pen(β̂)` at the accepted
    /// P-IRLS iterate, in the STABLE/TRANSFORMED coefficient basis, for engaging
    /// the inner-KKT envelope correction `Ṽ = V − ½·rᵀH⁻¹r` on the flexible-link
    /// outer path.
    ///
    /// The correction is engaged UNCONDITIONALLY whenever the residual is
    /// well-defined — it is the exact second-order value/gradient at the
    /// Newton-refined mode `β* = β̂ − H⁻¹r`, and it vanishes as `r → 0`, so a
    /// fully-converged inner solve is unchanged while a β̂ accepted at a
    /// first-order inner cap (the `outer_inner_cap` schedule the flexible-link
    /// optimizer uses) gets the stationary-mode gradient the raw capped β̂
    /// silently misreports. There is deliberately NO relative-residual gate: a
    /// threshold that suppressed the correction above some tolerance would
    /// silently present a non-stationary iterate as exact-KKT to the outer
    /// evaluator (gam#2062), which is exactly the failure this fix removes.
    ///
    /// Returns `None` (presenting exact-KKT, as before) only when the correction
    /// is genuinely inapplicable:
    ///   * Firth bias reduction is active — the penalized Hessian is a
    ///     reduced-rank Jeffreys-identifiable pseudo-object carrying the Firth
    ///     score; that correction needs the `HardPseudo` kernel and is out of
    ///     scope here (tracked separately as gam#1821);
    ///   * the stored residual is empty or non-finite.
    fn standard_inner_kkt_residual_transformed(
        &self,
        pirls_result: &PirlsResult,
        firth_active: bool,
    ) -> Option<Array1<f64>> {
        if firth_active {
            return None;
        }
        let r = &pirls_result.penalized_gradient_transformed;
        if r.is_empty() || r.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(r.clone())
    }

    /// Frame-mapped inner KKT residual for the ORIGINAL-basis assembly builders
    /// (`build_dense_original_assembly`, `build_sparse_assembly`). The stored
    /// residual lives in the transformed frame; the penalized score rotates with
    /// the orthogonal reparameterization exactly as `β` does (`β_o = Qs·β_t`),
    /// so `r_o = Qs·r_t`. These builders are only reached on the unconstrained
    /// QS frame, so there is no active-constraint normal component to strip.
    fn inner_kkt_residual_original_basis(
        &self,
        pirls_result: &PirlsResult,
        firth_active: bool,
    ) -> Option<crate::model_types::ProjectedKktResidual> {
        let r_t = self.standard_inner_kkt_residual_transformed(pirls_result, firth_active)?;
        let r_o = match pirls_result.coordinate_frame {
            pirls::PirlsCoordinateFrame::OriginalSparseNative => r_t,
            pirls::PirlsCoordinateFrame::TransformedQs => pirls_result.reparam_result.qs.dot(&r_t),
        };
        Some(crate::model_types::ProjectedKktResidual::from_active_projected(
            r_o,
        ))
    }

    /// Pack a `DerivativeContext` plus backend-specific pieces into an
    /// `InnerAssembly`. All three assembly builders (`build_dense_assembly`,
    /// `build_sparse_assembly`, `build_dense_original_assembly`) delegate
    /// here to avoid repeating the 18-field struct literal.
    pub(crate) fn finish_assembly(
        &self,
        pirls_result: &PirlsResult,
        ctx: DerivativeContext,
        hessian_op: std::sync::Arc<dyn super::reml_outer_engine::HessianOperator>,
        beta: Array1<f64>,
        penalty_logdet: super::reml_outer_engine::PenaltyLogdetDerivs,
        nullspace_dim: f64,
        hessian_logdet_correction: f64,
        penalty_subspace_trace: Option<
            std::sync::Arc<super::reml_outer_engine::PenaltySubspaceTrace>,
        >,
        free_basis: Option<&Array2<f64>>,
        inner_kkt_residual: Option<crate::model_types::ProjectedKktResidual>,
    ) -> super::assembly::InnerAssembly<'static> {
        // When a linear-inequality active set reduces the inner solve to the
        // free subspace `β = z β_f`, the penalty coordinates must be restricted
        // onto the same subspace so each `coord.dim()` matches the reduced
        // `beta.len()` that `InnerSolutionBuilder::build` asserts. The Hessian
        // operator and `e_for_logdet` are already projected by the caller; this
        // moves the penalty roots in lockstep (`R_k → R_k z`).
        //
        // Frame consistency (#509 second face): the free basis `z`, the
        // projected Hessian `ZᵀHZ`, the projected design `XZ`, and the reduced
        // `β = Zᵀβ_transformed` all live in the TRANSFORMED (post-Qs / post
        // box-reparam `T`) PIRLS frame. The penalty coordinates feed the inner
        // penalty quadratic `½βᵀS_kβ` and the IFT mode-response RHS `S_kβ`, so
        // they must live in that same transformed frame; otherwise a
        // non-orthogonal reparameterization (cumulative-sum `T` and the
        // stabilizing `Qs` for a monotone box-reparam smooth) desyncs the outer
        // gradient from the cost (analytic ≠ central-difference) and the Arc
        // trust-region rejects every step. `build_penalty_coords()` returns the
        // ORIGINAL-frame (pre-Qs) roots; under an active set, project the
        // TRANSFORMED-frame `reparam_result.canonical_transformed` roots instead
        // (the same per-component roots the projected `log|S|₊` derivatives now
        // read). When `Qs = I` (no reparameterization) the two frames coincide,
        // so this is a no-op for the ordinary active-set paths.
        let penalty_coords = match free_basis {
            Some(z) => {
                let original_coords = self.build_penalty_coords();
                let transformed = &pirls_result.reparam_result.canonical_transformed;
                let base_coords: Vec<_> = if transformed.len() == original_coords.len() {
                    transformed
                        .iter()
                        .map(|cp| cp.to_penalty_coordinate())
                        .collect()
                } else {
                    original_coords
                };
                base_coords
                    .iter()
                    .map(|coord| coord.project_into_subspace(z))
                    .collect()
            }
            None => self.build_penalty_coords(),
        };
        super::assembly::InnerAssembly {
            log_likelihood: ctx.log_likelihood,
            penalty_quadratic: pirls_result.stable_penalty_term,
            beta,
            // Effective sample size for the REML/LAML criterion. A prior weight
            // of exactly 0 makes a row contribute nothing to any weighted
            // cross-product (XᵀWX, XᵀWy) or to the weighted deviance, so it is
            // statistically equivalent to an absent row. The ONLY channel left
            // by which it could still perturb λ selection / EDF / dispersion is
            // the explicit observation count `n` that enters the profiled-scale
            // denominator `(n − M_p)` and the `(n−M_p)/2·log(2πφ̂)` REML term.
            // Counting zero-weight rows there puts a deviance numerator that
            // already excludes them over a denominator that does not, biasing φ̂
            // low, shifting the selected λ, and shrinking every SE. Use the
            // count of positive-weight rows (R's `n.ok = nobs − Σ[w==0]`, mgcv's
            // dropped zero-weight observations) — identical to the effective `n`
            // estimate.rs uses for the dispersion denominator (#584). With no
            // weights / all weights 1 this equals `self.y.len()`.
            n_observations: self.weights.iter().filter(|&&wi| wi > 0.0).count(),
            hessian_op,
            penalty_coords,
            penalty_logdet,
            dispersion: ctx.dispersion,
            rho_curvature_scale: 1.0,
            // The InnerSolution-carried prior must be the *effective* prior
            // (firth-general PC default substituted for unset coordinates), so
            // the unified EFS/LAML evaluation in `super::reml_outer_engine` shapes the λ
            // update with the same hyperprior the outer cost/gradient use. Using
            // the raw `self.rho_prior` here would leave the firth-general PC
            // default out of the inner update entirely.
            rho_prior: self.effective_rho_prior().into_owned(),
            hessian_logdet_correction,
            penalty_subspace_trace,
            deriv_provider: Some(ctx.deriv_provider),
            firth: ctx
                .firth_op
                .map(crate::estimate::reml::reml_outer_engine::ExactJeffreysTerm::new),
            nullspace_dim: Some(nullspace_dim),
            barrier_config: ctx.barrier_config,
            ext_coords: Vec::new(),
            ext_coord_pair_fn: None,
            rho_ext_pair_fn: None,
            fixed_drift_deriv: None,
            contracted_psi_second_order: None,
            kkt_residual: inner_kkt_residual,
            active_constraints: None,
        }
    }

    /// Build an `InnerAssembly` using the dense-transformed backend.
    ///
    /// Shared ingredient builder for all dense-spectral evaluation paths
    /// (normal, ext-coord, and EFS). Callers may set `ext_coords` and pair
    /// functions on the returned assembly before passing it to
    /// `assemble_and_evaluate` or `assemble_and_evaluate_efs`.
    pub(crate) fn build_dense_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        force_spectral_logdet: bool,
        populate_inner_kkt: bool,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::reml_outer_engine::{
            DenseCholeskyValueOnlyOperator, DenseSpectralOperator, PseudoLogdetMode,
        };
        use std::borrow::Cow;

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        let free_basis_opt = self.active_constraint_free_basis(pirls_result);
        let (h_for_operator, e_for_logdet) = if let Some(z) = free_basis_opt.as_ref() {
            (
                Cow::Owned(Self::projectwith_basis(bundle.h_total.as_ref(), z)),
                Cow::Owned(pirls_result.reparam_result.e_transformed.dot(z)),
            )
        } else {
            (
                Cow::Borrowed(bundle.h_total.as_ref()),
                Cow::Borrowed(&pirls_result.reparam_result.e_transformed),
            )
        };

        // When Firth bias reduction is active, the design column space may be
        // rank-deficient (e.g. aliased covariates).  The PIRLS penalized Hessian
        // H_total = X'WX + S − H_φ then inherits that rank deficiency on any
        // direction that is simultaneously null for X'WX, S, AND H_φ.  Under
        // `Smooth`, the small eigenvalues of H_total get a soft ε-floor that
        // leaks a spurious non-zero gradient contribution through the null
        // direction; both `log|H|` and its derivatives then count a direction
        // that carries no actual curvature.  Switching to `HardPseudo` masks
        // those `σ_j ≤ ε` eigenpairs consistently across `logdet`,
        // `trace_logdet_*`, and `H⁻¹` solves, so log|H|, its ρ-gradient, and
        // its ρ-Hessian all see the same reduced active subspace.
        //
        // Rationale: the Firth/Jeffreys penalty, the penalized score equation,
        // and therefore dβ̂/dρ all live on the identifiable subspace (matching
        // `firth_op.q_basis`); extending the logdet kernel to the null space
        // is both unphysical and inconsistent with the analytic subspace
        // derivatives.
        let hessian_mode = if bundle.firth_dense_operator.is_some() {
            PseudoLogdetMode::HardPseudo
        } else {
            PseudoLogdetMode::Smooth
        };

        let c_nontrivial = pirls_result.solve_c_array.iter().any(|&c| c != 0.0);

        // For ValueOnly evaluations on the SPD fast path (no Firth, no hard
        // linear constraints), use a Cholesky-backed operator.  LLT costs
        // O(p³/3) versus the O(9·p³) full eigendecomposition, giving a
        // multi-× speedup per line-search probe.  ValueOnly returns before
        // any gradient trace call, so the operator only needs to serve
        // `logdet()` and `solve()`/`solve_multi()` — both provided by LLT.
        //
        // Keep c-nontrivial non-Gaussian fits on the spectral path even for
        // value-only probes: their LAML value installs the intrinsic
        // pseudo-logdet correction below, and using the Cholesky full logdet
        // here would make cost-only line-search / FD probes evaluate a
        // different scalar from the value+gradient path (#901).
        //
        // #1376: the Cholesky fast path returns the EXACT log-determinant
        // `Σ ln σ_j`, while the gradient path's `DenseSpectralOperator` returns
        // the smooth-floored `Σ ln r_ε(σ_j)` and differentiates exactly THAT
        // floored object via `tr(G_ε Ḣ)`. The two agree only when every σ_j is
        // safely above the stability floor ε. A design-moving ψ coordinate (the
        // Matérn/Duchon log-κ axis) under the default double-penalty drives the
        // projected-kernel shrinkage block's near-null eigenvalues of
        // `H = XᵀWX + Sλ` down toward ε, so the exact and floored log-dets — and
        // hence their κ-derivatives — diverge. The outer FD audit then central-
        // differences the EXACT-logdet ValueOnly cost while the analytic gradient
        // reports the FLOORED-logdet derivative: an objective↔gradient DESYNC
        // (analytic ≠ FD on `psi_kappa[..]`, the headline #1376 symptom). Gate the
        // Cholesky fast path off whenever the outer vector carries a design-moving
        // ψ coordinate, so value and gradient share ONE smooth-floored `log|H|`.
        // Pure-ρ (penalty-only) fits keep the LLT speedup: their drifts live in
        // `range(Sλ)`, the floored/exact logdets and their ρ-derivatives match,
        // and there is no design-moving axis to desync. The caller passes
        // `force_spectral_logdet = true` exactly when the outer vector carries a
        // design-moving ψ coordinate.
        let hessian_op: std::sync::Arc<dyn super::reml_outer_engine::HessianOperator> = if mode
            == super::reml_outer_engine::EvalMode::ValueOnly
            && matches!(hessian_mode, PseudoLogdetMode::Smooth)
            && free_basis_opt.is_none()
            && !c_nontrivial
            && !force_spectral_logdet
        {
            match DenseCholeskyValueOnlyOperator::from_spd(h_for_operator.as_ref()) {
                Ok(chol_op) => std::sync::Arc::new(chol_op),
                Err(_) => std::sync::Arc::new(
                    DenseSpectralOperator::from_symmetric_with_mode(
                        h_for_operator.as_ref(),
                        hessian_mode,
                    )
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!(
                            "DenseSpectralOperator from PIRLS Hessian: {e}"
                        ))
                    })?,
                ),
            }
        } else {
            std::sync::Arc::new(
                DenseSpectralOperator::from_symmetric_with_mode(
                    h_for_operator.as_ref(),
                    hessian_mode,
                )
                .map_err(|e| {
                    EstimationError::InvalidInput(format!(
                        "DenseSpectralOperator from PIRLS Hessian: {e}"
                    ))
                })?,
            )
        };

        let uses_kron_penalty_logdet = self.kronecker_penalty_system.as_ref().is_some_and(|kron| {
            self.kronecker_factored.is_some() && kron.num_penalties() == rho.len()
        });
        // Only the penalty-side `log|S|₊` machinery consumes the penalty
        // subspace now; the Hessian-side kernel is intrinsic to H_pen (#901)
        // and no longer needs `range(S_+)`.
        let penalty_subspace = if !uses_kron_penalty_logdet {
            Some(self.compute_penalty_subspace(e_for_logdet.as_ref(), ridge_passport)?)
        } else {
            None
        };
        let (penalty_rank, penalty_logdet) = self.dense_penalty_logdet_derivs(
            rho,
            e_for_logdet.as_ref(),
            &[],
            ridge_passport,
            penalty_subspace.as_ref(),
            bundle,
            mode,
            free_basis_opt.as_ref(),
        )?;

        let beta = if let Some(z) = free_basis_opt.as_ref() {
            z.t().dot(pirls_result.beta_transformed.as_ref())
        } else {
            pirls_result.beta_transformed.as_ref().clone()
        };

        let nullspace_dim = h_for_operator.ncols().saturating_sub(penalty_rank) as f64;

        // Rank-deficient LAML fix (#901 corrected object): under `Smooth` the
        // spectral operator's cached logdet sums `ln r_ε(σ_j(H))` over ALL
        // eigenvalues, so genuinely null directions contribute `(p − rank) ·
        // ln ε` — very negative — and make `½(log|H| − log|S|_+)` diverge to
        // −∞ whenever `rank(X'WX) + rank(S) < p`. Replace it with the
        // intrinsic pseudo-logdet `log|H_pen|₊` over `range(H_pen)` (mgcv's
        // generalized determinant) by routing the scalar difference through
        // `hessian_logdet_correction`; `reml_laml_evaluate` already sums
        // `hop.logdet() + correction`.
        //
        // The matching gradient kernel is the SAME spectral object: `H_pen⁺`
        // carried as `PenaltySubspaceTrace { U_H, diag(1/σ) }`, whose trace
        // `tr(H_pen⁺ · Ḣ)` is the exact pseudo-logdet derivative for EVERY
        // drift — ρ-direction penalty drifts `λ_k S_k`, ψ-direction basis
        // drifts `λ_k ∂S_k/∂ψ` (whose `range(Sλ(ψ))` rotates with ψ), and the
        // non-Gaussian IFT correction `D_β H[v] = X' diag(c ⊙ X v) X` (which
        // leaks onto `null(S)` through the intercept column). The previous
        // realization projected value AND kernel onto `range(S_+)` — see
        // `intrinsic_hessian_pseudo_logdet_parts` for why that object drops
        // the θ-dependent Schur curvature `log det(C − BᵀA⁻¹B)` of the
        // penalty-null block and produced sign-flipped ρ-gradients and ~1e5
        // ψ-gradient blow-ups against FD (#901).
        //
        // Under `HardPseudo` the operator already masks null eigenpairs
        // consistently in `logdet`, `trace_logdet_*`, and `solve`, so the
        // correction is zero there and no kernel is needed.
        //
        // Gate on `c_nontrivial`: for canonical Gaussian (Identity link) the
        // IRLS weights are η-independent, `c ≡ 0`, `D_β H ≡ 0`, and the
        // classical Gaussian REML cost identity (`log|H|` through the smooth
        // spectral floor) is already FD-consistent — installing the exact
        // pseudo-logdet there would change the cost surface for no gradient
        // benefit. Every c-nontrivial family (Probit / Logit / cloglog /
        // Poisson / Gamma / SAS / GAMLSS noise blocks / …) gets the intrinsic
        // object unconditionally.
        let (hessian_logdet_correction, penalty_subspace_trace) =
            if matches!(hessian_mode, PseudoLogdetMode::Smooth) && c_nontrivial {
                let (log_det_h_plus, kernel) =
                    Self::intrinsic_hessian_pseudo_logdet_parts(h_for_operator.as_ref())?;
                (
                    log_det_h_plus - hessian_op.logdet(),
                    kernel.map(std::sync::Arc::new),
                )
            } else {
                (0.0, None)
            };

        // #1271 diagnostic: dump the REML logdet internals at every dense
        // evaluation so the rank/logdet mechanism is visible in the test log.
        // Pure logging — no numeric behavior change. Routed through `log::info!`
        // so it is harmless in production (default off) and captured by a
        // test-installed logger in the #1271 probe.
        if log::log_enabled!(log::Level::Info) {
            use faer::Side;
            use gam_linalg::faer_ndarray::FaerEigh;
            let h_logdet = hessian_op.logdet();
            let (h_above_one, h_min, h_max) = match h_for_operator.as_ref().eigh(Side::Lower) {
                Ok((evals, _)) => {
                    let above = evals.iter().filter(|&&s| s > 1.0).count();
                    let mn = evals.iter().copied().fold(f64::INFINITY, f64::min);
                    let mx = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    (above as i64, mn, mx)
                }
                Err(_) => (-1, f64::NAN, f64::NAN),
            };
            let rho_str = rho
                .iter()
                .map(|r| format!("{r:.4}"))
                .collect::<Vec<_>>()
                .join(",");
            let lam_str = rho
                .iter()
                .map(|r| format!("{:.3e}", r.exp()))
                .collect::<Vec<_>>()
                .join(",");
            log::info!(
                "[#1271-diag] p={p} penalty_rank={pr} nullspace_dim={nd} \
                 logS={ls:.6} logH={lh:.6} half_diff={hd:.6} \
                 h_above1={ha} h_min={hmin:.3e} h_max={hmax:.3e} \
                 rho=[{rho_str}] lambda=[{lam_str}]",
                p = h_for_operator.ncols(),
                pr = penalty_rank,
                nd = nullspace_dim,
                ls = penalty_logdet.value,
                lh = h_logdet,
                hd = 0.5 * (h_logdet - penalty_logdet.value),
                ha = h_above_one,
                hmin = h_min,
                hmax = h_max,
            );
        }

        let ctx =
            self.build_dense_derivative_context(pirls_result, bundle, &free_basis_opt, true)?;
        // Inner-KKT envelope residual (transformed frame). Only the
        // unconstrained branch maps a raw transformed residual directly onto the
        // Hessian operator; under an active-set free basis the stationarity
        // vector additionally carries a constraint-normal (Lagrange multiplier)
        // component this standard path does not strip, so present exact-KKT
        // there (unchanged behaviour) rather than a mis-projected residual.
        let inner_kkt_residual = if populate_inner_kkt && free_basis_opt.is_none() {
            self.standard_inner_kkt_residual_transformed(
                pirls_result,
                bundle.firth_dense_operator.is_some(),
            )
            .map(crate::model_types::ProjectedKktResidual::from_active_projected)
        } else {
            None
        };
        Ok(self.finish_assembly(
            pirls_result,
            ctx,
            hessian_op,
            beta,
            penalty_logdet,
            nullspace_dim,
            hessian_logdet_correction,
            penalty_subspace_trace,
            free_basis_opt.as_ref(),
            inner_kkt_residual,
        ))
    }

    /// Build an `InnerAssembly` using the sparse-exact backend.
    ///
    /// Shared ingredient builder for all sparse Cholesky evaluation paths.
    pub(crate) fn build_sparse_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        populate_inner_kkt: bool,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::reml_outer_engine::{
            HessianOperator, PenaltyLogdetDerivs, SparseCholeskyOperator,
        };
        let sparse = bundle.sparse_exact.as_ref().ok_or_else(|| {
            EstimationError::InvalidInput("missing sparse exact evaluation payload".to_string())
        })?;
        let pirls_result = bundle.pirls_result.as_ref();

        let beta = self.sparse_exact_beta_original(pirls_result);
        let p_dim = beta.len();
        let hessian_op: std::sync::Arc<dyn HessianOperator> = {
            let mut op = SparseCholeskyOperator::new(sparse.factor.clone(), sparse.logdet_h, p_dim);
            if let Some(ref taka) = sparse.takahashi {
                op = op.with_takahashi(taka.clone());
            }
            std::sync::Arc::new(op)
        };

        log::trace!(
            "SparseCholeskyOperator: dim={}, active_rank={}",
            hessian_op.dim(),
            hessian_op.active_rank()
        );

        let nullspace_dim = p_dim.saturating_sub(sparse.penalty_rank) as f64;
        let det2 = if mode == super::reml_outer_engine::EvalMode::ValueGradientHessian {
            let lambdas = rho.mapv(f64::exp);
            let (_, det2) =
                self.structural_penalty_logdet_derivatives_block_local(&lambdas, bundle)?;
            Some(det2)
        } else {
            None
        };
        let penalty_logdet = PenaltyLogdetDerivs {
            value: sparse.logdet_s_pos,
            first: sparse.det1_values.as_ref().clone(),
            second: det2,
        };

        let ctx = self.build_sparse_derivative_context(pirls_result, bundle)?;
        // Sparse-exact `log|H|` is Cholesky-derived from `X'WX + S_λ + δI`,
        // not a `PseudoLogdetMode::Smooth` eigenvalue sum, so the
        // `(p − rank) · ln ε` floor leakage that motivated e33c06be on
        // dense cannot occur here: Cholesky either succeeds (H is SPD,
        // logdet sums true log-diagonals) or, when H is genuinely non-PD,
        // `ensure_sparse_positive_definitewithridge` sets a ridge directly
        // from a Gershgorin bound on λ_min (surfaced via `log::warn`) so the
        // shift is provably sufficient rather than escalated blindly.
        //
        // The analogous rank-deficiency concern — that a nonzero ridge
        // δ would contribute `(p − rank(X'WX + S)) · log(δ)` to
        // `logdet_h` while the combined penalty pseudo-logdet excludes that
        // subspace from `log|ΣλS|_+` — is structurally out of reach on the
        // backends that select sparse-exact:
        //
        //   1. `estimate_sparse_native_decision` gates on H density, which
        //      implies large p and typically n ≫ p (large-scale) — i.e.
        //      `X'WX` is rank-full, so `X'WX + S` is rank-full, so
        //      `ridge_used = 0` and no leak term exists.
        //   2. The small-n/high-dim regime that motivated e33c06be (n=120,
        //      k=6 Duchon) is dense territory; it never chooses
        //      `SparseNative` because the sparse density heuristic rejects
        //      near-dense systems.
        //
        // No projection correction is emitted here; pass 0.0 through to
        // `finish_assembly`.  If a future test exercise ever finds a
        // sparse-native fit with `ridge_used > 0` AND genuine null
        // directions of `X'WX + S`, extend the combined penalty pseudo-logdet
        // to include the ridge-matched null contributions (keeping both
        // sides of the LAML ratio on the same p-dim space) rather than
        // reintroducing the range(S_+) projection; Cholesky can't cheaply
        // compute `U_S^T H U_S` without densifying H.
        // Sparse-exact assembles β and H in the original basis (see `beta =
        // sparse_exact_beta_original`), so the envelope residual is mapped to
        // that basis. Sparse-native fits are unconstrained on this path.
        let inner_kkt_residual = if populate_inner_kkt {
            self.inner_kkt_residual_original_basis(
                pirls_result,
                bundle.firth_dense_operator.is_some()
                    || bundle.firth_dense_operator_original.is_some(),
            )
        } else {
            None
        };
        Ok(self.finish_assembly(
            pirls_result,
            ctx,
            hessian_op,
            beta,
            penalty_logdet,
            nullspace_dim,
            0.0,
            None,
            None,
            inner_kkt_residual,
        ))
    }

    /// Build an `InnerAssembly` using the dense original-basis backend.
    ///
    /// Used when QS-transformed coordinates have no active constraints,
    /// allowing us to work in the original basis for better numerical
    /// stability with extended ψ coordinates. TK correction is zeroed
    /// in the assembly and applied post-hoc by `assemble_and_evaluate`.
    pub(crate) fn build_dense_original_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        force_spectral_logdet: bool,
        populate_inner_kkt: bool,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        use super::reml_outer_engine::{DenseSpectralOperator, PseudoLogdetMode};

        let pirls_result = bundle.pirls_result.as_ref();
        let ridge_passport = pirls_result.ridge_passport;

        let mut h_total_original =
            self.bundle_matrix_in_original_basis(pirls_result, bundle.h_total.as_ref());

        let beta = self.sparse_exact_beta_original(pirls_result);

        // Barrier Hessian diagonal: restore ORIGINAL-basis barrier curvature
        // on `h_total_original`.
        //
        // `prepare_dense_eval_bundle` (runtime.rs:2192-2202) adds
        // `τ · diag(1/Δ_j²)` to `bundle.h_total` IN THE TRANSFORMED BASIS,
        // gated on
        // `barrier_config_from_constraints(pirls_result
        //  .linear_constraints_transformed)`.  That gate only accepts
        // simple-bound rows (exactly one nonzero = 1.0 in the transformed
        // constraint matrix).  `build_transformed_lower_bound_constraints`
        // / `build_transformed_linear_constraints` rotate each original
        // simple bound `e_jᵀ β_o ≥ b` into `qs.row(j)ᵀ β_t ≥ b`; for
        // non-trivial `Qs` that row is dense, so the transformed-basis
        // gate typically returns `None` and no barrier is added to
        // `bundle.h_total`.  The edge case where it DOES fire is when
        // `qs.row(j)` happens to equal some standard basis vector `e_kᵀ`
        // (Qs permutation-or-identity on that row), and the bundle then
        // carries `τ / (β_t[k] − b)²` at position `[k, k]` in the
        // transformed Hessian.
        //
        // After `bundle_matrix_in_original_basis` rotates by `Qs · ... ·
        //  Qsᵀ`, any such transformed diagonal increment becomes `Qs ·
        //  diag(τ/Δ_t²) · Qsᵀ` in the original basis — which is a
        // DIFFERENT matrix than the correct original-basis barrier
        // `τ · diag(1/Δ_o²)` at the original simple-bound indices.  Even
        // in the identity-preserving edge case, double-adding is wrong.
        //
        // Clean invariant: strip the rotated transformed-basis barrier
        // first, then apply the ORIGINAL-basis barrier directly.  The
        // transformed-basis barrier is reproducible: we can rebuild its
        // `BarrierConfig` from `pirls_result.linear_constraints_
        //  transformed` and its β from `pirls_result.beta_transformed`,
        // rotate the resulting diagonal up into the original basis via
        // `Qs · D · Qsᵀ`, and subtract it.  Then add the correct
        // `τ · diag_o(1/Δ_o²)` using `self.linear_constraints` and
        // `β_original`.
        //
        // This makes `log|H|`, its ρ-gradient (`trace_logdet_gradient(
        //  dH/dρ)` in unified.rs), and the `barrier_cost` added at
        // unified.rs:3106-3112 all derive from the same penalized
        // Hessian surface `X'WX + S − H_φ + τ·diag_o(1/Δ_o²)`, which
        // the `BarrierDerivativeProvider` at unified.rs:3133-3150
        // differentiates in the original basis via
        // `self.linear_constraints`-derived config.
        let transformed_barrier_to_strip = pirls_result
            .linear_constraints_transformed
            .as_ref()
            .and_then(Self::barrier_config_from_constraints);
        if let Some(ref barrier_cfg_t) = transformed_barrier_to_strip {
            let mut diag_trans = Array2::<f64>::zeros((beta.len(), beta.len()));
            let beta_t = pirls_result.beta_transformed.as_ref();
            match barrier_cfg_t.add_barrier_hessian_diagonal(&mut diag_trans, beta_t) {
                Ok(()) => {
                    let qs = &pirls_result.reparam_result.qs;
                    // diag_orig_form = Qs · diag_trans · Qsᵀ — the rotated
                    // form of whatever transformed barrier already sits in
                    // bundle.h_total.
                    let tmp = qs.dot(&diag_trans);
                    let rotated = tmp.dot(&qs.t());
                    h_total_original -= &rotated;
                }
                Err(e) => {
                    log::warn!(
                        "Transformed-basis barrier-diagonal reconstruction failed ({e}); \
                         leaving bundle Hessian unchanged before adding original-basis barrier"
                    );
                }
            }
        }
        if let Some(barrier_cfg) = self
            .linear_constraints
            .as_ref()
            .and_then(Self::barrier_config_from_constraints)
            && let Err(e) = barrier_cfg.add_barrier_hessian_diagonal(&mut h_total_original, &beta)
        {
            log::warn!(
                "Original-basis barrier Hessian diagonal skipped: {e}; \
                     cost/gradient/logdet consistency may regress on infeasible \
                     candidates (slack ≤ 0).  BFGS line search must maintain \
                     feasibility for the barrier-aware outer objective."
            );
        }

        // Match `build_dense_assembly`: under Firth bias-reduction the penalized
        // Hessian `X'WX + S − H_φ` inherits the Jeffreys-identifiable subspace,
        // and `Smooth` would leak a spurious gradient contribution through the
        // null directions.  Keep `log|H|`, its gradient, its cross-traces, and
        // `H⁻¹` solves consistent on the same reduced active subspace by using
        // `HardPseudo` whenever a Firth operator is in play.
        let hessian_mode = if bundle.firth_dense_operator.is_some()
            || bundle.firth_dense_operator_original.is_some()
        {
            PseudoLogdetMode::HardPseudo
        } else {
            PseudoLogdetMode::Smooth
        };
        let c_nontrivial = pirls_result.solve_c_array.iter().any(|&c| c != 0.0);

        // Same Cholesky fast path as `build_dense_assembly`: for ValueOnly
        // evaluations with `Smooth` mode (no Firth and no beta-dependent
        // Hessian drift), LLT replaces eigh.
        // `build_dense_original_assembly` is only called when there is no
        // active constraint free-basis, so the no-hard-constraints condition
        // is always satisfied here.
        let hessian_op: std::sync::Arc<dyn super::reml_outer_engine::HessianOperator> = {
            use super::reml_outer_engine::DenseCholeskyValueOnlyOperator;
            if mode == super::reml_outer_engine::EvalMode::ValueOnly
                && matches!(hessian_mode, PseudoLogdetMode::Smooth)
                && !c_nontrivial
                && !force_spectral_logdet
            {
                match DenseCholeskyValueOnlyOperator::from_spd(&h_total_original) {
                    Ok(chol_op) => std::sync::Arc::new(chol_op),
                    Err(_) => std::sync::Arc::new(
                        DenseSpectralOperator::from_symmetric_with_mode(
                            &h_total_original,
                            hessian_mode,
                        )
                        .map_err(|e| {
                            EstimationError::InvalidInput(format!(
                                "DenseSpectralOperator from original-basis PIRLS Hessian: {e}"
                            ))
                        })?,
                    ),
                }
            } else {
                std::sync::Arc::new(
                    DenseSpectralOperator::from_symmetric_with_mode(
                        &h_total_original,
                        hessian_mode,
                    )
                    .map_err(|e| {
                        EstimationError::InvalidInput(format!(
                            "DenseSpectralOperator from original-basis PIRLS Hessian: {e}"
                        ))
                    })?,
                )
            }
        };

        let e_for_logdet = &pirls_result.reparam_result.e_transformed;
        let uses_kron_penalty_logdet = self.kronecker_penalty_system.as_ref().is_some_and(|kron| {
            self.kronecker_factored.is_some() && kron.num_penalties() == rho.len()
        });
        // Penalty-side `log|S|₊` machinery only; the Hessian-side kernel is
        // intrinsic to H_pen (#901) and no longer consumes `range(S_+)`.
        let penalty_subspace = if !uses_kron_penalty_logdet {
            Some(self.compute_penalty_subspace(e_for_logdet, ridge_passport)?)
        } else {
            None
        };
        let (penalty_rank, penalty_logdet) = self.dense_penalty_logdet_derivs(
            rho,
            e_for_logdet,
            &[],
            ridge_passport,
            penalty_subspace.as_ref(),
            bundle,
            mode,
            // Original-basis assembly is only used when there are no active
            // constraints, so no constraint-free projection applies here.
            None,
        )?;

        let nullspace_dim = beta.len().saturating_sub(penalty_rank) as f64;

        // Same rank-deficient LAML fix as `build_dense_assembly` (#901
        // corrected object), adapted to the original-basis Hessian. The
        // intrinsic pseudo-logdet `log|H_pen|₊` and its spectral kernel
        // `H_pen⁺` are properties of `H_pen` ALONE — no external `range(S_+)`
        // basis is involved — so the historical rotate-into-transformed-basis
        // / project / rotate-`U_S`-back-via-`Qs` dance is gone: build the
        // parts directly from `h_total_original` in the basis the ψ/τ drift
        // matrices are produced in. (The pseudo-logdet scalar is invariant
        // under the orthogonal Qs change of basis, so the value pairing with
        // `hessian_op.logdet()` — also original-basis — is unchanged.)
        //
        // Gate `c_nontrivial` and the `HardPseudo` skip: same rationale as
        // `build_dense_assembly`; see `intrinsic_hessian_pseudo_logdet_parts`
        // for the full #901 derivation (why range(Sλ)-projected value+kernel
        // was the wrong object, and why `tr(H_pen⁺ Ḣ)` is exact for every
        // drift including moving-subspace ψ directions).
        let (hessian_logdet_correction, penalty_subspace_trace) =
            if matches!(hessian_mode, PseudoLogdetMode::Smooth) && c_nontrivial {
                let (log_det_h_plus, kernel) =
                    Self::intrinsic_hessian_pseudo_logdet_parts(&h_total_original)?;
                (
                    log_det_h_plus - hessian_op.logdet(),
                    kernel.map(std::sync::Arc::new),
                )
            } else {
                (0.0, None)
            };

        // #1271 diagnostic (twin of the build_dense_assembly probe): this is the
        // path the unconstrained Gaussian tp fit actually takes. Dump REML
        // logdet internals per dense original-basis evaluation. Pure logging via
        // log::info! (default-off in production, no numeric behavior change).
        if log::log_enabled!(log::Level::Info) {
            use faer::Side;
            use gam_linalg::faer_ndarray::FaerEigh;
            let h_logdet = hessian_op.logdet();
            let (h_above_one, h_min, h_max, h_eig_str) = match h_total_original.eigh(Side::Lower) {
                Ok((evals, _)) => {
                    let above = evals.iter().filter(|&&s| s > 1.0).count();
                    let mn = evals.iter().copied().fold(f64::INFINITY, f64::min);
                    let mx = evals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sorted: Vec<f64> = evals.to_vec();
                    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let s = sorted
                        .iter()
                        .map(|v| format!("{v:.3e}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    (above as i64, mn, mx, s)
                }
                Err(_) => (-1, f64::NAN, f64::NAN, String::from("err")),
            };
            // Penalty Sλ = EᵀE eigenvalues (the SAME e_for_logdet that feeds
            // log|S|+). Sorted spectrum is basis-invariant; compare per-mode λ·d_i
            // against the sorted H spectrum to see which modes are data-dominated
            // (high EDF: λ·d ≪ H eig) vs penalty-crushed (λ·d ≈ H eig).
            let s_eig_str = {
                let s_lambda = e_for_logdet.t().dot(e_for_logdet);
                match s_lambda.eigh(Side::Lower) {
                    Ok((mut evals, _)) => {
                        let v = evals.as_slice_mut().unwrap();
                        v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                        v.iter()
                            .map(|x| format!("{x:.3e}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    }
                    Err(_) => String::from("err"),
                }
            };
            let rho_str = rho
                .iter()
                .map(|r| format!("{r:.4}"))
                .collect::<Vec<_>>()
                .join(",");
            let lam_str = rho
                .iter()
                .map(|r| format!("{:.3e}", r.exp()))
                .collect::<Vec<_>>()
                .join(",");
            let rss = pirls_result.deviance;
            let pen = pirls_result.stable_penalty_term;
            let dp = rss + pen;
            let pedf = pirls_result.edf;
            log::info!(
                "[#1271-diag] path=orig p={p} penalty_rank={pr} nullspace_dim={nd} \
                 rss={rss:.6} pen={pen:.6} Dp={dp:.6} pirls_edf={pedf:.4} \
                 logS={ls:.6} logH={lh:.6} half_diff={hd:.6} \
                 h_above1={ha} h_min={hmin:.3e} h_max={hmax:.3e} \
                 rho=[{rho_str}] lambda=[{lam_str}]\n  H_eig=[{h_eig_str}]\n  Slam_eig=[{s_eig_str}]",
                p = beta.len(),
                pr = penalty_rank,
                nd = nullspace_dim,
                ls = penalty_logdet.value,
                lh = h_logdet,
                hd = 0.5 * (h_logdet - penalty_logdet.value),
                ha = h_above_one,
                hmin = h_min,
                hmax = h_max,
            );
        }

        let ctx = self.build_sparse_derivative_context(pirls_result, bundle)?;
        // Original-basis envelope residual: `β` and `H` here are rotated into
        // the original basis, and `build_dense_original_assembly` is only ever
        // reached on the unconstrained QS frame, so the transformed residual
        // maps up cleanly via the same orthogonal `Qs`.
        let inner_kkt_residual = if populate_inner_kkt {
            self.inner_kkt_residual_original_basis(
                pirls_result,
                bundle.firth_dense_operator.is_some()
                    || bundle.firth_dense_operator_original.is_some(),
            )
        } else {
            None
        };
        Ok(self.finish_assembly(
            pirls_result,
            ctx,
            hessian_op,
            beta,
            penalty_logdet,
            nullspace_dim,
            hessian_logdet_correction,
            penalty_subspace_trace,
            None,
            inner_kkt_residual,
        ))
    }

    /// Build an `InnerAssembly` by auto-detecting the backend.
    ///
    /// - Sparse-exact SPD  → `build_sparse_assembly`
    /// - QS-transformed + unconstrained (with or without ext_coords)
    ///     → `build_dense_original_assembly`
    /// - Otherwise → `build_dense_assembly` (used only when active
    ///     inequality constraints force evaluation in the transformed frame)
    ///
    /// Coordinate-frame correctness.
    /// `self.canonical_penalties` — which `build_penalty_coords()` feeds into
    /// the outer gradient as `PenaltyCoordinate::{DenseRoot, BlockRoot}` — is
    /// stored in the ORIGINAL (pre-`Qs`) coefficient basis. PIRLS, however,
    /// runs in the QS-transformed frame and reports `β_transformed =
    /// Qsᵀ · β_original`. If we hand the assembly `β_transformed` together
    /// with original-basis penalty roots, then `coord.apply_penalty(β, λ)`
    /// silently returns `λ · S_orig · β_t`, which is **not** a valid
    /// `∂(½ βᵀ Sβ)/∂ρₖ` in any single basis (the scalar `βᵀSβ` is invariant,
    /// but `Sβ` is not). That inconsistency is what starved BFGS of descent
    /// steps on large-n binomial+logit — only 2/200 line searches accepted —
    /// because the analytic gradient disagreed with the cost by a term that
    /// grew with the off-diagonal structure of `Qs`.
    ///
    /// Dispatching the unconstrained dense path through
    /// `build_dense_original_assembly` keeps `β`, the Hessian operator, the
    /// GLM derivative design, and the penalty coordinates all in the same
    /// (original) basis, restoring cost–gradient consistency. The only path
    /// that still needs `build_dense_assembly` is the monotonicity/active-set
    /// branch, where `free_basis_opt` already rotates everything into a reduced QS
    /// subspace (penalty-coord fast path uses `build_dense_assembly` there because
    /// the projected-`Z` original-basis route is unavailable in that subspace).
    pub(crate) fn build_auto_assembly(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        force_spectral_logdet: bool,
        populate_inner_kkt: bool,
    ) -> Result<super::assembly::InnerAssembly<'static>, EstimationError> {
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            // Sparse-exact `log|H|` is Cholesky-derived and its gradient traces
            // are paired against the SAME exact factorization, so there is no
            // floored-vs-exact desync to suppress here (#1376 affects only the
            // dense smooth-floored spectral gradient paths).
            return self.build_sparse_assembly(rho, bundle, mode, populate_inner_kkt);
        }
        // The original-basis dense path keeps β, the Hessian operator, the GLM
        // derivative design, and the penalty coordinates in one (original)
        // basis. It applies only to an UNCONSTRAINED QS frame: an active-set /
        // monotonicity reduced subspace re-expresses the penalty coordinates
        // through its projected `Z`, so that case is assembled in the
        // transformed frame by `build_dense_assembly` — the correct routing for
        // that subspace, not a fallback pending further work.
        let unconstrained_qs_frame = matches!(
            bundle.pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && self
            .active_constraint_free_basis(bundle.pirls_result.as_ref())
            .is_none();
        if unconstrained_qs_frame {
            self.build_dense_original_assembly(
                rho,
                bundle,
                mode,
                force_spectral_logdet,
                populate_inner_kkt,
            )
        } else {
            // Reached for the transformed-QS active-set branch (where
            // `free_basis_opt` rotates into a reduced subspace) or a non-QS
            // frame: the original-basis penalty-coord fast path does not apply,
            // so assemble in the evaluation frame via `build_dense_assembly`.
            log::trace!(
                "[reml-assembly] build_dense_assembly path \
                 (transformed-QS active-set, or non-QS frame)"
            );
            self.build_dense_assembly(rho, bundle, mode, force_spectral_logdet, populate_inner_kkt)
        }
    }

    /// Augment a freshly-evaluated REML/LAML result with the ALO stabilization
    /// term, when the activation gate fires. The objective is bit-preserved
    /// (no `cost`/`gradient`/`hessian` mutation) whenever the gate is off, so
    /// every standard dense-Gaussian-identity caller — BFGS/Newton via
    /// `assemble_and_evaluate` and EFS via `assemble_and_evaluate_efs` — flows
    /// through this single shared augmentation and therefore optimizes the
    /// identical objective.
    pub(crate) fn apply_alo_stabilization_to_result(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        mut result: super::reml_outer_engine::RemlLamlResult,
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError> {
        let want_gradient = mode != super::reml_outer_engine::EvalMode::ValueOnly;
        let Some(alo_eval) = self.alo_stabilization_eval(rho, bundle, want_gradient)? else {
            return Ok(result);
        };
        result.cost += alo_eval.cost;
        if want_gradient {
            match (result.gradient.as_mut(), alo_eval.gradient.as_ref()) {
                (Some(gradient), Some(alo_gradient)) if gradient.len() >= alo_gradient.len() => {
                    for idx in 0..alo_gradient.len() {
                        gradient[idx] += alo_gradient[idx];
                    }
                    // The ALO augmentation contributes only a cost and gradient
                    // term; its Gauss-Newton curvature term was dropped as dead
                    // (see "drop dead ALO-stabilization Hessian computation"), so
                    // the base-REML analytic Hessian is retained unchanged. ARC's
                    // adaptive cubic regularization still converges on the exact
                    // augmented cost+gradient via its ratio test, and keeping the
                    // base Hessian preserves the `HessianSource::Analytic` invariant
                    // the outer plan relies on.
                }
                _ => {
                    log::warn!(
                        "[ALO-STABILIZED-REML] unstable ALO detected but analytic gradient \
                         augmentation was unavailable; retaining REML gradient and adding \
                         value term only (n={} max_h={:.3} min_denom={:.3})",
                        self.y.len(),
                        alo_eval.max_leverage,
                        alo_eval.min_denominator,
                    );
                }
            }
        }
        log::info!(
            "[ALO-STABILIZED-REML] active cost_add={:.6e} max_h={:.3} min_denom={:.3} k_hat={:?}",
            alo_eval.cost,
            alo_eval.max_leverage,
            alo_eval.min_denominator,
            alo_eval.k_hat,
        );
        Ok(result)
    }

    /// ρ-independent certificate that the Gaussian-identity ALO-stabilization
    /// augmentation can never activate on this surface (#1689). See the
    /// `alo_provably_inactive` field docs for the monotone-bound derivation.
    /// Computed at most once and memoized. Returns `false` (not certified)
    /// whenever the bound is not strictly below the activation threshold or the
    /// unpenalized Gram is not factorizable — in either case the caller falls
    /// through to the exact per-evaluation gate, preserving prior behavior.
    pub(crate) fn alo_provably_inactive_certificate(&self) -> Result<bool, EstimationError> {
        if let Some(cached) = *self.alo_provably_inactive.read().unwrap() {
            return Ok(cached);
        }
        let verdict = self.compute_alo_provably_inactive();
        *self.alo_provably_inactive.write().unwrap() = Some(verdict);
        Ok(verdict)
    }

    /// Compute the non-activation certificate. The leverage `xᵢᵀ H⁻¹ xᵢ` is
    /// invariant under any invertible column reparameterization, so this works
    /// in the original design basis with the fixed Gaussian-identity weights.
    fn compute_alo_provably_inactive(&self) -> bool {
        let x = match self
            .x
            .try_to_dense_arc("ALO non-activation certificate requires dense design")
        {
            Ok(x) => x,
            Err(_) => return false,
        };
        unpenalized_hat_bound_below_threshold(x.as_ref(), self.weights, ALO_MAX_LEVERAGE_THRESHOLD)
    }

    pub(crate) fn alo_stabilization_eval(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        want_gradient: bool,
    ) -> Result<Option<AloStabilizationEval>, EstimationError> {
        // ρ-posterior sampling suppression (#979). The ALO-stabilization
        // augmentation is an OUTER-OPTIMIZER aid (#813/#821): a leverage barrier
        // that keeps the smoothing-parameter search off pathological high-
        // leverage λ regions. The marginal smoothing-parameter posterior
        // `π(ρ|y) ∝ exp(−LAML(ρ))` (#938) is a property of the genuine model
        // criterion, and its Laplace proposal is built from the BASE REML
        // Hessian (the augmentation's curvature term is dropped as dead, see
        // `apply_alo_stabilization_to_result`). Evaluating the augmented cost
        // while sampling therefore drives a target inconsistent with its own
        // proposal AND pays the full ALO diagnostic suite on every leapfrog
        // step — the dominant cost of `compute_inference` on poorly-identified ρ
        // posteriors. Suppressing the augmentation during the certificate /
        // NUTS evaluations (`without_alo_stabilization`) restores proposal↔target
        // consistency and removes that cost.
        if self
            .alo_stabilization_suppression
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
        {
            return Ok(None);
        }
        if self.config.firth_bias_reduction
            || !matches!(
                self.config.likelihood.spec.response,
                ResponseFamily::Gaussian
            )
            || !matches!(self.config.link_function(), LinkFunction::Identity)
            || !matches!(
                bundle.pirls_result.coordinate_frame,
                pirls::PirlsCoordinateFrame::TransformedQs
            )
            || bundle.backend_kind() == GeometryBackendKind::SparseExactSpd
        {
            return Ok(None);
        }
        let n = self.y.len();
        if n < ALO_STABILIZATION_MIN_N {
            return Ok(None);
        }
        // #1033 n-free κ-loop: when the inner Gaussian solve was served by the
        // ψ-keyed sufficient-statistic Gram cache (`row_prediction_is_stale`),
        // the surface's realized rows — `x_transformed`, `finalmu`, the working
        // weights — are FROZEN at the pinning ψ of the last `reset_surface`, NOT
        // this trial's ψ. `compute_alo_diagnostics_from_pirls` would (a) cost a
        // full O(n·k) leverage pass (the dense design materialization + n hat
        // values `h_ii = w_i x_iᵀ H⁻¹ x_i`) on EVERY in-window κ-trial — the last
        // O(n) term defeating the issue's sufficient-statistic invariant — and
        // (b) read those stale rows, so the leverage it returns describes the
        // wrong ψ. The stabilizer is an OUTER-OPTIMIZER aid, never part of the
        // genuine REML/LAML criterion (see the header comment), so skipping it on
        // the stale-row lane changes no fitted result: the slow-path anchor (one
        // realization per design revision) still carries a non-stale cache and
        // keeps the leverage barrier engaged at the pinning ψ. Gate strictly on
        // the installed cache being the stale-row tensor cache, so every
        // realized-design eval (slow path, off-window, non-Gaussian) keeps the
        // exact augmentation.
        if self
            .installed_gaussian_fixed_cache()
            .is_some_and(|cache| cache.row_prediction_is_stale)
        {
            return Ok(None);
        }
        // Suppress the stabilizer on near-saturated / over-parameterized designs
        // (e.g. small-n tensor-product `te()` whose marginal-product column count
        // rivals n). There leverage is high for *every* row from basis geometry,
        // not from a few influential observations, so the augmentation can never
        // clear its own leverage gate and instead leaves a near-flat ill-
        // conditioned ridge on the outer surface that the optimizer crawls to its
        // iteration cap (#813 / #821). This check is cheap (one scalar ratio) and
        // is intentionally placed *before* `compute_alo_diagnostics_from_pirls`,
        // so the per-outer-evaluation dense Hessian factorization and n column
        // solves the diagnostics require are skipped entirely in this regime —
        // restoring `te()` to the same per-eval cost profile as `duchon`/`matern`.
        let edf = bundle.pirls_result.edf;
        if edf.is_finite() && edf > ALO_EDF_FRACTION_SATURATION * (n as f64) {
            return Ok(None);
        }
        // ρ-independent non-activation certificate (#1689). The augmentation can
        // only engage when some row's *penalized* leverage
        //   h_i = w_i · xᵢᵀ H_λ⁻¹ xᵢ ,   H_λ = XᵀWX + S_λ + ridge·I
        // reaches `ALO_MAX_LEVERAGE_THRESHOLD`. Since `S_λ + ridge·I ⪰ 0`,
        // `H_λ ⪰ XᵀWX`, hence `H_λ⁻¹ ⪯ (XᵀWX)⁻¹` and
        //   h_i ≤ w_i · xᵢᵀ (XᵀWX)⁻¹ xᵢ =: h̄_i,
        // the *unpenalized* weighted hat diagonal. For Gaussian identity (the
        // only family reaching here) W is the fixed prior-weight diagonal, so h̄
        // is independent of ρ and of the smoothing search. If max_i h̄_i is below
        // the activation threshold, no ρ can trip the leverage gate, so the
        // augmentation provably never engages and the per-outer-evaluation
        // O(n·p²) ALO diagnostic (otherwise computed and discarded on EVERY cost
        // and gradient evaluation — the dominant cost of ordinary P-spline /
        // thin-plate Gaussian fits, #1689) is skipped with zero behavioral
        // change. Computed lazily, at most once per surface.
        if self.alo_provably_inactive_certificate()? {
            return Ok(None);
        }
        // Graceful degradation when the ALO diagnostic itself is not computable
        // at this ρ (#1191). The stabilizer is an OUTER-OPTIMIZER aid, never part
        // of the genuine criterion (see the header comment), so a failure to form
        // its leverage / leave-one-out diagnostic must NOT abort the outer
        // objective evaluation — it must behave like every other "conditions not
        // right for stabilization" branch above and simply skip the augmentation.
        //
        // The diagnostic legitimately fails at *intermediate* ρ on designs whose
        // penalized Hessian is indefinite or near-saturated, which is exactly the
        // transient state shape-constrained smooths (monotone/convex/concave) pass
        // through during the smoothing-parameter search: the constrained Hessian
        // is indefinite away from the optimum, so the exact frozen-curvature ALO
        // fixed point receives a non-finite per-row score/curvature and reports
        // `LooComputationFailed`. Before this guard that `?` propagated as an
        // `EstimationError` that the seed loop classified as a *domain* rejection,
        // so every candidate seed was rejected and the entire fit aborted with
        // "no candidate seeds passed outer startup validation" — making all four
        // shape-constrained smooths unfittable from `gamfit.fit`, even though the
        // genuine REML criterion (and the `gam` CLI, which only computes ALO as a
        // post-convergence diagnostic on the well-conditioned final Hessian) fits
        // the identical data fine. Skipping the augmentation here returns the
        // outer eval to the plain REML surface for that ρ; the augmentation
        // re-engages automatically once the search reaches well-conditioned ρ
        // where the diagnostic is computable again.
        let alo = match crate::inference::alo::compute_alo_diagnostics_from_pirls(
            bundle.pirls_result.as_ref(),
            self.y,
            self.config.link_function(),
        ) {
            Ok(alo) => alo,
            Err(err) => {
                log::debug!(
                    "[ALO-STABILIZED-REML] skipping augmentation at this ρ: ALO diagnostic not computable ({err}); falling back to the plain REML criterion for this evaluation",
                );
                return Ok(None);
            }
        };
        let max_leverage = alo
            .leverage
            .iter()
            .copied()
            .fold(0.0_f64, |acc, h| acc.max(h));
        let min_denominator = alo
            .leverage
            .iter()
            .copied()
            .map(|h| 1.0 - h)
            .fold(f64::INFINITY, |acc, d| acc.min(d));
        if max_leverage < ALO_MAX_LEVERAGE_THRESHOLD
            && min_denominator > ALO_DENOM_INSTABILITY_THRESHOLD
        {
            return Ok(None);
        }
        if self.high_leverage_is_pure_parametric(&alo.leverage, bundle.pirls_result.as_ref())? {
            return Ok(None);
        }

        // Pervasiveness guard. The stabilizer is designed for a *minority* of
        // genuinely influential rows on an identified design; there the leverage
        // barrier `Σ(h_i − 0.80)₊²` can be cleared by a modest λ bump that the
        // REML criterion tolerates, and the augmented surface stays well-
        // conditioned. When instead a *large fraction* of rows clears the
        // leverage threshold, the high leverage is a basis-geometry artifact of a
        // near-interpolating tensor-product `te()` basis (every local B-spline
        // cell carries one or two near-interpolated rows), not outliers: no
        // finite λ can pull all of them below 0.80, so the barrier turns the
        // outer surface into a near-flat ill-conditioned ridge that the optimizer
        // crawls to its iteration cap (the #821 "cost decreasing ~1e-4 per step
        // over thousands of evals" signature). RKHS smooths (`duchon`/`matern`)
        // spread leverage globally and keep this fraction near zero on identical
        // data — which is precisely why they were fast while `te()` was not.
        // Suppressing the augmentation here returns `te()` to the plain REML
        // surface (which already controls λ) and to the others' convergence
        // profile, while leaving every concentrated-outlier fit fully stabilized.
        let n_high_leverage = alo
            .leverage
            .iter()
            .filter(|&&h| h > ALO_MAX_LEVERAGE_THRESHOLD)
            .count();
        if (n_high_leverage as f64) > ALO_PERVASIVE_LEVERAGE_FRACTION * (n as f64) {
            return Ok(None);
        }

        let raw_influence: Vec<f64> = alo
            .leverage
            .iter()
            .copied()
            .map(|h| h.max(0.0) / (1.0 - h).max(1e-12))
            .collect();
        let psis = crate::psis::pareto_smooth_weights(&raw_influence);
        let smoothed_influence = psis
            .as_ref()
            .map(|p| p.smoothed.as_slice())
            .unwrap_or(raw_influence.as_slice());
        let mean_influence =
            smoothed_influence.iter().sum::<f64>() / smoothed_influence.len() as f64;
        // Clamp the per-observation reweight to [0.25, 4.0]. This conservative
        // 4× band bounds how far a single PSIS-smoothed influence weight can
        // pull the deviance term, guaranteeing no point can dominate the
        // augmented objective even if the tail fit is imperfect.
        let fresh_influence_scale: Vec<f64> = smoothed_influence
            .iter()
            .map(|w| (*w / mean_influence.max(f64::MIN_POSITIVE)).clamp(0.25, 4.0))
            .collect();
        let fresh_phi = match self.config.likelihood.scale.fixed_phi() {
            Some(phi) if phi.is_finite() && phi > 0.0 => phi,
            Some(_) => 1.0,
            None => {
                let dp = bundle.pirls_result.deviance + bundle.pirls_result.stable_penalty_term;
                let denom = (n as f64 - bundle.pirls_result.edf).max(1.0);
                (dp / denom).max(f64::MIN_POSITIVE)
            }
        };
        // Freeze the reweight and dispersion at their first-activation values
        // for this REML surface so the augmented cost and its analytic gradient
        // stay consistent. Recomputing them per outer evaluation while the
        // gradient holds them constant is the #821/#813 inconsistency. The
        // cache must be owned by this RemlState, not a static pointer-keyed map:
        // cold model sweeps commonly allocate a new state at the same address
        // with the same n, and reusing another formula's ALO weights recreates
        // the #862 smooth+linear outer-REML grind.
        let (influence_scale, phi) = {
            let mut frozen = self.alo_frozen_nuisance.write().unwrap();
            match frozen.as_ref() {
                Some(cached) if cached.n_obs == n && cached.influence_scale.len() == n => {
                    (cached.influence_scale.clone(), cached.phi)
                }
                _ => {
                    *frozen = Some(super::AloFrozenNuisance {
                        n_obs: n,
                        influence_scale: fresh_influence_scale.clone(),
                        phi: fresh_phi,
                    });
                    (fresh_influence_scale, fresh_phi)
                }
            }
        };
        let mut cost = 0.0_f64;
        for i in 0..n {
            cost += ALO_TAU * alo_leverage_barrier(alo.leverage[i]);
            cost += ALO_GAMMA
                * influence_scale[i]
                * gaussian_alo_deviance(self.y[i], alo.eta_tilde[i], self.weights[i], phi);
        }
        if !(cost.is_finite() && cost >= 0.0) {
            return Ok(None);
        }
        let gradient = if want_gradient
            && rho.len()
                == bundle
                    .pirls_result
                    .reparam_result
                    .canonical_transformed
                    .len()
            && n.saturating_mul(bundle.pirls_result.beta_transformed.as_ref().len())
                <= ALO_GRADIENT_MAX_WORK
        {
            // Factor the stabilized penalized Hessian and form H⁻¹Xᵀ exactly once
            // here, then hand the factor + solve to the gradient. The leverage
            // diagnostics above already factored the same matrix internally, so
            // recomputing dense X + a fresh Cholesky + the H⁻¹Xᵀ solve inside the
            // gradient is a full duplicate O(np² + p³) per outer evaluation —
            // the dominant per-eval cost once ALO engages on small-n smooth+linear
            // fits (#862). One factorization feeds both the value and the gradient.
            let x = bundle.pirls_result.x_transformed.to_dense();
            match bundle
                .pirls_result
                .dense_stabilizedhessian_transformed("ALO-stabilized REML gradient")?
                .cholesky(Side::Lower)
            {
                Ok(chol) => {
                    let sensitivity =
                        crate::sensitivity::FitSensitivity::from_faer_cholesky(&chol, x.ncols());
                    let h_inv_xt = sensitivity.leverage_block(&x);
                    self.alo_stabilization_gradient(
                        rho,
                        bundle,
                        &alo,
                        &influence_scale,
                        phi,
                        &AloFactoredHessian {
                            x: &x,
                            sensitivity: &sensitivity,
                            h_inv_xt: &h_inv_xt,
                        },
                    )?
                }
                Err(_) => None,
            }
        } else {
            None
        };
        Ok(Some(AloStabilizationEval {
            cost,
            gradient,
            k_hat: psis.as_ref().map(|p| p.k_hat),
            max_leverage,
            min_denominator,
        }))
    }

    pub(crate) fn high_leverage_is_pure_parametric(
        &self,
        alo_leverage: &Array1<f64>,
        pirls_result: &PirlsResult,
    ) -> Result<bool, EstimationError> {
        let pure_parametric_cols = self.pure_parametric_column_indices();
        if pure_parametric_cols.len() <= 1 {
            return Ok(false);
        }
        let high_rows: Vec<usize> = alo_leverage
            .iter()
            .enumerate()
            .filter_map(|(idx, &h)| (h > ALO_MAX_LEVERAGE_THRESHOLD).then_some(idx))
            .collect();
        if high_rows.is_empty() {
            return Ok(false);
        }
        let parametric_leverage =
            self.pure_parametric_projection_leverage(&pure_parametric_cols, pirls_result)?;
        let all_high_rows_parametric = high_rows.iter().all(|&idx| {
            let h = alo_leverage[idx];
            let hp = parametric_leverage[idx];
            hp >= ALO_MAX_LEVERAGE_THRESHOLD && hp >= ALO_PARAMETRIC_LEVERAGE_SHARE * h
        });
        if all_high_rows_parametric {
            log::info!(
                "[ALO-STABILIZED-REML] suppressed: {} high-leverage rows are explained by {} pure-parametric columns",
                high_rows.len(),
                pure_parametric_cols.len(),
            );
        }
        Ok(all_high_rows_parametric)
    }

    pub(crate) fn pure_parametric_column_indices(&self) -> Vec<usize> {
        let mut covered_by_penalty = vec![false; self.p];
        for penalty in self.canonical_penalties.iter() {
            for col in penalty.col_range.clone() {
                if col < self.p {
                    covered_by_penalty[col] = true;
                }
            }
        }
        covered_by_penalty
            .iter()
            .enumerate()
            .filter_map(|(col, &covered)| (!covered).then_some(col))
            .collect()
    }

    pub(crate) fn pure_parametric_projection_leverage(
        &self,
        cols: &[usize],
        pirls_result: &PirlsResult,
    ) -> Result<Array1<f64>, EstimationError> {
        let x_dense = match self
            .x
            .try_to_dense_arc("ALO pure-parametric activation gate requires dense design")
        {
            Ok(x_dense) => x_dense,
            Err(reason) => {
                log::debug!("[ALO-STABILIZED-REML] pure-parametric gate skipped: {reason}");
                return Ok(Array1::<f64>::zeros(pirls_result.finalweights.len()));
            }
        };
        let n = x_dense.nrows();
        if n != pirls_result.finalweights.len() || cols.iter().any(|&col| col >= x_dense.ncols()) {
            crate::bail_invalid_estim!(
                "ALO pure-parametric activation gate received inconsistent dimensions"
            );
        }
        let q = cols.len();
        let mut gram = Array2::<f64>::zeros((q, q));
        for i in 0..n {
            let wi = pirls_result.finalweights[i];
            if !wi.is_finite() || wi <= 0.0 {
                return Ok(Array1::<f64>::zeros(n));
            }
            for (a, &ca) in cols.iter().enumerate() {
                let xa = x_dense[[i, ca]];
                for (b, &cb) in cols.iter().take(a + 1).enumerate() {
                    gram[[a, b]] += wi * xa * x_dense[[i, cb]];
                }
            }
        }
        for a in 0..q {
            for b in 0..a {
                gram[[b, a]] = gram[[a, b]];
            }
        }
        let factor = match StableSolver::new("ALO pure-parametric activation gate").factorize(&gram)
        {
            Ok(factor) => factor,
            Err(_) => return Ok(Array1::<f64>::zeros(n)),
        };
        let mut gram_inv = Array2::<f64>::eye(q);
        let mut gram_inv_view = array2_to_matmut(&mut gram_inv);
        factor.solve_in_place(gram_inv_view.as_mut());
        let mut leverage = Array1::<f64>::zeros(n);
        for i in 0..n {
            let wi = pirls_result.finalweights[i];
            let mut h = 0.0;
            for (a, &ca) in cols.iter().enumerate() {
                let xa = x_dense[[i, ca]];
                for (b, &cb) in cols.iter().enumerate() {
                    h += xa * gram_inv[[a, b]] * x_dense[[i, cb]];
                }
            }
            leverage[i] = (wi * h).max(0.0);
        }
        Ok(leverage)
    }

    /// Analytic ρ-gradient of the ALO augmentation.
    ///
    /// The augmentation is `C(ρ) = Σ_i [ τ·b(h_i(ρ)) + γ·s_i·w_i·(y_i −
    /// η̃_i(ρ))²/φ ]`, with `b(h) = (h − 0.80)₊²` the leverage barrier and
    /// `η̃_i` the leave-one-out predictor. Writing `u_i,k = ∂η̃_i/∂ρ_k` and
    /// `v_i,k = ∂h_i/∂ρ_k`, the exact gradient is
    ///   `g_k = Σ_i [ τ·b'(h_i)·v_i,k − 2γ·s_i·w_i·(y_i − η̃_i)/φ · u_i,k ]`.
    /// The base REML analytic Hessian is reused unchanged as the second-order
    /// model for ARC's adaptive cubic regularization (the ratio test runs on
    /// the exact augmented cost+gradient produced here).
    ///
    /// The stabilized-Hessian factorization is supplied via `factored` — the
    /// value path factors the same matrix once and shares it here, so the
    /// gradient adds no extra dense factorization (#862).
    pub(crate) fn alo_stabilization_gradient(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        alo: &crate::inference::alo::AloDiagnostics,
        influence_scale: &[f64],
        phi: f64,
        factored: &AloFactoredHessian<'_>,
    ) -> Result<Option<Array1<f64>>, EstimationError> {
        let &AloFactoredHessian {
            x,
            sensitivity,
            h_inv_xt,
        } = factored;
        let beta = bundle.pirls_result.beta_transformed.as_ref();
        let k = rho.len();
        let nrows = x.nrows();
        let phi_safe = phi.max(f64::MIN_POSITIVE);
        // Per-observation sensitivities of η̃_i and h_i to each ρ_k.
        let mut deta_loo = Array2::<f64>::zeros((nrows, k));
        let mut lev_deriv = Array2::<f64>::zeros((nrows, k));
        for penalty_idx in 0..k {
            let lambda = rho[penalty_idx].exp();
            let sbeta = transformed_penalty_matvec(
                &bundle.pirls_result.reparam_result.canonical_transformed[penalty_idx],
                beta,
            )
            .mapv(|v| lambda * v);
            let mut mode_deriv = sensitivity.apply(&sbeta);
            mode_deriv.mapv_inplace(|v| -v);
            let eta_deriv = x.dot(&mode_deriv);

            for i in 0..nrows {
                let m_i = h_inv_xt.column(i);
                let penalty_m = transformed_penalty_matvec(
                    &bundle.pirls_result.reparam_result.canonical_transformed[penalty_idx],
                    &m_i.to_owned(),
                );
                let h_i = alo.leverage[i];
                let denom = (1.0 - h_i).max(1e-12);
                let residual = self.y[i] - bundle.pirls_result.final_eta[i];
                let v_ik = -self.weights[i] * lambda * m_i.dot(&penalty_m);
                lev_deriv[[i, penalty_idx]] = v_ik;
                deta_loo[[i, penalty_idx]] =
                    eta_deriv[i] / denom - residual * v_ik / (denom * denom);
            }
        }

        let mut grad = Array1::<f64>::zeros(k);
        for i in 0..nrows {
            let h_i = alo.leverage[i];
            // ∂d_i/∂η̃_i for the raw standardized squared LOO residual d_i.
            let raw_dev_eta_grad =
                -2.0 * self.weights[i] * (self.y[i] - alo.eta_tilde[i]) / phi_safe;
            // Chain the smooth saturator: ∂g(d_i)/∂η̃_i = g'(d_i)·∂d_i/∂η̃_i so
            // the gradient matches the saturated cost used in the value path. A
            // saturated (geometry-driven) high-leverage point has g'(d_i) → 0,
            // so it stops pulling λ up — the analytic correlate of bounding its
            // influence on the criterion.
            let raw_dev =
                gaussian_alo_raw_deviance(self.y[i], alo.eta_tilde[i], self.weights[i], phi_safe);
            let dev_eta_grad = gaussian_alo_deviance_saturation_factor(raw_dev) * raw_dev_eta_grad;
            let b_prime = alo_leverage_barrier_derivative(h_i);
            for kk in 0..k {
                let u_ik = deta_loo[[i, kk]];
                let v_ik = lev_deriv[[i, kk]];
                grad[kk] +=
                    ALO_TAU * b_prime * v_ik + ALO_GAMMA * influence_scale[i] * dev_eta_grad * u_ik;
            }
        }
        if grad.iter().all(|g| g.is_finite()) {
            Ok(Some(grad))
        } else {
            Ok(None)
        }
    }

    /// Build the soft prior tuple for the given mode.
    pub(crate) fn build_prior(
        &self,
        rho: &Array1<f64>,
        mode: super::reml_outer_engine::EvalMode,
    ) -> Option<(f64, Array1<f64>, Option<Array2<f64>>)> {
        let configured = self.configured_rho_prior_atom(rho);
        let soft = self.soft_rho_guard_prior_atom(rho);
        let cost = soft.cost() + configured.cost();
        if mode == super::reml_outer_engine::EvalMode::ValueOnly {
            return (cost.abs() > 0.0).then(|| (cost, Array1::zeros(rho.len()), None));
        }

        let gradient = soft.gradient() + configured.gradient();
        let hessian = if mode == super::reml_outer_engine::EvalMode::ValueGradientHessian {
            let mut hess = soft
                .hessian()
                .unwrap_or_else(|| Array2::<f64>::zeros((rho.len(), rho.len())));
            if let Some(configured_hess) = configured.hessian() {
                hess += configured_hess;
            }
            hess.iter().any(|&v| v != 0.0).then_some(hess)
        } else {
            None
        };

        (cost.abs() > 0.0 || gradient.iter().any(|&v| v != 0.0))
            .then_some((cost, gradient, hessian))
    }

    /// Single assembly point: evaluate an `InnerAssembly`, compute prior, and
    /// apply TK correction.
    ///
    /// All `evaluate_unified*` methods funnel through here.
    pub(crate) fn assemble_and_evaluate(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
        assembly: super::assembly::InnerAssembly<'static>,
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError> {
        let prior = self.build_prior(rho, mode);
        self.validate_tk_ext_coords(mode, &assembly.ext_coords)?;
        let tk_atom = self.tierney_kadane_terms(rho, bundle, mode, &assembly.ext_coords)?;
        let trace_state = self.hypergradient_trace_state();
        Self::reset_hypergradient_trace_telemetry(&trace_state);
        let assembly_ext_len = assembly.ext_coords.len();
        let mut inner_solution = assembly.build();
        inner_solution.stochastic_trace_state = trace_state;
        inner_solution.gaussian_weight_log_sum_half = self.gaussian_weight_log_sum_half();
        inner_solution.dp_floor_scale = self.gaussian_dp_floor_scale();
        let solution_beta = inner_solution.beta.clone();
        let result = super::assembly::evaluate_solution(
            &inner_solution,
            rho.as_slice().unwrap(),
            mode,
            prior,
        )
        .map_err(EstimationError::InvalidInput)?;
        let result = self.apply_theta_correction_atom_to_result(result, &tk_atom)?;
        // Adaptive, block-local Laplace-to-sampling fallback (issue #784): where
        // a curvature direction is too non-Gaussian for the Laplace summary,
        // splice in the importance-sampled block marginal correction. Reuses
        // the same value+gradient splicing contract as the TK correction so the
        // outer REML/LAML stays consistent. A no-op when every direction is
        // Laplace-trustworthy.
        let block_terms = self.block_local_sampled_correction(rho, bundle, assembly_ext_len)?;
        let block_atom = super::atoms::ThetaOnlyCorrectionAtom::from_tk_terms(
            "sampled_block_marginal",
            block_terms,
        );
        let result = self.apply_theta_correction_atom_to_result(result, &block_atom)?;
        let result = self.apply_alo_stabilization_to_result(rho, bundle, mode, result)?;
        self.store_ift_mode_response_cache_from_result(rho, bundle, &result);
        if let Some(polish_step) = result.inner_polish_step.as_ref() {
            self.apply_inner_polish_step_to_warm_start(bundle, &solution_beta, polish_step);
        }
        Ok(result)
    }

    pub(crate) fn assemble_and_evaluate_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        assembly: super::assembly::InnerAssembly<'static>,
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        use super::reml_outer_engine::{compute_efs_update, compute_hybrid_efs_update};

        let beta_for_barrier = assembly.beta.clone();
        let has_psi = assembly.ext_coords.iter().any(|c| !c.is_penalty_like);
        // Always evaluate cost + gradient: the EFS update uses the
        // universal-form `Δρ = log(1 − 2·g_full / q_eff)` for ρ and τ
        // coordinates, which folds Tierney–Kadane corrections, smoothing-
        // parameter priors, Firth bias-reduction, monotonicity barriers,
        // and SAS log-δ ridge contributions into the multiplicative
        // target through their gradient channel. Without the gradient
        // the EFS step targets the wrong stationarity equation whenever
        // any of those terms are active.
        let eval_mode = super::reml_outer_engine::EvalMode::ValueAndGradient;
        self.validate_tk_ext_coords(eval_mode, &assembly.ext_coords)?;
        let tk_atom = self.tierney_kadane_terms(rho, bundle, eval_mode, &assembly.ext_coords)?;
        let assembly_ext_len = assembly.ext_coords.len();
        let mut inner_solution = assembly.build();
        inner_solution.gaussian_weight_log_sum_half = self.gaussian_weight_log_sum_half();
        inner_solution.dp_floor_scale = self.gaussian_dp_floor_scale();
        let inner_hessian_scale = super::reml_outer_engine::hessian_operator_geometric_scale(
            inner_solution.hessian_op.as_ref(),
        );

        let prior = self.build_prior(rho, eval_mode);
        let cost_result = super::assembly::evaluate_solution(
            &inner_solution,
            rho.as_slice().unwrap(),
            eval_mode,
            prior,
        )
        .map_err(EstimationError::InvalidInput)?;
        let cost_result = self.apply_theta_correction_atom_to_result(cost_result, &tk_atom)?;
        // Fold the #784 adaptive block-local Laplace-to-sampling correction into
        // the EFS objective too, so the EFS fixed-point and the BFGS/Newton path
        // (`assemble_and_evaluate`) optimize the SAME marginal-likelihood
        // surface. The correction enters through the gradient channel exactly
        // like TK, which the universal EFS step already folds in. No-op when no
        // direction is non-Gaussian.
        let block_terms = self.block_local_sampled_correction(rho, bundle, assembly_ext_len)?;
        let block_atom = super::atoms::ThetaOnlyCorrectionAtom::from_tk_terms(
            "sampled_block_marginal",
            block_terms,
        );
        let cost_result = self.apply_theta_correction_atom_to_result(cost_result, &block_atom)?;
        // Augment with the ALO stabilization term BEFORE the gradient is read,
        // so the EFS step (which is driven by `cost_result.gradient` and
        // `cost_result.cost`) targets the same stabilized stationarity equation
        // that BFGS/Newton optimize via `assemble_and_evaluate`. Without this
        // the two outer strategies would optimize different objectives on a
        // high-leverage Gaussian-identity fit. `eval_mode` here is always
        // `ValueAndGradient`, so the augmented gradient term flows through.
        let cost_result =
            self.apply_alo_stabilization_to_result(rho, bundle, eval_mode, cost_result)?;
        self.store_ift_mode_response_cache_from_result(rho, bundle, &cost_result);
        // SINGLE SOURCE OF TRUTH for the outer objective VALUE across optimizer
        // strategies: the EFS `cost` reported here must equal the BFGS/Newton
        // path (`compute_outer_eval_with_order`) and the `eval_cost`
        // line-search value (`compute_cost`), all of which apply the
        // `screening_residual_penalty` barrier. The EFS step itself is driven by
        // the gradient channel (the barrier carries no analytic ρ/ψ-gradient and
        // is inactive at every converged point), so wrapping only the reported
        // `cost` keeps the EFS fixed-point unchanged while making EFS
        // backtracking and the cross-strategy cost comparison consume the
        // identical objective — closing the objective↔gradient desync on the EFS
        // arm too (#1122).
        let efs_cost = screening_residual_penalty(cost_result.cost, bundle.pirls_result.as_ref());
        let gradient =
            cost_result
                .gradient
                .as_ref()
                .ok_or(EstimationError::GradientUnavailable {
                    context: concat!(
                        "[outer-efs-first-order-fallback] EFS needs gradient; ",
                        "switch to BFGS"
                    ),
                    mode: "ValueAndGradient",
                })?;

        let efs_eval = if has_psi {
            let hybrid = compute_hybrid_efs_update(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
            );
            let diagnostics = super::reml_outer_engine::efs_single_loop_diagnostics(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
                &hybrid.steps,
                bundle.pirls_result.relative_gradient_norm(),
            );
            self.record_efs_single_loop_bias(rho, diagnostics)?;
            let psi_gradient = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(ndarray::Array1::from_vec(hybrid.psi_gradient))
            };
            let psi_indices = if hybrid.psi_indices.is_empty() {
                None
            } else {
                Some(hybrid.psi_indices)
            };
            gam_problem::EfsEval {
                cost: efs_cost,
                steps: hybrid.steps,
                beta: Some(beta_for_barrier),
                psi_gradient,
                psi_indices,
                inner_hessian_scale,
                logdet_enclosure_gap: None,
            }
        } else {
            let steps = compute_efs_update(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
            );
            let diagnostics = super::reml_outer_engine::efs_single_loop_diagnostics(
                &inner_solution,
                rho.as_slice().unwrap(),
                gradient.as_slice().unwrap(),
                &steps,
                bundle.pirls_result.relative_gradient_norm(),
            );
            self.record_efs_single_loop_bias(rho, diagnostics)?;
            gam_problem::EfsEval {
                cost: efs_cost,
                steps,
                beta: Some(beta_for_barrier),
                psi_gradient: None,
                psi_indices: None,
                inner_hessian_scale,
                logdet_enclosure_gap: None,
            }
        };

        Ok(efs_eval)
    }

    /// Evaluate the REML/LAML objective (and optionally gradient) through the
    /// unified evaluator, using a pre-obtained eval bundle.
    ///
    /// This is the single bridge from the existing PIRLS infrastructure to the
    /// unified `reml_laml_evaluate` function. Both `compute_cost` and
    /// `compute_gradient` ultimately delegate here, ensuring that cost and
    /// gradient share the exact same formula.
    pub fn evaluate_unified(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError> {
        // Dispatch through `build_auto_assembly` rather than hard-wiring
        // `build_dense_assembly`.
        //
        // `build_dense_assembly` keeps β and H in PIRLS's QS-transformed frame
        // while `self.canonical_penalties` (consumed by the outer gradient via
        // `build_penalty_coords()`) stays in the ORIGINAL basis. The scalar
        // cost is invariant to that rotation, but `coord.apply_penalty(β, λ)`
        // is not: it returns `λ S_orig β_t`, which is neither `λ S_orig β_orig`
        // nor `λ S_t β_t`. That fed a miscalibrated `½ βᵀ Sβ` first partial
        // into BFGS on unconstrained dense problems (n=320k, p=71, duchon),
        // which accepted only 2/200 line-search steps before stalling while
        // an independent numeric check on the same cost showed descent
        // progress. `build_auto_assembly` routes unconstrained
        // `TransformedQs` fits through `build_dense_original_assembly`, which
        // maps β and H back to the original basis so every ingredient agrees
        // with the canonical-penalty coordinate frame.
        // No design-moving ψ on this bridge (pure-ρ cost/gradient): the LLT
        // value-only fast path stays eligible. Pure-ρ fits present exact-KKT
        // (no inner-KKT envelope), so this path is byte-unchanged.
        let assembly = self.build_auto_assembly(rho, bundle, mode, false, false)?;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    /// Sparse-exact bridge: delegates to `build_sparse_assembly` + `assemble_and_evaluate`.
    pub fn evaluate_unified_sparse(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        mode: super::reml_outer_engine::EvalMode,
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError> {
        let assembly = self.build_sparse_assembly(rho, bundle, mode, false)?;
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    pub(crate) fn evaluate_unified_value_only_with_synthetic_ext_count(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        synthetic_ext_count: usize,
        force_sparse: bool,
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError> {
        assert!(synthetic_ext_count > 0);
        let mode = super::reml_outer_engine::EvalMode::ValueOnly;
        // #1376: a synthetic ext count is installed only when the outer vector
        // carries design-moving ψ coordinates (the Matérn/Duchon log-κ / aniso
        // axes whose value-only probes must dimension-match the gradient path).
        // Force the smooth-floored spectral `log|H|` here so this value-only
        // probe evaluates the SAME floored determinant the gradient path
        // differentiates — otherwise the LLT fast path's exact `Σ ln σ` desyncs
        // from the analytic `tr(G_ε Ḣ)` on the ψ block (the headline #1376 gap).
        let mut assembly = if force_sparse {
            self.build_sparse_assembly(rho, bundle, mode, false)?
        } else {
            self.build_auto_assembly(rho, bundle, mode, true, false)?
        };
        let p_dim = assembly.beta.len();
        assembly.ext_coords = (0..synthetic_ext_count)
            .map(|_| super::reml_outer_engine::HyperCoord {
                a: 0.0,
                g: Array1::zeros(p_dim),
                drift: super::reml_outer_engine::HyperCoordDrift::none(),
                ld_s: 0.0,
                b_depends_on_beta: false,
                is_penalty_like: false,
                firth_g: None,
                tk_eta_fixed: None,
                tk_x_fixed: None,
            })
            .collect();
        self.assemble_and_evaluate(rho, bundle, mode, assembly)
    }

    /// Evaluate the unified REML/LAML objective with anisotropic ψ ext_coords
    /// injected into the `InnerSolution`.
    ///
    /// This is the primary bridge for joint [ρ, ψ] optimization of anisotropic
    /// spatial smooths. It:
    /// 1. Obtains the PIRLS bundle at the current ρ (with the design already
    ///    rebuilt for the current ψ).
    /// 2. Builds `HyperCoord` ext_coords from the supplied `DirectionalHyperParam`
    ///    list via the dense/transformed or sparse/native τ builders, depending
    ///    on the active backend.
    /// 3. Builds the full `InnerSolution` (mirroring `evaluate_unified`), injects
    ///    the ext_coords, and calls `reml_laml_evaluate`.
    ///
    /// The caller is responsible for rebuilding the `RemlState` with the design
    /// X(ψ) and penalties S(ψ) corresponding to the current ψ values before
    /// calling this method. The unified evaluator then preserves the backend:
    /// dense spectral paths stay dense, and sparse exact paths stay sparse.
    pub fn evaluate_unified_with_psi_ext(
        &self,
        rho: &Array1<f64>,
        cache_theta: Option<&Array1<f64>>,
        mode: super::reml_outer_engine::EvalMode,
        hyper_dirs: &[crate::estimate::reml::DirectionalHyperParam],
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError> {
        let t0 = std::time::Instant::now();
        let bundle = if let Some(theta) = cache_theta {
            self.obtain_eval_bundle_for_outer_theta(rho, theta)?
        } else {
            self.obtain_eval_bundle(rho)?
        };
        let pirls_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let (ext_coords, ext_pair_fn, rho_ext_pair_fn, fixed_drift_deriv) = if !hyper_dirs
            .is_empty()
        {
            if mode == super::reml_outer_engine::EvalMode::ValueGradientHessian {
                let (coords, epf, repf, fixed_drift_deriv) =
                    self.build_tau_unified_objects_from_bundle(rho, &bundle, hyper_dirs)?;
                (coords, Some(epf), Some(repf), fixed_drift_deriv)
            } else if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
                (
                    self.build_tau_hyper_coords_sparse_exact(rho, &bundle, hyper_dirs, false)?,
                    None,
                    None,
                    None,
                )
            } else if matches!(
                bundle.pirls_result.coordinate_frame,
                pirls::PirlsCoordinateFrame::TransformedQs
            ) && self
                .active_constraint_free_basis(bundle.pirls_result.as_ref())
                .is_none()
            {
                (
                    self.build_tau_hyper_coords_original_basis(rho, &bundle, hyper_dirs, false)?,
                    None,
                    None,
                    None,
                )
            } else {
                (
                    self.build_tau_hyper_coords(rho, &bundle, hyper_dirs, false)?,
                    None,
                    None,
                    None,
                )
            }
        } else {
            (Vec::new(), None, None, None)
        };
        let tau_build_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let t2 = std::time::Instant::now();
        // #1376: when this evaluation carries a design-moving ψ coordinate
        // (e.g. the Matérn/Duchon log-κ axis), force the smooth-floored spectral
        // `log|H|` even on a value-only probe so the cost matches the floored
        // determinant the gradient path differentiates via `tr(G_ε Ḣ)`.
        let force_spectral_logdet = ext_coords.iter().any(|c| !c.is_penalty_like);
        // ψ/aniso ext coordinates are equally hypersensitive to a capped inner
        // β̂, but engaging the envelope correction here touches the whole
        // Matérn/Duchon/tensor ψ suite; keep this path on the exact-KKT
        // assumption for now (the SAS/mixture link-ext path below carries the
        // #1876 fix). Present exact-KKT (populate_inner_kkt = false).
        let mut assembly =
            self.build_auto_assembly(rho, &bundle, mode, force_spectral_logdet, false)?;
        assembly.ext_coords = ext_coords;
        assembly.ext_coord_pair_fn = ext_pair_fn;
        assembly.rho_ext_pair_fn = rho_ext_pair_fn;
        assembly.fixed_drift_deriv = fixed_drift_deriv;
        let result = self.assemble_and_evaluate(rho, &bundle, mode, assembly);
        let reml_eval_ms = t2.elapsed().as_secs_f64() * 1000.0;

        log::debug!(
            "[outer-timing] evaluate_unified_with_psi_ext: PIRLS={:.1}ms  tau_build={:.1}ms  reml_eval={:.1}ms  total={:.1}ms",
            pirls_ms,
            tau_build_ms,
            reml_eval_ms,
            t0.elapsed().as_secs_f64() * 1000.0,
        );
        result
    }

    /// Compute EFS (Extended Fellner-Schall) evaluation: runs the inner solve
    /// at the given rho, computes the REML/LAML cost, and returns the EFS
    /// step vector from `compute_efs_update`.
    ///
    /// This is the entry point called by the EFS branch of `run_outer` via
    /// the `OuterObjective::eval_efs` method.
    pub fn compute_efs_steps(
        &self,
        p: &Array1<f64>,
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        if self.large_n_efs_single_loop_lane() {
            self.cache_manager.invalidate_eval_bundle();
            let previous_cap = self
                .outer_inner_cap
                .swap(efs_single_loop_encoded_cap(), Ordering::Relaxed);
            let result = self.compute_efs_steps_inner(p);
            self.outer_inner_cap.store(previous_cap, Ordering::Relaxed);
            self.cache_manager.invalidate_eval_bundle();
            return result;
        }

        self.compute_efs_steps_inner(p)
    }

    pub(crate) fn compute_efs_steps_inner(
        &self,
        p: &Array1<f64>,
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(EstimationError::RemlOptimizationFailed(
                    "inner solve ill-conditioned during EFS evaluation".to_string(),
                ));
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };

        self.evaluate_efs(p, &bundle, Vec::new())
    }

    /// EFS evaluation with anisotropic or isotropic ψ ext_coords injected.
    pub fn compute_efs_steps_with_psi_ext(
        &self,
        rho: &Array1<f64>,
        hyper_dirs: &[crate::estimate::reml::DirectionalHyperParam],
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        if self.large_n_efs_single_loop_lane() {
            self.cache_manager.invalidate_eval_bundle();
            let previous_cap = self
                .outer_inner_cap
                .swap(efs_single_loop_encoded_cap(), Ordering::Relaxed);
            let result = self.compute_efs_steps_with_psi_ext_inner(rho, hyper_dirs);
            self.outer_inner_cap.store(previous_cap, Ordering::Relaxed);
            self.cache_manager.invalidate_eval_bundle();
            return result;
        }
        self.compute_efs_steps_with_psi_ext_inner(rho, hyper_dirs)
    }

    pub(crate) fn compute_efs_steps_with_psi_ext_inner(
        &self,
        rho: &Array1<f64>,
        hyper_dirs: &[crate::estimate::reml::DirectionalHyperParam],
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        let bundle = match self.obtain_eval_bundle(rho) {
            Ok(bundle) => bundle,
            Err(EstimationError::ModelIsIllConditioned { .. }) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(EstimationError::RemlOptimizationFailed(
                    "inner solve ill-conditioned during psi-ext EFS evaluation".to_string(),
                ));
            }
            Err(e) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(e);
            }
        };

        let ext_coords = if !hyper_dirs.is_empty() {
            if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
                self.build_tau_hyper_coords_sparse_exact(rho, &bundle, hyper_dirs, false)?
            } else if matches!(
                bundle.pirls_result.coordinate_frame,
                pirls::PirlsCoordinateFrame::TransformedQs
            ) && self
                .active_constraint_free_basis(bundle.pirls_result.as_ref())
                .is_none()
            {
                self.build_tau_hyper_coords_original_basis(rho, &bundle, hyper_dirs, false)?
            } else {
                self.build_tau_hyper_coords(rho, &bundle, hyper_dirs, false)?
            }
        } else {
            Vec::new()
        };

        if ext_coords.is_empty() {
            return self.compute_efs_steps(rho);
        }

        self.evaluate_efs(rho, &bundle, ext_coords)
    }

    pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        self.arena
            .lastgradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
        let t_eval_start = std::time::Instant::now();
        {
            let prefix: Vec<String> = p.iter().take(4).map(|v| format!("{:.3}", v)).collect();
            log::debug!(
                "[REML] grad-only begin | rho[..4]=[{}] | k={}",
                prefix.join(","),
                p.len()
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::debug!(
                "[REML] grad-only cache hit | |g| {:.3e} | elapsed {:.1}ms",
                gnorm,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(eval.gradient);
        }
        let t_pirls = std::time::Instant::now();
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
        let pirls_ms = t_pirls.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "[REML] grad-only pirls done | elapsed {:.1}ms | backend {:?}",
            pirls_ms,
            bundle.backend_kind()
        );
        let t_assemble = std::time::Instant::now();
        if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            // Exact analytic LAML ρ-gradient for any penalty layout, including
            // repeated/shared col_range `by`-factor smooths. `rho_derivatives_
            // from_penalties` maps every component into its merged eigenspace
            // block span and forms tr(W'S_kW) and the cross-terms tr(Y_kY_l)
            // within that shared span, so penalties that share a coefficient
            // block contribute the correct coupled logdet derivative. This is
            // the exact gradient of the (un-barriered) REML cost `result.cost`;
            // the screening barrier `+0.5·r_g²` carries no ρ-gradient and is a
            // partial-inner-state tie-break, not part of the objective being
            // descended, so the analytic gradient is the honest REML descent
            // direction even at a screened seed.
            let result = self.evaluate_unified_sparse(
                p,
                &bundle,
                super::reml_outer_engine::EvalMode::ValueAndGradient,
            )?;
            let ift_residual_energy = result.ift_residual_energy;
            store_ift_residual_energy_for_outer_theta(p, ift_residual_energy);
            let grad = result
                .gradient
                .ok_or(EstimationError::GradientUnavailable {
                    context: "REML sparse gradient evaluation requires gradient",
                    mode: "ValueAndGradient",
                })?;
            let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::debug!(
                "[REML] grad-only sparse done | |g| {:.3e} | assemble {:.1}ms | total {:.1}ms",
                gnorm,
                t_assemble.elapsed().as_secs_f64() * 1000.0,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            self.update_hypergradient_budget_after_outer_eval(p, &grad, ift_residual_energy);
            return Ok(grad);
        }
        let result = self.evaluate_unified(
            p,
            &bundle,
            super::reml_outer_engine::EvalMode::ValueAndGradient,
        )?;
        let ift_residual_energy = result.ift_residual_energy;
        store_ift_residual_energy_for_outer_theta(p, ift_residual_energy);
        let grad = result
            .gradient
            .ok_or(EstimationError::GradientUnavailable {
                context: "REML dense gradient evaluation requires gradient",
                mode: "ValueAndGradient",
            })?;
        let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        log::debug!(
            "[REML] grad-only dense done | |g| {:.3e} | assemble {:.1}ms | total {:.1}ms",
            gnorm,
            t_assemble.elapsed().as_secs_f64() * 1000.0,
            t_eval_start.elapsed().as_secs_f64() * 1000.0
        );
        self.update_hypergradient_budget_after_outer_eval(p, &grad, ift_residual_energy);
        Ok(grad)
    }

    pub(crate) fn compute_cost_and_gradient(
        &self,
        p: &Array1<f64>,
    ) -> Result<(f64, Array1<f64>), EstimationError> {
        let eval = self.compute_outer_eval_with_order(
            p,
            crate::rho_optimizer::OuterEvalOrder::ValueAndGradient,
        )?;
        Ok((eval.cost, eval.gradient))
    }

    pub fn compute_outer_eval_with_order(
        &self,
        p: &Array1<f64>,
        order: crate::rho_optimizer::OuterEvalOrder,
    ) -> Result<OuterEval, EstimationError> {
        self.arena
            .lastgradient_used_stochastic_fallback
            .store(false, Ordering::Relaxed);
        let allow_second_order = matches!(
            order,
            crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian
        ) && self.analytic_outer_hessian_enabled();
        let t_eval_start = std::time::Instant::now();
        {
            let prefix: Vec<String> = p.iter().take(4).map(|v| format!("{:.3}", v)).collect();
            log::debug!(
                "[REML] outer-eval begin {:?} | rho[..4]=[{}] | k={} | 2nd-order={}",
                order,
                prefix.join(","),
                p.len(),
                allow_second_order
            );
        }
        let rho_key = EvalCacheManager::sanitized_rhokey(p);
        if let Some(eval) = self.cache_manager.cached_outer_eval(&rho_key) {
            let cache_satisfies_request = !allow_second_order || eval.hessian.is_analytic();
            if cache_satisfies_request {
                let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
                log::debug!(
                    "[REML] outer-eval cache hit | cost {:.6e} | |g| {:.3e} | elapsed {:.1}ms",
                    eval.cost,
                    gnorm,
                    t_eval_start.elapsed().as_secs_f64() * 1000.0
                );
                return Ok(eval);
            }
        }

        let t_pirls = std::time::Instant::now();
        let bundle = match self.obtain_eval_bundle(p) {
            Ok(bundle) => bundle,
            Err(err) if err.is_inner_solve_retreat() => {
                self.cache_manager.invalidate_eval_bundle();
                log::debug!(
                    "P-IRLS inner-solve retreat at current rho ({}); returning infeasible outer eval.",
                    err
                );
                return Ok(OuterEval::infeasible(p.len()));
            }
            Err(err) => {
                self.cache_manager.invalidate_eval_bundle();
                return Err(err);
            }
        };

        // Genuinely value-only fulfilment (#979). A `Value` request never needs
        // the outer gradient — the continuation pre-warm asks for it purely to
        // run the inner P-IRLS and warm `warm_start_beta`, which the inner solve
        // above (`obtain_eval_bundle`) has already done. The previous code rode
        // the value+gradient path for `Value`, paying the full k²·n·p² LAML
        // gradient assembly at EVERY continuation step (the dominant cost of the
        // ~35s/seed marginal-slope pre-warm and the centers=20 non-finish). Skip
        // the gradient assembly entirely: assemble the cost with `ValueOnly`,
        // return a zero-length gradient (no outer optimiser ever consumes a
        // `Value`-order gradient — the runner only requests `Value` for cost
        // probes / pre-warm), and surface the warmed β so the continuation walk's
        // `inner_beta_hint` forwarding is unchanged. ValueAndGradient and
        // ValueGradientHessian are byte-identical to before.
        if matches!(order, crate::rho_optimizer::OuterEvalOrder::Value) {
            let t_assemble = std::time::Instant::now();
            let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
                self.evaluate_unified_sparse(
                    p,
                    &bundle,
                    super::reml_outer_engine::EvalMode::ValueOnly,
                )?
            } else {
                self.evaluate_unified(p, &bundle, super::reml_outer_engine::EvalMode::ValueOnly)?
            };
            store_ift_residual_energy_for_outer_theta(p, result.ift_residual_energy);
            // SINGLE SOURCE OF TRUTH for the outer objective VALUE: apply the
            // exact same `screening_residual_penalty` wrap that `compute_cost`
            // (the `eval_cost` line-search value) applies. Without this, a ρ/ψ
            // at which the inner P-IRLS caps out (`is_failed_max_iterations`)
            // would be reported with cost `V + 0.5·r_g²` through `eval_cost`
            // but raw `V` through this `OuterEval` path — the classic
            // objective↔gradient desync that lets the line-search and the
            // gradient descend two different functions (the #1122 matern stall
            // signature: nonzero final_grad_norm at a capped-inner regime).
            let cost = screening_residual_penalty(result.cost, bundle.pirls_result.as_ref());
            log::debug!(
                "[REML] outer-eval value-only done | cost {:.6e} | assemble {:.1}ms | total {:.1}ms",
                cost,
                t_assemble.elapsed().as_secs_f64() * 1000.0,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
            // Deliberately NOT cached into `store_outer_eval`: that cache serves
            // gradient/Hessian-bearing requests, and a gradient-free entry would
            // force a re-evaluation the moment the optimiser asks for a real
            // gradient at the same ρ. A value-only probe is cheap to repeat.
            return Ok(OuterEval::value_only(
                cost,
                p.len(),
                self.current_original_basis_beta(),
            ));
        }

        let decision = match order {
            // Value+gradient: this evaluator's assembly contract requires a
            // gradient (see the `result.gradient` demand below), so fulfil it as
            // value+gradient with the Hessian skipped.
            crate::rho_optimizer::OuterEvalOrder::Value
            | crate::rho_optimizer::OuterEvalOrder::ValueAndGradient => None,
            crate::rho_optimizer::OuterEvalOrder::ValueGradientHessian => {
                if allow_second_order {
                    Some(self.selecthessian_strategy_policy(&bundle))
                } else {
                    None
                }
            }
        };
        let eval_mode = match decision.as_ref().map(|decision| decision.strategy) {
            Some(HessianEvalStrategyKind::SpectralExact) => {
                super::reml_outer_engine::EvalMode::ValueGradientHessian
            }
            _ => super::reml_outer_engine::EvalMode::ValueAndGradient,
        };

        let pirls_ms = t_pirls.elapsed().as_secs_f64() * 1000.0;
        log::debug!(
            "[REML] outer-eval pirls done | elapsed {:.1}ms | backend {:?} | mode {:?}",
            pirls_ms,
            bundle.backend_kind(),
            eval_mode
        );

        let t_assemble = std::time::Instant::now();
        let result = if bundle.backend_kind() == GeometryBackendKind::SparseExactSpd {
            self.evaluate_unified_sparse(p, &bundle, eval_mode)?
        } else {
            self.evaluate_unified(p, &bundle, eval_mode)?
        };
        let assemble_ms = t_assemble.elapsed().as_secs_f64() * 1000.0;
        let ift_residual_energy = result.ift_residual_energy;
        store_ift_residual_energy_for_outer_theta(p, ift_residual_energy);

        let gradient = result.gradient.ok_or_else(|| {
            EstimationError::InvalidInput(format!(
                "unified evaluator returned no gradient in {:?} mode",
                eval_mode
            ))
        })?;

        let hessian = match decision.map(|decision| decision.strategy) {
            Some(HessianEvalStrategyKind::SpectralExact) => result.hessian,
            None => HessianValue::Unavailable,
        };

        // SINGLE SOURCE OF TRUTH for the outer objective VALUE: this
        // value+gradient (and value+gradient+Hessian) path must report the
        // EXACT same cost as `compute_cost` (the `eval_cost` line-search value),
        // so apply the identical `screening_residual_penalty` wrap. The penalty
        // is a continuous barrier (`+0.5·r_g²`) that is non-zero ONLY when the
        // inner P-IRLS capped out at this ρ/ψ; it vanishes at every converged
        // point, so the analytic `gradient` below — which differentiates the
        // un-penalized LAML objective `V` — is the exact gradient wherever the
        // barrier is inactive. Reporting raw `V` here while `eval_cost` reports
        // `V + 0.5·r_g²` was an objective↔gradient desync: the line-search and
        // the BFGS direction descended two different functions at capped-inner
        // ρ/ψ, producing the stall-with-nonzero-final-grad-norm signature of
        // the #1122 matern iso-κ optimizer.
        let cost = screening_residual_penalty(result.cost, bundle.pirls_result.as_ref());
        // The reported cost carries the screening barrier `+0.5·r_g²` (active
        // only on a partial, screened inner state) to keep the seed-screening
        // VALUE monotonic, but the gradient is the exact analytic ρ-gradient of
        // the un-barriered REML objective on every backend — including the
        // sparse-exact `by`-factor layout, whose shared-block logdet derivative
        // is handled exactly by `rho_derivatives_from_penalties`. The barrier
        // has no ρ-gradient and vanishes at every converged point, so this is
        // the honest REML descent direction; no numerical-difference fallback.
        let eval = OuterEval {
            cost,
            gradient,
            hessian,
            inner_beta_hint: self.current_original_basis_beta(),
        };
        {
            let gnorm = eval.gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            log::debug!(
                "[REML] outer-eval done | cost {:.6e} | |g| {:.3e} | assemble {:.1}ms | total {:.1}ms",
                eval.cost,
                gnorm,
                assemble_ms,
                t_eval_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.update_hypergradient_budget_after_outer_eval(p, &eval.gradient, ift_residual_energy);
        self.cache_manager.store_outer_eval(&rho_key, &eval);
        Ok(eval)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Unified REML evaluation with link-parameter ext_coords
    // ═══════════════════════════════════════════════════════════════════════════

    /// Evaluate the unified REML/LAML objective with SAS or mixture link
    /// parameters injected as ext_coords into the `InnerSolution`.
    ///
    /// This replaces the manual gradient computation in the outer optimizer's
    /// eval_fn closure. The link parameter derivatives (du/dθ, dW/dθ, dℓ/dθ)
    /// are wrapped as `HyperCoord` objects and passed through the same unified
    /// evaluator that handles ρ coordinates, giving a single Newton/BFGS step
    /// over `[ρ, θ_link]` jointly.
    ///
    /// # Parameters
    ///
    /// - `rho`: log-smoothing parameters (penalty coordinates only, length k)
    /// - `mode`: evaluation mode (cost-only, cost+gradient, cost+gradient+Hessian)
    ///
    /// # Returns
    ///
    /// `RemlLamlResult` whose gradient (when requested) has length `k + aux_dim`,
    /// with the link parameters appended after the ρ coordinates. The caller is
    /// responsible for:
    /// - Applying SAS epsilon reparameterization chain rule (`grad[k] *= d_eps/d_raw`)
    /// - Adding SAS ridge/barrier gradient contributions to `grad[k+1]`
    /// - Adding SAS ridge/barrier cost contributions to `result.cost`
    /// Build link ext_coords from the current runtime link state.
    pub(crate) fn build_link_ext_coords(
        &self,
        bundle: &EvalShared,
    ) -> Result<Vec<super::reml_outer_engine::HyperCoord>, EstimationError> {
        if let Some(sas_state) = &self.runtime_sas_link_state {
            let is_beta_logistic = matches!(
                self.config.link_function(),
                gam_problem::LinkFunction::BetaLogistic
            );
            self.build_sas_link_ext_coords(bundle, sas_state, is_beta_logistic)
        } else if let Some(mix_state) = &self.runtime_mixture_link_state {
            self.build_mixture_link_ext_coords(bundle, mix_state)
        } else {
            Ok(Vec::new())
        }
    }

    /// Check that Firth/Jeffreys is not active (incompatible with link ext_coords).
    pub(crate) fn reject_firth_link_ext(&self) -> Result<(), EstimationError> {
        if reml_robust_jeffreys_link(&self.config).is_some() {
            crate::bail_invalid_estim!(
                "link-parameter ext_coord optimization is incompatible with \
                 Firth-adjusted outer gradients"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Rotate link-parameter `HyperCoord`s from the PIRLS transformed basis
    /// into the original coefficient basis, gated on the same predicate
    /// `build_auto_assembly` uses to select `build_dense_original_assembly`.
    ///
    /// The SAS / mixture / beta-logistic link-ext builders
    /// (`build_sas_link_ext_coords`, `build_mixture_link_ext_coords`) form
    /// `g_j = −X_tᵀ·du/dθ_j` and `B_j = X_tᵀ·diag(dW/dθ_j)·X_t` from
    /// `pirls_result.x_transformed`, i.e. in the transformed basis.  When
    /// PIRLS is in `TransformedQs` with no active constraints,
    /// `build_auto_assembly` routes through `build_dense_original_assembly`
    /// — which reports `β` and `H` in the ORIGINAL basis.  Attaching
    /// transformed-basis `(g_j, B_j)` to an original-basis assembly would
    /// evaluate `coord.g·β` and `coord.drift·v` across mismatched frames.
    ///
    /// Under an orthogonal rotation `Qs` with `β_original = Qs·β_transformed`
    /// (matching `sparse_exact_beta_original` /
    /// `bundle_matrix_in_original_basis`), the covariant / bilinear
    /// transformation laws give
    ///
    ///   g_original = Qs · g_transformed                   (covariant vector)
    ///   B_original = Qs · B_transformed · Qsᵀ            (bilinear form)
    ///
    /// derived from scalar invariance `⟨g_o, β_o⟩ = ⟨g_t, β_t⟩` and
    /// `β_oᵀ B_o β_o = β_tᵀ B_t β_t`, both under `β_o = Qs β_t`.
    ///
    /// Scalars (`a`, `ld_s`) and semantic flags (`b_depends_on_beta`,
    /// `is_penalty_like`) are basis-invariant by construction.  The
    /// `HyperCoordDrift::block_local` and `operator` variants are not
    /// handled by this helper — the link-ext builders must emit only
    /// the `dense` variant (`HyperCoordDrift::from_dense`), and the
    /// helper returns an error if that invariant is broken so a future
    /// builder change trips it immediately rather than silently
    /// regressing SAS/mixture optimisation.
    pub(crate) fn rotate_link_ext_coords_to_original(
        &self,
        bundle: &EvalShared,
        coords: &mut [super::reml_outer_engine::HyperCoord],
    ) -> Result<(), EstimationError> {
        if coords.is_empty() {
            return Ok(());
        }
        let pirls_result = bundle.pirls_result.as_ref();
        // Sparse-native (OriginalSparseNative) path is structurally
        // identity-rotated: `make_reparam_operator` sets
        // `x_transformed = x_original` and `sparse_exact_beta_original`
        // returns β verbatim, so link-ext coords built from
        // `pirls_result.x_transformed` are already in the original basis
        // that `build_sparse_assembly` reports `β` and `H` in.  The gate
        // below therefore returns Ok(()) without touching coords whenever
        // the frame is OriginalSparseNative — the analogue of 4ec0208f's
        // dense rotation is a no-op on sparse, not a missing fix.
        let needs_rotation = matches!(
            pirls_result.coordinate_frame,
            pirls::PirlsCoordinateFrame::TransformedQs
        ) && self.active_constraint_free_basis(pirls_result).is_none();
        if !needs_rotation {
            return Ok(());
        }
        let qs = &pirls_result.reparam_result.qs;
        let qs_t = qs.t();
        for (coord_idx, coord) in coords.iter_mut().enumerate() {
            if coord.g.len() == qs.nrows() {
                coord.g = qs.dot(&coord.g);
            }
            // Invariant guard: link-ext builders must emit ONLY the
            // `dense` variant of `HyperCoordDrift`.  `block_local` and
            // `operator` variants would require distinct rotations
            // (block-local would lose its block structure under
            // arbitrary `Qs`; operator-backed drifts would need a
            // wrapping `RotatedHyperOperator`).  If a future link-ext
            // builder ever populates either variant, this guard
            // refuses rather than silently running a partial rotation
            // that regresses SAS/mixture optimisation.
            if coord.drift.block_local.is_some() || coord.drift.operator.is_some() {
                crate::bail_invalid_estim!(
                    "link-ext HyperCoord[{coord_idx}] carries a non-dense drift \
                     variant (block_local={} operator={}); the coord-frame rotation \
                     helper only handles `HyperCoordDrift::from_dense`.  Update the \
                     link-ext builder (or extend `rotate_link_ext_coords_to_original`) \
                     before attaching non-dense drifts to original-basis assemblies.",
                    coord.drift.block_local.is_some(),
                    coord.drift.operator.is_some(),
                );
            }
            if let Some(b) = coord.drift.dense.as_mut()
                && b.nrows() == qs.nrows()
                && b.ncols() == qs.nrows()
            {
                // B_original = Qs · B_transformed · Qsᵀ
                let tmp = qs.dot(&*b);
                *b = tmp.dot(&qs_t);
            }
        }
        Ok(())
    }

    pub fn evaluate_unified_with_link_ext(
        &self,
        rho: &Array1<f64>,
        mode: super::reml_outer_engine::EvalMode,
    ) -> Result<super::reml_outer_engine::RemlLamlResult, EstimationError> {
        self.reject_firth_link_ext()?;
        let bundle = self.obtain_eval_bundle(rho)?;
        let mut ext_coords = self.build_link_ext_coords(&bundle)?;

        if ext_coords.is_empty() {
            return self.evaluate_unified(rho, &bundle, mode);
        }

        // Rotate link-ext coords to match the basis `build_auto_assembly`
        // will pick for the assembly (original basis under TransformedQs +
        // no active constraints).  See `rotate_link_ext_coords_to_original`.
        self.rotate_link_ext_coords_to_original(&bundle, &mut ext_coords)?;

        // #1376: a non-empty ext block here means a value-only probe must agree
        // with the gradient path's smooth-floored `log|H|`; force the spectral
        // logdet so the LLT exact-determinant fast path cannot desync the cost.
        let force_spectral_logdet = ext_coords.iter().any(|c| !c.is_penalty_like);
        // #1876: the SAS/mixture flexible-link optimizer accepts the inner β̂ at
        // a first-order cap (`outer_inner_cap`); the link-parameter gradient is
        // hypersensitive to that capped β̂ (`coord.g ~ 1e4`+), so a ~1e-4 β̂ error
        // silently collapses the ε gradient to near-zero and the optimizer
        // stalls at the wrong skewness. Engage the inner-KKT envelope correction
        // `Ṽ = V − ½·rᵀH⁻¹r` (populate_inner_kkt = true) so the outer value AND
        // gradient are the true stationary-mode quantities rather than the raw
        // capped-β̂ values. The optimizer's cost closure routes through this same
        // evaluator (value-only), so cost and gradient stay in sync. The
        // correction vanishes as `r → 0`, so a converged inner solve is
        // unchanged.
        let mut assembly =
            self.build_auto_assembly(rho, &bundle, mode, force_spectral_logdet, true)?;
        let ext_dim = ext_coords.len();
        let p_dim = ext_coords.first().map(|coord| coord.g.len()).unwrap_or(0);
        assembly.ext_coords = ext_coords;
        if mode == super::reml_outer_engine::EvalMode::ValueGradientHessian {
            // Link-shape parameters do not directly differentiate the penalty
            // matrices, so their explicit rho-link second partials are zero.
            // Installing this callback is still required: the unified Hessian
            // builder then evaluates the nonzero profiled cross terms from the
            // first-order link drifts and IFT mode responses.
            assembly.rho_ext_pair_fn = Some(Box::new(move |_, _| {
                super::reml_outer_engine::HyperCoordPair {
                    a: 0.0,
                    g: Array1::zeros(p_dim),
                    b_mat: Array2::zeros((p_dim, p_dim)),
                    b_operator: None,
                    ld_s: 0.0,
                }
            }));
            assembly.ext_coord_pair_fn = Some(Box::new(move |_, _| {
                super::reml_outer_engine::HyperCoordPair {
                    a: 0.0,
                    g: Array1::zeros(p_dim),
                    b_mat: Array2::zeros((p_dim, p_dim)),
                    b_operator: None,
                    ld_s: 0.0,
                }
            }));
            assert!(ext_dim > 0);
        }
        self.assemble_and_evaluate(rho, &bundle, mode, assembly)
    }

    /// EFS evaluation with link-parameter ext_coords injected.
    pub fn compute_efs_steps_with_link_ext(
        &self,
        rho: &Array1<f64>,
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        self.reject_firth_link_ext()?;
        let bundle = self.obtain_eval_bundle(rho)?;
        let mut ext_coords = self.build_link_ext_coords(&bundle)?;

        if ext_coords.is_empty() {
            return self.compute_efs_steps(rho);
        }

        // Same rotation gate as `evaluate_unified_with_link_ext`: the EFS
        // path eventually builds its assembly via `build_auto_assembly`
        // inside `evaluate_efs`, which picks original basis under the same
        // predicate.
        self.rotate_link_ext_coords_to_original(&bundle, &mut ext_coords)?;

        self.evaluate_efs(rho, &bundle, ext_coords)
    }

    /// Unified EFS bridge: auto-dispatches backend, injects ext_coords,
    /// and evaluates via `assemble_and_evaluate_efs`.
    pub(crate) fn evaluate_efs(
        &self,
        rho: &Array1<f64>,
        bundle: &EvalShared,
        ext_coords: Vec<super::reml_outer_engine::HyperCoord>,
    ) -> Result<gam_problem::EfsEval, EstimationError> {
        // #1376: force the smooth-floored spectral `log|H|` whenever a
        // design-moving ψ ext coordinate is present so the value matches the
        // gradient path's floored determinant (the LLT exact-`Σ ln σ` fast path
        // desyncs from `tr(G_ε Ḣ)` on the ψ block otherwise).
        let force_spectral_logdet = ext_coords.iter().any(|c| !c.is_penalty_like);
        let mut assembly = self.build_auto_assembly(
            rho,
            bundle,
            super::reml_outer_engine::EvalMode::ValueOnly,
            force_spectral_logdet,
            // EFS is a universal-form fixed-point λ update, not a
            // gradient-descent step against the corrected objective; leave the
            // inner-KKT envelope correction to the gradient/Hessian path.
            false,
        )?;
        assembly.ext_coords = ext_coords;
        self.assemble_and_evaluate_efs(rho, bundle, assembly)
    }
}

/// ρ-independent certificate that the Gaussian-identity ALO-stabilization
/// augmentation can never activate (#1689).
///
/// The augmentation engages only when some row's *penalized* leverage
/// `h_i = w_i · xᵢᵀ H_λ⁻¹ xᵢ` reaches `threshold`, where the penalized Hessian
/// `H_λ = XᵀWX + S_λ + ridge·I ⪰ XᵀWX` (the penalty and ridge are PSD). By
/// Loewner monotonicity `H_λ⁻¹ ⪯ (XᵀWX)⁻¹`, so for every ρ
/// `h_i ≤ w_i · xᵢᵀ (XᵀWX)⁻¹ xᵢ =: h̄_i`, the *unpenalized* weighted hat
/// diagonal. Returns `true` iff `max_i h̄_i < threshold` — in which case no ρ
/// can trip the activation gate. Returns `false` (cannot certify) when the
/// design is empty, the weights are inconsistent / non-finite / negative, the
/// unpenalized weighted Gram `XᵀWX` is not factorizable (the bound degenerates),
/// or the bound reaches the threshold. A `false` result is never wrong — the
/// caller then falls through to the exact per-evaluation leverage gate.
///
/// The leverage `xᵢᵀ H⁻¹ xᵢ` is invariant under invertible column
/// reparameterization, so this may be evaluated in the original design basis
/// with the fixed prior weights `W` (for Gaussian identity these equal the
/// converged Hessian weights and are independent of ρ).
pub(crate) fn unpenalized_hat_bound_below_threshold(
    x: &Array2<f64>,
    weights: ArrayView1<'_, f64>,
    threshold: f64,
) -> bool {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 || weights.len() != n || !(threshold.is_finite() && threshold > 0.0) {
        return false;
    }
    // Unpenalized weighted Gram G = XᵀWX. Any non-finite or negative weight
    // makes the monotone bound uncertifiable.
    let mut gram = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let wi = weights[i];
        if !wi.is_finite() || wi < 0.0 {
            return false;
        }
        if wi == 0.0 {
            continue;
        }
        for a in 0..p {
            let xa = wi * x[[i, a]];
            for b in a..p {
                gram[[a, b]] += xa * x[[i, b]];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            gram[[a, b]] = gram[[b, a]];
        }
    }
    let factor = match StableSolver::new("ALO non-activation certificate").factorize(&gram) {
        Ok(factor) => factor,
        // Rank-deficient / ill-conditioned unpenalized Gram: the monotone bound
        // degenerates, so we cannot certify.
        Err(_) => return false,
    };
    let mut gram_inv = Array2::<f64>::eye(p);
    let mut gram_inv_view = array2_to_matmut(&mut gram_inv);
    factor.solve_in_place(gram_inv_view.as_mut());
    // h̄_i = w_i · xᵢᵀ G⁻¹ xᵢ ≥ h_i for every ρ.
    for i in 0..n {
        let wi = weights[i];
        if wi <= 0.0 {
            continue;
        }
        let mut quad = 0.0f64;
        for a in 0..p {
            let xa = x[[i, a]];
            if xa == 0.0 {
                continue;
            }
            let mut row_acc = 0.0f64;
            for b in 0..p {
                row_acc += gram_inv[[a, b]] * x[[i, b]];
            }
            quad = xa.mul_add(row_acc, quad);
        }
        let hbar = wi * quad;
        if !hbar.is_finite() || hbar >= threshold {
            return false;
        }
    }
    true
}

pub(crate) fn positive_penalty_rank_and_logdet(eigenvalues: &[f64]) -> (usize, f64) {
    let threshold = super::reml_outer_engine::positive_eigenvalue_threshold(eigenvalues);
    let rank = eigenvalues.iter().filter(|&&ev| ev > threshold).count();
    let log_det = super::reml_outer_engine::exact_pseudo_logdet(eigenvalues, threshold);
    (rank, log_det)
}

#[cfg(test)]
mod tk_math_tests {
    use super::*;
    use crate::mixture_link::{
        beta_logistic_inverse_link_jet, beta_logistic_inverse_link_pdffourth_derivative,
        beta_logistic_inverse_link_pdfthird_derivative, inverse_link_jet_for_inverse_link,
        inverse_link_pdffourth_derivative_for_inverse_link,
        inverse_link_pdfthird_derivative_for_inverse_link, mixture_inverse_link_jet,
        sas_inverse_link_jet, sas_inverse_link_pdffourth_derivative,
        sas_inverse_link_pdfthird_derivative, state_fromspec,
    };
    use crate::pirls::{VarianceJet, e_obs_from_jets};
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;
    use gam_problem::{LinkComponent, MixtureLinkSpec};
    use ndarray::array;
    use num_dual::{Dual3_64, Dual64, DualNum, third_derivative};

    /// The unpenalized hat bound h̄_i = w_i·xᵢᵀ(XᵀWX)⁻¹xᵢ must dominate the
    /// penalized leverage h_i(λ) = w_i·xᵢᵀ(XᵀWX + λS)⁻¹xᵢ for EVERY λ ≥ 0 and
    /// every row — the soundness property the #1689 non-activation certificate
    /// relies on. We check it against the exact penalized leverage at a spread
    /// of λ on a deterministic design, so a future change that breaks the
    /// monotone-bound direction reddens here rather than silently skipping a
    /// stabilization that should have engaged.
    #[test]
    pub(crate) fn unpenalized_hat_bound_dominates_penalized_leverage_for_all_lambda() {
        let n = 40usize;
        let p = 5usize;
        // Deterministic, well-conditioned design + a PSD penalty S = DᵀD.
        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            let t = i as f64 / (n as f64 - 1.0);
            (1.7 * t * (j as f64 + 1.0) + 0.3 * j as f64).sin() + 0.11 * (j as f64)
        });
        let weights = Array1::from_shape_fn(n, |i| 0.4 + 0.5 * ((i as f64) * 0.21).sin().abs());
        let mut s = Array2::<f64>::zeros((p, p));
        for k in 0..p - 1 {
            // second-difference-like penalty block contribution
            s[[k, k]] += 1.0;
            s[[k + 1, k + 1]] += 1.0;
            s[[k, k + 1]] -= 1.0;
            s[[k + 1, k]] -= 1.0;
        }
        // Unpenalized weighted Gram and its bound diagonal.
        let mut gram0 = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            for a in 0..p {
                for b in 0..p {
                    gram0[[a, b]] += weights[i] * x[[i, a]] * x[[i, b]];
                }
            }
        }
        let g0_factor = StableSolver::new("test gram0").factorize(&gram0).unwrap();
        let mut g0_inv = Array2::<f64>::eye(p);
        let mut g0_inv_v = array2_to_matmut(&mut g0_inv);
        g0_factor.solve_in_place(g0_inv_v.as_mut());
        let hbar: Vec<f64> = (0..n)
            .map(|i| {
                let mut q = 0.0;
                for a in 0..p {
                    for b in 0..p {
                        q += x[[i, a]] * g0_inv[[a, b]] * x[[i, b]];
                    }
                }
                weights[i] * q
            })
            .collect();
        // For every λ ≥ 0 the penalized leverage must not exceed the bound.
        for &lambda in &[0.0_f64, 0.5, 2.0, 17.0, 1.0e3, 1.0e6] {
            let mut h = gram0.clone();
            for a in 0..p {
                for b in 0..p {
                    h[[a, b]] += lambda * s[[a, b]];
                }
            }
            let factor = StableSolver::new("test penalized H").factorize(&h).unwrap();
            let mut h_inv = Array2::<f64>::eye(p);
            let mut h_inv_v = array2_to_matmut(&mut h_inv);
            factor.solve_in_place(h_inv_v.as_mut());
            for i in 0..n {
                let mut q = 0.0;
                for a in 0..p {
                    for b in 0..p {
                        q += x[[i, a]] * h_inv[[a, b]] * x[[i, b]];
                    }
                }
                let h_i = weights[i] * q;
                assert!(
                    h_i <= hbar[i] + 1e-9,
                    "penalized leverage h_{i}(λ={lambda})={h_i:.6} exceeded unpenalized \
                     bound h̄={:.6}; monotone certificate would be unsound",
                    hbar[i]
                );
            }
        }

        // A threshold safely above every h̄ certifies inactivity; one below the
        // observed maximum must refuse to certify.
        let max_hbar = hbar.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            unpenalized_hat_bound_below_threshold(&x, weights.view(), max_hbar + 0.05),
            "bound below a generous threshold must certify inactivity"
        );
        assert!(
            !unpenalized_hat_bound_below_threshold(&x, weights.view(), max_hbar * 0.5),
            "a threshold under the max bound must NOT certify (would skip a live gate)"
        );
    }

    /// A near-isolated row (its own near-orthogonal basis column) drives the
    /// unpenalized hat bound to ≈ 1, so the certificate must refuse to certify
    /// and fall through to the exact gate — the high-leverage regime the ALO
    /// stabilization exists to handle.
    #[test]
    pub(crate) fn high_leverage_row_is_not_certified_inactive() {
        let n = 12usize;
        let p = 3usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            x[[i, 0]] = 1.0;
            x[[i, 1]] = i as f64 / (n as f64 - 1.0);
        }
        // Third column is a near-indicator for the last row only: that row gets
        // (near-)unit leverage in the unpenalized fit.
        x[[n - 1, 2]] = 1.0;
        let weights = Array1::<f64>::ones(n);
        assert!(
            !unpenalized_hat_bound_below_threshold(&x, weights.view(), 0.80),
            "a near-unit-leverage row must not be certified inactive"
        );
    }

    #[test]
    pub(crate) fn nonpositive_or_nonfinite_threshold_cannot_certify_inactive() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]];
        let weights = Array1::<f64>::ones(x.nrows());
        for threshold in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                !unpenalized_hat_bound_below_threshold(&x, weights.view(), threshold),
                "invalid threshold {threshold} must not certify ALO inactivity"
            );
        }
    }

    #[test]
    pub(crate) fn firth_default_pc_prior_fills_flat_holes() {
        let pc = firth_default_pc_prior();
        let configured = RhoPrior::Normal { mean: 0.1, sd: 2.0 };
        let flat_gamma = RhoPrior::GammaPrecision {
            shape: 1.0,
            rate: 0.0,
        };
        // A whole `Flat` prior becomes the weak PC default.
        assert_eq!(*resolve_effective_rho_prior(&RhoPrior::Flat), pc);
        // Gamma(1, 0) is the same flat MAP-in-λ prior and follows the same path.
        assert_eq!(*resolve_effective_rho_prior(&flat_gamma), pc);
        // An explicitly-configured prior is honored unchanged (no override).
        assert_eq!(*resolve_effective_rho_prior(&configured), configured);
        // Only flat coordinates of an Independent prior inherit the PC.
        let indep =
            RhoPrior::Independent(vec![RhoPrior::Flat, configured.clone(), flat_gamma.clone()]);
        assert_eq!(
            *resolve_effective_rho_prior(&indep),
            RhoPrior::Independent(vec![pc.clone(), configured.clone(), pc.clone()])
        );
        // An Independent prior with no Flat holes is left untouched.
        let no_holes = RhoPrior::Independent(vec![configured.clone(), configured.clone()]);
        assert_eq!(*resolve_effective_rho_prior(&no_holes), no_holes);
    }

    #[test]
    pub(crate) fn firth_default_coord_mask_marks_only_flat_coordinates() {
        // Whole-Flat → every coordinate is a firth default.
        assert_eq!(firth_default_coord_mask(&RhoPrior::Flat, 3), vec![true; 3]);
        // Independent → only mathematically flat holes are defaults.
        let indep = RhoPrior::Independent(vec![
            RhoPrior::Flat,
            RhoPrior::Normal { mean: 0.0, sd: 1.0 },
            RhoPrior::GammaPrecision {
                shape: 1.0,
                rate: 0.0,
            },
        ]);
        assert_eq!(firth_default_coord_mask(&indep, 3), vec![true, false, true]);
        // An explicitly-configured scalar prior defaults nothing.
        assert_eq!(
            firth_default_coord_mask(&RhoPrior::Normal { mean: 0.0, sd: 1.0 }, 2),
            vec![false; 2]
        );
    }

    #[test]
    pub(crate) fn firth_default_barrier_is_byte_zero_on_identified_side() {
        use crate::rho_prior_eval::{firth_default_barrier_terms, pc_prior_rate};
        let upper = FIRTH_DEFAULT_PC_UPPER;
        let theta = pc_prior_rate(upper, FIRTH_DEFAULT_PC_TAIL_PROB);
        let rho_gate = -2.0 * upper.ln();

        // Identified side (ρ ≥ ρ_gate, distance d = e^{-ρ/2} ≤ upper): the barrier
        // contributes EXACTLY zero — strict zero-downside, no Occam pull. This is
        // the whole point of FIX B: the plain PC gradient would be ≈ +1/2 here.
        for &r in &[rho_gate, rho_gate + 1e-9, 0.0, 5.0, 50.0] {
            let (c, g, h) = firth_default_barrier_terms(theta, upper, r);
            assert_eq!((c, g, h), (0.0, 0.0, 0.0), "must be byte-zero at ρ={r}");
        }

        // Degenerate side (ρ < ρ_gate, λ → 0): a convex wall that pushes ρ UP
        // (negative gradient) with positive curvature, and is C⁰ at the gate.
        for &r in &[rho_gate - 1.0, rho_gate - 5.0, -20.0] {
            let (c, g, h) = firth_default_barrier_terms(theta, upper, r);
            assert!(c > 0.0, "cost must be positive below the gate at ρ={r}");
            assert!(g < 0.0, "gradient must push ρ up (away from λ→0) at ρ={r}");
            assert!(h > 0.0, "curvature must be positive at ρ={r}");
        }

        // C⁰ continuity at the gate: cost → 0 as ρ ↑ ρ_gate.
        let (c_below, _, _) = firth_default_barrier_terms(theta, upper, rho_gate - 1e-6);
        assert!(
            c_below.abs() < 1e-9,
            "cost continuous at the gate, got {c_below}"
        );

        // Finite-difference check of grad and (diagonal) Hessian below the gate.
        let r = rho_gate - 2.0;
        let cost_at = |dr: f64| firth_default_barrier_terms(theta, upper, r + dr).0;
        let grad_at = |dr: f64| firth_default_barrier_terms(theta, upper, r + dr).1;
        let (_, g, h) = firth_default_barrier_terms(theta, upper, r);
        let fd_g = (cost_at(1e-6) - cost_at(-1e-6)) / 2e-6;
        let fd_h = (grad_at(1e-5) - grad_at(-1e-5)) / 2e-5;
        assert!((fd_g - g).abs() < 1e-6, "grad FD {fd_g} vs {g}");
        assert!((fd_h - h).abs() < 1e-5, "hess FD {fd_h} vs {h}");
    }

    #[test]
    pub(crate) fn penalty_rank_uses_actual_positive_eigenspace_not_root_rows() {
        let e = array![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],];
        let s = e.t().dot(&e);
        let (evals, _) = s.eigh(Side::Lower).expect("penalty eigensystem");
        let (rank, log_det) = positive_penalty_rank_and_logdet(evals.as_slice().unwrap());

        assert_eq!(rank, 1);
        assert!(
            (log_det - 2.0_f64.ln()).abs() < 1e-12,
            "logdet should use the single positive eigenvalue, got {log_det}"
        );
    }

    pub(crate) fn solve_vec(h: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
        h.cholesky(Side::Lower).expect("chol(H)").solvevec(rhs)
    }

    pub(crate) fn solve_xt(h: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
        let mut xt = x.t().to_owned();
        h.cholesky(Side::Lower)
            .expect("chol(H)")
            .solve_mat_in_place(&mut xt);
        xt
    }

    /// Closed-form TK scalar V_TK in any DualNum-compatible field, for a
    /// scalar (1×1) Hessian H.  Restricting to p = 1 keeps the matrix
    /// solve to a single division, which lets us evaluate the entire
    /// V_TK formula in Dual64 and read the analytic derivative directly
    /// from the dual part.
    ///
    /// All quantities derive from the same closed forms used by
    /// `tk_scalar_from_shared`, just expanded for the n×1 case:
    ///   K_ij = (x_i * x_j) / h
    ///   h_ii = K_ii = x_i² / h
    ///   m_i  = c_i * h_ii
    ///   y    = (Σ x_i m_i) / h          (since H⁻¹ is 1/h)
    ///   q    = Σ x_i m_i
    ///   V_TK = −¹⁄₈ Σ d_i h_ii²
    ///          + ¹⁄₁₂ Σᵢⱼ c_i c_j K_ij³
    ///          + ¹⁄₈ q · y
    pub(crate) fn tk_scalar_dual<D: DualNum<f64> + Copy>(h: D, x: &[D], c: &[D], d_arr: &[D]) -> D {
        let n = x.len();
        let inv_h = D::one() / h;
        let mut h_diag = vec![D::from(0.0); n];
        for i in 0..n {
            h_diag[i] = x[i] * x[i] * inv_h;
        }
        let mut d_term = D::from(0.0);
        for i in 0..n {
            d_term += d_arr[i] * h_diag[i] * h_diag[i];
        }
        d_term *= D::from(-0.125);

        let mut c_term = D::from(0.0);
        for i in 0..n {
            for j in 0..n {
                let k_ij = x[i] * x[j] * inv_h;
                c_term += c[i] * c[j] * k_ij * k_ij * k_ij;
            }
        }
        c_term *= D::from(1.0 / 12.0);

        let mut q = D::from(0.0);
        for i in 0..n {
            q += x[i] * c[i] * h_diag[i];
        }
        let q_term = D::from(0.125) * q * q * inv_h;

        d_term + c_term + q_term
    }

    /// Verify that `tk_gradient_from_shared` computes the exact analytic
    /// directional derivative ∂V_TK/∂θ along a hand-chosen direction
    /// (h_dot, x_dot, c_dot = d⊙η_dot, d_dot = e⊙η_dot).
    ///
    /// Reference is computed via dual-number AD on the closed-form
    /// scalar in [`tk_scalar_dual`].  The setup uses p = 1 so the matrix solve
    /// inside H reduces to a scalar division, which Dual64 handles
    /// natively.
    #[test]
    pub(crate) fn tierney_kadane_gradient_matches_dual_ad_reference_scalarhessian() {
        // p = 1, n = 4 toy problem.
        let x_vec = vec![1.3_f64, -0.4, 0.7, -0.9];
        let h_val = 2.5_f64;
        let c = array![0.21_f64, -0.13, 0.18, 0.07];
        let d = array![-0.05_f64, 0.09, 0.04, -0.07];
        let e = array![0.03_f64, -0.04, 0.02, 0.018];
        let eta_dot = array![0.11_f64, -0.06, 0.07, -0.085];
        let x_dot_vec = vec![0.025_f64, -0.018, 0.022, -0.014];
        let h_dot_val = 0.07_f64;

        // Build dual versions of (h, x, c, d) at θ = 0 with derivatives
        // (h_dot, x_dot, c_dot = d ⊙ η_dot, d_dot = e ⊙ η_dot).  This is
        // the same chain rule the production code applies internally.
        let h_dual = Dual64::new(h_val, h_dot_val);
        let x_dual: Vec<Dual64> = x_vec
            .iter()
            .zip(x_dot_vec.iter())
            .map(|(&v, &dv)| Dual64::new(v, dv))
            .collect();
        let c_dual: Vec<Dual64> = c
            .iter()
            .zip(d.iter().zip(eta_dot.iter()))
            .map(|(&cv, (&dv, &edot))| Dual64::new(cv, dv * edot))
            .collect();
        let d_dual: Vec<Dual64> = d
            .iter()
            .zip(e.iter().zip(eta_dot.iter()))
            .map(|(&dv, (&ev, &edot))| Dual64::new(dv, ev * edot))
            .collect();

        let v_tk_dual = tk_scalar_dual(h_dual, &x_dual, &c_dual, &d_dual);
        let dv_tk_ad = v_tk_dual.eps;
        let v_tk_value = v_tk_dual.re;

        // Sanity-check the closed-form scalar against the production
        // `tk_scalar_from_shared` at θ = 0.
        let x_mat = Array2::from_shape_vec((4, 1), x_vec.clone()).unwrap();
        let h_mat = Array2::from_shape_vec((1, 1), vec![h_val]).unwrap();
        let z_mat = solve_xt(&h_mat, &x_mat);
        let solve = |rhs: &Array1<f64>| -> Result<Array1<f64>, EstimationError> {
            Ok(solve_vec(&h_mat, rhs))
        };
        let shared = RemlState::tk_shared_intermediates(
            &x_mat,
            &z_mat,
            &c,
            "tk scalar dual baseline",
            &solve,
        )
        .expect("shared TK intermediates");
        let mut gram = Array2::<f64>::zeros((TK_BLOCK_SIZE, TK_BLOCK_SIZE));
        let v_tk_prod = RemlState::tk_scalar_from_shared(&x_mat, &z_mat, &d, &shared, &mut gram)
            .expect("TK scalar (production)");
        let scalar_rel =
            (v_tk_value - v_tk_prod).abs() / v_tk_value.abs().max(v_tk_prod.abs()).max(1.0e-14);
        assert!(
            scalar_rel < 1.0e-12,
            "TK scalar mismatch (production vs closed-form): prod={v_tk_prod:.12e}, dual_re={v_tk_value:.12e}, rel={scalar_rel:.3e}"
        );

        // Production analytic derivative through the same
        // `tk_gradient_from_shared` path that runs in REML evaluation.
        // We treat the derivative as a single ext coordinate so the
        // returned gradient[0] is precisely ∂V_TK/∂θ.
        let h_dot_mat = Array2::from_shape_vec((1, 1), vec![h_dot_val]).unwrap();
        let x_dot_mat = Array2::from_shape_vec((4, 1), x_dot_vec.clone()).unwrap();
        let gradient = RemlState::tk_gradient_from_shared(
            &x_mat,
            &z_mat,
            &c,
            &d,
            &e,
            &[],
            &[],
            &[h_dot_mat],
            &[Some(eta_dot.clone())],
            &[Some(x_dot_mat)],
            &[Array1::<f64>::zeros(x_mat.nrows())],
            &[Array1::<f64>::zeros(x_mat.ncols())],
            None,
            &shared,
            &mut gram,
        )
        .expect("analytic TK derivative");
        let analytic = gradient[0];

        let rel = (analytic - dv_tk_ad).abs() / analytic.abs().max(dv_tk_ad.abs()).max(1.0e-14);
        assert!(
            rel < 1.0e-12,
            "Tierney-Kadane analytic c/d propagation does not match Dual64 AD reference: analytic={analytic:.12e}, ad={dv_tk_ad:.12e}, rel={rel:.3e}"
        );
    }

    // ---------------------------------------------------------------------
    //  Analytic e_i = ∂³W_obs/∂η³ vs. Dual3 AD across non-Logit links.
    //
    //  These tests verify the general observed-Hessian path that
    //  `RemlState::hessian_cde_arrays` invokes for every non-canonical link
    //  (Probit, CLogLog, SAS, BetaLogistic, Mixture).  The reference is
    //  `num_dual::third_derivative`, which evaluates W_obs(η₀+δ) in
    //  third-order dual arithmetic and extracts ∂³W_obs/∂δ³ at δ=0
    //  symbolically.
    //
    //  The trick is that Dual3 only carries (re, v1, v2, v3) so any
    //  Taylor truncation of μ to order δ³ is propagated *exactly* through
    //  Dual3 arithmetic.  We therefore parametrise μ(δ), μ'(δ), μ''(δ)
    //  as length-(4..6) Taylor polynomials whose coefficients come from
    //  the *independently derived* closed-form helpers in
    //  `mixture_link.rs` (μ⁽ᵏ⁾(η₀) for k ∈ 0..5).  This makes the test
    //  link-agnostic: the same AD machinery validates the polynomial
    //  formula in `pirls::e_obs_from_jets` for every link.
    // ---------------------------------------------------------------------

    /// Build a Dual3 polynomial in δ representing
    ///   μ(η₀ + δ) ≈ Σ_{k=0..5} μ⁽ᵏ⁾(η₀)·δᵏ/k!
    /// truncated at δ³ – Dual3's arithmetic discards higher-order terms
    /// anyway, so the propagated (re, v1, v2, v3) match the analytic
    /// values μ⁽⁰⁾..μ⁽³⁾(η₀) exactly.
    pub(crate) fn taylor_mu_dual3(delta: Dual3_64, h0: f64, h1: f64, h2: f64, h3: f64) -> Dual3_64 {
        let inv2 = 0.5_f64;
        let inv6 = 1.0_f64 / 6.0_f64;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        Dual3_64::from_re(h0) + delta * h1 + d2 * (h2 * inv2) + d3 * (h3 * inv6)
    }

    /// Same Taylor expansion as [`taylor_mu_dual3`] but starting one order
    /// later in the η-jet, so that the resulting Dual3 represents
    ///   ∂μ/∂η at η₀+δ ≈ Σ_{k=0..4} μ⁽ᵏ⁺¹⁾(η₀)·δᵏ/k!
    /// (used as the closed-form `h1(δ)` carrier inside W_obs).
    pub(crate) fn taylor_mu_eta_dual3(
        delta: Dual3_64,
        h1: f64,
        h2: f64,
        h3: f64,
        h4: f64,
    ) -> Dual3_64 {
        let inv2 = 0.5_f64;
        let inv6 = 1.0_f64 / 6.0_f64;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        Dual3_64::from_re(h1) + delta * h2 + d2 * (h3 * inv2) + d3 * (h4 * inv6)
    }

    /// Closed-form `h2(δ)` carrier (= ∂²μ/∂η² at η₀+δ) as a Dual3 Taylor
    /// polynomial – picks up `h5` in its δ³ coefficient.
    pub(crate) fn taylor_mu_etaeta_dual3(
        delta: Dual3_64,
        h2: f64,
        h3: f64,
        h4: f64,
        h5: f64,
    ) -> Dual3_64 {
        let inv2 = 0.5_f64;
        let inv6 = 1.0_f64 / 6.0_f64;
        let d2 = delta * delta;
        let d3 = d2 * delta;
        Dual3_64::from_re(h2) + delta * h3 + d2 * (h4 * inv2) + d3 * (h5 * inv6)
    }

    /// Closed-form W_obs for a binomial likelihood with arbitrary link:
    ///   W_obs(η) = h₁²/(φV) − (y−μ)·(h₂·V − h₁²·V_μ)/(φV²)
    /// where V(μ)=μ(1−μ) and V_μ(μ)=1−2μ.  Built entirely from the three
    /// Taylor carriers (mu, mu_eta, mu_etaeta) using only num_dual ops.
    pub(crate) fn binomial_w_obs_dual3(
        mu: Dual3_64,
        mu_eta: Dual3_64,
        mu_etaeta: Dual3_64,
        y: f64,
        phi: f64,
    ) -> Dual3_64 {
        let one = Dual3_64::from_re(1.0);
        let two = Dual3_64::from_re(2.0);
        let v = mu * (one - mu);
        let v_mu = one - two * mu;
        let h1sq = mu_eta * mu_eta;
        // φ enters as a constant scalar — fold it into the existing
        // Dual3 multiplications via `Mul<F>` instead of allocating a new
        // Dual3 carrier (mathematically identical, just lighter).
        let phi_v = v * phi;
        let fisher = h1sq / phi_v;
        let resid = Dual3_64::from_re(y) - mu;
        let t_prime = (mu_etaeta * v - h1sq * v_mu) / (phi_v * v);
        fisher - resid * t_prime
    }

    /// Drive `pirls::e_obs_from_jets` and Dual3 AD for the same scalar
    /// fixture and assert the analytic 3rd-η-derivative of W_obs matches
    /// the AD reference to floating-point tolerance.
    pub(crate) fn assert_e_obs_matches_dual3_ad(
        link_label: &str,
        eta0: f64,
        y: f64,
        h0: f64,
        h1: f64,
        h2: f64,
        h3: f64,
        h4: f64,
        h5: f64,
        phi: f64,
        tol: f64,
    ) {
        let v = h0 * (1.0 - h0);
        let vj = VarianceJet::bernoulli(h0);
        assert!(
            v.is_finite() && v > 0.0,
            "fixture {link_label} (eta={eta0}) has degenerate V(μ)={v}; pick a non-saturated point"
        );
        let analytic = e_obs_from_jets(y, h0, h1, h2, h3, h4, h5, vj, phi, 1.0);

        let (_, _, _, d3_w) = third_derivative(
            |delta| {
                let mu = taylor_mu_dual3(delta, h0, h1, h2, h3);
                let mu_eta = taylor_mu_eta_dual3(delta, h1, h2, h3, h4);
                let mu_etaeta = taylor_mu_etaeta_dual3(delta, h2, h3, h4, h5);
                binomial_w_obs_dual3(mu, mu_eta, mu_etaeta, y, phi)
            },
            0.0_f64,
        );

        let scale = analytic.abs().max(d3_w.abs()).max(1.0e-12);
        let rel = (analytic - d3_w).abs() / scale;
        assert!(
            rel < tol,
            "{link_label}: analytic e_obs ({analytic:.12e}) does not match Dual3 AD ({d3_w:.12e}) at eta={eta0}, y={y}; rel={rel:.3e}"
        );
    }

    /// Probit + Bernoulli: closed-form e_obs (via `pirls::e_obs_from_jets`)
    /// matches the Dual3-AD third η-derivative of W_obs at every fixture
    /// point.  This is the production link for large-scale GAMLSS
    /// marginal-slope inference, so we cover both shoulders and the
    /// origin to exercise the (η³,η²,η) polynomial structure of the
    /// Probit jets.
    #[test]
    pub(crate) fn e_obs_from_jets_matches_dual3_ad_probit_bernoulli() {
        let etas = [-1.4_f64, -0.4, 0.0, 0.6, 1.3];
        let ys = [0.0_f64, 1.0];
        let probit = InverseLink::Standard(StandardLink::Probit);
        for &eta in &etas {
            let jet = inverse_link_jet_for_inverse_link(&probit, eta).expect("probit jet");
            let h4 =
                inverse_link_pdfthird_derivative_for_inverse_link(&probit, eta).expect("probit h4");
            let h5 = inverse_link_pdffourth_derivative_for_inverse_link(&probit, eta)
                .expect("probit h5");
            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "probit", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }

    /// CLogLog + Bernoulli: same AD vs analytic check.  CLogLog is the
    /// canonical hazard-link for survival models so its W_obs has a
    /// non-trivial residual term that exercises the full (h₁..h₅,
    /// V₀..V₂) polynomial composition inside `e_obs_from_jets`.
    #[test]
    pub(crate) fn e_obs_from_jets_matches_dual3_ad_cloglog_bernoulli() {
        let etas = [-1.6_f64, -0.5, 0.0, 0.4, 1.2];
        let ys = [0.0_f64, 1.0];
        let cloglog = InverseLink::Standard(StandardLink::CLogLog);
        for &eta in &etas {
            let jet = inverse_link_jet_for_inverse_link(&cloglog, eta).expect("cloglog jet");
            let h4 = inverse_link_pdfthird_derivative_for_inverse_link(&cloglog, eta)
                .expect("cloglog h4");
            let h5 = inverse_link_pdffourth_derivative_for_inverse_link(&cloglog, eta)
                .expect("cloglog h5");
            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "cloglog", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }

    /// Logit + Bernoulli (canonical fast-path consistency check):
    /// `hessian_cde_arrays` ships canonical Logit through a dedicated
    /// fast path that returns `pw·μ⁽⁴⁾(η)` (since W = h₁ for the
    /// canonical pairing).  Independently, the *general* observed-Hessian
    /// path in `pirls::e_obs_from_jets` must produce the identical value
    /// when evaluated on the same Logit jets, otherwise the two paths
    /// would disagree on outer-gradient propagation.  We additionally
    /// sanity-check both quantities against Dual3 AD.
    #[test]
    pub(crate) fn e_obs_from_jets_matches_dual3_ad_logit_bernoulli_canonical_consistency() {
        let etas = [-1.7_f64, -0.6, 0.0, 0.5, 1.4];
        let ys = [0.0_f64, 1.0];
        let logit = InverseLink::Standard(StandardLink::Logit);
        for &eta in &etas {
            let jet = inverse_link_jet_for_inverse_link(&logit, eta).expect("logit jet");
            let h4 =
                inverse_link_pdfthird_derivative_for_inverse_link(&logit, eta).expect("logit h4");
            let h5 =
                inverse_link_pdffourth_derivative_for_inverse_link(&logit, eta).expect("logit h5");

            // The (y−μ) residual term in the general formula must drop
            // out when y is replaced by μ; pick y = μ to isolate
            // W_F = h₁²/V = h₁ on the Logit canonical pairing and confirm
            // that path returns h⁽⁴⁾, which is exactly the value the
            // canonical fast-path stores in `e_array`.
            let canonical_fast_path = h4; // W = h₁ ⇒ ∂³W/∂η³ = h⁽⁴⁾.
            let general_at_y_eq_mu = e_obs_from_jets(
                jet.mu,
                jet.mu,
                jet.d1,
                jet.d2,
                jet.d3,
                h4,
                h5,
                VarianceJet::bernoulli(jet.mu),
                1.0,
                1.0,
            );
            let rel_fast = (general_at_y_eq_mu - canonical_fast_path).abs()
                / canonical_fast_path.abs().max(1.0e-12);
            assert!(
                rel_fast < 1.0e-10,
                "Logit canonical fast path ({canonical_fast_path:.12e}) does not match general e_obs at y=μ ({general_at_y_eq_mu:.12e}) at eta={eta}; rel={rel_fast:.3e}"
            );

            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "logit", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }

    /// SAS + Bernoulli: validate `e_obs_from_jets` against Dual3 AD using
    /// the closed-form SAS jets `(μ, h₁..h₃)` from `sas_inverse_link_jet`
    /// plus the analytic `h₄, h₅` helpers.  Two SAS shape parameters are
    /// covered to exercise both the symmetric (ε≈0) and skewed regimes.
    #[test]
    pub(crate) fn e_obs_from_jets_matches_dual3_ad_sas_bernoulli() {
        let etas = [-1.1_f64, -0.3, 0.0, 0.5, 1.0];
        let ys = [0.0_f64, 1.0];
        let configs = [(-0.25_f64, 0.35_f64), (0.4_f64, -0.2_f64)];
        for &(epsilon, log_delta) in &configs {
            let state = crate::mixture_link::sas_link_state_from_raw(epsilon, log_delta)
                .expect("sas state");
            let link = InverseLink::Sas(state);
            for &eta in &etas {
                let jet = sas_inverse_link_jet(eta, state.epsilon, state.log_delta);
                let h4 = sas_inverse_link_pdfthird_derivative(eta, state.epsilon, state.log_delta);
                let h5 = sas_inverse_link_pdffourth_derivative(eta, state.epsilon, state.log_delta);
                // Cross-check the Result-returning dispatch path used by
                // `hessian_cde_arrays` against the direct closed-form
                // helpers — both must return the same h₄, h₅.
                let h4_dispatch = inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                    .expect("sas h4 via dispatch");
                let h5_dispatch = inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                    .expect("sas h5 via dispatch");
                assert!(
                    (h4 - h4_dispatch).abs() <= 1.0e-12 * h4.abs().max(1.0),
                    "sas h4 dispatch mismatch at eta={eta}, eps={epsilon}, log_delta={log_delta}: direct={h4} dispatch={h4_dispatch}"
                );
                assert!(
                    (h5 - h5_dispatch).abs() <= 1.0e-12 * h5.abs().max(1.0),
                    "sas h5 dispatch mismatch at eta={eta}, eps={epsilon}, log_delta={log_delta}: direct={h5} dispatch={h5_dispatch}"
                );
                for &y in &ys {
                    assert_e_obs_matches_dual3_ad(
                        "sas", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                    );
                }
            }
        }
    }

    /// BetaLogistic + Bernoulli: the third production-grade flexible link
    /// shipped by the GAMLSS marginal-slope family.  Same Dual3 AD test
    /// against `e_obs_from_jets`, exercised at two (delta, ε) shape pairs.
    #[test]
    pub(crate) fn e_obs_from_jets_matches_dual3_ad_beta_logistic_bernoulli() {
        let etas = [-1.0_f64, -0.25, 0.0, 0.4, 0.9];
        let ys = [0.0_f64, 1.0];
        let configs = [(0.18_f64, -0.22_f64), (-0.3_f64, 0.4_f64)];
        for &(epsilon, log_delta) in &configs {
            let delta = log_delta.exp();
            let state = SasLinkState {
                epsilon,
                log_delta,
                delta,
            };
            let link = InverseLink::BetaLogistic(state);
            for &eta in &etas {
                let jet = beta_logistic_inverse_link_jet(eta, state.log_delta, state.epsilon);
                let h4 = beta_logistic_inverse_link_pdfthird_derivative(
                    eta,
                    state.log_delta,
                    state.epsilon,
                );
                let h5 = beta_logistic_inverse_link_pdffourth_derivative(
                    eta,
                    state.log_delta,
                    state.epsilon,
                );
                let h4_dispatch = inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                    .expect("beta-logistic h4 dispatch");
                let h5_dispatch = inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                    .expect("beta-logistic h5 dispatch");
                assert!(
                    (h4 - h4_dispatch).abs() <= 1.0e-12 * h4.abs().max(1.0),
                    "beta-logistic h4 dispatch mismatch at eta={eta}: direct={h4} dispatch={h4_dispatch}"
                );
                assert!(
                    (h5 - h5_dispatch).abs() <= 1.0e-12 * h5.abs().max(1.0),
                    "beta-logistic h5 dispatch mismatch at eta={eta}: direct={h5} dispatch={h5_dispatch}"
                );
                for &y in &ys {
                    assert_e_obs_matches_dual3_ad(
                        "beta-logistic",
                        eta,
                        y,
                        jet.mu,
                        jet.d1,
                        jet.d2,
                        jet.d3,
                        h4,
                        h5,
                        1.0,
                        1.0e-9,
                    );
                }
            }
        }
    }

    /// Mixture-of-links + Bernoulli: μ is a convex combination of standard
    /// component inverse links, so every η-derivative is the matching
    /// convex combination of the per-component derivatives.  We cover a
    /// 4-component Probit/Logit/CLogLog/Cauchit mixture to exercise the
    /// dispatch path inside `inverse_link_pdf{third,fourth}_derivative_for_inverse_link`.
    #[test]
    pub(crate) fn e_obs_from_jets_matches_dual3_ad_mixture_bernoulli() {
        let spec = MixtureLinkSpec {
            components: vec![
                LinkComponent::Probit,
                LinkComponent::Logit,
                LinkComponent::CLogLog,
                LinkComponent::Cauchit,
            ],
            initial_rho: Array1::from_vec(vec![0.35, -0.45, 0.2]),
        };
        let state = state_fromspec(&spec).expect("mixture state");
        let link = InverseLink::Mixture(state.clone());
        let etas = [-1.2_f64, -0.4, 0.0, 0.5, 1.1];
        let ys = [0.0_f64, 1.0];
        for &eta in &etas {
            let jet = mixture_inverse_link_jet(&state, eta);
            let h4 = inverse_link_pdfthird_derivative_for_inverse_link(&link, eta)
                .expect("mixture h4 dispatch");
            let h5 = inverse_link_pdffourth_derivative_for_inverse_link(&link, eta)
                .expect("mixture h5 dispatch");
            for &y in &ys {
                assert_e_obs_matches_dual3_ad(
                    "mixture", eta, y, jet.mu, jet.d1, jet.d2, jet.d3, h4, h5, 1.0, 1.0e-9,
                );
            }
        }
    }
}

#[cfg(test)]
mod adaptive_lm_lambda_tests {
    use super::adaptive_lm_lambda_hint;

    #[test]
    pub(crate) fn pathological_cached_lambdas_fall_through_to_cold_default() {
        // Non-finite or non-positive cached values must return None so
        // the cold default `1e-6` is used. Locks the historical
        // contract that pathological cache slots can't poison the warm
        // start.
        assert_eq!(adaptive_lm_lambda_hint(f64::NAN, 5, true), None);
        assert_eq!(adaptive_lm_lambda_hint(f64::INFINITY, 5, true), None);
        assert_eq!(adaptive_lm_lambda_hint(0.0, 5, true), None);
        assert_eq!(adaptive_lm_lambda_hint(-1e-3, 5, true), None);
    }

    #[test]
    pub(crate) fn no_feedback_yet_falls_through_to_cold_default() {
        // (last_iters == 0, last_converged == false) is the sentinel
        // `clear_warm_start_adaptive_signals` writes — there's no
        // feedback to drive an adaptive regime, so the cold default
        // `1e-6` is preferred over a possibly-stale cache slot.
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 0, false), None);
    }

    #[test]
    pub(crate) fn newton_friendly_regime_admits_floor_to_1e_minus_9() {
        // Previous fit converged in 1 iter — well-conditioned local
        // geometry. The cached λ is allowed down to 1e-9 (the LM-
        // internal floor) so the next fit can leverage the previous
        // Newton-like trajectory.
        assert_eq!(
            adaptive_lm_lambda_hint(1e-9, 1, true),
            Some(1e-9),
            "cached λ at the LM floor must pass through unchanged"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(1e-12, 1, true),
            Some(1e-9),
            "below-floor cached λ clamped up to 1e-9"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(1e-2, 1, true),
            Some(1e-3),
            "above-ceiling cached λ clamped down to 1e-3 even in Newton-friendly regime"
        );
        // last_iters == 2 still counts as Newton-friendly.
        assert_eq!(adaptive_lm_lambda_hint(1e-9, 2, true), Some(1e-9));
    }

    #[test]
    pub(crate) fn hard_fit_regime_preserves_heavy_damping_signal() {
        // Previous fit didn't converge OR took many iters — the cached
        // λ is in the heavy-damping regime, and we want to preserve
        // that signal up to gradient-descent (1.0).
        assert_eq!(
            adaptive_lm_lambda_hint(0.5, 12, true),
            Some(0.5),
            "heavy-damping cached λ passes through unchanged"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(2.0, 12, true),
            Some(1.0),
            "above-ceiling cached λ clamped to 1.0"
        );
        assert_eq!(
            adaptive_lm_lambda_hint(1e-6, 12, true),
            Some(1e-3),
            "below-floor cached λ clamped up to 1e-3 in hard-fit regime"
        );
        // Non-converged (cap exhausted) path takes the same regime.
        assert_eq!(adaptive_lm_lambda_hint(0.5, 5, false), Some(0.5));
    }

    #[test]
    pub(crate) fn default_regime_matches_historical_static_clamp() {
        // 1-9 iters AND converged — a "moderate" fit. The clamp is
        // [1e-6, 1e-3], matching the historical static clamp before
        // the adaptive layer was introduced. Locks behavior so a
        // future commit can't silently widen the default range.
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 5, true), Some(1e-5));
        assert_eq!(adaptive_lm_lambda_hint(1e-9, 5, true), Some(1e-6));
        assert_eq!(adaptive_lm_lambda_hint(1e-1, 5, true), Some(1e-3));
        // Boundary: last_iters=3 (above Newton-friendly cap of 2,
        // below hard-fit floor of 10) goes to default.
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 3, true), Some(1e-5));
        assert_eq!(adaptive_lm_lambda_hint(1e-5, 9, true), Some(1e-5));
    }
}

#[cfg(test)]
mod ift_warm_start_tests {
    use super::*;
    use gam_linalg::matrix::SymmetricMatrix;
    use gam_terms::construction::CanonicalPenalty;
    use ndarray::Array2;

    #[test]
    pub(crate) fn joint_ift_cache_rejects_pending_theta_when_extended_hyperparameters_change() {
        let cache = super::IftJointModeResponseRuntimeCache {
            theta: Array1::from_vec(vec![0.1, -0.2, 0.5]),
            rho_dim: 2,
            beta_original: Array1::from_vec(vec![1.0]),
            mode_response_cols: Array2::zeros((1, 3)),
            active_constraints: false,
        };
        let pending_theta = Array1::from_vec(vec![0.1, -0.2, 0.9]);
        let new_rho = Array1::from_vec(vec![0.1, -0.2]);

        assert!(
            !super::joint_ift_cache_matches_theta(&cache, &pending_theta, &new_rho),
            "extended hyperparameters must participate in joint IFT cache validity"
        );
    }

    /// Test-only entry into `predict_warm_start_beta_ift_inner_with_outcome`
    /// that drops the outcome — the production
    /// `RemlState::predict_warm_start_beta_ift_with_outcome` path uses the
    /// outcome to bookkeep the IFT cache, but the regression tests below
    /// only assert on `Coefficients`.
    pub(crate) fn predict_warm_start_beta_ift_inner(
        cache: &super::IftWarmStartCache,
        canonical_penalties: &[CanonicalPenalty],
        new_rho: &Array1<f64>,
        p: usize,
        last_ift_residual: Option<f64>,
    ) -> Option<Coefficients> {
        super::predict_warm_start_beta_ift_inner_with_outcome(
            cache,
            canonical_penalties,
            new_rho,
            p,
            last_ift_residual,
            None,
            None,
        )
        .map(|(coef, _outcome)| coef)
    }

    /// Variant taking a pre-built `&dyn FactorizedSystem` so the caller can
    /// reuse a cached Cholesky factor across successive predict calls. The
    /// negative-test below verifies the factor argument is actually consumed
    /// (vs silently ignored) by the inner-with-outcome path.
    pub(super) fn predict_warm_start_beta_ift_with_factor(
        cache: &super::IftWarmStartCache,
        canonical_penalties: &[CanonicalPenalty],
        new_rho: &Array1<f64>,
        p: usize,
        last_ift_residual: Option<f64>,
        factor_override: &dyn gam_linalg::matrix::FactorizedSystem,
    ) -> Option<Coefficients> {
        super::predict_warm_start_beta_ift_inner_with_outcome(
            cache,
            canonical_penalties,
            new_rho,
            p,
            last_ift_residual,
            None,
            Some(factor_override),
        )
        .map(|(coef, _outcome)| coef)
    }

    /// Build a CanonicalPenalty from a dense p×p SPD-ish penalty matrix by
    /// taking its eigendecomposition and packing the positive-eigenvalue
    /// components into the `rank × p` root.
    pub(crate) fn dense_canonical_from_local(local: Array2<f64>, p: usize) -> CanonicalPenalty {
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerEigh;
        let (evals, evecs) = local.eigh(Side::Lower).expect("eigh penalty");
        let mut rows: Vec<Array1<f64>> = Vec::new();
        let mut positive_eigenvalues: Vec<f64> = Vec::new();
        for (idx, &lam) in evals.iter().enumerate() {
            if lam > 1e-12 {
                let scale = lam.sqrt();
                let v = evecs.column(idx).to_owned();
                rows.push(v.mapv(|x| x * scale));
                positive_eigenvalues.push(lam);
            }
        }
        let rank = rows.len();
        let mut root = Array2::<f64>::zeros((rank, p));
        for (i, row) in rows.iter().enumerate() {
            for j in 0..p {
                root[[i, j]] = row[j];
            }
        }
        let nullity = p - rank;
        CanonicalPenalty {
            root,
            col_range: 0..p,
            total_dim: p,
            nullity,
            local,
            prior_mean: Array1::zeros(p),
            positive_eigenvalues,
            op: None,
        }
    }

    /// Verify the IFT predictor satisfies the linearized FOC:
    /// `H_pen · (β_predict − β_cur) ≈ −Σ_k Δρ_k · e^{ρ_k} · S_k · (β_cur-μ_k)`.
    /// Tested in the original-basis path (`frame_was_original = true`).
    #[test]
    pub(crate) fn ift_predictor_satisfies_linearized_foc_original_basis() {
        let p = 5usize;
        // Hand-built S_1, S_2 — diagonal-dominant SPD blocks so the
        // local matrices have full rank and the FOC is non-trivial.
        let s1 = Array2::from_shape_vec(
            (p, p),
            vec![
                2.0, 0.3, 0.0, 0.0, 0.0, 0.3, 2.5, 0.4, 0.0, 0.0, 0.0, 0.4, 1.8, 0.2, 0.0, 0.0,
                0.0, 0.2, 1.2, 0.1, 0.0, 0.0, 0.0, 0.1, 1.5,
            ],
        )
        .unwrap();
        let s2 = Array2::from_shape_vec(
            (p, p),
            vec![
                1.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.9, 0.0, 0.3, 0.0, 0.2, 0.0, 1.4, 0.0, 0.2, 0.0,
                0.3, 0.0, 1.7, 0.0, 0.1, 0.0, 0.2, 0.0, 2.1,
            ],
        )
        .unwrap();
        let cp1 = dense_canonical_from_local(s1.clone(), p);
        let cp2 = dense_canonical_from_local(s2.clone(), p);
        let canonical = vec![cp1, cp2];

        // Cached β at converged solve.
        let beta_cur = ndarray::array![0.4, -0.7, 0.2, 0.9, -0.3];
        // Cached ρ.
        let rho_cur = ndarray::array![0.2_f64, -0.1];
        // H_pen at the cached solve. Doesn't have to literally equal
        // ∇²D + Σ e^ρ_k S_k for the linear-FOC check; it just has to
        // be SPD and consistent with what the predictor uses. Keep it
        // simple: use ∇²D = identity and add the penalties.
        let mut h_pen = Array2::<f64>::eye(p);
        for (k_idx, cp) in canonical.iter().enumerate() {
            let lam = rho_cur[k_idx].exp();
            for i in 0..p {
                for j in 0..p {
                    h_pen[[i, j]] += lam * cp.local[[i, j]];
                }
            }
        }

        let cache = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };

        // Small Δρ — well within the 2.0 cap.
        let new_rho = ndarray::array![0.25_f64, -0.05];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, None)
            .expect("IFT predictor should accept small Δρ");

        // Check the linearized FOC residual:
        //   H_pen · (β_pred − β_cur) + Σ_k Δρ_k · e^{ρ_k} · S_k · (β_cur-μ_k) ≈ 0
        let dbeta = &predicted.0 - &beta_cur;
        let lhs = h_pen.dot(&dbeta);
        let mut rhs = Array1::<f64>::zeros(p);
        for (k_idx, cp) in canonical.iter().enumerate() {
            let drho = new_rho[k_idx] - rho_cur[k_idx];
            let scale = drho * rho_cur[k_idx].exp();
            let sb = cp.local.dot(&beta_cur);
            for i in 0..p {
                rhs[i] += scale * sb[i];
            }
        }
        // residual should be tiny (linear-FOC is exactly satisfied by the IFT
        // predictor up to factorization round-off).
        let residual: f64 = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(&a, &b)| (a + b) * (a + b))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = rhs.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        assert!(
            residual / scale < 1e-9,
            "IFT predictor violates linearized FOC: residual {residual:.3e}, scale {scale:.3e}"
        );

        // β_predict should be different from β_cur (otherwise the predictor
        // is a no-op and the test is meaningless).
        let dbeta_norm: f64 = dbeta.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            dbeta_norm > 1e-6,
            "IFT predictor produced zero β-update; check that Δρ propagated"
        );
    }

    /// The lambda_s_beta_blocks precompute path (commit 2f2670b7) must
    /// produce BIT-EQUIVALENT output to the inline-mat-vec fallback.
    /// The 4 existing IFT tests pass `None` for the precompute field,
    /// leaving the fast path untested. This test populates the field
    /// from the same penalties the fallback would use and asserts both
    /// β_predict are identical to ~floating-point round-off.
    #[test]
    pub(crate) fn ift_predictor_precompute_path_matches_inline_path() {
        let p = 5usize;
        let s1 = Array2::from_shape_vec(
            (p, p),
            vec![
                1.0, 0.2, 0.0, 0.0, 0.0, 0.2, 1.5, 0.1, 0.0, 0.0, 0.0, 0.1, 1.3, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let s2 = Array2::from_shape_vec(
            (p, p),
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.05, 0.0, 0.0,
                0.0, 0.05, 1.7, 0.0, 0.0, 0.0, 0.0, 0.0, 2.1,
            ],
        )
        .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let cp2 = dense_canonical_from_local(s2, p);
        let canonical = vec![cp1.clone(), cp2.clone()];

        let beta_cur = ndarray::array![0.4_f64, -0.7, 0.2, 0.9, -0.3];
        let rho_cur = ndarray::array![0.2_f64, -0.1];

        // Build H_pen consistently with the cached solve.
        let mut h_pen = Array2::<f64>::eye(p);
        for (k_idx, cp) in canonical.iter().enumerate() {
            let lam = rho_cur[k_idx].exp();
            for i in 0..p {
                for j in 0..p {
                    h_pen[[i, j]] += lam * cp.local[[i, j]];
                }
            }
        }

        // Precompute the per-penalty `S_k · (β_cur-μ_k)` blocks the same way
        // updatewarm_start_from does at cache-write time.
        let lambda_s_beta_blocks: Vec<ndarray::Array1<f64>> = canonical
            .iter()
            .map(|cp| {
                let r = &cp.col_range;
                let beta_block = beta_cur.slice(s![r.start..r.end]);
                let centered = &beta_block - &cp.prior_mean;
                cp.local.dot(&centered)
            })
            .collect();

        let cache_inline = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        let cache_precompute = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: Some(lambda_s_beta_blocks),
        };

        let new_rho = ndarray::array![0.25_f64, -0.05];
        let predicted_inline =
            predict_warm_start_beta_ift_inner(&cache_inline, &canonical, &new_rho, p, None)
                .expect("inline-path predict");
        let predicted_precompute =
            predict_warm_start_beta_ift_inner(&cache_precompute, &canonical, &new_rho, p, None)
                .expect("precompute-path predict");

        // Bit-equivalent: the precompute path multiplies and accumulates
        // in the same order as the inline path, so the two should agree
        // to round-off at most a few ULPs.
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_precompute.0[i]).abs();
            assert!(
                diff < 1e-12,
                "precompute and inline IFT paths diverged at index {i}: \
                 inline={} precompute={} diff={:.3e}",
                predicted_inline.0[i],
                predicted_precompute.0[i],
                diff,
            );
        }

        // Defensive: if the precompute length disagrees with
        // canonical_penalties.len(), the predictor must fall back to
        // inline-mat-vec rather than mis-indexing.
        let cache_wrong_len = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: Some(vec![ndarray::Array1::zeros(0)]), // length 1, expected 2
        };
        let predicted_fallback =
            predict_warm_start_beta_ift_inner(&cache_wrong_len, &canonical, &new_rho, p, None)
                .expect("wrong-length precompute should still predict via inline fallback");
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_fallback.0[i]).abs();
            assert!(
                diff < 1e-12,
                "wrong-length precompute fell back incorrectly at index {i}",
            );
        }

        // Exercise the `_with_factor` production path: build the
        // factor manually, pass it in, verify the output matches the
        // inline-factorize path. This covers the H_pen factor cache
        // (commit ec18559d) which the integration tests only touch
        // transitively.
        let pre_built_factor = SymmetricMatrix::Dense(h_pen.clone())
            .factorize()
            .expect("factorize for _with_factor test");
        let predicted_via_factor = predict_warm_start_beta_ift_with_factor(
            &cache_inline,
            &canonical,
            &new_rho,
            p,
            None,
            pre_built_factor.as_ref(),
        )
        .expect("with_factor predict");
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_via_factor.0[i]).abs();
            assert!(
                diff < 1e-12,
                "_with_factor and inline paths diverged at index {i}: \
                 inline={} factor={} diff={:.3e}",
                predicted_inline.0[i],
                predicted_via_factor.0[i],
                diff,
            );
        }

        // Negative test: pass a DIFFERENT factor than the one matching
        // cache.penalized_hessian_transformed, and verify the predictor
        // actually USES it (vs silently ignoring and re-factorizing
        // internally). With a perturbed H, the prediction must
        // differ measurably from the inline-path prediction.
        let mut h_perturbed = h_pen.clone();
        for i in 0..p {
            // Add 5× to diagonal — produces a factor that maps
            // back-solves to noticeably smaller magnitudes.
            h_perturbed[[i, i]] *= 5.0;
        }
        let perturbed_factor = SymmetricMatrix::Dense(h_perturbed)
            .factorize()
            .expect("factorize perturbed H");
        let predicted_with_wrong_factor = predict_warm_start_beta_ift_with_factor(
            &cache_inline,
            &canonical,
            &new_rho,
            p,
            None,
            perturbed_factor.as_ref(),
        )
        .expect("with_factor predict (wrong factor)");
        let mut max_diff = 0.0_f64;
        for i in 0..p {
            let diff = (predicted_inline.0[i] - predicted_with_wrong_factor.0[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff > 1e-3,
            "wrong factor produced bit-identical output (max_diff={:.3e}); \
             _with_factor path is silently ignoring its factor argument",
            max_diff,
        );
    }

    /// Verify the basis-conversion path: when frame_was_original=false,
    /// solving with H in transformed basis and round-tripping through qs
    /// (which is orthogonal) must produce the SAME prediction as solving
    /// in original basis with H_orig = qs · H_tfd · qs^T.
    #[test]
    pub(crate) fn ift_predictor_basis_conversion_matches_original() {
        let p = 4usize;
        let s1 = Array2::from_shape_vec(
            (p, p),
            vec![
                1.5, 0.2, 0.0, 0.0, 0.2, 1.8, 0.1, 0.0, 0.0, 0.1, 1.2, 0.05, 0.0, 0.0, 0.05, 1.4,
            ],
        )
        .unwrap();
        let cp1 = dense_canonical_from_local(s1.clone(), p);
        let canonical = vec![cp1];

        let beta_cur = ndarray::array![0.5, -0.3, 0.8, 0.2];
        let rho_cur = ndarray::array![0.1_f64];

        // H_orig = I + e^ρ · S
        let mut h_orig = Array2::<f64>::eye(p);
        let lam = rho_cur[0].exp();
        for i in 0..p {
            for j in 0..p {
                h_orig[[i, j]] += lam * canonical[0].local[[i, j]];
            }
        }

        // Build a non-trivial orthogonal qs (Gram-Schmidt of a random-ish basis).
        let raw = Array2::from_shape_vec(
            (p, p),
            vec![
                1.0, 0.5, 0.0, 0.1, 0.0, 1.0, 0.3, 0.0, 0.2, 0.0, 1.0, 0.4, 0.0, 0.1, 0.0, 1.0,
            ],
        )
        .unwrap();
        let (q, _r) = {
            use gam_linalg::faer_ndarray::FaerQr;
            raw.qr().expect("QR")
        };
        let qs: Array2<f64> = q;
        // Sanity: qs is orthogonal.
        let qq = qs.t().dot(&qs);
        for i in 0..p {
            for j in 0..p {
                let target = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qq[[i, j]] - target).abs() < 1e-10,
                    "qs not orthogonal at [{},{}]: {}",
                    i,
                    j,
                    qq[[i, j]]
                );
            }
        }

        let h_tfd = qs.t().dot(&h_orig).dot(&qs);

        let cache_tfd = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_tfd),
            qs: qs.clone(),
            frame_was_original: false,
            lambda_s_beta_blocks: None,
        };
        let cache_orig = super::super::IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur.clone(),
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_orig.clone()),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };

        let new_rho = ndarray::array![0.3_f64];
        let predicted_tfd =
            predict_warm_start_beta_ift_inner(&cache_tfd, &canonical, &new_rho, p, None)
                .expect("tfd predict");
        let predicted_orig =
            predict_warm_start_beta_ift_inner(&cache_orig, &canonical, &new_rho, p, None)
                .expect("orig predict");

        for i in 0..p {
            assert!(
                (predicted_tfd.0[i] - predicted_orig.0[i]).abs() < 1e-10,
                "basis-conversion path mismatch at index {i}: tfd={}, orig={}",
                predicted_tfd.0[i],
                predicted_orig.0[i],
            );
        }
    }

    /// Δρ above the safety cap must reject (return None), so the caller
    /// falls through to the tangent-line / flat warm-start.
    #[test]
    pub(crate) fn ift_predictor_rejects_large_drho() {
        let p = 3usize;
        let s1 = Array2::from_shape_vec((p, p), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let canonical = vec![cp1];
        let beta_cur = ndarray::array![1.0, 1.0, 1.0];
        let rho_cur = ndarray::array![0.0_f64];
        let h_pen = Array2::<f64>::eye(p) * 2.0;
        let cache = super::super::IftWarmStartCache {
            beta_original: beta_cur,
            rho: rho_cur,
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        // |Δρ| = 3 > IFT_WARM_START_DEFAULT_MAX_DRHO = 2.0 → reject under
        // default cap (no quality history).
        let new_rho = ndarray::array![3.0_f64];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, None);
        assert!(
            predicted.is_none(),
            "predictor should reject Δρ above default cap, got {:?}",
            predicted
        );
        // With excellent prior quality (residual=0.005) the adaptive cap
        // expands to 4.0, so |Δρ|=3 is now ACCEPTED.
        let predicted_good_history =
            predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, Some(0.005));
        assert!(
            predicted_good_history.is_some(),
            "predictor should accept Δρ=3 under expanded cap (good prior quality)",
        );
        // With poor prior quality (residual=0.6) the adaptive cap
        // tightens to 0.5, so even modest |Δρ|=1 is REJECTED.
        let modest_rho = ndarray::array![1.0_f64];
        let predicted_bad_history =
            predict_warm_start_beta_ift_inner(&cache, &canonical, &modest_rho, p, Some(0.6));
        assert!(
            predicted_bad_history.is_none(),
            "predictor should reject Δρ=1 under tightened cap (poor prior quality)",
        );
    }

    /// All Δρ below the eps floor (effectively-zero outer step) must
    /// return Some(β_cur, Noop) — preserving the cached β unchanged.
    /// The inner predictor short-circuits identity predictions before
    /// paying a fresh O(p³)/3 Cholesky; this test exercises that
    /// contract directly. The production
    /// `RemlState::predict_warm_start_beta_ift_with_outcome` wraps the
    /// same check in front of the H_pen factor lookup so the cache miss
    /// is also avoided.
    #[test]
    pub(crate) fn ift_predictor_returns_noop_when_all_drho_below_eps() {
        use crate::estimate::reml::IftWarmStartCache;

        let p = 3usize;
        let s1 = Array2::from_shape_vec((p, p), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let canonical = vec![cp1];
        let beta_cur = ndarray::array![0.5_f64, -0.3, 1.7];
        let rho_cur = ndarray::array![0.0_f64];
        let h_pen = Array2::<f64>::eye(p) * 2.0;
        let cache = IftWarmStartCache {
            beta_original: beta_cur.clone(),
            rho: rho_cur,
            penalized_hessian_transformed: SymmetricMatrix::Dense(h_pen),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        // Δρ = 1e-15 is well below IFT_WARM_START_DRHO_EPS (1e-12);
        // every component of the new ρ is essentially the cached ρ.
        let new_rho = ndarray::array![1e-15_f64];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, p, None)
            .expect("noop must return Some(β_cur)");
        for i in 0..p {
            assert_eq!(
                predicted.0[i], beta_cur[i],
                "noop must return cached β bit-equal at idx {i}: got {} vs cached {}",
                predicted.0[i], beta_cur[i]
            );
        }
    }

    /// Empty ρ in the cache (the in-between state where
    /// updatewarm_start_from has populated β/H but record_warm_start_rho
    /// has not stamped ρ yet) must return None.
    #[test]
    pub(crate) fn ift_predictor_rejects_unstamped_cache() {
        let p = 2usize;
        let cache = super::super::IftWarmStartCache {
            beta_original: ndarray::array![1.0, 2.0],
            rho: Array1::zeros(0),
            penalized_hessian_transformed: SymmetricMatrix::Dense(Array2::eye(p)),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        let new_rho = ndarray::array![0.1_f64];
        let predicted = predict_warm_start_beta_ift_inner(&cache, &[], &new_rho, p, None);
        assert!(predicted.is_none());
    }

    /// `adaptive_ift_max_drho` must follow the documented piecewise
    /// policy: small residual → looser cap; large residual → tighter
    /// cap; non-finite or missing residual → default 2.0.
    #[test]
    pub(crate) fn adaptive_ift_max_drho_follows_quality_tiers() {
        use super::adaptive_ift_max_drho;
        // No history (first solve at this surface) → default 2.0.
        assert_eq!(adaptive_ift_max_drho(None), 2.0);
        // Pathological residuals (NaN, negative) → default 2.0.
        assert_eq!(adaptive_ift_max_drho(Some(f64::NAN)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(-1.0)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(f64::INFINITY)), 0.5);
        // Tier boundaries.
        assert_eq!(adaptive_ift_max_drho(Some(0.0)), 4.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.005)), 4.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.01)), 3.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.04)), 3.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.05)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.10)), 2.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.20)), 1.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.30)), 1.0);
        assert_eq!(adaptive_ift_max_drho(Some(0.50)), 0.5);
        assert_eq!(adaptive_ift_max_drho(Some(1.5)), 0.5);
        // Policy is monotone non-increasing in residual.
        let residuals = [0.001, 0.02, 0.10, 0.30, 0.80];
        let caps: Vec<f64> = residuals
            .iter()
            .map(|&r| adaptive_ift_max_drho(Some(r)))
            .collect();
        for w in caps.windows(2) {
            assert!(
                w[0] >= w[1],
                "adaptive cap is not monotone non-increasing in residual: {caps:?}"
            );
        }
    }

    /// Parallel `S_k · (β-μ_k)` mat-vec across penalties (the rayon par_iter
    /// pattern used at IFT cache-write time in updatewarm_start_from)
    /// must produce bit-equivalent output to the serial version. The
    /// parallelization is across penalties (each penalty's mat-vec is
    /// independent and writes to its own Vec slot — no shared state),
    /// so reordering doesn't change the floating-point result. This
    /// test pins that invariant down: 8 penalties × 50-coefficient
    /// blocks, par_iter vs serial iter, must agree to bit-equality
    /// (not just within-tolerance).
    #[test]
    pub(crate) fn parallel_lambda_s_beta_blocks_matches_serial() {
        use rayon::prelude::*;
        let p = 50usize;
        let n_penalties = 8usize;
        let mut canonical = Vec::with_capacity(n_penalties);
        for k in 0..n_penalties {
            let scale = (k as f64 + 1.0) * 0.5;
            let mut s = Array2::<f64>::zeros((p, p));
            for i in 0..p {
                s[[i, i]] = scale;
                if i + 1 < p {
                    s[[i, i + 1]] = scale * 0.1;
                    s[[i + 1, i]] = scale * 0.1;
                }
            }
            canonical.push(dense_canonical_from_local(s, p));
        }
        let beta_cur =
            ndarray::Array1::from_shape_fn(p, |i| (i as f64 * 0.1).sin() + (i as f64 * 0.05).cos());
        // Serial reference.
        let serial: Vec<ndarray::Array1<f64>> = canonical
            .iter()
            .map(|cp| {
                let r = &cp.col_range;
                let beta_block = beta_cur.slice(s![r.start..r.end]);
                let centered = &beta_block - &cp.prior_mean;
                cp.local.dot(&centered)
            })
            .collect();
        // Parallel (matches the writer at line ~2272).
        let parallel: Vec<ndarray::Array1<f64>> = canonical
            .par_iter()
            .map(|cp| {
                let r = &cp.col_range;
                let beta_block = beta_cur.slice(s![r.start..r.end]);
                let centered = &beta_block - &cp.prior_mean;
                cp.local.dot(&centered)
            })
            .collect();
        assert_eq!(serial.len(), parallel.len());
        for (s_block, p_block) in serial.iter().zip(parallel.iter()) {
            assert_eq!(s_block.len(), p_block.len());
            for (a, b) in s_block.iter().zip(p_block.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "parallel mat-vec diverged from serial: {a} vs {b}",
                );
            }
        }
    }

    /// Pin down the NaN-sentinel discipline for the IFT residual
    /// atomic. The `0` sentinel previously used would have collided
    /// with `f64::to_bits(0.0) == 0`, making a residual of exactly 0
    /// indistinguishable from "no signal yet". The new sentinel
    /// (`IFT_RESIDUAL_NO_SIGNAL_BITS`, a quiet-NaN bit pattern)
    /// encodes "no signal" via NaN's self-inequality, so any finite
    /// non-negative value is unambiguously genuine signal.
    #[test]
    pub(crate) fn ift_residual_sentinel_is_distinguishable_from_zero() {
        use super::IFT_RESIDUAL_NO_SIGNAL_BITS;
        // Sentinel decodes to NaN.
        assert!(
            f64::from_bits(IFT_RESIDUAL_NO_SIGNAL_BITS).is_nan(),
            "sentinel must decode to NaN so reads can detect 'no signal yet'",
        );
        // 0.0 round-trips as 0 bits — distinct from the sentinel, so
        // a stored residual of 0.0 would not be confused with the
        // no-signal state.
        assert_eq!(0.0_f64.to_bits(), 0, "f64::to_bits(0.0) is 0 by IEEE 754");
        assert_ne!(
            IFT_RESIDUAL_NO_SIGNAL_BITS, 0,
            "sentinel must not collide with f64::to_bits(0.0)",
        );
        // The reader's `r.is_finite() && r >= 0.0` predicate accepts
        // 0.0 (a real signal) and rejects the NaN sentinel.
        let r_zero = f64::from_bits(0.0_f64.to_bits());
        assert!(r_zero.is_finite() && r_zero >= 0.0, "0.0 is genuine signal");
        let r_sentinel = f64::from_bits(IFT_RESIDUAL_NO_SIGNAL_BITS);
        assert!(
            !(r_sentinel.is_finite() && r_sentinel >= 0.0),
            "sentinel must fail the reader's accept predicate",
        );
        // Any finite non-negative residual round-trips through
        // `to_bits` / `from_bits` losslessly.
        for &val in &[0.0_f64, 1e-10, 0.05, 0.5, 4.0] {
            let bits = val.to_bits();
            let back = f64::from_bits(bits);
            assert_eq!(back, val, "round-trip failed for {val}");
        }
    }

    /// Dim-mismatch rejection paths must return None (so the caller
    /// falls through to tangent-line / flat warm-start) rather than
    /// dereferencing into mismatched arrays. The new structured
    /// markers added in this commit make these silent failure modes
    /// visible in production logs as bug signals.
    #[test]
    pub(crate) fn ift_predictor_rejects_dim_mismatches() {
        let p = 3usize;
        let s1 = Array2::from_shape_vec((p, p), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
        let cp1 = dense_canonical_from_local(s1, p);
        let canonical = vec![cp1];
        let beta_cur = ndarray::array![1.0_f64, 1.0, 1.0];
        let rho_cur = ndarray::array![0.0_f64];
        let cache = super::super::IftWarmStartCache {
            beta_original: beta_cur,
            rho: rho_cur,
            penalized_hessian_transformed: SymmetricMatrix::Dense(Array2::<f64>::eye(p) * 2.0),
            qs: Array2::eye(p),
            frame_was_original: true,
            lambda_s_beta_blocks: None,
        };
        // new_rho dim mismatch: cache has 1 ρ, caller passes 2.
        let bad_new_rho = ndarray::array![0.1_f64, 0.2];
        assert!(
            predict_warm_start_beta_ift_inner(&cache, &canonical, &bad_new_rho, p, None).is_none(),
            "new_rho dim mismatch must reject",
        );
        // penalty_penalties dim mismatch: cache has 1 ρ, caller passes 0 penalties.
        let new_rho = ndarray::array![0.1_f64];
        assert!(
            predict_warm_start_beta_ift_inner(&cache, &[], &new_rho, p, None).is_none(),
            "penalty dim mismatch must reject",
        );
        // beta dim mismatch: cache has p=3 β, caller passes p=4.
        assert!(
            predict_warm_start_beta_ift_inner(&cache, &canonical, &new_rho, 4, None).is_none(),
            "beta dim mismatch must reject",
        );
    }

    /// `adaptive_tangent_alpha_cap` follows the same quality-tier
    /// pattern as `adaptive_ift_max_drho`, but with values calibrated
    /// to the tangent-line predictor's α scale (multiples of the
    /// previous ρ-step). Pin down each tier transition + defensive
    /// handling.
    #[test]
    pub(crate) fn adaptive_tangent_alpha_cap_follows_quality_tiers() {
        use super::adaptive_tangent_alpha_cap;
        // No history → default 1.5 (the original hardcoded constant).
        assert_eq!(adaptive_tangent_alpha_cap(None), 1.5);
        // NaN / negative → default (defensive against torn atomics).
        assert_eq!(adaptive_tangent_alpha_cap(Some(f64::NAN)), 1.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(-1.0)), 1.5);
        // INFINITY → tightest tier (catastrophic prediction).
        assert_eq!(adaptive_tangent_alpha_cap(Some(f64::INFINITY)), 0.5);
        // Tier boundaries.
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.0)), 2.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.005)), 2.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.01)), 1.75);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.04)), 1.75);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.05)), 1.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.10)), 1.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.20)), 1.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.49)), 1.0);
        assert_eq!(adaptive_tangent_alpha_cap(Some(0.50)), 0.5);
        assert_eq!(adaptive_tangent_alpha_cap(Some(2.0)), 0.5);
        // Monotone non-increasing in residual: same shape as
        // adaptive_ift_max_drho. The two predictors share the
        // residual signal so their caps move together.
        let residuals = [0.001, 0.02, 0.10, 0.30, 0.80];
        let caps: Vec<f64> = residuals
            .iter()
            .map(|&r| adaptive_tangent_alpha_cap(Some(r)))
            .collect();
        for w in caps.windows(2) {
            assert!(
                w[0] >= w[1],
                "tangent α cap is not monotone non-increasing in residual: {caps:?}"
            );
        }
    }
}

#[cfg(test)]
mod warm_start_key_stability_tests {
    use super::analytic_penalty_registry_fingerprint;
    use gam_terms::analytic_penalties::{
        AnalyticPenaltyKind, AnalyticPenaltyRegistry, IsometryPenalty, PsiSlice,
    };
    use ndarray::{Array2, Array3};
    use std::sync::Arc;

    pub(crate) fn duchon_like_registry() -> AnalyticPenaltyRegistry {
        // A single isometry (Duchon-style) penalty, matching the marginal-slope
        // `duchon(...)` fit in issue #1048. Its Jacobian-derivative caches start
        // empty (`None`) exactly as a cold fit opens its warm-start session.
        let mut registry = AnalyticPenaltyRegistry::new();
        let target = PsiSlice::full(6, Some(2));
        registry.push(AnalyticPenaltyKind::Isometry(Arc::new(
            IsometryPenalty::new_euclidean(target, 3),
        )));
        registry
    }

    pub(crate) fn populate_lazy_caches(registry: &AnalyticPenaltyRegistry) {
        // Mimic the SAE/IFT outer driver refreshing the θ-dependent Jacobian
        // derivative caches mid-fit. These are pure functions of basis + θ and
        // are recomputable; their *population state* must never reach the key.
        for penalty in &registry.penalties {
            if let AnalyticPenaltyKind::Isometry(p) = penalty {
                let jac = Arc::new(Array2::from_elem((4, 6), 0.123_f64));
                let jac2 = Arc::new(Array2::from_elem((4, 12), -0.456_f64));
                p.refresh_caches(Some(jac), Some(jac2));
                let jac3 = Arc::new(Array3::from_elem((4, 3, 8), 0.789_f64));
                p.set_third_decoder_derivative(Some(jac3));
            }
        }
    }

    /// Regression for #1048: populating the lazily-refreshed, θ-dependent
    /// isometry Jacobian caches (`jacobian_cache` / `jacobian_second_cache` /
    /// `third_decoder_derivative`) must NOT change the persistent warm-start
    /// fingerprint. Before the fix the key hashed the live cache snapshot, so a
    /// repeat fit — which opens its session with those caches already populated
    /// from the prior run's converged θ — drifted to a different key and missed
    /// the outer `skip-outer-validation` warm hit, re-paying the full outer
    /// REML optimization (~30× slower repeat fits).
    #[test]
    pub(crate) fn lazy_jacobian_caches_do_not_perturb_warm_start_fingerprint() {
        let registry = duchon_like_registry();
        let cold = analytic_penalty_registry_fingerprint(&registry);
        populate_lazy_caches(&registry);
        let warm = analytic_penalty_registry_fingerprint(&registry);
        assert_eq!(
            cold, warm,
            "warm-start key drifted after lazily-refreshed Jacobian caches were \
             populated; the outer skip-outer-validation hit would be lost (#1048)"
        );
    }

    /// Two registries that are structurally identical but differ only in
    /// whether their derived caches are populated must hash identically — this
    /// is the cross-fit invariant the repeat-fit warm hit relies on (cold fit:
    /// caches empty; repeat fit: caches full).
    #[test]
    pub(crate) fn cold_and_warm_registries_share_one_fingerprint() {
        let cold_registry = duchon_like_registry();
        let warm_registry = duchon_like_registry();
        populate_lazy_caches(&warm_registry);
        assert_eq!(
            analytic_penalty_registry_fingerprint(&cold_registry),
            analytic_penalty_registry_fingerprint(&warm_registry),
            "cold (empty-cache) and warm (full-cache) registries must produce the \
             same warm-start key so the second identical fit hits the warm skip (#1048)"
        );
    }
}
