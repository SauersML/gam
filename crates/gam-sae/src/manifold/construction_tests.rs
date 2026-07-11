#[cfg(test)]
mod amortized_encoder_tests {
    use crate::manifold::tests::small_two_atom_periodic_term;

    /// The fitted encoder is reachable end-to-end and returns one coordinate
    /// block per atom plus one honest joint-convergence verdict per row.
    #[test]
    fn amortized_encode_fitted_is_reachable_and_jointly_solved() {
        let (term, target, _rho_unused) = small_two_atom_periodic_term();
        let n = term.n_obs();
        let k = term.k_atoms();

        let results = term
            .amortized_encode_fitted(target.view())
            .expect("amortized encode of the fit-time target runs end-to-end");
        assert_eq!(results.coords.len(), k, "one coordinate block per atom");

        for (atom_idx, result) in results.coords.iter().enumerate() {
            assert_eq!(
                result.nrows(),
                n,
                "atom {atom_idx} encode must produce one coordinate per row"
            );
            assert_eq!(
                result.ncols(),
                term.atoms[atom_idx].latent_dim,
                "atom {atom_idx} encode coords must match its latent dim"
            );
        }
        assert_eq!(
            results.converged.len(),
            n,
            "joint verdict must cover every row"
        );
        assert_eq!(
            results.unconverged_count,
            results.converged.iter().filter(|ok| !**ok).count()
        );
    }

    /// The fitted amplitudes the encoder derives are exactly the posterior gate
    /// coordinates used by reconstruction. Decoder magnitude stays in `B`, so
    /// there is no second radial-scale channel to fold into these values.
    #[test]
    fn fitted_assignment_amplitudes_equal_posterior_gates() {
        let (term, _target, _rho_unused) = small_two_atom_periodic_term();
        let n = term.n_obs();
        let k = term.k_atoms();
        let amplitudes = term
            .fitted_assignment_amplitudes()
            .expect("fitted amplitudes derive from posterior assignments");
        assert_eq!(amplitudes.dim(), (n, k));
        for row in 0..n {
            let a = term
                .assignment
                .try_assignments_row(row)
                .expect("assignment row resolves");
            for atom_idx in 0..k {
                assert_eq!(
                    amplitudes[[row, atom_idx]],
                    a[atom_idx],
                    "amplitude[{row},{atom_idx}] must equal its posterior gate"
                );
            }
        }
    }
}

#[cfg(test)]
mod outer_gradient_error_classification_1451_tests {
    use super::OuterGradientError;

    /// #1451 — the three numerical/linear-algebra failure sites inside the
    /// deflation path (`apply_cached_arrow_hessian`, the projected `h_span.eigh`,
    /// and `DeflatedArrowSolver::from_orthonormal_gauges`) must distinguish a
    /// genuine near-singular conditioning trip (`IllConditioned`) from an
    /// internal-invariant defect — a shape/dimension mismatch or a non-finite
    /// intermediate (`InternalInvariant`). Both propagate if the projected
    /// implicit solve cannot complete, but the typed diagnosis must stay exact.
    ///
    /// `OuterGradientError::classify_arrow_solver_error` is the helper all three
    /// sites route through. Before the #1451 fix every failure there was
    /// re-labelled `IllConditioned` (the original `conditioning_err`), so the
    /// shape/non-finite cases below would have been misdiagnosed as numerical
    /// conditioning. This test pins that a shape/non-finite error classifies to
    /// `InternalInvariant` while a genuine finite, correctly-shaped
    /// near-singular failure stays `IllConditioned`.
    #[test]
    fn classify_arrow_solver_error_routes_shape_and_nonfinite_to_internal_1451() {
        let conditioning = || OuterGradientError::IllConditioned {
            reason: "near-singular joint Hessian (min/max pivot ratio 5.3e-16)".to_string(),
        };

        // Shape/dimension-mismatch markers emitted by the deflation helpers must
        // classify as InternalInvariant.
        let shape_messages = [
            "apply_cached_arrow_hessian: vector shapes (t=3, beta=2) != cache shapes (t=4, beta=2)",
            "DeflatedArrowSolver: gauge length 5 != cache full length 6",
            "DeflatedArrowSolver: solution length 5 != cache full length 6",
        ];
        for msg in shape_messages {
            let classified = OuterGradientError::classify_arrow_solver_error(msg, conditioning());
            assert!(
                matches!(classified, OuterGradientError::InternalInvariant { .. }),
                "shape mismatch must classify to InternalInvariant (#1451); got {classified}"
            );
        }

        // Non-finite-intermediate markers must likewise propagate as internal.
        let nonfinite_messages = [
            "DeflatedArrowSolver: gauge stiffness must be finite and positive; got NaN",
            "outer_gradient_arrow_solver: non-finite entry in projected gauge Hessian",
        ];
        for msg in nonfinite_messages {
            let classified = OuterGradientError::classify_arrow_solver_error(msg, conditioning());
            assert!(
                matches!(classified, OuterGradientError::InternalInvariant { .. }),
                "non-finite intermediate must classify to InternalInvariant (#1451); \
                 got {classified}"
            );
        }

        // A genuine near-singular linear-algebra failure on a finite, correctly
        // shaped input (back-solve / Cholesky/Woodbury factor that tripped on
        // rank-deficiency) is the legitimate #1273 conditioning case: it must
        // KEEP IllConditioned.
        let conditioning_messages = [
            "DeflatedArrowSolver: gauge Woodbury factor failed: matrix is not positive definite",
            "DeflatedArrowSolver: gauge back-solve: singular factor",
        ];
        for msg in conditioning_messages {
            let classified = OuterGradientError::classify_arrow_solver_error(msg, conditioning());
            assert!(
                matches!(classified, OuterGradientError::IllConditioned { .. }),
                "a finite, correctly-shaped near-singular failure must KEEP \
                 IllConditioned (#1451 / #1273); got {classified}"
            );
        }
    }
}

#[cfg(test)]
mod softmax_majorizer_active_entry_1410_tests {
    //! #1410 — the active-only softmax-entropy curvature helpers
    //! ([`super::active_softmax_gershgorin_majorizer_entry`],
    //! [`super::softmax_dense_entropy_hessian_entry`],
    //! [`super::softmax_majorizer_log_mean`]) let the compact assembly /
    //! θ-adjoint / exact-Hessian-correction paths read one `(k)` diagonal or
    //! `(k,j)` matrix entry WITHOUT materialising the full-`K` `d` vector or the
    //! `K×K` dense entropy/majorizer blocks per row — the residual per-worker
    //! `O(K)`/`O(K²)` scratch that defeated the compact `O(top_k·d)`-per-token
    //! contract.
    //!
    //! Correctness is single-sourced: these helpers MUST reproduce the
    //! `SoftmaxAssignmentSparsityPenalty` dense library routines
    //! (`psd_majorizer_abs_row_sums`, `row_psd_majorizer`, `row_dense_hessian`)
    //! BIT-FOR-BIT, because the assembled `B`, the criterion's `log|H|`, and the
    //! #1006 θ-adjoint all differentiate ONE operator. If the dense library
    //! formula ever changes, this oracle fails and forces the helpers back into
    //! sync (preventing the value↔adjoint desync the compact rewrite must not
    //! introduce).

    use gam_terms::analytic_penalties::SoftmaxAssignmentSparsityPenalty;

    /// Deterministic, well-spread softmax logit rows (a long tail plus a few
    /// peaks) so the abs-row-sum / dense-Hessian algebra is exercised across
    /// near-zero and near-one assignment masses.
    fn logit_rows(k: usize) -> Vec<Vec<f64>> {
        let mut rows = Vec::new();
        // Row a: a few sharp peaks spread across K, deep floor elsewhere.
        let mut a = vec![-7.0_f64; k];
        for &peak in &[0usize, k / 3, 2 * k / 3, k - 1] {
            a[peak] = 5.0 + (peak as f64) * 0.001;
        }
        rows.push(a);
        // Row b: smoothly varying logits (no degenerate ties).
        let b: Vec<f64> = (0..k)
            .map(|i| ((i as f64) * 0.37).sin() * 2.0 - (i as f64) / (k as f64))
            .collect();
        rows.push(b);
        // Row c: near-uniform (entropy Hessian indefinite here — the regime the
        // Gershgorin majorizer exists for).
        rows.push(vec![0.01; k]);
        rows
    }

    #[test]
    fn active_softmax_gershgorin_matches_dense_majorizer_1410() {
        let k = 64usize;
        let temperature = 0.8_f64;
        let scale = 1.7_f64;
        let penalty = SoftmaxAssignmentSparsityPenalty::new(k, temperature);
        for row in logit_rows(k) {
            // Dense reference: full-K abs-row-sum diagonal `d`.
            let d_dense = penalty.psd_majorizer_abs_row_sums(&row, scale);
            // The helper consumes the softmax row `a`, not raw logits, exactly as
            // the assembly/adjoint feed it `assignments`. Build `a` the same way
            // the penalty does internally.
            let a = crate::assignment::softmax_row(
                ndarray::ArrayView1::from(row.as_slice()),
                temperature,
            );
            let a = a.as_slice().expect("softmax row contiguous");
            let m = super::softmax_majorizer_log_mean(a);
            for kk in 0..k {
                let got = super::active_softmax_gershgorin_majorizer_entry(a, kk, m, scale);
                assert_eq!(
                    got, d_dense[kk],
                    "active Gershgorin majorizer entry must equal the dense \
                     psd_majorizer_abs_row_sums[{kk}] BIT-FOR-BIT (single-source #1410/#1419)"
                );
            }
        }
    }

    #[test]
    fn active_softmax_dense_entropy_hessian_entry_matches_dense_block_1410() {
        let k = 48usize;
        let temperature = 1.3_f64;
        let scale = 0.9_f64;
        let penalty = SoftmaxAssignmentSparsityPenalty::new(k, temperature);
        for row in logit_rows(k) {
            let h_dense = penalty.row_dense_hessian(&row, scale);
            let a = crate::assignment::softmax_row(
                ndarray::ArrayView1::from(row.as_slice()),
                temperature,
            );
            let a = a.as_slice().expect("softmax row contiguous");
            let m = super::softmax_majorizer_log_mean(a);
            for kk in 0..k {
                for jj in 0..k {
                    let got = super::softmax_dense_entropy_hessian_entry(a, kk, jj, m, scale);
                    assert_eq!(
                        got,
                        h_dense[[kk, jj]],
                        "active dense entropy-Hessian entry ({kk},{jj}) must equal \
                         row_dense_hessian BIT-FOR-BIT (single-source #1410/#1418)"
                    );
                }
            }
        }
    }

    #[test]
    fn active_softmax_majorizer_logit_derivative_matches_dense_1410() {
        let k = 40usize;
        let temperature = 0.7_f64;
        let scale = 1.1_f64;
        let inv_tau = 1.0 / temperature;
        let penalty = SoftmaxAssignmentSparsityPenalty::new(k, temperature);
        for row in logit_rows(k) {
            let a = crate::assignment::softmax_row(
                ndarray::ArrayView1::from(row.as_slice()),
                temperature,
            );
            let a = a.as_slice().expect("softmax row contiguous");
            let m = super::softmax_majorizer_log_mean(a);
            // Pin the active diagonal entry against the dense library derivative
            // matrix (which is diagonal: `out[[kk, kk]]`) for several `w`.
            for w in [0usize, k / 2, k - 1] {
                let dense = penalty.row_psd_majorizer_logit_derivative(&row, scale, w);
                for kk in 0..k {
                    let got = super::active_softmax_majorizer_logit_derivative_entry(
                        a, kk, w, m, scale, inv_tau,
                    );
                    assert_eq!(
                        got,
                        dense[[kk, kk]],
                        "active majorizer logit-derivative ∂D_({kk},{kk})/∂z_{w} must equal \
                         row_psd_majorizer_logit_derivative diagonal BIT-FOR-BIT \
                         (single-source #1410/#1419/#1006)"
                    );
                }
            }
        }
    }
}

/// #1418: the implicit-function (IFT) back-substitution must invert the EXACT
/// stationarity Jacobian `A = ∇²_θθ L`, not the assembled surrogate `B`.
#[cfg(test)]
mod exact_stationarity_solve_1418_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::manifold::tests::{diagonal_latent_cache, gamma_fd_tiny_fixture};
    use ndarray::Array1;

    /// Build a converged tiny SAE state whose inner residual is genuinely
    /// nonzero (an unmodellable target perturbation on a curved periodic basis),
    /// so the dropped curvature `ΔC = A − B = ⟨r, ∂²f⟩ + (H_entropy − D) + min(V'',0)`
    /// is materially nonzero and `A ≠ B`. Returns the term, the perturbed target,
    /// the rho, and the converged cache.
    fn converged_state_with_residual() -> (
        SaeManifoldTerm,
        Array2<f64>,
        SaeManifoldRho,
        ArrowFactorCache,
    ) {
        let (mut term, mut target, mut rho) = gamma_fd_tiny_fixture();
        // Perturb the target off the model manifold so the inner optimum has a
        // real residual `r`, hence a real `⟨r, ∂²f⟩` curvature delta.
        let (n, p) = (target.nrows(), target.ncols());
        for row in 0..n {
            for col in 0..p {
                let phase = (row as f64 + 0.35) / n as f64;
                let theta = std::f64::consts::TAU * phase;
                target[[row, col]] += 0.6 * (3.0 * theta + 0.5 * col as f64).sin();
            }
        }
        // Activate the sparsity / smoothness / ARD prior strengths so the softmax
        // entropy delta and the periodic-ARD `min(V'',0)` delta are live too.
        rho.log_lambda_sparse = -0.5;
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for v in axis.iter_mut() {
                *v = -0.5;
            }
        }
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged cache with residual");
        (term, target, rho, cache)
    }

    /// `‖A x − rhs‖` for the exact stationarity Jacobian `A` (the matrix-free
    /// `B v + ΔC v` apply).
    fn a_residual_norm(
        term: &SaeManifoldTerm,
        rho: &SaeManifoldRho,
        target: ArrayView2<'_, f64>,
        cache: &ArrowFactorCache,
        x: &SaeArrowVector,
        rhs: &SaeArrowVector,
    ) -> f64 {
        let ax = term
            .apply_exact_hessian(rho, target, cache, x)
            .expect("A matvec");
        let resid = SaeArrowVector {
            t: &ax.t - &rhs.t,
            beta: &ax.beta - &rhs.beta,
        };
        sae_norm(&resid)
    }

    /// `solve_exact_stationarity` returns the EXACT solve of `A x = rhs` (small
    /// `A`-residual), AND the surrogate solve `x_B = B⁻¹ rhs` leaves a LARGE
    /// `A`-residual — so the certificate is non-vacuous (`A ≠ B`) and the IFT
    /// step genuinely inverts `A`. The surrogate solve `x_B = B⁻¹ rhs` leaves
    /// the large `A`-residual asserted below, so this test passes only when the
    /// implicit solve targets the exact stationarity Jacobian.
    #[test]
    fn solve_exact_stationarity_inverts_a_not_b_1418() {
        let (term, target, rho, cache) = converged_state_with_residual();
        let solver = DeflatedArrowSolver::plain(&cache);

        // A deterministic, nonzero rhs spanning both the latent (t) and decoder
        // (β) blocks.
        let total_t = cache.delta_t_len();
        let rhs = SaeArrowVector {
            t: Array1::from_shape_fn(total_t, |i| 0.3 + 0.1 * ((i % 5) as f64) - 0.02 * i as f64),
            beta: Array1::from_shape_fn(cache.k, |j| 0.2 - 0.05 * ((j % 3) as f64)),
        };
        let rhs_norm = sae_norm(&rhs).max(1.0);

        // Exact A-solve via the #1418 path.
        let x = term
            .solve_exact_stationarity(&rho, target.view(), &cache, &solver, &rhs)
            .expect("exact stationarity solve");
        let exact_resid = a_residual_norm(&term, &rho, target.view(), &cache, &x, &rhs);

        // Surrogate solve x_B = B⁻¹ rhs (the pre-#1418 implicit step).
        let x_b = solver
            .solve(rhs.t.view(), rhs.beta.view())
            .expect("B inverse");
        let surrogate_resid = a_residual_norm(&term, &rho, target.view(), &cache, &x_b, &rhs);

        // 1) The exact solve drives the A-residual to ~0.
        assert!(
            exact_resid <= 1.0e-6 * rhs_norm,
            "solve_exact_stationarity must invert the EXACT A: ‖A x − rhs‖/‖rhs‖ = {:.3e} \
             (rhs_norm={rhs_norm:.3e}) — the IFT step is not solving A x = rhs (#1418)",
            exact_resid / rhs_norm
        );

        // 2) Non-vacuity: the surrogate B-solve leaves a materially large
        //    A-residual, so A ≠ B is genuinely exercised. The pre-#1418 code used
        //    x_B for the implicit step, so this is exactly the error #1418 removed.
        assert!(
            surrogate_resid >= 1.0e-2 * rhs_norm,
            "the surrogate B-solve must leave a large A-residual so the A≠B fix is \
             non-vacuous: ‖A x_B − rhs‖/‖rhs‖ = {:.3e} — ΔC = A − B is too small to \
             distinguish the exact stationarity Jacobian from the surrogate",
            surrogate_resid / rhs_norm
        );

        // 3) The exact solve is a strict, large improvement over the surrogate.
        assert!(
            exact_resid < 1.0e-3 * surrogate_resid,
            "exact A-solve residual {exact_resid:.3e} must be far below surrogate {surrogate_resid:.3e}"
        );
    }

    /// #2253 production-wiring regression: the operator-generic core used by
    /// `solve_exact_stationarity` must install the solver's closed-form gauge
    /// stiffness on BOTH raw operators and invert `A_Q = A + κQQᵀ`, not raw
    /// `A`. Deterministic diagonal `A` and `B` isolate that production seam from
    /// stochastic inner fitting and its unrelated dictionary-collapse guards.
    #[test]
    fn solve_exact_stationarity_uses_solver_gauge_fix_2253() {
        // B=diag(2,5), A=diag(3,7), q=e0, κ=5. The raw pencil is healthy, and
        // the quotient pencil is A_Q=diag(8,7), B_Q=diag(7,5). A gauge-bearing
        // rhs makes the solution of A_Q observably different from raw A⁻¹rhs.
        let cache = diagonal_latent_cache(&[2.0_f64, 5.0]);
        let gauge = Array1::from_vec(vec![1.0_f64, 0.0]);
        let stiffness = 5.0;
        let solver =
            DeflatedArrowSolver::from_orthonormal_gauges(&cache, vec![gauge], stiffness)
                .expect("gauge-fixed exact-stationarity preconditioner");
        let rhs = SaeArrowVector {
            t: Array1::from_vec(vec![4.0_f64, 6.0]),
            beta: Array1::zeros(0),
        };
        let apply_raw_a = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            Ok(SaeArrowVector {
                t: Array1::from_vec(vec![3.0 * v.t[0], 7.0 * v.t[1]]),
                beta: Array1::zeros(0),
            })
        };
        let apply_raw_b = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            Ok(SaeArrowVector {
                t: Array1::from_vec(vec![2.0 * v.t[0], 5.0 * v.t[1]]),
                beta: Array1::zeros(0),
            })
        };

        let solved = solve_exact_stationarity_on_gauge_quotient(
            &solver,
            &rhs,
            &apply_raw_a,
            &apply_raw_b,
        )
        .expect("gauge-fixed exact stationarity solve");
        let raw_ax = apply_raw_a(&solved).expect("raw A apply");
        let raw_residual = SaeArrowVector {
            t: &raw_ax.t - &rhs.t,
            beta: &raw_ax.beta - &rhs.beta,
        };
        let mut gauge_fixed_ax = raw_ax;
        solver
            .add_gauge_stiffness(&solved, &mut gauge_fixed_ax)
            .expect("κQQᵀ action");
        let gauge_fixed_residual = SaeArrowVector {
            t: &gauge_fixed_ax.t - &rhs.t,
            beta: &gauge_fixed_ax.beta - &rhs.beta,
        };
        let rhs_norm = sae_norm(&rhs).max(1.0);
        let fixed_norm = sae_norm(&gauge_fixed_residual);
        let raw_norm = sae_norm(&raw_residual);
        assert!(
            fixed_norm <= 1.0e-6 * rhs_norm,
            "solve_exact_stationarity must solve the gauge-fixed A_Q system: \
             ‖A_Qx-rhs‖/‖rhs‖={:.3e}",
            fixed_norm / rhs_norm
        );
        assert!(
            raw_norm >= 1.0e-3 * rhs_norm,
            "test must distinguish A_Q from raw A: raw residual was only {:.3e}",
            raw_norm / rhs_norm
        );
    }

    /// #2253: a near-zero Rayleigh quotient of the complete response is not by
    /// itself a numerical null. Resolved positive and negative pencil components
    /// can cancel exactly. The inverse-power discriminator must recognize the
    /// resolved negative direction and keep the full finite response.
    #[test]
    fn ift_solve_keeps_resolved_indefinite_rayleigh_cancellation_2253() {
        // B=I, A=diag(-1/2, 2), x=(2,1), rhs=A x=(-1,2). Although
        // x^T A x = 0 exactly, both pencil eigenvalues are far above the
        // sqrt(epsilon) identifiability floor in magnitude.
        let cache = diagonal_latent_cache(&[1.0_f64, 1.0]);
        let solver = DeflatedArrowSolver::plain(&cache);
        let rhs = SaeArrowVector {
            t: Array1::from_vec(vec![-1.0_f64, 2.0]),
            beta: Array1::zeros(0),
        };
        let apply_raw_a = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            Ok(SaeArrowVector {
                t: Array1::from_vec(vec![-0.5 * v.t[0], 2.0 * v.t[1]]),
                beta: Array1::zeros(0),
            })
        };
        let apply_raw_b = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> {
            Ok(v.clone())
        };

        let solved = solve_exact_stationarity_on_gauge_quotient(
            &solver,
            &rhs,
            &apply_raw_a,
            &apply_raw_b,
        )
        .expect("resolved indefinite response");
        assert_abs_diff_eq!(solved.t[0], 2.0, epsilon = 1.0e-10);
        assert_abs_diff_eq!(solved.t[1], 1.0, epsilon = 1.0e-10);
    }

    /// #2080 defect 4 — with a SATURATED gate logit, the exact stationarity
    /// Jacobian `A` develops a near-null pencil direction (data curvature
    /// `∝ σ'(ℓ)² ≈ 0` against an O(1) majorizer entry in `B`), and the raw
    /// GMRES solve of `A x = rhs` amplifies any rhs mass there by `1/μ` —
    /// the objective↔gradient desync class (#931) that flipped the analytic
    /// λ-gradient's sign. `solve_exact_stationarity` must DEFLATE it: the
    /// returned solution's generalized Rayleigh quotient `xᵀAx/xᵀBx` must sit
    /// at or above the identifiability floor. Non-vacuity is asserted first:
    /// the UNDEFLATED solve must actually collapse below the floor on this
    /// fixture, so the test can only pass through genuine deflation.
    #[test]
    fn ift_solve_deflates_saturated_gate_near_null_direction_2080() {
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
        // Saturate atom 1's gate logits hard OFF: σ'(−40)² ≈ 1e-35 kills the
        // data curvature along those logit coordinates while the assembled
        // majorizer keeps an O(1) diagonal there.
        for row in 0..term.n_obs() {
            term.assignment.logits[[row, 1]] = -40.0;
        }
        rho.log_lambda_sparse = -1.0;
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged saturated-gate IBP cache");
        let solver = DeflatedArrowSolver::plain(&cache);

        // Deterministic rhs with mass on every coordinate (including the
        // saturated logit slots).
        let total_t = cache.delta_t_len();
        let rhs = SaeArrowVector {
            t: Array1::from_shape_fn(total_t, |i| 0.3 + 0.1 * ((i % 5) as f64) - 0.02 * i as f64),
            beta: Array1::from_shape_fn(cache.k, |j| 0.2 - 0.05 * ((j % 3) as f64)),
        };

        let pencil_mu = |x: &SaeArrowVector| -> f64 {
            let ax = term
                .apply_exact_hessian(&rho, target.view(), &cache, x)
                .expect("A matvec");
            let bx =
                apply_cached_arrow_hessian(&cache, x.t.view(), x.beta.view()).expect("B matvec");
            sae_inner(x, &ax) / sae_inner(x, &bx)
        };

        // Non-vacuity: the raw (undeflated) exact solve is dominated by the
        // saturated near-null direction.
        let raw = solve_b_preconditioned_gmres_with(
            &rhs,
            |v| term.apply_exact_hessian(&rho, target.view(), &cache, v),
            |vector| solver.solve(vector.t.view(), vector.beta.view()),
        )
        .expect("raw exact solve");
        let raw_mu = pencil_mu(&raw);
        assert!(
            raw_mu > 0.0,
            "fixture must be a stable near-null minimum, not a negative-curvature \
             stationary point: raw pencil Rayleigh {raw_mu:.3e}"
        );
        assert!(
            raw_mu < sae_ift_min_curvature_fraction(),
            "fixture must exercise the defect: raw solve pencil Rayleigh {raw_mu:.3e} \
             should collapse below the {:.1e} floor \
             (saturated gate produced no near-null direction — strengthen the fixture)",
            sae_ift_min_curvature_fraction()
        );

        // The production solve deflates: identifiable-curvature fraction is
        // restored at or above the floor, and the solution is finite.
        let deflated = term
            .solve_exact_stationarity(&rho, target.view(), &cache, &solver, &rhs)
            .expect("deflated exact stationarity solve");
        assert!(
            deflated
                .t
                .iter()
                .chain(deflated.beta.iter())
                .all(|v| v.is_finite()),
            "deflated IFT solution must be finite"
        );
        let deflated_mu = pencil_mu(&deflated);
        assert!(
            deflated_mu >= sae_ift_min_curvature_fraction(),
            "deflated solve must remove the unidentifiable component: pencil Rayleigh \
             {deflated_mu:.3e} still below the {:.1e} floor",
            sae_ift_min_curvature_fraction()
        );
        // Deflation only removes, never adds: the deflated solution is no
        // larger than the raw one in the B-metric.
        let b_norm = |x: &SaeArrowVector| -> f64 {
            let bx =
                apply_cached_arrow_hessian(&cache, x.t.view(), x.beta.view()).expect("B matvec");
            sae_inner(x, &bx).max(0.0).sqrt()
        };
        assert!(
            b_norm(&deflated) <= b_norm(&raw) * (1.0 + 1.0e-8),
            "deflation must be a projection (B-norm non-increasing)"
        );
    }

    /// Build a converged IBP-MAP tiny SAE state whose cache carries the exact
    /// cross-row rank-`R` Woodbury (`H_full = H₀' + U D Uᵀ`), with a genuinely
    /// nonzero inner residual so `ΔC = A − B` is also live.
    fn converged_ibp_state_with_woodbury() -> (
        SaeManifoldTerm,
        Array2<f64>,
        SaeManifoldRho,
        ArrowFactorCache,
    ) {
        let (mut term, mut target, mut rho) = gamma_fd_tiny_fixture();
        // Off-manifold target perturbation ⇒ real residual ⇒ live ΔC.
        let (n, p) = (target.nrows(), target.ncols());
        for row in 0..n {
            for col in 0..p {
                let phase = (row as f64 + 0.35) / n as f64;
                let theta = std::f64::consts::TAU * phase;
                target[[row, col]] += 0.12 * (3.0 * theta + 0.5 * col as f64).sin();
            }
        }
        // IBP-MAP assignment with an ACTIVE sparsity strength so the empirical-mass
        // prior's cross-row curvature `d_k = w·s'_k` is live (≠ 0) ⇒ a real
        // Woodbury is emitted and downdated into `H₀'`.
        term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, false);
        rho.log_lambda_sparse = -1.0;
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged IBP cache with cross-row Woodbury");
        (term, target, rho, cache)
    }

    /// `U D Uᵀ v` on the latent (`t`) block, computed INDEPENDENTLY from the
    /// carrier's public dense `U`/`d` (β-part of `U` is structurally zero).
    fn woodbury_forward_t(cache: &ArrowFactorCache, v: &SaeArrowVector) -> Array1<f64> {
        let w = cache
            .cross_row_woodbury
            .as_ref()
            .expect("IBP cache must carry the cross-row Woodbury");
        let total_t = cache.delta_t_len();
        let r = w.d.len();
        // p_k = d_k · (Uᵀ v_t)_k.
        let mut pk = vec![0.0_f64; r];
        for k in 0..r {
            let mut acc = 0.0_f64;
            for g in 0..total_t {
                acc += w.u[[g, k]] * v.t[g];
            }
            pk[k] = w.d[k] * acc;
        }
        // out_t = U p.
        let mut out_t = Array1::<f64>::zeros(total_t);
        for g in 0..total_t {
            let mut acc = 0.0_f64;
            for k in 0..r {
                acc += w.u[[g, k]] * pk[k];
            }
            out_t[g] = acc;
        }
        out_t
    }

    /// Solve a small dense symmetric system `M x = b` by partial-pivot LU
    /// (independent of the production solver; `M` here is the inner solver's own
    /// exact inverse `H_full⁻¹`, so `solve_dense(M, v) = H_full v`).
    fn solve_dense(m: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let dim = m.nrows();
        let mut a = m.clone();
        let mut x = b.clone();
        for col in 0..dim {
            let mut piv = col;
            let mut best = a[[col, col]].abs();
            for row in (col + 1)..dim {
                let mag = a[[row, col]].abs();
                if mag > best {
                    best = mag;
                    piv = row;
                }
            }
            if piv != col {
                for c in 0..dim {
                    a.swap((col, c), (piv, c));
                }
                x.swap(col, piv);
            }
            let pivot = a[[col, col]];
            for row in (col + 1)..dim {
                let factor = a[[row, col]] / pivot;
                for c in col..dim {
                    let v = a[[col, c]];
                    a[[row, c]] -= factor * v;
                }
                let xc = x[col];
                x[row] -= factor * xc;
            }
        }
        for col in (0..dim).rev() {
            let mut sum = x[col];
            for c in (col + 1)..dim {
                sum -= a[[col, c]] * x[c];
            }
            x[col] = sum / a[[col, col]];
        }
        x
    }

    /// #1038 / #1418 regression: for an IBP-MAP cache the EXACT-Hessian forward
    /// apply `apply_exact_hessian` must equal the DENSE exact joint Hessian
    /// `A_true = H_full + ΔC`, where `H_full = H₀' + U D Uᵀ` is the operator the
    /// inner solver (`full_inverse_apply`) inverts and `ΔC = ⟨r, ∂²f⟩` is the
    /// dropped residual curvature.
    ///
    /// The dense `H_full` oracle is built independently of the apply path by
    /// inverting the inner solver's own exact inverse (columns of
    /// `cache.full_inverse_apply(e_j)` give `H_full⁻¹`; `solve_dense` against it
    /// gives `H_full·v`). Before the fix `apply_exact_hessian` used only `H₀'`, so
    /// the residual equals exactly the dropped cross-row term `U D Uᵀ v` (asserted
    /// to be materially nonzero, so the test is non-vacuous).
    #[test]
    fn apply_exact_hessian_includes_ibp_cross_row_woodbury_1038() {
        let (term, target, rho, cache) = converged_ibp_state_with_woodbury();

        // (2) The production IBP path must actually carry the Woodbury.
        assert!(
            cache.cross_row_woodbury.is_some(),
            "a converged IBP-MAP cache with active sparsity must carry the cross-row \
             Woodbury — otherwise the bug is unreachable on this path"
        );

        let total_t = cache.delta_t_len();
        let kdim = cache.k;
        let m = total_t + kdim;

        // Build the dense inner-solver inverse `Minv = H_full⁻¹` column-by-column
        // from `full_inverse_apply` (the operator whose log-det the evidence
        // reports). This is fully independent of `apply_cached_arrow_hessian`.
        let mut minv = Array2::<f64>::zeros((m, m));
        for j in 0..m {
            let mut e_t = Array1::<f64>::zeros(total_t);
            let mut e_b = Array1::<f64>::zeros(kdim);
            if j < total_t {
                e_t[j] = 1.0;
            } else {
                e_b[j - total_t] = 1.0;
            }
            let (sol_t, sol_b) = cache
                .full_inverse_apply(e_t.view(), e_b.view())
                .expect("full_inverse_apply column");
            for i in 0..total_t {
                minv[[i, j]] = sol_t[i];
            }
            for i in 0..kdim {
                minv[[total_t + i, j]] = sol_b[i];
            }
        }
        // Symmetrize away back-substitution rounding asymmetry.
        for i in 0..m {
            for j in (i + 1)..m {
                let avg = 0.5 * (minv[[i, j]] + minv[[j, i]]);
                minv[[i, j]] = avg;
                minv[[j, i]] = avg;
            }
        }

        // Deterministic probe spanning the latent (t) and decoder (β) blocks.
        let v = SaeArrowVector {
            t: Array1::from_shape_fn(total_t, |i| {
                0.37 + 0.11 * ((i % 4) as f64) - 0.017 * i as f64
            }),
            beta: Array1::from_shape_fn(kdim, |j| 0.21 - 0.043 * ((j % 3) as f64)),
        };
        let v_flat = flatten_arrow_parts(v.t.view(), v.beta.view());

        // Dense H_full·v = solve(Minv, v); ΔC·v from the matrix-free dropped-curvature.
        let hfull_v = solve_dense(&minv, &v_flat);
        let dc_v = term
            .apply_exact_hessian_minus_b(&rho, target.view(), &cache, &v)
            .expect("ΔC apply");
        let mut a_true = hfull_v.clone();
        for i in 0..total_t {
            a_true[i] += dc_v.t[i];
        }
        for i in 0..kdim {
            a_true[total_t + i] += dc_v.beta[i];
        }

        // The code under test.
        let ae = term
            .apply_exact_hessian(&rho, target.view(), &cache, &v)
            .expect("apply_exact_hessian");
        let ae_flat = flatten_arrow_parts(ae.t.view(), ae.beta.view());

        // Non-vacuity: the missing cross-row term must be materially nonzero, and
        // it must equal the H_full−H₀' gap the dense oracle carries.
        let udut_v = woodbury_forward_t(&cache, &v);
        let udut_norm = udut_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let a_true_norm = a_true.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        assert!(
            udut_norm > 1.0e-3 * a_true_norm,
            "the IBP cross-row term U D Uᵀ v must be materially nonzero (‖UDUᵀv‖={udut_norm:.3e}, \
             ‖A_true v‖={a_true_norm:.3e}) — otherwise this regression is vacuous"
        );

        // Primary assertion: apply_exact_hessian == dense exact joint Hessian.
        let resid = (0..m)
            .map(|i| (ae_flat[i] - a_true[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        // For diagnostics: the pre-fix residual equals exactly ‖UDUᵀ v‖.
        let resid_vs_udut = {
            let mut s = 0.0_f64;
            for g in 0..total_t {
                s += (ae.t[g] - a_true[g] + udut_v[g]).powi(2);
            }
            for i in 0..kdim {
                s += (ae.beta[i] - a_true[total_t + i]).powi(2);
            }
            s.sqrt()
        };
        eprintln!(
            "[#1038] ‖apply_exact_hessian − A_true‖ = {resid:.6e}; ‖UDUᵀv‖ = {udut_norm:.6e}; \
             ‖(apply_exact_hessian − A_true) + UDUᵀv‖ = {resid_vs_udut:.6e}"
        );
        assert!(
            resid <= 1.0e-9 * a_true_norm,
            "apply_exact_hessian must equal the dense exact joint Hessian H_full + ΔC for an \
             IBP cache: ‖A_apply v − A_true v‖ = {resid:.3e} (rel {:.3e}); the omitted term is \
             the cross-row Woodbury U D Uᵀ v (‖·‖={udut_norm:.3e}), confirmed by \
             ‖residual + UDUᵀv‖ = {resid_vs_udut:.3e} ≈ 0 (#1038)",
            resid / a_true_norm
        );

        // Operator/preconditioner consistency: stripping ΔC must round-trip
        // through the inner solver's exact inverse back to v.
        let (rt, rb) = cache
            .full_inverse_apply((&ae.t - &dc_v.t).view(), (&ae.beta - &dc_v.beta).view())
            .expect("round-trip inverse");
        let round_trip = {
            let mut s = 0.0_f64;
            for g in 0..total_t {
                s += (rt[g] - v.t[g]).powi(2);
            }
            for i in 0..kdim {
                s += (rb[i] - v.beta[i]).powi(2);
            }
            s.sqrt()
        };
        let v_norm = v_flat.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        assert!(
            round_trip <= 1.0e-9 * v_norm,
            "H_full⁻¹·(apply_exact_hessian − ΔC)·v must return v (operator and preconditioner on \
             the SAME H_full): round-trip residual {round_trip:.3e} (rel {:.3e})",
            round_trip / v_norm
        );
    }
}

/// Validates the matrix-free Hutchinson stochastic-trace estimator that replaces
/// the exact `Σ_k M_k·r_k`-solve per-atom decoder-smoothness effective-dof at
/// massive `K` (the `O(K³·M·p)` wall). The estimator is exercised here on a
/// small (`K = 2`) fixture — where the exact column-solve is the ground truth —
/// so the block-restricted one-solve-per-probe identity
/// `E[z_kᵀ (S_β⁻¹ M z)_k] = tr((S_β⁻¹)_{kk} M_k)` (including cross-atom
/// cancellation, which only a `K ≥ 2` fixture can exhibit) is checked against the
/// exact trace, plus determinism for a fixed seed.
#[cfg(test)]
mod smoothness_dof_hutchinson_tests {
    use super::*;
    use crate::manifold::tests::small_two_atom_periodic_term;

    /// Rebuild the exact function's `(offsets, out_dim)` β-layout so the estimator
    /// is fed the identical geometry.
    fn beta_layout(term: &SaeManifoldTerm) -> (Vec<usize>, Box<dyn Fn(usize) -> usize>) {
        let p = term.output_dim();
        if term.frames_active() {
            let ranks: Vec<usize> = term.atoms.iter().map(|a| a.border_frame_rank()).collect();
            (
                term.factored_beta_offsets(),
                Box::new(move |k: usize| ranks[k]),
            )
        } else {
            (term.beta_offsets(), Box::new(move |_k: usize| p))
        }
    }

    #[test]
    fn hutchinson_smoothness_dof_matches_exact_and_is_deterministic() {
        let (mut term, target, rho) = small_two_atom_periodic_term();
        let (_value, _loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged cache for the two-atom fixture");
        let lambda = rho.lambda_smooth_vec();

        // Ground truth: the exact column-by-column trace (the `K < threshold`
        // path this fixture actually takes).
        let exact = term
            .decoder_smoothness_effective_dof_per_atom(&cache, &lambda)
            .expect("exact per-atom smoothness edof");
        assert_eq!(exact.len(), 2, "two-atom fixture must return two edofs");

        let (offsets, out_dim) = beta_layout(&term);
        let solve = |rhs: ndarray::ArrayView1<'_, f64>| {
            cache
                .schur_inverse_apply(rhs)
                .map_err(|e| format!("schur_inverse_apply: {e:?}"))
        };

        // Many probes so the Monte-Carlo band is tight enough to pin the math.
        let probes = 6000;
        let seed = 0xC0FFEE_1234;
        let est = term
            .decoder_smoothness_effective_dof_per_atom_hutchinson(
                cache.k,
                &offsets,
                out_dim.as_ref(),
                &lambda,
                probes,
                seed,
                solve,
            )
            .expect("hutchinson per-atom smoothness edof");

        // Total trace tr(S_β⁻¹ M) — the sum averages the per-atom variance, so it
        // pins tightly to the exact total.
        let exact_sum: f64 = exact.iter().sum();
        let est_sum: f64 = est.iter().sum();
        assert!(
            (est_sum - exact_sum).abs() <= 0.03 * exact_sum.abs().max(1.0e-3),
            "hutchinson total edof {est_sum:.6} vs exact {exact_sum:.6}"
        );

        // Per-atom: looser Monte-Carlo band (per-atom carries the cross-atom
        // coupling variance), but tight enough that a block-indexing bug — which
        // would scramble the per-atom split by O(1) — cannot pass.
        for k in 0..2 {
            assert!(
                (est[k] - exact[k]).abs() <= 0.10 * exact[k].abs().max(1.0e-2) + 0.05,
                "atom {k}: hutchinson edof {:.6} vs exact {:.6}",
                est[k],
                exact[k]
            );
        }

        // Determinism: a second run with the SAME seed is bit-identical (the REML
        // outer-loop reproducibility contract).
        let solve2 = |rhs: ndarray::ArrayView1<'_, f64>| {
            cache
                .schur_inverse_apply(rhs)
                .map_err(|e| format!("schur_inverse_apply: {e:?}"))
        };
        let est2 = term
            .decoder_smoothness_effective_dof_per_atom_hutchinson(
                cache.k,
                &offsets,
                out_dim.as_ref(),
                &lambda,
                probes,
                seed,
                solve2,
            )
            .expect("hutchinson rerun");
        assert_eq!(
            est, est2,
            "hutchinson smoothness edof must be bit-reproducible for a fixed seed"
        );
    }
}



#[cfg(test)]
mod shape_uncertainty_joint_recompute_tests {
    use super::*;
    use crate::manifold::tests::gamma_fd_tiny_fixture;

    /// After a structure-search / finalization change, the shape bands are
    /// rebuilt at the FINAL state by `recompute_joint_shape_uncertainty`, which
    /// must return the JOINT inverse-Hessian covariance — NOT the per-atom
    /// inner-Hessian marginal the pre-fix path fell back to. Two properties
    /// distinguish the two and are pinned here:
    ///   1. the recompute reproduces the direct-Schur `assemble_shape_uncertainty`
    ///      bands (it IS the joint path); and
    ///   2. the joint band carries per-output-channel variance AND differs
    ///      materially from the per-atom-marginal completion — so replacing the
    ///      marginal with the joint recompute genuinely changes the reported band.
    #[test]
    fn recompute_reproduces_joint_and_differs_from_per_atom_marginal() {
        // A reliably-converging tiny state: the fixture target was assembled under
        // a softmax gate, so switching to an IBP-MAP gate at the PD-region ρ
        // (`log_lambda_sparse = 0.5`, the deflation-regression config) leaves a
        // genuine reconstruction residual — a real dispersion and nonzero bands —
        // while the state stays near its inner optimum so the undamped joint
        // factor converges in a few steps.
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        term.assignment.mode = AssignmentMode::ibp_map(0.7, 0.9, true);
        rho.log_lambda_sparse = 0.5;

        // Reference joint bands via the direct Schur path.
        let (_c, loss, cache) = term
            .reml_criterion_with_cache(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
            .expect("converged joint cache");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, None)
            .expect("dispersion");
        assert!(dispersion > 0.0, "a real residual ⇒ positive dispersion");
        let joint = term
            .assemble_shape_uncertainty(&cache, dispersion)
            .expect("direct joint bands");

        // Property 1: the final-state recompute reproduces the joint path (it IS
        // the joint path, rebuilt from the term + ρ rather than a cached factor).
        let recomputed = term
            .recompute_joint_shape_uncertainty(target.view(), &rho, None, 5, 0.4, 1.0e-6, 1.0e-6)
            .expect("joint recompute");
        assert_eq!(recomputed.atoms.len(), joint.atoms.len());
        for (k, (a, b)) in recomputed.atoms.iter().zip(joint.atoms.iter()).enumerate() {
            assert_eq!(a.band_sd.dim(), b.band_sd.dim(), "atom {k} band shape");
            for (x, y) in a.band_sd.iter().zip(b.band_sd.iter()) {
                assert!(
                    (x - y).abs() <= 1e-9 * (1.0 + y.abs()),
                    "atom {k}: recompute must reproduce the joint band ({x} vs {y})"
                );
            }
        }

        // The joint per-channel SD genuinely varies across the p output channels
        // (the coordinate-Schur coupling makes each channel's decoder covariance
        // differ) — a per-atom marginal `φ·Φᵀ H_k⁻¹ Φ` is IDENTICAL across
        // channels. Measured scale-free as the within-row max/min ratio so a tiny
        // dispersion (which scales every band equally) does not hide the spread.
        let mut joint_channel_spread = 0.0_f64;
        for atom in &joint.atoms {
            for gi in 0..atom.band_sd.nrows() {
                let row = atom.band_sd.row(gi);
                let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max > 0.0 {
                    joint_channel_spread = joint_channel_spread.max((max - min) / max);
                }
            }
        }
        assert!(
            joint_channel_spread > 1e-6,
            "the JOINT band must carry per-output-channel variance (relative spread \
             {joint_channel_spread:.3e}); a constant-across-channel band is the per-atom \
             marginal the fix replaced"
        );

        // Property 2 (non-vacuity): the per-atom inner-Hessian marginal the
        // fallback produces DIFFERS materially from the joint band. Compared
        // scale-free (both scale by the same dispersion), so the gap reflects the
        // dropped cross-atom / coordinate couplings, not the dispersion.
        term.set_atom_inner_fits(target.view(), dispersion)
            .expect("inner fits harvested");
        let mut marginal = joint.clone();
        marginal.invalidate_bands_for_recompute();
        term.complete_born_atom_shape_bands(&mut marginal)
            .expect("per-atom marginal completion");
        let mut max_rel_gap = 0.0_f64;
        let mut compared = 0usize;
        for (a, b) in marginal.atoms.iter().zip(joint.atoms.iter()) {
            for (x, y) in a.band_sd.iter().zip(b.band_sd.iter()) {
                if x.is_finite() && y.is_finite() && *y > 0.0 {
                    max_rel_gap = max_rel_gap.max((x - y).abs() / y);
                    compared += 1;
                }
            }
        }
        assert!(
            compared > 0,
            "the per-atom marginal must fill finite bands to compare"
        );
        assert!(
            max_rel_gap > 1e-3,
            "the per-atom marginal must differ materially from the joint band \
             (max rel gap {max_rel_gap:.3e}); otherwise the joint recompute changes nothing"
        );
    }
}
