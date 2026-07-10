#[cfg(test)]
mod amortized_encoder_tests {
    use crate::manifold::tests::small_two_atom_periodic_term;

    /// The fitted encoder is reachable end-to-end and returns one coordinate
    /// block per atom plus one honest joint-convergence verdict per row.
    #[test]
    fn amortized_encode_fitted_is_reachable_and_jointly_solved() {
        let (term, target, rho) = small_two_atom_periodic_term();
        let n = term.n_obs();
        let k = term.k_atoms();

        let results = term
            .amortized_encode_fitted(target.view(), &rho)
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

    /// The fitted amplitudes the encoder derives are the realised intensity
    /// coordinates the reconstruction uses: the existence gate times the explicit
    /// cone radial scale.  Returning the bare gate would silently re-couple
    /// presence and intensity for amortized encoding as soon as `log_amplitude`
    /// moves away from zero.
    #[test]
    fn fitted_assignment_amplitudes_include_explicit_log_amplitude_1939() {
        let (mut term, _target, rho) = small_two_atom_periodic_term();
        term.atoms[0].log_amplitude = 2.0_f64.ln();
        term.atoms[1].log_amplitude = 0.25_f64.ln();
        let n = term.n_obs();
        let k = term.k_atoms();
        let amplitudes = term
            .fitted_assignment_amplitudes(&rho)
            .expect("fitted amplitudes derive from assignment and cone scale");
        assert_eq!(amplitudes.dim(), (n, k));
        for row in 0..n {
            let a = term
                .assignment
                .try_assignments_row_for_rho(row, &rho)
                .expect("assignment row resolves");
            for atom_idx in 0..k {
                let expected = a[atom_idx] * term.atoms[atom_idx].log_amplitude.exp();
                assert_eq!(
                    amplitudes[[row, atom_idx]],
                    expected,
                    "amplitude[{row},{atom_idx}] must equal gate times explicit cone scale"
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
    use crate::manifold::tests::gamma_fd_tiny_fixture;
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
        let raw = solve_b_preconditioned_gmres(&solver, &rhs, |v| {
            term.apply_exact_hessian(&rho, target.view(), &cache, v)
        })
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

// ============================================================================
// #2022 STEP2 FD / behavior gate. Paste as a new submodule INTO
// crates/gam-sae/src/manifold/construction_tests.rs (that file is already
// #[cfg(test)]; this `mod` nests cleanly, mirroring its existing
// `use crate::manifold::tests::small_two_atom_periodic_term;` submodules).
//
// Parallel-safe: exercises the retract+peel via `absorb_*` DIRECTLY (not the
// GAM_SAE_QUOTIENT_SCALE env lever), so no process-global env mutation. Because
// absorb sets s≠0 and the assembly's β a_phi × exp(s) is ALWAYS-ON, this covers
// the FINDING-1 exp(s) gradient/loss consistency without the lever.
// ============================================================================
#[cfg(test)]
mod step2_quotient_scale_tests {
    use crate::manifold::SaeManifoldTerm;
    use crate::manifold::tests::small_two_atom_periodic_term;

    fn frob(b: &ndarray::Array2<f64>) -> f64 {
        b.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// #2022 — the SCALE-gauge quotient (unit-Frobenius decoder + explicit
    /// log-amplitude `s`) is exp(s)-consistent:
    ///   (i)  peeling ‖B_k‖ into `s_k` preserves the reconstruction data-fit
    ///        (the always-on `fill_decoded_* · exp(s)` from STEP1),
    ///   (ii) every decoder is unit-Frobenius afterward and `s_k` is finite,
    ///   (iii) FINDING-1: with `s≠0`, the assembled β Jacobian carries `exp(s)`
    ///        (`a_phi × exp(s)`), so a Newton step assembles with finite deltas
    ///        and a damped step does not blow up / materially increase the loss.
    ///        A missing `exp(s)` on `a_phi` (the desync STEP2 fixes) misscales the
    ///        β gradient/Hessian and breaks this.
    #[test]
    fn step2_amplitude_peel_is_exp_s_consistent() {
        let (mut term, target, rho) = small_two_atom_periodic_term();
        let loss0 = term.loss(target.view(), &rho).expect("baseline loss");

        // Retract + peel each atom onto the unit sphere (scale → s). Same op the
        // gated apply_newton_step performs; called directly for a lever-free,
        // parallel-safe test.
        for atom in term.atoms.iter_mut() {
            atom.absorb_decoder_norm_into_log_amplitude(f64::MIN_POSITIVE);
        }

        // (i) reconstruction data-fit preserved by the peel.
        let loss_peeled = term.loss(target.view(), &rho).expect("peeled loss");
        assert!(
            (loss_peeled.data_fit - loss0.data_fit).abs() <= 1e-9 * (1.0 + loss0.data_fit.abs()),
            "peel must preserve reconstruction data-fit: {} vs {}",
            loss0.data_fit,
            loss_peeled.data_fit
        );

        // (ii) unit-Frobenius decoders + finite log-amplitude.
        for atom in &term.atoms {
            let nrm = frob(&atom.decoder_coefficients);
            assert!(
                (nrm - 1.0).abs() <= 1e-9,
                "‖B_k‖ must be 1 after peel; got {nrm}"
            );
            assert!(
                atom.log_amplitude.is_finite(),
                "log_amplitude must be finite after peel"
            );
        }

        // (iii) FINDING-1 consistency: assemble+solve a Newton step on the peeled
        // (s≠0) state — the assembly must produce finite deltas, and a damped step
        // must keep the loss finite without materially increasing it.
        let (dt, db) = term
            .solve_newton_step(target.view(), &rho, None, 0.0, 0.0)
            .expect("newton step assembles on the exp(s)-scaled decoder");
        assert!(
            dt.iter().chain(db.iter()).all(|v| v.is_finite()),
            "step deltas must be finite (exp(s) assembly well-formed)"
        );
        term.apply_newton_step(dt.view(), db.view(), 0.1)
            .expect("apply damped newton step");
        let loss_step = term.loss(target.view(), &rho).expect("post-step loss");
        assert!(
            loss_step.total().is_finite()
                && loss_step.total() <= loss_peeled.total() * (1.0 + 1e-6) + 1e-6,
            "a damped Newton step on the exp(s)-scaled decoder must not blow up / \
             materially increase loss: {} -> {}",
            loss_peeled.total(),
            loss_step.total()
        );
    }

    /// #2099 — collapse guards must measure the atom's physical contribution
    /// scale `exp(s_k)‖B_k‖_F`, not the raw decoder norm alone. After the quotient
    /// peels both decoders to unit Frobenius, `‖B_k‖` cannot distinguish a live atom
    /// from one whose explicit log-amplitude has collapsed.
    #[test]
    fn quotient_collapse_scale_tracks_log_amplitude_not_unit_decoder_norm() {
        let (mut term, _target, _rho) = small_two_atom_periodic_term();
        for atom in term.atoms.iter_mut() {
            atom.absorb_decoder_norm_into_log_amplitude(f64::MIN_POSITIVE);
        }
        let live_scale = term.atoms[0].contribution_frobenius_scale();
        term.atoms[1].log_amplitude = term.atoms[0].log_amplitude - 4.0_f64.ln();
        let collapsed_scale = term.atoms[1].contribution_frobenius_scale();

        assert!(
            (frob(&term.atoms[0].decoder_coefficients) - 1.0).abs() <= 1e-9
                && (frob(&term.atoms[1].decoder_coefficients) - 1.0).abs() <= 1e-9,
            "fixture must exercise equal unit decoders"
        );
        assert!(
            collapsed_scale < live_scale,
            "physical scale must see the explicit-amplitude collapse: live={live_scale}, collapsed={collapsed_scale}"
        );
    }

    /// #2022 gate (i) — lever DEFAULT-OFF ⇒ the step never peels ⇒ `s` stays 0 ⇒
    /// bit-for-bit. Verified by construction: `absorb_*` is only invoked from the
    /// step under the `quotient_scale` kwarg (default false) and from the gated
    /// seed peel; with the kwarg unset a freshly-built term has every
    /// `log_amplitude == 0.0`. (No env mutation here — parallel-safe.)
    #[test]
    fn step2_default_off_leaves_log_amplitude_zero() {
        let (term, _target, _rho) = small_two_atom_periodic_term();
        for atom in &term.atoms {
            assert_eq!(
                atom.log_amplitude, 0.0,
                "default (lever off) must leave log_amplitude at 0 (bit-for-bit)"
            );
        }
    }

    /// #2022 FINDING-1 (rigorous) — the assembled β gradient carries exp(s):
    /// gb == central-diff ∂(data_fit+smoothness)/∂β on the peeled (s≠0) state.
    /// A missing exp(s) on a_phi would scale gb by exp(-s) vs the FD and fail.
    /// Parallel-safe: drives the quotient via absorb_* directly (no kwarg/env).
    #[test]
    fn step2_beta_gradient_matches_central_diff_on_peeled_state() {
        let (mut term, target, rho) = small_two_atom_periodic_term();
        for atom in term.atoms.iter_mut() {
            atom.absorb_decoder_norm_into_log_amplitude(f64::MIN_POSITIVE);
        }
        let sys = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("assemble arrow-schur on peeled state");
        assert_eq!(
            sys.gb.len(),
            term.beta_dim(),
            "expected full-B gb layout (no frames) on the fixture"
        );
        let gb = sys.gb.clone();
        let offsets = term.beta_offsets();
        let p = term.output_dim();
        // `sys.gb` is the β-gradient of the objective the STEP-2 inner Newton step
        // descends. We FD the terms `gb` carries a NONZERO gradient for: the
        // data-fit `Jᵀr` + smoothness `λ S B` (both carry the peel's `exp(s)` via
        // `fill_decoded_row` / the cached `S`) and the #1026 decoder-repulsion force
        // (frozen-gate `∂P_rep/∂B`, ~120× the data-fit leg here — omitting it made
        // the historical `data_fit + smoothness` FD disagree at ~basis 0).
        //
        // The #1026/#1522 SEPARATION barrier `P_sep = −μ q w(c²) log(1−c²+ε)` is
        // deliberately EXCLUDED from this FD objective. This fixture is p = 1
        // (single output dim), and for rank-1 (p = 1) decoders the normalized cross
        // `c²_jk = ‖B_jB_kᵀ‖²_F / (‖B_j‖²‖B_k‖²)` collapses to the ALGEBRAIC IDENTITY
        // `1` (`‖B_jB_kᵀ‖²_F = ‖B_j‖²‖B_k‖²` exactly), independent of the decoder
        // entries. So `∂c²/∂B ≡ 0` and the barrier's β-gradient is IDENTICALLY ZERO:
        // `add_sae_separation_barrier` writes exactly 0 into `sys.gb` here (the
        // `mb·inv − (c²/n_j²)B_j` factor cancels term-for-term), and `gb` correctly
        // carries none of it. Its VALUE, though, is a ~1e7 constant
        // (`−μq·log(1−c²+ε) = −μq·log ε`, `ε = 1e-6`); central-differencing that
        // constant is catastrophic cancellation — fp wobble in `c²` at the ~1e-16
        // level, amplified by `μq/ε`, injects a spurious ~40-magnitude "gradient"
        // into a total-objective FD that `gb` (correctly) does not match. FDing the
        // barrier-free objective — exactly the terms `gb` differentiates to a
        // nonzero gradient, with the barrier's genuinely-zero contribution left
        // out — is the consistent contract (no tolerance is loosened).
        let objective = |t: &SaeManifoldTerm| -> f64 {
            t.loss(target.view(), &rho).expect("loss").total() + t.decoder_repulsion_value(1.0)
        };
        let eps = 1e-6;
        let mut checked = 0usize;
        for k in 0..term.k_atoms() {
            let m = term.atoms[k].basis_size();
            for &(bc, oc) in &[(0usize, 0usize), (m.saturating_sub(1), 0usize)] {
                if bc >= m {
                    continue;
                }
                let idx = offsets[k] + bc * p + oc;
                let orig = term.atoms[k].decoder_coefficients[[bc, oc]];
                term.atoms[k].decoder_coefficients[[bc, oc]] = orig + eps;
                let lp = objective(&term);
                term.atoms[k].decoder_coefficients[[bc, oc]] = orig - eps;
                let lm = objective(&term);
                term.atoms[k].decoder_coefficients[[bc, oc]] = orig;
                let fd = (lp - lm) / (2.0 * eps);
                let g = gb[idx];
                // Sign: gb = +∂(objective)/∂β (data-fit `Jᵀ(fitted−target)`
                // + smoothness + decoder-repulsion force). If CI shows a uniform
                // flip, compare to `-g` (1-char fix).
                assert!(
                    (fd - g).abs() <= 1e-4 * (1.0 + g.abs()),
                    "β gradient exp(s)-consistency (atom {k}, basis {bc}, out {oc}): \
                     FD {fd} vs gb {g}"
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 2,
            "must check at least two β entries; checked {checked}"
        );
    }

    /// #2022 transport-peel — the no-refresh peel normalizes the decoder into
    /// log_amplitude while leaving `smooth_penalty` BYTE-IDENTICAL. This is what
    /// makes it safe at the transport/reparam sites, whose transported penalty
    /// (`T⁻ᵀ S_old T⁻¹`, decoder-magnitude-independent) must survive the peel.
    #[test]
    fn step2_without_refresh_peel_keeps_smooth_penalty() {
        let (mut term, _target, _rho) = small_two_atom_periodic_term();
        // Non-unit decoder so the peel does real work.
        for v in term.atoms[0].decoder_coefficients.iter_mut() {
            *v *= 3.0;
        }
        let penalty_before = term.atoms[0].smooth_penalty.clone();
        let s_before = term.atoms[0].log_amplitude;
        let norm_before = term.atoms[0]
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        term.atoms[0].absorb_decoder_norm_into_log_amplitude_without_refresh(f64::MIN_POSITIVE);
        // smooth_penalty untouched (the refresh was skipped).
        assert_eq!(
            term.atoms[0].smooth_penalty, penalty_before,
            "without_refresh peel must NOT change smooth_penalty"
        );
        // Decoder normalized; magnitude moved into log_amplitude.
        let norm_after = term.atoms[0]
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            (norm_after - 1.0).abs() <= 1e-9,
            "‖B‖ must be 1 after peel; got {norm_after}"
        );
        assert!(
            (term.atoms[0].log_amplitude - (s_before + norm_before.ln())).abs() <= 1e-9,
            "log_amplitude must gain ln‖B‖"
        );
    }
}

// ============================================================================
// #2072 default-off LEVER-WIRING coverage. Both `quotient_scale` and
// `data_row_reseed` are typed `pub(crate) bool` per-fit opt-ins that default
// false (bit-inert). Their leaf producers are unit-tested, but the
// behavior-changing DRIVER path each flag gates was never exercised. These
// tests run the SAME driver op twice — flag OFF (default) then ON — and assert
// the output differs in the exact way the flag is meant to change it, so a
// driver that ignored the flag would fail. Parallel-safe (typed setters, no env).
// ============================================================================
#[cfg(test)]
mod lever_wiring_2072_tests {
    use crate::manifold::tests::small_two_atom_periodic_term;
    use crate::manifold::{
        AssignmentMode, LatentManifold, SaeAssignment, SaeAtomBasisKind, SaeManifoldAtom,
        SaeManifoldRho, SaeManifoldTerm, sae_data_row_anchored_euclidean_coords,
        sae_pca_seed_initial_coords_with_pc_offset,
    };
    use ndarray::{Array2, Array3, array};

    fn frob(b: &Array2<f64>) -> f64 {
        b.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// #2072 — `quotient_scale` DRIVER (the `if self.quotient_scale` arm of
    /// `refit_decoder_least_squares_at_current_state`, construction.rs). The LSQ
    /// refit always writes the ABSOLUTE decoder `B_abs`. With the lever OFF the
    /// write stays that way (`log_amplitude == 0`, decoder magnitude untouched).
    /// With the lever ON the driver resets `s = 0` then peels ‖B_k‖ into
    /// `log_amplitude`, so every decoder is renormalized to unit Frobenius and
    /// #1939 — the per-fit `cone_atom_recovery` opt-in must be carried across
    /// clones (the stagewise driver clones the term for every birth candidate /
    /// backfit, so a dropped field would silently disable recovery mid-fit — a hole
    /// the flag-ON==flag-OFF no-op proof cannot catch, since both read `false`).
    #[test]
    fn clone_preserves_cone_atom_recovery() {
        let (mut term, _target, _rho) = small_two_atom_periodic_term();
        assert!(
            !term.cone_atom_recovery(),
            "cone_atom_recovery must default to off"
        );
        term.set_cone_atom_recovery(true);
        let cloned = term.clone();
        assert!(
            cloned.cone_atom_recovery(),
            "Clone must carry cone_atom_recovery — a dropped field silently disables \
             recovery on every stagewise birth/backfit clone"
        );
    }

    /// #2072 — `cone_atom_recovery` DRIVER (the
    /// `if (self.cone_atom_recovery || self.quotient_scale)
    ///     && self.retract_collapsed_decoders_in_loop() > 0` boundary arm at the
    /// accepted OUTER-iterate seam, fit_drivers.rs). With a CO-VANISHED atom in the
    /// dictionary the lever RECOVERS it: the collapsed decoder is retracted onto the
    /// unit-Frobenius sphere (its magnitude folded into `log_amplitude`) and the
    /// amplitudes re-solved. OFF short-circuits the `&&` (both flags false), so the
    /// retraction helper is never even called and the collapsed decoder stays
    /// collapsed. This runs the EXACT driver gate both ways on one collapsed fixture
    /// and asserts the recovery fires in the intended direction ONLY when the lever
    /// is ON — a driver that ignored the flag (or a clone that dropped it) fails.
    #[test]
    fn cone_atom_recovery_driver_recovers_covanished_decoder() {
        // Shared 2-atom periodic fixture, then artificially CO-VANISH atom 0's
        // decoder to ~1e-6 of its healthy peer: below the breach floor
        // (SAE_ATOM_DECODER_NORM_COLLAPSE_RATIO = 1e-3 · max‖B‖) yet far above the
        // 1e-12 · max direction floor, so it is a RETRACTABLE collapse (not an
        // unretractable literal zero).
        fn make_collapsed() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
            let (mut term, target, rho) = small_two_atom_periodic_term();
            let collapsed = &term.atoms[0].decoder_coefficients * 1.0e-6;
            term.atoms[0].decoder_coefficients.assign(&collapsed);
            (term, target, rho)
        }

        // Healthy peer norm sets the breach floor the driver measures collapse
        // against (the MAX survivor, gauge.rs).
        let peer_norm = {
            let (term, _t, _r) = make_collapsed();
            frob(&term.atoms[1].decoder_coefficients)
        };
        let breach_floor = 1.0e-3 * peer_norm;

        // OFF (default): the gate `(false || false) && …` short-circuits — the
        // retraction helper is NEVER invoked, so atom 0 stays co-vanished.
        let (mut off, off_target, off_rho) = make_collapsed();
        // #2228: the SCALE-gauge (`quotient_scale`) now DEFAULTS ON, so this baseline
        // explicitly disables BOTH levers to isolate the `cone_atom_recovery` driver
        // under test (else the default-on quotient would drive the same retraction).
        off.set_quotient_scale(false);
        assert!(
            !off.cone_atom_recovery() && !off.quotient_scale(),
            "baseline must have both scale-gauge levers OFF"
        );
        if (off.cone_atom_recovery() || off.quotient_scale())
            && off.retract_collapsed_decoders_in_loop() > 0
        {
            off.optimize_log_amplitudes_closed_form(off_target.view(), &off_rho)
                .expect("amplitude solve (off path is unreachable, but must type-check)");
        }
        let off_norm = frob(&off.atoms[0].decoder_coefficients);
        assert!(
            off_norm < breach_floor,
            "OFF: the co-vanished decoder must stay collapsed (‖B_0‖ = {off_norm:e} \
             < breach floor {breach_floor:e})"
        );

        // ON: engage the lever — the SAME gate now retracts the co-vanished atom
        // onto the unit-Frobenius sphere and re-solves the amplitudes.
        let (mut on, on_target, on_rho) = make_collapsed();
        on.set_cone_atom_recovery(true);
        let recovered = (on.cone_atom_recovery() || on.quotient_scale())
            && on.retract_collapsed_decoders_in_loop() > 0;
        assert!(
            recovered,
            "ON: the co-vanished atom must be flagged as collapsed and retracted"
        );
        on.optimize_log_amplitudes_closed_form(on_target.view(), &on_rho)
            .expect("amplitude solve (on)");
        let on_norm = frob(&on.atoms[0].decoder_coefficients);
        assert!(
            (on_norm - 1.0).abs() <= 1e-9,
            "ON: the recovered decoder must sit on the unit-Frobenius sphere; \
             got ‖B_0‖ = {on_norm}"
        );

        // The lever produced a genuinely different, recovered decoder state — the
        // collapsed norm was lifted by many orders of magnitude, not left inert.
        assert!(
            on_norm > off_norm * 1.0e3,
            "ON must lift the co-vanished decoder far above the OFF (collapsed) \
             norm; on = {on_norm:e}, off = {off_norm:e}"
        );
    }

    /// `s_k = ln‖B_abs,k‖` — a PURE GAUGE move: the reconstruction
    /// `a·exp(s)·Φ·B_unit == a·Φ·B_abs` is preserved to machine precision.
    #[test]
    fn quotient_scale_driver_peels_decoder_norm_and_preserves_reconstruction() {
        // OFF baseline — #2228: `quotient_scale` now defaults ON, so disable it
        // explicitly to exercise the lever-off refit path this test contrasts against.
        let (mut off, target, rho) = small_two_atom_periodic_term();
        off.set_quotient_scale(false);
        off.refit_decoder_least_squares_at_current_state(target.view(), Some(&rho))
            .expect("LSQ refit (lever off)");
        let fitted_off = off
            .try_fitted_for_rho(&rho)
            .expect("reconstruction after off-refit");

        // ON: identical fixture + refit, lever engaged.
        let (mut on, _t, _r) = small_two_atom_periodic_term();
        on.set_quotient_scale(true);
        on.refit_decoder_least_squares_at_current_state(target.view(), Some(&rho))
            .expect("LSQ refit (lever on)");
        let fitted_on = on
            .try_fitted_for_rho(&rho)
            .expect("reconstruction after on-refit");

        // (1) OFF keeps log_amplitude at 0 and leaves at least one non-unit decoder
        //     (else the peel would be vacuous — this guards that the ON change is
        //     the peel, not the solve).
        let mut off_has_nonunit = false;
        for atom in &off.atoms {
            assert_eq!(atom.log_amplitude, 0.0, "off: log_amplitude must stay 0");
            if (frob(&atom.decoder_coefficients) - 1.0).abs() > 1e-3 {
                off_has_nonunit = true;
            }
        }
        assert!(
            off_has_nonunit,
            "off: the LSQ solve must leave a non-unit decoder (else the peel is vacuous)"
        );

        // (2) ON renormalizes EVERY decoder to unit Frobenius with a finite
        //     log_amplitude, and at least one amplitude actually moved off 0.
        let mut on_amp_moved = false;
        for atom in &on.atoms {
            let nrm = frob(&atom.decoder_coefficients);
            assert!(
                (nrm - 1.0).abs() <= 1e-9,
                "on: ‖B_k‖ must be 1 after the quotient peel; got {nrm}"
            );
            assert!(
                atom.log_amplitude.is_finite(),
                "on: log_amplitude must be finite after the peel"
            );
            if atom.log_amplitude.abs() > 1e-6 {
                on_amp_moved = true;
            }
        }
        assert!(
            on_amp_moved,
            "on: the peel must move a decoder magnitude into log_amplitude"
        );

        // (3) The peel is gauge-invariant: reconstruction preserved to machine tol.
        assert_eq!(fitted_on.dim(), fitted_off.dim(), "recon shape preserved");
        let max_gap = (&fitted_on - &fitted_off)
            .iter()
            .fold(0.0_f64, |m, d| m.max(d.abs()));
        assert!(
            max_gap <= 1e-9,
            "quotient peel must preserve the reconstruction within tol; max gap {max_gap:e}"
        );
    }

    /// A K=1 caller-managed FLAT (EuclideanPatch) term with n ≫ p. Zero decoder
    /// ⇒ fitted = 0 ⇒ the reconstruction residual is `-target` (a real,
    /// structured residual the reseed producers read). `p = 4`, `n = 8` ⇒
    /// `pc_pairs = min(n, p)/2 = 2`, so any `pc_pair_offset ≥ 2` is an
    /// exhausted-pool retry — the only regime the `data_row_reseed` lever gates.
    fn small_flat_euclidean_term() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let n = 8usize;
        let p = 4usize;
        let m = 2usize;
        let phi = Array2::<f64>::ones((n, m));
        let jet = Array3::<f64>::zeros((n, m, 1));
        let decoder = Array2::<f64>::zeros((m, p));
        let smooth = Array2::<f64>::eye(m);
        let atom = SaeManifoldAtom::new(
            "flat0",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            smooth,
        )
        .unwrap();
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![Array2::<f64>::zeros((n, 1))],
            vec![LatentManifold::Euclidean],
            AssignmentMode::softmax(1.0),
        )
        .unwrap();
        let term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
        let mut target = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                target[[i, j]] = ((i * 7 + j) as f64).sin() + 0.25 * ((i + 3 * j) as f64).cos();
            }
        }
        let rho = SaeManifoldRho::new(0.0, 0.0, vec![array![0.0]]);
        (term, target, rho)
    }

    /// #2072 — `data_row_reseed` DRIVER (the `if data_row_reseed && all_flat && …`
    /// arm of `reseed_atoms_onto_distinct_residual_pcs`, fit_drivers.rs). On an
    /// exhausted-pool retry (`offset ≥ pc_pairs`) over all-FLAT atoms the lever
    /// routes the reseed from the ~p/2-capped PCA pool to the DATA-ROW-anchored
    /// producer (unbounded n-row diversity). OFF stays on the PCA path. The test
    /// runs the driver both ways and pins WHICH producer each path took.
    #[test]
    fn data_row_reseed_driver_routes_exhausted_pool_retry_to_data_row_anchor() {
        let atoms = [0usize];
        let offset = 7usize; // ≥ pc_pairs = 2 ⇒ exhausted-pool retry.

        // OFF (default): PCA pool even on the exhausted-pool retry.
        let (mut off, target, rho) = small_flat_euclidean_term();
        off.reseed_atoms_onto_distinct_residual_pcs(&atoms, target.view(), &rho, offset)
            .expect("reseed (lever off)");
        let coords_off = off.assignment.coords[0].as_matrix();

        // ON: same retry routed to the data-row-anchored producer.
        let (mut on, _t, _r) = small_flat_euclidean_term();
        on.set_data_row_reseed(true);
        on.reseed_atoms_onto_distinct_residual_pcs(&atoms, target.view(), &rho, offset)
            .expect("reseed (lever on)");
        let coords_on = on.assignment.coords[0].as_matrix();

        // The two producers write genuinely different seeds.
        let max_diff = (&coords_on - &coords_off)
            .iter()
            .fold(0.0_f64, |m, d| m.max(d.abs()));
        assert!(
            max_diff > 1e-6,
            "the lever must route to a different seed on the exhausted-pool retry; \
             max |on - off| = {max_diff:e}"
        );

        // Pin WHICH producer each path took (non-vacuous): recompute both
        // reference seeds from a fresh term at the driver's exact anchor/offset.
        let (reference, _rt, _rr) = small_flat_euclidean_term();
        let residual = reference
            .reconstruction_residual(target.view(), &rho)
            .expect("residual for reference seeds");
        let n = reference.n_obs();
        let dims = [1usize];
        let kinds = [SaeAtomBasisKind::EuclideanPatch];
        let pca =
            sae_pca_seed_initial_coords_with_pc_offset(residual.view(), &kinds, &dims, offset)
                .expect("pca reference seed");
        // Driver anchor for slot 0: (0 + offset·atoms.len()) % n.
        let anchor_rows = [(0usize + offset.wrapping_mul(atoms.len().max(1))) % n];
        let data_row = sae_data_row_anchored_euclidean_coords(residual.view(), &dims, &anchor_rows)
            .expect("data-row reference seed");

        for row in 0..n {
            assert!(
                (coords_off[[row, 0]] - pca[[0, row, 0]]).abs() <= 1e-12,
                "off path must equal the PCA producer at row {row}"
            );
            assert!(
                (coords_on[[row, 0]] - data_row[[0, row, 0]]).abs() <= 1e-12,
                "on path must equal the data-row producer at row {row}"
            );
        }

        // Guard the >1e-6 gap above: the two reference producers are distinct.
        let mut ref_gap = 0.0_f64;
        for row in 0..n {
            ref_gap = ref_gap.max((pca[[0, row, 0]] - data_row[[0, row, 0]]).abs());
        }
        assert!(
            ref_gap > 1e-6,
            "the PCA and data-row producers must differ at this offset; gap {ref_gap:e}"
        );
    }

    /// #1939 — the DEFAULT-ON ARD/evidence prior on `log s_k` + SCAD-on-amplitude
    /// penalty inside `optimize_log_amplitudes_closed_form`. A spurious atom whose
    /// TRUE amplitude is zero (its only support is orthogonal-complement noise) is
    /// shrunk to the amplitude FLOOR by SCAD's selection, while a genuine
    /// large-amplitude atom is recovered essentially UNBIASED (it sits in SCAD's
    /// flat tail; the evidence prior's log-shrinkage is negligible against the
    /// strong data curvature). No flag toggles this — both effects are on by default
    /// and self-tune from the fit's own evidence/noise.
    #[test]
    fn cone_amplitude_ard_scad_shrinks_spurious_atom_1939() {
        let (mut term, _t, rho) = small_two_atom_periodic_term();
        assert_eq!(term.k_atoms(), 2);
        // Per-atom UNIT-amplitude designs C_k, read off the term's own
        // reconstruction exactly as the solver's probe does (atom k on at s = 0, the
        // rest off). This fixture installs no RowMetric / row-weight ⇒ the amplitude
        // Gram is plain Euclidean over the flattened R^5 outputs (p = 1).
        const OFF: f64 = -700.0;
        let unit_design = |term: &mut SaeManifoldTerm, k: usize| -> Array2<f64> {
            for (j, a) in term.atoms.iter_mut().enumerate() {
                a.log_amplitude = if j == k { 0.0 } else { OFF };
            }
            term.try_fitted_for_rho(&rho).expect("unit design")
        };
        let c0 = unit_design(&mut term, 0);
        let c1 = unit_design(&mut term, 1);
        let v0: Vec<f64> = c0.iter().copied().collect();
        let v1: Vec<f64> = c1.iter().copied().collect();
        let n = v0.len();
        let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };

        // ε in the ORTHOGONAL COMPLEMENT of span(C0, C1) (Gram–Schmidt): it perturbs
        // neither atom's least-squares amplitude (β0_ls = 8, β1_ls = 0 EXACTLY) yet
        // supplies a nonzero residual variance σ̂² > 0 for the evidence/SCAD scales to
        // read — so atom 1 is a genuinely SPURIOUS (zero-amplitude) atom that only
        // noise could support.
        let mut eps = vec![0.30, -0.22, 0.18, -0.26, 0.11];
        assert_eq!(eps.len(), n);
        for basis in [&v0, &v1] {
            let bb = dot(basis, basis);
            if bb > 0.0 {
                let proj = dot(&eps, basis) / bb;
                for i in 0..n {
                    eps[i] -= proj * basis[i];
                }
            }
        }
        for basis in [&v0, &v1] {
            assert!(
                dot(&eps, basis).abs() < 1e-9,
                "ε must be orthogonal to the atom designs"
            );
        }
        assert!(
            dot(&eps, &eps).sqrt() > 1e-2,
            "ε must carry real residual energy so σ̂² > 0"
        );

        // Ground truth: atom 0 STRONG (β0 = 8), atom 1 ABSENT (β1 = 0). target =
        // 8·C0 + ε.
        const BETA0: f64 = 8.0;
        let mut target = Array2::<f64>::zeros(c0.dim());
        for (i, t) in target.iter_mut().enumerate() {
            *t = BETA0 * v0[i] + eps[i];
        }

        // Neutral default state, then the production penalized amplitude solve.
        term.atoms[0].log_amplitude = 0.0;
        term.atoms[1].log_amplitude = 0.0;
        term.optimize_log_amplitudes_closed_form(target.view(), &rho)
            .expect("penalized amplitude solve");
        let b0 = term.atoms[0].log_amplitude.exp();
        let b1 = term.atoms[1].log_amplitude.exp();

        // (1) The true large amplitude is recovered essentially unbiased (SCAD leaves
        //     β ≫ γλ untouched; the evidence prior barely moves it): within 15%.
        assert!(
            (b0 - BETA0).abs() <= 0.15 * BETA0,
            "true large-amplitude atom must be preserved (β0 = {b0}, want ≈ {BETA0})"
        );
        // (2) The spurious atom (zero true amplitude, noise-only support below the
        //     universal SCAD threshold) is driven to the amplitude floor — turned
        //     OFF — not merely nudged.
        assert!(
            b1 < 1e-6,
            "spurious zero-amplitude atom must be shrunk to the floor (β1 = {b1})"
        );
        // (3) The two verdicts are decisively separated.
        assert!(
            b0 / b1 > 1e6,
            "true vs spurious amplitude must separate (ratio {})",
            b0 / b1
        );
    }

    /// #1939 / #2134 — K=3 MULTI-ACTIVE co-collapse RECOVERY through the amplitude
    /// scale-gauge [`SaeManifoldTerm::pin_scale_gauge`]. THREE concepts are planted
    /// on atoms 0, 1, 2, each on its OWN ambient output column (disjoint,
    /// unit-Frobenius decoders → orthogonal gated designs `C_k`), and an IBP-MAP
    /// gate caps each atom's activation by its COLUMN INDEX:
    /// `a_k = σ(l_k/τ)·π_k`, `π_k = (α/(α+1))^{k+1} = {0.5, 0.25, 0.125}` at α=1.
    /// With the logits saturated (`σ ≈ 1`) the gate is a pure ordered-prior cap, so
    /// the UNPENALIZED-amplitude reconstruction (`s_k = 0`) delivers only `π_k` of
    /// each planted concept and CO-COLLAPSES per active atom (`top_k>1` divergence).
    ///
    /// The pin peels each unit-Frobenius decoder into the unpenalized `s_k` and
    /// re-homes the magnitude by the JOINT amplitude solve. Because the ordered
    /// prior over-shrinks atom `k` by `1/π_k` PER COLUMN INDEX (not a single shared
    /// scale), the recovery must be per-atom: `exp(s_k) ≈ 1/π_k` (2, 4, 8), each
    /// atom's gated reconstruction must reach its planted concept, and the aggregate
    /// EV must lift from the co-collapsed regime to ≈ 1. This is the generalized
    /// (any-K) pin, the case the old `k_atoms()==1` gate could not serve.
    #[test]
    fn pin_scale_gauge_recovers_k3_multi_active_co_collapse_1939() {
        use crate::assignment::ibp_map_row;
        use crate::manifold::tests::{TestPeriodicEvaluator, periodic_basis};
        use ndarray::Array1;
        use std::sync::Arc;

        // Three periodic atoms at distinct coords; each decoder writes ONLY its own
        // ambient column `k` (block-diagonal in p = 3) and is unit-Frobenius, so the
        // three gated designs `C_k` are exactly ORTHOGONAL — the joint K×K amplitude
        // solve decouples and `exp(s_k)` reads off `1/a_k` with no cross-talk. The
        // sin/cos loadings are nonzero so every output column VARIES across rows (a
        // target with real, non-constant variance to explain).
        let coords = [
            array![[0.05], [0.20], [0.55], [0.80], [0.35]],
            array![[0.15], [0.30], [0.65], [0.90], [0.45]],
            array![[0.10], [0.42], [0.70], [0.25], [0.88]],
        ];
        let raw_cols = [
            [0.20_f64, 0.70, -0.50],
            [-0.30, 0.55, 0.40],
            [0.45, -0.35, 0.60],
        ];
        let n = 5usize;
        let p = 3usize;
        let mut atoms = Vec::new();
        let mut coords_vec = Vec::new();
        for k in 0..3 {
            let (phi, jet) = periodic_basis(&coords[k]);
            let mut dec = Array2::<f64>::zeros((3, p));
            for r in 0..3 {
                dec[[r, k]] = raw_cols[k][r];
            }
            let fro = frob(&dec);
            dec.mapv_inplace(|v| v / fro); // unit-Frobenius decoder B̂_k
            let atom = SaeManifoldAtom::new(
                format!("periodic{k}"),
                SaeAtomBasisKind::Periodic,
                1,
                phi,
                jet,
                dec,
                Array2::<f64>::eye(3),
            )
            .unwrap()
            .with_basis_evaluator(Arc::new(TestPeriodicEvaluator));
            atoms.push(atom);
            coords_vec.push(coords[k].clone());
        }

        // Saturated logits ⇒ σ(l/τ) ≈ 1 ⇒ a_k ≈ π_k: the gate is a PURE ordered
        // prior cap by column index, the exact IBP-MAP over-shrink the pin targets.
        let mut logits = Array2::<f64>::zeros((n, 3));
        logits.fill(20.0);
        let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
            logits,
            coords_vec,
            vec![LatentManifold::Circle { period: 1.0 }; 3],
            AssignmentMode::ibp_map(1.0, 1.0, false),
        )
        .unwrap();
        let mut term = SaeManifoldTerm::new(atoms, assignment).unwrap();
        assert_eq!(term.k_atoms(), 3);
        let rho = SaeManifoldRho::new(0.0, -6.0, vec![array![0.0], array![0.0], array![0.0]]);

        // Gate values a_k = σ(20)·π_k (σ(20) = 0.999999998 ⇒ a_k = π_k to 1e-9).
        let gate = ibp_map_row(Array1::from(vec![20.0_f64; 3]).view(), 1.0);

        // Per-atom UNIT-amplitude gated designs C_k = a_k·Φ_k·B̂_k, read off the
        // term's OWN reconstruction (atom k on at s = 0, the rest off) exactly as
        // the solver's probe does. C_k is nonzero only in ambient column k.
        const OFF: f64 = -700.0;
        let unit_design = |term: &mut SaeManifoldTerm, kk: usize| -> Array2<f64> {
            for (j, a) in term.atoms.iter_mut().enumerate() {
                a.log_amplitude = if j == kk { 0.0 } else { OFF };
            }
            term.try_fitted_for_rho(&rho).expect("unit design probe")
        };
        let c: Vec<Array2<f64>> = (0..3).map(|k| unit_design(&mut term, k)).collect();

        // Planted target: each concept at UNIT amplitude, gate removed:
        // z = Σ_k Φ_k·B̂_k = Σ_k C_k / a_k. The gate can only reach a_k = π_k of it.
        let mut z = Array2::<f64>::zeros((n, p));
        for k in 0..3 {
            z.scaled_add(1.0 / gate[k], &c[k]);
        }
        // A tiny residual in the ORTHOGONAL COMPLEMENT of span(C_0,C_1,C_2) so the
        // penalized solve has a well-posed σ̂² > 0 (a noiseless-exact target would
        // floor σ̂² and destabilize λ); Gram–Schmidt against each flattened C_k
        // keeps it orthogonal, so it perturbs no atom's LS amplitude (exp(s_k) = 1/a_k
        // is preserved) yet supplies real residual energy.
        let mut eps = Array2::<f64>::zeros((n, p));
        for (i, v) in eps.iter_mut().enumerate() {
            *v = (0.7 * i as f64 + 0.3).sin();
        }
        for ck in &c {
            let cf: Vec<f64> = ck.iter().copied().collect();
            let cc: f64 = cf.iter().map(|x| x * x).sum();
            if cc > 0.0 {
                let ef: Vec<f64> = eps.iter().copied().collect();
                let proj = ef.iter().zip(&cf).map(|(e, x)| e * x).sum::<f64>() / cc;
                for (e, x) in eps.iter_mut().zip(cf.iter()) {
                    *e -= proj * x;
                }
            }
        }
        let znorm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
        let enorm = eps.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        eps.mapv_inplace(|v| v * 0.03 * znorm / enorm); // ≈ 3% orthogonal residual
        z = &z + &eps;

        // Co-collapsed regime: unpenalized amplitude (s_k = 0) ⇒ the gate delivers
        // only a_k = π_k of each concept ⇒ the reconstruction cannot reach z.
        for atom in term.atoms.iter_mut() {
            atom.log_amplitude = 0.0;
        }
        let ev_before = term
            .dictionary_reconstruction_ev(z.view(), &rho)
            .expect("EV defined");
        assert!(
            ev_before < 0.5,
            "the s_k = 0 gate-capped reconstruction must CO-COLLAPSE (EV = {ev_before:.4}); \
             each atom delivers only π_k of its concept"
        );

        // The cure: peel each unit decoder into s_k, re-home by the joint solve.
        term.pin_scale_gauge(z.view(), &rho)
            .expect("multi-active scale-gauge pin");

        // (1) exp(s_k) ≈ 1/π_k emerges PER ATOM (2, 4, 8) — per-column-index
        //     compensation, not one shared scale.
        let mut amp = [0.0_f64; 3];
        for k in 0..3 {
            amp[k] = term.atoms[k].log_amplitude.exp();
            let pi_k = 0.5_f64.powi(k as i32 + 1);
            let want = 1.0 / gate[k]; // = 1/a_k ≈ 1/π_k
            assert!(
                (amp[k] - want).abs() <= 0.08 * want,
                "atom {k}: exp(s_k) = {:.5} must recover 1/a_k = {want:.5} (≈ 1/π_k = {:.3})",
                amp[k],
                1.0 / pi_k
            );
        }
        // (2) The per-atom amplitudes are DISTINCT and ordered by column index — the
        //     hallmark of per-π_k compensation (a single shared scale cannot do this).
        assert!(
            amp[0] < amp[1] && amp[1] < amp[2],
            "per-column-index compensation must be ordered: {amp:?}"
        );

        // (3) Per-atom gated reconstruction now REACHES the planted concept: the
        //     recovered fitted output matches z within the seeded residual.
        let fitted = term.try_fitted_for_rho(&rho).expect("recovered fitted");
        for k in 0..3 {
            let col_sig: f64 = (0..n).map(|r| z[[r, k]] * z[[r, k]]).sum::<f64>().sqrt();
            let col_res: f64 = (0..n)
                .map(|r| (fitted[[r, k]] - z[[r, k]]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                col_res <= 0.15 * col_sig.max(1e-12),
                "atom {k} gated reconstruction must reach its concept column \
                 (residual {col_res:.4} vs signal {col_sig:.4})"
            );
        }

        // (4) Aggregate reconstruction EV lifts out of the co-collapse to ≈ 1.
        let ev_after = term
            .dictionary_reconstruction_ev(z.view(), &rho)
            .expect("EV defined");
        assert!(
            ev_after > 0.9,
            "the scale-gauge pin must recover the multi-active reconstruction \
             (EV {ev_before:.4} → {ev_after:.4})"
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
        term.set_atom_inner_fits(target.view(), &rho, dispersion)
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
