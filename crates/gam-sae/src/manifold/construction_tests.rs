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

    /// PATH C (#2253) — the exact fixed-stratum Hessian block for the solver-free
    /// explicit channels (decoder-smoothness with its Occam renormalization + ARD
    /// log-precision prior) must equal a central finite difference of the SAME
    /// production gradient channels at a frozen inner state. First HVP channel
    /// gate: it exercises the smoothness renormalization's rank-one cross-coupling
    /// and the periodic-ARD normalizer second derivative on the two-atom circle
    /// fixture. The rank-charge, assignment, log-determinant, and third-order
    /// channels are gated separately; this reference omits them.
    #[test]
    fn outer_explicit_smoothness_ard_hessian_matches_finite_difference_2253() {
        use ndarray::Array1;
        let (term, _target, rho) = small_two_atom_periodic_term();
        let n_params = rho.to_flat().len();
        let lambda = rho.lambda_smooth_vec().unwrap();
        let frozen_smoothness: f64 = term
            .decoder_smoothness_value_per_atom(&lambda)
            .iter()
            .sum();

        let analytic = term
            .outer_explicit_smoothness_ard_hessian(&rho, frozen_smoothness)
            .expect("explicit smoothness/ARD Hessian block assembles");

        let base = rho.to_flat();
        let eps = 1.0e-6;
        for j in 0..n_params {
            // Solver-free reference gradient (smoothness renormalized to the
            // FROZEN energy + ARD explicit derivative) at ρ ± ε e_j.
            let gradient = |sign: f64| -> Array1<f64> {
                let mut flat = base.clone();
                flat[j] += sign * eps;
                let r = rho.from_flat(flat.view()).unwrap();
                let mut v = Array1::<f64>::zeros(n_params);
                let lam = r.lambda_smooth_vec().unwrap();
                let se = term.decoder_smoothness_value_per_atom(&lam);
                let s: f64 = se.iter().sum();
                for a in 0..r.log_lambda_smooth.len() {
                    v[r.smooth_flat_index(a)] = if s.abs() > 0.0 {
                        frozen_smoothness * se[a] / s
                    } else {
                        se[a]
                    };
                }
                let ard = term.ard_log_precision_explicit_derivatives(&r).unwrap();
                for (atom, axes) in ard.iter().enumerate() {
                    for axis in 0..axes.len() {
                        v[r.ard_flat_index(atom, axis)] += ard[atom][axis];
                    }
                }
                v
            };
            let fd_col = (gradient(1.0) - gradient(-1.0)) / (2.0 * eps);
            for i in 0..n_params {
                let analytic_ij = analytic[[i, j]];
                let fd_ij = fd_col[i];
                assert!(
                    (analytic_ij - fd_ij).abs() < 1.0e-6 + 1.0e-5 * analytic_ij.abs(),
                    "explicit smoothness/ARD Hessian [{i},{j}] mismatch: \
                     analytic={analytic_ij}, fd={fd_ij}"
                );
            }
        }
    }

    /// PATH C (#2253) — the exact fixed-stratum second derivative of the
    /// rank-charge `direct_rho` channel must equal a central finite difference of
    /// `production_rank_charge_derivative(...).direct_rho` at a frozen inner state
    /// (frozen `loss`/`cache`), on the converged two-atom circle fixture. Second
    /// HVP channel gate; exercises the `A⁻¹G (A⁻¹S)²` curvature trace.
    #[test]
    fn rank_charge_direct_rho_hessian_matches_finite_difference_2253() {
        use crate::manifold::tests::gamma_fd_tiny_fixture;
        use ndarray::Array1;
        // small_two_atom_periodic_term co-collapses through the full inner fit at
        // current HEAD (K=2 unsupported for that tiny target); gamma_fd_tiny is the
        // converging fixture the sibling criterion tests use. rank-charge is
        // smooth-index-only and assignment-mode-agnostic, so its second derivative
        // is exercised identically.
        let (mut term, target, rho) = gamma_fd_tiny_fixture();
        let (_cost, loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                40,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("converged joint cache for the frozen stratum");

        let n_params = rho.to_flat().len();
        let analytic = term
            .rank_charge_direct_rho_hessian(target.view(), &rho, &loss, &cache)
            .expect("rank-charge direct_rho Hessian assembles");
        assert!(
            analytic.iter().any(|&x| x.abs() > 1.0e-6),
            "fixture must exercise a non-trivial rank-charge curvature (interior EDF), \
             else this gate is vacuous"
        );

        let base = rho.to_flat();
        let eps = 1.0e-6;
        for j in 0..n_params {
            // FD of the production rank-charge direct_rho gradient at ρ ± ε e_j
            // with the inner state (loss, cache) held frozen.
            let direct_rho = |sign: f64| -> Array1<f64> {
                let mut flat = base.clone();
                flat[j] += sign * eps;
                let r = rho.from_flat(flat.view()).unwrap();
                term.production_rank_charge_derivative(target.view(), &r, &loss, &cache)
                    .unwrap()
                    .direct_rho
            };
            let fd_col = (direct_rho(1.0) - direct_rho(-1.0)) / (2.0 * eps);
            for i in 0..n_params {
                let analytic_ij = analytic[[i, j]];
                let fd_ij = fd_col[i];
                assert!(
                    (analytic_ij - fd_ij).abs() < 1.0e-5 + 1.0e-4 * analytic_ij.abs(),
                    "rank-charge direct_rho Hessian [{i},{j}] mismatch: \
                     analytic={analytic_ij}, fd={fd_ij}"
                );
            }
        }
    }

    /// PATH C (#2253) — the exact fixed-stratum second derivative of the outer
    /// gradient's log-determinant Daleckii–Krein trace channel (`logdet_trace`)
    /// must equal a central finite difference of that SAME production channel at a
    /// frozen inner state. Third HVP channel gate; it exercises the full-`H⁻¹`
    /// selected-inverse curvature (`−tr(G C_j G Cᵢ)`) for both the decoder
    /// smoothness EDF trace and the periodic-ARD log-precision Hessian trace, plus
    /// their cross coupling and the rank-charge coordinate-block subtraction. The
    /// FD rebuilds the fixed-θ̂ cache at each ρ ± h so `H⁻¹` MOVES with ρ — the
    /// Daleckii–Krein term the analytic block carries.
    #[test]
    fn logdet_daleckii_krein_hessian_matches_finite_difference_2253() {
        use crate::manifold::arrow_solver::DeflatedArrowSolver;
        use crate::manifold::tests::gamma_fd_tiny_fixture;
        use ndarray::{array, Array1};
        // gamma_fd_tiny sits at a ρ = −6 floor with no interior PD minimum (the
        // sibling rank-charge gate documents the same). Lift ρ_sparse into the PD
        // basin AND lift ρ_smooth / ρ_ard off the floor so the decoder-smoothness
        // EDF (λ_smooth ≈ 1.35) and the periodic-ARD log-precision Hessian traces
        // (α ≈ 0.9, 1.1) carry a non-trivial curvature signal. A ρ perturbation
        // scales only α, never the frozen circle coordinate t, so the max(·,0)
        // majorizer active set is invariant — no subgradient ambiguity at the FD.
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.5;
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = 0.3;
        }
        rho.log_ard = vec![array![-0.1_f64], array![0.1_f64]];
        // Converge to a well-conditioned PD stationary state (mutates `term` to the
        // fitted θ̂), then re-derive the fixed-θ̂ cache with a zero inner budget:
        // H(ρ) = H_data(θ̂) + penalty(ρ) with θ̂ frozen — the fixed stratum the
        // analytic Hessian differentiates.
        term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            200,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("converged PD stationary cache");
        let (_value, _loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                0,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("fixed-theta base cache");

        let n_params = rho.to_flat().len();
        let analytic = term
            .logdet_daleckii_krein_hessian(&rho, &cache)
            .expect("logdet Daleckii-Krein Hessian block assembles");

        // The smooth + ARD coordinates this channel covers (sparse/block excluded).
        let mut coord_indices: Vec<usize> = Vec::new();
        for a in 0..rho.log_lambda_smooth.len() {
            coord_indices.push(rho.smooth_flat_index(a));
        }
        for kk in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[kk].len() {
                let idx = rho.ard_flat_index(kk, axis);
                if !coord_indices.contains(&idx) {
                    coord_indices.push(idx);
                }
            }
        }

        // Non-vacuity: a smoothing AND an ARD diagonal must carry real curvature,
        // else the gate would pass on an all-zero block.
        let smooth0 = rho.smooth_flat_index(0);
        let ard0 = rho.ard_flat_index(0, 0);
        assert!(
            analytic[[smooth0, smooth0]].abs() > 1.0e-6,
            "smoothing logdet curvature must be non-trivial: {}",
            analytic[[smooth0, smooth0]]
        );
        assert!(
            analytic[[ard0, ard0]].abs() > 1.0e-6,
            "ARD logdet curvature must be non-trivial: {}",
            analytic[[ard0, ard0]]
        );

        // The production `logdet_trace` channel in ISOLATION, reproduced exactly as
        // `analytic_outer_rho_gradient_components` assembles it (smooth EDF trace +
        // ARD joint minus coordinate-block trace), so this validates CH4
        // independently of the rank-charge / third-order channels.
        let base = rho.to_flat();
        let h = 1.0e-5;
        let logdet_trace_at = |sign: f64, j: usize| -> Array1<f64> {
            let mut flat = base.clone();
            flat[j] += sign * h;
            let r = rho.from_flat(flat.view()).unwrap();
            let mut t = term.clone();
            let (_value, _loss, cache) = t
                .penalized_quasi_laplace_criterion_with_cache(
                    target.view(),
                    &r,
                    None,
                    0,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                )
                .expect("perturbed fixed-theta cache");
            let solver = DeflatedArrowSolver::plain(&cache);
            let lambda = r.lambda_smooth_vec().unwrap();
            let smooth_logdet = t
                .decoder_smoothness_effective_dof_with_solver_per_atom(&cache, &solver, &lambda)
                .expect("smooth EDF trace");
            let ard_joint = t
                .ard_log_precision_hessian_trace(&r, &cache, &solver)
                .expect("ard joint logdet trace");
            let ard_coord = t
                .coordinate_block_ard_log_precision_hessian_trace(&r, &cache)
                .expect("ard coordinate-block logdet trace");
            let mut v = Array1::<f64>::zeros(n_params);
            for a in 0..r.log_lambda_smooth.len() {
                v[r.smooth_flat_index(a)] = 0.5 * smooth_logdet[a];
            }
            for kk in 0..r.log_ard.len() {
                for axis in 0..r.log_ard[kk].len() {
                    v[r.ard_flat_index(kk, axis)] += ard_joint[kk][axis] - ard_coord[kk][axis];
                }
            }
            v
        };
        for &j in &coord_indices {
            let fd_col = (logdet_trace_at(1.0, j) - logdet_trace_at(-1.0, j)) / (2.0 * h);
            for &i in &coord_indices {
                let analytic_ij = analytic[[i, j]];
                let fd_ij = fd_col[i];
                assert!(
                    (analytic_ij - fd_ij).abs() < 1.0e-5 + 1.0e-4 * analytic_ij.abs(),
                    "logdet Daleckii-Krein Hessian [{i},{j}] mismatch: \
                     analytic={analytic_ij}, fd={fd_ij}"
                );
            }
        }
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
    use crate::manifold::tests::{diagonal_latent_cache, gamma_fd_tiny_fixture};
    use approx::assert_abs_diff_eq;
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
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                40,
                0.4,
                1.0e-6,
                1.0e-6,
            )
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
        let solver = DeflatedArrowSolver::from_orthonormal_gauges(&cache, vec![gauge], stiffness)
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

        let solved =
            solve_exact_stationarity_on_gauge_quotient(&solver, &rhs, &apply_raw_a, &apply_raw_b)
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
        let apply_raw_b = |v: &SaeArrowVector| -> Result<SaeArrowVector, String> { Ok(v.clone()) };

        let solved =
            solve_exact_stationarity_on_gauge_quotient(&solver, &rhs, &apply_raw_a, &apply_raw_b)
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
        term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, false);
        // Saturate atom 1's gate logits hard OFF: σ'(−40)² ≈ 1e-35 kills the
        // data curvature along those logit coordinates while the assembled
        // majorizer keeps an O(1) diagonal there.
        for row in 0..term.n_obs() {
            term.assignment.logits[[row, 1]] = -40.0;
        }
        rho.log_lambda_sparse = -1.0;
        let (_value, _loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                40,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("converged saturated-gate ordered Beta--Bernoulli cache");
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
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                40,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("converged cache for the two-atom fixture");
        let lambda = rho.lambda_smooth_vec().unwrap();

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
    /// must return the exact JOINT inverse-Hessian covariance.
    #[test]
    fn recompute_reproduces_joint_shape_band() {
        // A reliably-converging tiny state: the fixture target was assembled under
        // a softmax gate, so switching to an ordered Beta--Bernoulli gate at the PD-region ρ
        // (`log_lambda_sparse = 0.5`, the deflation-regression config) leaves a
        // genuine reconstruction residual — a real dispersion and nonzero bands —
        // while the state stays near its inner optimum so the undamped joint
        // factor converges in a few steps.
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        term.assignment.mode = AssignmentMode::ordered_beta_bernoulli(0.7, 0.9, true);
        rho.log_lambda_sparse = 0.5;

        // Reference joint bands via the direct Schur path.
        let (_c, loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                5,
                0.4,
                1.0e-6,
                1.0e-6,
            )
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
            let a_sd = a.band_sd.as_ref().expect("recomputed joint band");
            let b_sd = b.band_sd.as_ref().expect("direct joint band");
            assert_eq!(a_sd.dim(), b_sd.dim(), "atom {k} band shape");
            for (x, y) in a_sd.iter().zip(b_sd.iter()) {
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
            let band_sd = atom.band_sd.as_ref().expect("joint band");
            for gi in 0..band_sd.nrows() {
                let row = band_sd.row(gi);
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
    }
}
