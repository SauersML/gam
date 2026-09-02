#[cfg(test)]
mod exact_hessian_fixture_tests {
    use super::*;

    /// One authoritative off-manifold, fixed-stratum fixture for the #1418 exact
    /// stationarity and #2253 exact outer-Hessian gates.
    ///
    /// The target excitation makes residual, entropy, and curvature-delta channels
    /// genuinely live. Its fit must happen in this known non-vanishing
    /// regularization basin; derivative gates may subsequently freeze this state
    /// and assemble a Hessian at a different evaluation `rho`, but must never use
    /// that evaluation point to construct the fitted state.
    pub(super) fn converged_state_with_residual() -> (
        SaeManifoldTerm,
        Array2<f64>,
        SaeManifoldRho,
        ArrowFactorCache,
    ) {
        use crate::manifold::tests::gamma_fd_tiny_fixture;

        // #2253/#2330 re-anchor: after #2330 Phase-2a ranks the EXACT +/-log|A|
        // and refuses an indefinite A, the historical (sparse -0.5, smooth -1,
        // ard -0.5) basin plus the sin excitation lands on an exact-A SADDLE and
        // the criterion refuses at construction. Rank this softmax fixture in a
        // MODERATE-penalty basin (all log-lambda = -1) where the majorizer-
        // converged mode is exact-A positive definite with a real margin
        // (min_eig ~ 1.7e-1, >> the 1e-9 relative PD floor), the residual /
        // curvature-delta channels stay LIVE (‖ΔC‖ ~ 2.0, Daleckii–Krein cross
        // ~ 1.9e-1, both far above FD noise), and the +/-1e-5 FD perturbation of
        // every rho coordinate stays PD (so the fixed-theta FD gates below never
        // trip the refusal at a probe point). The softmax gate is retained (NOT
        // the ordered-Beta--Bernoulli PD specimen) because the logdet
        // Daleckii–Krein and full outer-Hessian channels model the SOFTMAX sparse
        // log-strength row but refuse an OBB sparse coordinate. The exact-A saddle
        // refusal branch is preserved reachably by
        // `converged_state_with_residual_a_saddle_2336`.
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.0;
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for value in axis.iter_mut() {
                *value = -1.0;
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
            .expect("softmax PD-basin re-anchor fixture must converge with both atoms alive");
        (term, target, rho, cache)
    }

    /// #2330/#2336 PRICING-branch companion to `converged_state_with_residual`.
    /// The historical softmax `gamma_fd_tiny_fixture` target is NOT ordered-Beta--
    /// Bernoulli reachable, so its majorizer-converged mode is an exact-A saddle
    /// (a joint-block eigenvalue below the shared PD floor). MEASURED (#2336): that
    /// negative curvature is FULLY attributable to the bounded ARD periodic
    /// concave-clamp wrinkle E (`λ+e_v(ARD)=+0.026`, `e_v=0.041 ≥ |λ|=0.015`), so
    /// under the value-side E-attributability semantics the criterion PRICES it
    /// finite at its basin curvature rather than refusing. This is therefore the
    /// canonical E-attributable wrinkle-saddle PRICING specimen
    /// (`exact_observed_information_prices_e_attributable_a_saddle_2336`); the
    /// genuine (non-attributable) refusal specimen is `obb_patchd_fixture` at a
    /// window-scan saddle scale.
    pub(super) fn converged_state_with_residual_a_saddle_2336()
    -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        use crate::manifold::tests::gamma_fd_tiny_fixture;

        let (term, mut target, mut rho) = gamma_fd_tiny_fixture();
        let (n, p) = (target.nrows(), target.ncols());
        for row in 0..n {
            for col in 0..p {
                let phase = (row as f64 + 0.35) / n as f64;
                let theta = std::f64::consts::TAU * phase;
                target[[row, col]] += 0.6 * (3.0 * theta + 0.5 * col as f64).sin();
            }
        }

        rho.log_lambda_sparse = -0.5;
        for value in rho.log_lambda_smooth.iter_mut() {
            *value = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for value in axis.iter_mut() {
                *value = -0.5;
            }
        }
        (term, target, rho)
    }

}

#[cfg(test)]
mod amortized_encoder_tests {
    use crate::manifold::EvidenceOperator;
    use crate::manifold::tests::small_two_atom_periodic_term;

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
            .expect("smoothness evaluation must preserve CUDA failures")
            .iter()
            .sum();

        let analytic = term
            .outer_explicit_smoothness_ard_hessian(&rho, frozen_smoothness)
            .expect("explicit smoothness/ARD Hessian block assembles");

        // The sparse explicit channel must be LIVE on this softmax fixture, else its
        // Hessian row is a vacuous ~0-vs-~0 comparison.
        let sparse_index = rho
            .sparse_flat_index()
            .expect("softmax fixture must carry a sparse log-strength coordinate");
        assert!(
            analytic[[sparse_index, sparse_index]].abs() > 1.0e-6,
            "sparse explicit ∂² must be non-trivial (λ_sparse·E): {}",
            analytic[[sparse_index, sparse_index]]
        );

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
                let se = term
                    .decoder_smoothness_value_per_atom(&lam)
                    .expect("smoothness evaluation must preserve CUDA failures");
                // TRUE gradient `g_a = renorm·se_a` with `renorm = C/Σse =
                // penalty_scale` ρ-INVARIANT (`C = loss.smoothness = penalty_scale·
                // Σse`, construction.rs:4995). The FD must hold `renorm` fixed and
                // let `se` move — it must NOT re-divide by the moving `Σse`, which
                // is the frozen-`C` convention that manufactured the spurious Occam
                // cross term. This fixture uses `C = Σse(base)` (penalty_scale = 1),
                // so `g_a = se_a`.
                for a in 0..r.log_lambda_smooth.len() {
                    v[r.smooth_flat_index(a)] = se[a];
                }
                let ard = term.ard_log_precision_explicit_derivatives(&r).unwrap();
                for (atom, axes) in ard.iter().enumerate() {
                    for axis in 0..axes.len() {
                        v[r.ard_flat_index(atom, axis)] += ard[atom][axis];
                    }
                }
                // Sparse (softmax log-strength) explicit gradient, the channel CH6
                // adds to the Hessian: the assignment prior value = λ_sparse·E.
                if let Some(si) = r.sparse_flat_index() {
                    v[si] = crate::assignment::assignment_prior_log_strength_derivative_weighted(
                        &term.assignment,
                        &r,
                        term.row_loss_weights.as_deref(),
                    )
                    .unwrap();
                }
                v
            };
            let fd_col = (gradient(1.0) - gradient(-1.0)) / (2.0 * eps);
            for i in 0..n_params {
                let analytic_ij = analytic[[i, j]];
                let fd_ij = fd_col[i];
                assert!(
                    (analytic_ij - fd_ij).abs() < 1.0e-6 + 1.0e-5 * analytic_ij.abs(),
                    "explicit smoothness/ARD/sparse Hessian [{i},{j}] mismatch: \
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
        let (mut term, target, mut rho) = gamma_fd_tiny_fixture();
        // #2253 re-anchor: rank into the moderate-penalty PD basin (all
        // log-lambda = -1, see `converged_state_with_residual`) so the exact-A
        // criterion is positive definite and returns a cache here too.
        rho.log_lambda_sparse = -1.0;
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for v in axis.iter_mut() {
                *v = -1.0;
            }
        }
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
        use ndarray::{Array1, array};
        // Construct θ̂ through the shared, independently exercised PD-basin
        // authority.  This is deliberately distinct from the lifted evaluation ρ
        // below: fitting at that point drives this tiny decoder to co-collapse.
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();

        // Evaluation ρ: a DIFFERENT off-stationary point from the fit basin, chosen
        // so the Daleckii–Krein CROSS terms — which scale as O(λ²) and O(α²) — sit
        // well above FD noise (here λ_smooth = e^-1.5, α = e^-1.2/e^-1, cross ~1e-1,
        // far above the ~1e-4 gate). At the deep floor the cross term is ~6e-6, under
        // the FD tolerance, and the gate would pass on the δ self-term alone while
        // the D-K machinery went entirely unchecked. The fixed-stratum Hessian is
        // exact at ANY frozen θ̂ — it does not require θ̂ to be stationary for this ρ
        // — and a ZERO inner budget assembles H(ρ) = H_data(θ̂) + penalty(ρ) without
        // re-running the fit, so no co-collapse guard is tripped. #2253 re-anchor:
        // this eval also keeps the exact A positive definite with a real margin
        // (min_eig ~6e-2) so the #2330 Phase-2a exact-½log|A| criterion returns a
        // cache here rather than refusing an indefinite mode.
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
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

        // The smooth + ARD coordinates this fixture materially exercises.  The
        // softmax log-strength operator is assembled by this channel too, but the
        // joint and row-block traces cancel on this frozen state (audited below),
        // so including it in the FD matrix would be a vacuous zero-vs-zero gate.
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
        let sparse_index = rho
            .sparse_flat_index()
            .expect("the softmax fixture must carry a live sparse log-strength coordinate");

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
        // Non-vacuity of the Daleckii–Krein term SPECIFICALLY. An OFF-diagonal entry
        // has no δ self-term, so it is pure `−½(tr(G Cᵢ G C_j) − tr(H_bd⁻¹ Cᵢ H_bd⁻¹ C_j))`
        // — the selected-inverse curvature this channel exists to compute. It must
        // exceed the FD tolerance by a real margin, else the gate would be satisfied
        // by the self-term alone and the D-K math would ride through unvalidated.
        let max_off_diagonal = coord_indices
            .iter()
            .flat_map(|&i| coord_indices.iter().map(move |&j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| analytic[[i, j]].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_off_diagonal > 1.0e-4,
            "the Daleckii-Krein cross term must be materially exercised (off-diagonal \
             entries carry no delta self-term); max |off-diagonal| = {max_off_diagonal}"
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
                .ard_log_precision_hessian_trace(&r, &cache, &solver, EvidenceOperator::Majorizer)
                .expect("ard joint logdet trace");
            let ard_coord = t
                .coordinate_block_ard_log_precision_hessian_trace(
                    &r,
                    &cache,
                    EvidenceOperator::Majorizer,
                )
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
            if let Some(si) = r.sparse_flat_index() {
                let joint = t
                    .assignment_log_strength_hessian_trace(&r, &cache, &solver)
                    .expect("sparse joint logdet trace");
                let coord = t
                    .coordinate_block_assignment_log_strength_hessian_trace(
                    &r,
                    &cache,
                    EvidenceOperator::Majorizer,
                )
                    .expect("sparse coordinate-block logdet trace");
                v[si] = joint - coord;
            }
            v
        };
        // Scope audit for the omitted sparse row.  Even on the independently
        // converged residual fixture, the full-joint and coordinate-block
        // logdet traces cancel to roundoff for this free-logit direction.  Record
        // that fact explicitly and require the analytic D-K row to agree with it;
        // the live sparse second derivative is covered by the independent
        // explicit-channel gate above.  This test therefore makes no false claim
        // that a zero-vs-zero sparse FD validates the selected-inverse algebra.
        let base_trace = logdet_trace_at(0.0, sparse_index);
        eprintln!(
            "CH4 sparse logdet leg (inert on this fixture): logdet_trace[sparse]={:.6e}, \
             H[sparse,sparse]={:.6e}",
            base_trace[sparse_index],
            analytic[[sparse_index, sparse_index]]
        );
        assert!(
            base_trace[sparse_index].abs() <= 1.0e-12
                && analytic
                    .row(sparse_index)
                    .iter()
                    .all(|value| value.abs() <= 1.0e-12)
                && analytic
                    .column(sparse_index)
                    .iter()
                    .all(|value| value.abs() <= 1.0e-12),
            "the sparse logdet scope audit must remain an explicitly inert row: \
             logdet_trace[sparse]={}, analytic_row={:?}",
            base_trace[sparse_index],
            analytic.row(sparse_index).to_vec(),
        );

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

    /// PATH C (#2253) / #2339 DIAGNOSTIC — NAME the FD-kink site the softplus did
    /// not heal. fix-2253 attributed the smooth×ARD FD instability to the periodic
    /// `max(V'',0)` cos-basis majorizer clamp; #2339 smoothed that clamp and the
    /// two FD gates stayed red. This dumps, at the failing tests' frozen-θ̂ eval ρ,
    /// the per-row SPECTRAL-DEFLATION set (`cache.deflated_row_directions`) and the
    /// criterion value at ρ−h and ρ+h for every ρ coordinate, so the branch that
    /// flips across the ±h stencil is named directly. Pure diagnostic — asserts
    /// only that the caches build and the coord set is non-empty.
    #[test]
    fn kink_site_deflation_flip_diagnostic_2339() {
        use ndarray::{Array1, array};
        let (term, target, rho, _sc) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
        let base = rho.to_flat();
        let h = 1.0e-5;

        let mut coords: Vec<usize> = Vec::new();
        for a in 0..rho.log_lambda_smooth.len() {
            coords.push(rho.smooth_flat_index(a));
        }
        for kk in 0..rho.log_ard.len() {
            for axis in 0..rho.log_ard[kk].len() {
                let idx = rho.ard_flat_index(kk, axis);
                if !coords.contains(&idx) {
                    coords.push(idx);
                }
            }
        }
        if let Some(sparse) = rho.sparse_flat_index() {
            coords.push(sparse);
        }
        let label = |idx: usize| -> String {
            if rho.sparse_flat_index() == Some(idx) {
                "sparse".to_string()
            } else if (rho.smooth_flat_start()
                ..rho.smooth_flat_start() + rho.log_lambda_smooth.len())
                .contains(&idx)
            {
                format!("smooth{}", idx - rho.smooth_flat_start())
            } else {
                format!("ard@{idx}")
            }
        };

        // (value, per-row deflated-direction counts, total deflated, spectrally
        // deflated rows with their min raw eigenvalue).
        let dump = |flat: &Array1<f64>| -> (f64, Vec<usize>, usize, Vec<(usize, f64)>) {
            let r = rho.from_flat(flat.view()).unwrap();
            let mut t = term.clone();
            let (value, _loss, cache) = t
                .penalized_quasi_laplace_criterion_with_cache(
                    target.view(),
                    &r,
                    None,
                    0,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                )
                .expect("diagnostic cache");
            let per_row: Vec<usize> =
                cache.deflated_row_directions.iter().map(Vec::len).collect();
            let total: usize = per_row.iter().sum();
            let spectra: Vec<(usize, f64)> = cache
                .deflation_row_spectra
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    s.as_ref().map(|sp| {
                        (
                            i,
                            sp.raw_evals.iter().copied().fold(f64::INFINITY, f64::min),
                        )
                    })
                })
                .collect();
            (value, per_row, total, spectra)
        };

        for &j in &coords {
            let mut fm = base.clone();
            fm[j] -= h;
            let mut fp = base.clone();
            fp[j] += h;
            let (vm, prm, tm, spm) = dump(&fm);
            let (vp, prp, tp, spp) = dump(&fp);
            let flipped: Vec<usize> = (0..prm.len().min(prp.len()))
                .filter(|&i| prm[i] != prp[i])
                .collect();
            eprintln!(
                "KINKDIAG coord {}: value(-h)={vm:.9e} value(+h)={vp:.9e} d(value)={:.3e} | deflated_total {tm}->{tp} | rows_flipped={flipped:?}",
                label(j),
                vp - vm
            );
            if !spm.is_empty() || !spp.is_empty() {
                eprintln!("    spectrally-deflated rows (row,min_raw_eval): -h={spm:?} +h={spp:?}");
            }
        }
        assert!(!coords.is_empty(), "coord set must be non-empty");
    }

    /// PATH C (#2253) DIAGNOSTIC — localize the smooth↔ARD non-conservation of the
    /// production third-order gradient `g3[j] = −½⟨A⁺Γ_eff, g_ρ,j⟩`. `g3` is
    /// `∂Φ/∂ρ − ∂L/∂ρ` for a scalar `Φ`, so it MUST be conservative
    /// (`∂g3[ard]/∂ρ_smooth == ∂g3[smooth]/∂ρ_ard`); the full-set gate shows it is
    /// not. This splits `g3` by `Γ_eff = Γ_joint − Γ_tt + 2∇R` and prints each
    /// part's cross asymmetry so ONE run names the offending adjoint. Pure
    /// diagnostic — asserts only finiteness so it never masks the defect.
    #[test]
    fn third_order_conservation_bisection_2253() {
        use crate::manifold::arrow_solver::DeflatedArrowSolver;
        use ndarray::array;
        let (term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
        let base = rho.to_flat();
        let h = 1.0e-5;
        let smooth0 = rho.smooth_flat_index(0);
        let ard0 = rho.ard_flat_index(0, 0);

        // g3 restricted to ONE Γ_eff part (0 = Γ_joint, 1 = Γ_tt, 2 = 2∇R),
        // component `j`, at a REBUILT fixed-θ̂ cache at ρ + sign·h·e_perturb.
        let g3_part = |sign: f64, perturb: usize, j: usize, part: usize| -> f64 {
            let mut flat = base.clone();
            flat[perturb] += sign * h;
            let r = rho.from_flat(flat.view()).unwrap();
            let mut t = term.clone();
            let (_v, loss, cache) = t
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
            let gamma = match part {
                0 => t.logdet_theta_adjoint(&r, &cache, &solver).unwrap(),
                1 => t
                    .coordinate_block_logdet_theta_adjoint(&r, &cache, EvidenceOperator::Majorizer, None)
                    .unwrap(),
                _ => {
                    let rc = t
                        .production_rank_charge_derivative(target.view(), &r, &loss, &cache)
                        .unwrap();
                    crate::manifold::arrow_solver::SaeArrowVector {
                        t: &rc.theta.t * 2.0,
                        beta: &rc.theta.beta * 2.0,
                    }
                }
            };
            let a = t
                .solve_exact_stationarity(&r, target.view(), &cache, &gamma)
                .unwrap();
            let g_rho = t.outer_rho_gradient_ift_rhs(&r, j, &cache).unwrap();
            let dot: f64 = a.t.dot(&g_rho.t) + a.beta.dot(&g_rho.beta);
            -0.5 * dot
        };

        for (name, part) in [("Gamma_joint", 0usize), ("Gamma_tt", 1), ("2_grad_R", 2)] {
            let d_ard_by_smooth = (g3_part(1.0, smooth0, ard0, part)
                - g3_part(-1.0, smooth0, ard0, part))
                / (2.0 * h);
            let d_smooth_by_ard = (g3_part(1.0, ard0, smooth0, part)
                - g3_part(-1.0, ard0, smooth0, part))
                / (2.0 * h);
            eprintln!(
                "CH5 conservation bisect [{name}]: d g3[ard0]/d rho_smooth0={d_ard_by_smooth:.9e} \
                 d g3[smooth0]/d rho_ard0={d_smooth_by_ard:.9e} asym={:.3e}",
                (d_ard_by_smooth - d_smooth_by_ard).abs()
            );
            assert!(
                d_ard_by_smooth.is_finite() && d_smooth_by_ard.is_finite(),
                "conservation bisection [{name}] produced non-finite cross derivatives"
            );
        }
    }

    /// #2330 — IFT-residual arbiter. The θ-adjoint `Γ_joint` is exact (arbiter
    /// green), so the g3 non-conservation lives in `θ̂_ρ,j = −A⁺ g_ρ,j`. This
    /// tests the leading hypothesis: the #2080-d4 pencil deflation drops the
    /// near-null component of the DEFLATED t-block `g_ρ` (ARD) while the β-block
    /// `g_ρ` (smooth) is fully resolved — a built-in smooth↔ARD asymmetry.
    ///
    /// `x = A⁺ g_ρ,j`; since `A⁺` deflates, `A·x = P·g_ρ,j` and the residual
    /// `A·x − g_ρ,j = −(deflated component of g_ρ,j)`. A LARGE ARD residual with a
    /// ~0 smooth residual is the asymmetry root (`θ̂_ρ,ard` drops a response that
    /// `θ̂_ρ,smooth` keeps). Also confirms the solve uses the EXACT stationarity
    /// operator `A = H + ΔC` (`|A·x − g|` small) and not the cached majorizer `H`
    /// (`|H·x − g|` would then be the small one). Diagnostic: prints the norms.
    #[test]
    fn third_order_ift_deflation_residual_2330() {
        use crate::manifold::arrow_solver::{
            SaeArrowVector, apply_cached_arrow_hessian,
        };
        use ndarray::array;
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
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
            .expect("deflated fixed-state cache");
        assert!(
            cache
                .deflated_row_directions
                .iter()
                .any(|dirs| !dirs.is_empty()),
            "IFT residual arbiter requires per-row deflation to be present"
        );
        let norm = |v: &SaeArrowVector| (v.t.dot(&v.t) + v.beta.dot(&v.beta)).sqrt();
        let smooth0 = rho.smooth_flat_index(0);
        let ard0 = rho.ard_flat_index(0, 0);
        for (name, j) in [("smooth0", smooth0), ("ard0", ard0)] {
            let g_rho = term
                .outer_rho_gradient_ift_rhs(&rho, j, &cache)
                .expect("ift rhs");
            let x = term
                .solve_exact_stationarity(&rho, target.view(), &cache, &g_rho)
                .expect("A+ g_rho");
            let hx = apply_cached_arrow_hessian(&cache, x.t.view(), x.beta.view()).expect("H x");
            let dc = term
                .apply_exact_hessian_minus_b(&rho, target.view(), &cache, &x)
                .expect("dC x");
            // A·x = H·x + ΔC·x (the exact stationarity operator A = B + ΔC, B = H).
            let ax = SaeArrowVector {
                t: &hx.t + &dc.t,
                beta: &hx.beta + &dc.beta,
            };
            let r_exact = SaeArrowVector {
                t: &ax.t - &g_rho.t,
                beta: &ax.beta - &g_rho.beta,
            };
            let r_maj = SaeArrowVector {
                t: &hx.t - &g_rho.t,
                beta: &hx.beta - &g_rho.beta,
            };
            eprintln!(
                "IFT[{name}] |g_rho|={:.6e} |x|={:.6e} |A.x-g|={:.6e} |H.x-g|={:.6e}",
                norm(&g_rho),
                norm(&x),
                norm(&r_exact),
                norm(&r_maj)
            );
            assert!(
                norm(&r_exact).is_finite() && norm(&r_maj).is_finite(),
                "IFT residual arbiter [{name}] produced a non-finite residual"
            );
        }
    }

    /// #2330 — twist-vs-∂A split of the g3 cross non-conservation. The off-diagonal
    /// `H3[i,j] = −½⟨∂_iΓ_eff − (∂_iA)·a, b_j⟩` has two pieces; conservation needs
    /// EACH to be cross-symmetric between the (i,j) and (j,i) orientations. This
    /// prints both:
    /// * `twist[i,j] = ⟨∂_iΓ_eff, b_j⟩`, with `∂_iΓ_eff` a central difference of the
    ///   ASSEMBLED `Γ_eff` at ρ±h with the cache REBUILT per leg (so the deflation
    ///   is re-discovered — this is the deflated adjoint's actual ρ-dependence,
    ///   membership shifts included, which the frozen `−G Mᵢ G` twist may miss).
    /// * `dA[i,j] = ⟨(∂_iA)·a, b_j⟩`, using the exact `∂A/∂ρ` operators.
    ///
    /// The guilty piece should reproduce the round-6b transpose fingerprint
    /// (`piece[i,j] ≈ piece_fd[j,i]`), not just an asymmetric magnitude. Diagnostic
    /// (prints; asserts finiteness). Pairs (smooth0,ard0) and (smooth1,ard1).
    #[test]
    fn third_order_cross_symmetry_split_2330() {
        use crate::manifold::arrow_solver::{DeflatedArrowSolver, SaeArrowVector};
        use ndarray::{Array1, array};
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
        let base = rho.to_flat();
        let h = 1.0e-5;
        let (_value, loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                0,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("deflated base cache");
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let flatten = |v: &SaeArrowVector| -> Array1<f64> {
            let mut out = Array1::<f64>::zeros(dim);
            for (k, &x) in v.t.iter().enumerate() {
                out[k] = x;
            }
            for (k, &x) in v.beta.iter().enumerate() {
                out[total_t + k] = x;
            }
            out
        };
        let solver = DeflatedArrowSolver::plain(&cache);

        // Γ_eff = Γ_joint − Γ_tt + 2∇R, the gradient's effective adjoint.
        let mut gamma_eff = term
            .logdet_theta_adjoint(&rho, &cache, &solver)
            .expect("gamma_joint");
        {
            let gtt = term
                .coordinate_block_logdet_theta_adjoint(&rho, &cache, EvidenceOperator::Majorizer, None)
                .expect("gamma_tt");
            gamma_eff.t -= &gtt.t;
            gamma_eff.beta -= &gtt.beta;
            let rc = term
                .production_rank_charge_derivative(target.view(), &rho, &loss, &cache)
                .expect("rank charge");
            gamma_eff.t.scaled_add(2.0, &rc.theta.t);
            gamma_eff.beta.scaled_add(2.0, &rc.theta.beta);
        }
        let a = term
            .solve_exact_stationarity(&rho, target.view(), &cache, &gamma_eff)
            .expect("a = A+ Gamma");
        let a_flat = flatten(&a);
        let m_ops = term
            .penalty_curvature_operators_by_flat(&rho, &cache)
            .expect("dH/drho operators");
        let deltas = term
            .exact_stationarity_penalty_derivative_delta_by_flat(&rho, &cache)
            .expect("dC/drho deltas");

        let coords = [
            ("smooth0", rho.smooth_flat_index(0)),
            ("smooth1", rho.smooth_flat_index(1)),
            ("ard0", rho.ard_flat_index(0, 0)),
            ("ard1", rho.ard_flat_index(1, 0)),
        ];
        // Per-coordinate: b_j = A+ g_ρ,j ; (∂_iA)a ; ∂_iΓ_eff (FD, cache rebuilt).
        let mut b_flat = Vec::new();
        let mut da_a = Vec::new();
        let mut d_gamma = Vec::new();
        for &(_name, i) in &coords {
            let g_rho = term
                .outer_rho_gradient_ift_rhs(&rho, i, &cache)
                .expect("ift rhs");
            let b = term
                .solve_exact_stationarity(&rho, target.view(), &cache, &g_rho)
                .expect("b_j");
            b_flat.push(flatten(&b));
            let mut da = m_ops[&i].dot(&a_flat);
            if let Some(delta) = deltas.get(&i) {
                da += &delta.dot(&a_flat);
            }
            da_a.push(da);
            let gamma_at = |sign: f64| -> Array1<f64> {
                let mut flat = base.clone();
                flat[i] += sign * h;
                let r = rho.from_flat(flat.view()).unwrap();
                let mut t = term.clone();
                let (_v, loss, cache) = t
                    .penalized_quasi_laplace_criterion_with_cache(
                        target.view(),
                        &r,
                        None,
                        0,
                        0.4,
                        1.0e-6,
                        1.0e-6,
                    )
                    .expect("perturbed cache");
                let solver = DeflatedArrowSolver::plain(&cache);
                let mut g = t
                    .logdet_theta_adjoint(&r, &cache, &solver)
                    .expect("gamma_joint");
                let gtt = t
                    .coordinate_block_logdet_theta_adjoint(&r, &cache, EvidenceOperator::Majorizer, None)
                    .expect("gamma_tt");
                g.t -= &gtt.t;
                g.beta -= &gtt.beta;
                let rc = t
                    .production_rank_charge_derivative(target.view(), &r, &loss, &cache)
                    .expect("rank charge");
                g.t.scaled_add(2.0, &rc.theta.t);
                g.beta.scaled_add(2.0, &rc.theta.beta);
                flatten(&g)
            };
            d_gamma.push((gamma_at(1.0) - gamma_at(-1.0)) / (2.0 * h));
        }

        let twist = |i: usize, j: usize| -> f64 { d_gamma[i].dot(&b_flat[j]) };
        let d_a = |i: usize, j: usize| -> f64 { da_a[i].dot(&b_flat[j]) };
        for (ia, ib) in [(0usize, 2usize), (1usize, 3usize)] {
            eprintln!(
                "SPLIT pair ({},{}): twist[i,j]={:.6e} twist[j,i]={:.6e} (asym {:.3e}) | \
                 dA[i,j]={:.6e} dA[j,i]={:.6e} (asym {:.3e})",
                coords[ia].0,
                coords[ib].0,
                twist(ia, ib),
                twist(ib, ia),
                (twist(ia, ib) - twist(ib, ia)).abs(),
                d_a(ia, ib),
                d_a(ib, ia),
                (d_a(ia, ib) - d_a(ib, ia)).abs()
            );
        }
        assert!(
            twist(0, 2).is_finite() && d_a(0, 2).is_finite(),
            "cross-symmetry split produced non-finite terms"
        );
    }

    /// #2330 DIAGNOSTIC — why is the deflation-eigen twist term inert? Prints per
    /// row: deflation direction count, whether the DK map is SPECTRAL
    /// (`spectrum Some`) or gauge-only (`None`), and `∂H_tt/∂ρ_ard` (the row
    /// t-block max of each ARD `∂H/∂ρ` operator). If every deflated row has
    /// `∂H_tt/∂ρ_ard = 0`, the deflation does not move under ρ_ard (deflated rows
    /// are ARD-inactive, majorizer 0) and the eigen term is correctly zero — so
    /// the twist asymmetry has a DIFFERENT cause. If deflated rows are gauge-only
    /// (`spectrum None`), the eigen route returns 0 by construction.
    #[test]
    fn deflation_type_and_htt_probe_2330() {
        use ndarray::array;
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
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
            .expect("deflated cache");
        let m_ops = term
            .penalty_curvature_operators_by_flat(&rho, &cache)
            .expect("operators");
        let ard0 = rho.ard_flat_index(0, 0);
        let ard1 = rho.ard_flat_index(1, 0);
        let block_max = |op: &ndarray::Array2<f64>, base: usize, q: usize| -> f64 {
            let mut mx = 0.0_f64;
            for a in 0..q {
                for b in 0..q {
                    mx = mx.max(op[[base + a, base + b]].abs());
                }
            }
            mx
        };
        for row in 0..term.n_obs() {
            let dirs = cache
                .deflated_row_directions
                .get(row)
                .map(|d| d.len())
                .unwrap_or(0);
            let spec = cache
                .deflation_row_spectra
                .get(row)
                .and_then(|s| s.as_ref())
                .is_some();
            let base = cache.row_offsets[row];
            let q = cache.row_dims[row];
            let ht0 = block_max(&m_ops[&ard0], base, q);
            let ht1 = block_max(&m_ops[&ard1], base, q);
            eprintln!(
                "DEFL row={row} q={q} defl_dirs={dirs} spectrum_some={spec} \
                 dHtt_ard0_max={ht0:.3e} dHtt_ard1_max={ht1:.3e}"
            );
            // The whole reading of this probe is "is `∂H_tt/∂ρ_ard` zero or not",
            // so the reported block maxima must be well-posed magnitudes: a |·|-max
            // is finite and non-negative by construction, and a NaN would print as
            // a plausible-looking number while destroying the zero/non-zero verdict.
            assert!(
                ht0.is_finite() && ht0 >= 0.0 && ht1.is_finite() && ht1 >= 0.0,
                "row {row}: the ARD ∂H_tt/∂ρ block maxima are absolute-value maxima and must be \
                 finite and non-negative, got dHtt_ard0_max={ht0}, dHtt_ard1_max={ht1}"
            );
            // The row must actually have coordinates for a block max to mean
            // anything; a zero-width block would make every printed max a vacuous 0.
            assert!(
                q > 0 && base + q <= m_ops[&ard0].nrows(),
                "row {row}: the cache row block [{base}, {base}+{q}) must be a nonempty slice of \
                 the {}-dimensional ARD operator",
                m_ops[&ard0].nrows()
            );
        }
        assert!(
            term.n_obs() > 0,
            "the deflation probe must inspect at least one row, otherwise it reports nothing"
        );
    }

    /// #2330 — attribute the g3 cross non-conservation to the trace vs frozen-DK
    /// piece of the twist. Splits `dΓ_joint/dρ` into 4 legs (part-a/part-b ×
    /// trace-only/DK) and prints each leg's cross pair `⟨leg_smooth, b_ard⟩` vs
    /// `⟨leg_ard, b_smooth⟩` with the asymmetry. The asymmetric leg is the leak;
    /// the strong suspect is `part_a_dk` (frozen `deflation_block_correction` fed
    /// the twisted inverse `−G Mᵢ G`, which is not a valid selected inverse).
    #[test]
    fn twist_leg_cross_split_2330() {
        use ndarray::array;
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
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
            .expect("deflated cache");
        let smooth0 = rho.smooth_flat_index(0);
        let ard0 = rho.ard_flat_index(0, 0);
        let legs = term
            .ch5_twist_leg_cross(&rho, target.view(), &cache, smooth0, ard0)
            .expect("twist leg cross");
        let names = ["part_a_tr", "part_a_dk", "part_b_tr", "part_b_dk"];
        // The attribution only works if every named leg is present: a shortened
        // return silently drops the suspect leg (`part_a_dk`) from the report while
        // `zip` still prints a tidy-looking table of the survivors.
        assert_eq!(
            legs.len(),
            names.len(),
            "the twist split must return one cross pair per named leg {names:?}, got {}",
            legs.len()
        );
        for (name, (ij, ji)) in names.iter().zip(legs.iter()) {
            eprintln!(
                "LEG {name}: <leg_smooth,b_ard>={ij:.6e} <leg_ard,b_smooth>={ji:.6e} \
                 asym={:.3e}",
                (ij - ji).abs()
            );
            // The asymmetry printed per leg is the measurement; it is only readable
            // if both inner products are finite numbers.
            assert!(
                ij.is_finite() && ji.is_finite(),
                "leg {name}: both cross pairings must be finite, got \
                 <leg_smooth,b_ard>={ij}, <leg_ard,b_smooth>={ji}"
            );
        }
    }

    /// #2330 — the EXACT observed-information Laplace log-dets `(log|A|, log|A_tt|)`
    /// from the strict-Cholesky production path (`exact_observed_information_log_dets`)
    /// equal the independent dense eigendecomposition oracle `Σ ln λ_i(A)`, and `A`
    /// is certified positive definite (min eigenvalue > 0) at the converged mode.
    /// This pins the `log|A|` VALUE the dense capability route ranks against the
    /// exact observed information `A = ∇²_θθ L`, NOT the majorized surrogate `B`.
    #[test]
    fn exact_observed_information_log_det_matches_eigendecomposition_2330() {
        use ndarray::{Array1, Array2, array, s};
        // This module does not `use super::*`; the arbiter is the first test here
        // to build a `SaeArrowVector`, call `.eigh` (FaerEigh), and name `Side`.
        use super::{
            ArrowMetric, FaerEigh, SaeArrowVector, SaeCriterionError, Side,
            sae_exact_a_direction_floor,
        };
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = -0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
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
            .expect("fixed-theta cache");
        // Independent oracle: materialize A densely via the exact-Hessian apply,
        // then eigendecompose FIRST. The eigen spectrum decides which parity to
        // assert — because a majorizer-converged fixture mode need NOT be an
        // exact-A maximum. `B`-Newton stops where `B`'s gradient vanishes; in the
        // ARD negative-curvature region the exact `A = B + ΔC` (ΔC subtracts the
        // clamped `min(V'',0)` the majorizer drops) can be INDEFINITE there. That
        // indefinite point is exactly the #2330 mispricing made visible — not a
        // true max — so `½log|A|` is undefined and the typed refusal MUST fire.
        let total_t = cache.delta_t_len();
        let k = cache.k;
        let dim = total_t + k;
        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut unit = SaeArrowVector {
            t: Array1::<f64>::zeros(total_t),
            beta: Array1::<f64>::zeros(k),
        };
        for col in 0..dim {
            if col < total_t {
                unit.t[col] = 1.0;
            } else {
                unit.beta[col - total_t] = 1.0;
            }
            let av = term
                .apply_exact_hessian(&rho, target.view(), &cache, &unit)
                .expect("exact-Hessian apply");
            if col < total_t {
                unit.t[col] = 0.0;
            } else {
                unit.beta[col - total_t] = 0.0;
            }
            for r in 0..total_t {
                a[[r, col]] = av.t[r];
            }
            for r in 0..k {
                a[[total_t + r, col]] = av.beta[r];
            }
        }
        let sym = (&a + &a.t()) * 0.5;
        let (eigs, vecs) = sym.eigh(Side::Lower).expect("A eigendecomposition");
        let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let n_nonpos = eigs.iter().filter(|&&l| l <= 0.0).count();
        eprintln!(
            "A spectrum: min_eig={min_eig:.6e} max_eig={max_eig:.6e} n_nonpos={n_nonpos}/{dim}"
        );
        let result = term.exact_observed_information_log_dets(&rho, target.view(), &cache);
        // #2330 Phase-2: the value path classifies the spectrum three ways against
        // the SHARED floor — kept (λ>floor, contributes ln λ), null band
        // (|λ|≤floor, contributes 0), refused (λ<−floor). The arbiter mirrors
        // that classification exactly, so a future null-band-PD A is judged
        // correctly rather than binary PD-vs-refuse.
        //
        // #2673 — the band is PER DIRECTION now, because the metric it is
        // relative to is. This oracle keeps its own operands (its own dense `A`,
        // its own plain `eigh`, its own `B`-applies) and shares only the scalar
        // rule, so it still oracles the classification while a second copy of the
        // rule cannot drift from production's.
        let spectral_norm = eigs.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        let joint_metric = ArrowMetric::Joint(&cache);
        let floors: Vec<f64> = (0..dim)
            .map(|index| {
                let vbv = joint_metric
                    .quadratic_form(vecs.column(index))
                    .expect("B quadratic form on the joint block");
                sae_exact_a_direction_floor(dim, spectral_norm, vbv)
            })
            .collect();
        let worst_floor = floors.iter().copied().fold(0.0_f64, f64::max);
        if min_eig >= -worst_floor
            && eigs
                .iter()
                .enumerate()
                .all(|(index, &lambda)| lambda >= -floors[index])
        {
            // PD on the gauge quotient (min_eig may be a gauge null in [−floor, floor]).
            let (log_a, log_a_tt) =
                result.expect("A is PD on the quotient so the log-dets must be Ok");
            let kept: f64 = eigs
                .iter()
                .enumerate()
                .filter(|&(index, l)| *l > floors[index])
                .map(|(_, l)| l.ln())
                .sum();
            assert!(
                (log_a - kept).abs() <= 1.0e-9 * (1.0 + kept.abs()),
                "log|A| kept-eigenvalue sum {log_a} != oracle {kept}"
            );
            let a_tt = sym.slice(s![..total_t, ..total_t]).to_owned();
            let (eigs_tt, vecs_tt) = a_tt.eigh(Side::Lower).expect("A_tt eigendecomposition");
            let tt_norm = eigs_tt.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
            let tt_metric = ArrowMetric::Coordinate(&cache);
            let kept_tt: f64 = eigs_tt
                .iter()
                .enumerate()
                .filter(|&(index, l)| {
                    let vbv = tt_metric
                        .quadratic_form(vecs_tt.column(index))
                        .expect("B quadratic form on the coordinate block");
                    *l > sae_exact_a_direction_floor(total_t, tt_norm, vbv)
                })
                .map(|(_, l)| l.ln())
                .sum();
            assert!(
                (log_a_tt - kept_tt).abs() <= 1.0e-9 * (1.0 + kept_tt.abs()),
                "log|A_tt| kept-eigenvalue sum {log_a_tt} != oracle {kept_tt}"
            );
        } else {
            // A is non-PD. Under #2336 value-side E-attributability the classification
            // is three-way: an indefinite direction attributable to the bounded ARD
            // concave-clamp (λ+e_v ≥ −floor) is PRICED at its basin curvature, only a
            // genuinely indefinite one (λ+e_v < −floor) REFUSES. The earlier version of
            // this branch asserted unconditional refusal and said so in a comment: the
            // fixture was PD on the gauge quotient then, so the branch was unreached and
            // the assertion was left as a placeholder to be "split by attributability"
            // if a future fixture ever landed here. #2267's inner-solve step fix moves
            // this fixture's converged state, so it lands here now — and the split is
            // written out rather than assumed.
            //
            // The oracle reads the SAME clamp diagonal the value path reads
            // (`materialize_ard_concave_clamp_diagonal`) and applies the same
            // predicate, so the two cannot drift; the assertion is STRICTER than the
            // placeholder, because it pins WHICH way each negative direction is
            // classified and what the priced log-det then has to equal.
            let e_diag = term
                .materialize_ard_concave_clamp_diagonal(&rho, &cache)
                .expect("ARD concave-clamp diagonal");
            let mut all_attributable = true;
            let mut priced_log_a = 0.0_f64;
            for (idx, &lambda) in eigs.iter().enumerate() {
                let floor = floors[idx];
                let priced = if lambda < -floor {
                    let v = vecs.column(idx);
                    let mut e_v = 0.0_f64;
                    for j in 0..total_t {
                        e_v += e_diag[j] * v[j] * v[j];
                    }
                    let basin = lambda + e_v;
                    if basin < -floor {
                        all_attributable = false;
                    }
                    basin
                } else {
                    lambda
                };
                if priced > floor {
                    priced_log_a += priced.ln();
                }
            }
            eprintln!(
                "A non-PD: min_eig={min_eig:.6e} all_attributable={all_attributable} \
                 priced_log|A|={priced_log_a:.9e}"
            );
            if all_attributable {
                let (log_a, _log_a_tt) = result.expect(
                    "every sub-floor negative direction is ARD-clamp attributable, so the \
                     value path must PRICE the basin curvature instead of refusing",
                );
                assert!(
                    (log_a - priced_log_a).abs() <= 1.0e-9 * (1.0 + priced_log_a.abs()),
                    "priced log|A| {log_a} != attributability oracle {priced_log_a}"
                );
            } else {
                match result {
                    Err(SaeCriterionError::IndefiniteObservedInformation { block }) => {
                        assert_eq!(block, "joint", "refusal fired on the wrong block: {block}");
                    }
                    other => panic!(
                        "A has a genuinely indefinite direction (min_eig={min_eig:.3e}, not \
                         clamp-attributable) but exact_observed_information_log_dets did not \
                         refuse: {other:?}"
                    ),
                }
            }
        }
    }

    /// #2330/#2336 PRICING arbiter (reachable companion to the PD parity test
    /// above). MEASURED correction to the earlier premise: the a_saddle specimen's
    /// two exact-A negatives are FULLY attributable to the bounded ARD periodic
    /// concave-clamp wrinkle (`λ+e_v(ARD)=+0.026`, verified in
    /// `zz_measure_e_attributability_2336`) — NOT a residual-rooted genuine saddle.
    /// So under the value-side E-attributability semantics (#2336) the criterion
    /// PRICES it at the basin curvature and returns a FINITE value. This is the
    /// price half of the (price ⟺ E-attributable, refuse ⟺ genuine) contract; the
    /// refuse half is `genuine_saddle_is_infeasible_probe_not_fatal_2336` on the
    /// obb window-scan specimen.
    #[test]
    fn exact_observed_information_prices_e_attributable_a_saddle_2336() {
        let (mut term, target, rho) =
            super::exact_hessian_fixture_tests::converged_state_with_residual_a_saddle_2336();
        let result = term.penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        );
        assert!(
            matches!(&result, Ok((value, _, _)) if value.is_finite()),
            "the E-attributable a_saddle specimen must PRICE FINITE under #2336, not refuse; got: {:?}",
            result.as_ref().map(|(value, _, _)| *value).map_err(|e| format!("{e:?}"))
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

}

/// #1418: the implicit-function (IFT) back-substitution must invert the EXACT
/// stationarity Jacobian `A = ∇²_θθ L`, not the assembled surrogate `B`.
#[cfg(test)]
mod exact_stationarity_solve_1418_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    /// `‖A x − rhs‖` for the exact stationarity Jacobian `A` (the matrix-free
    /// `B v + ΔC v` apply).
    ///
    /// #2674 — the FULL ambient residual. This used to project the residual onto
    /// the complement of the analytic chart-gauge orbit, matching a solve that
    /// deleted that orbit before inverting; both the projection and the deletion
    /// are gone, so the residual the caller asserts on is now the whole thing.
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

    /// A synthetic spectral block whose per-direction band is one chosen
    /// constant (#2673).
    ///
    /// Production classifies direction `i` at
    /// `max(dim·ε·‖A‖₂, √ε·vᵢᵀBvᵢ)`, so a UNIFORM metric
    /// `vᵢᵀBvᵢ = floor/√ε` reproduces the scalar band the fixtures below were
    /// written against. That is asserted here rather than assumed: a block whose
    /// realised floor is not the requested one would silently re-tune every
    /// fixture that uses this helper.
    fn spectral_block_with_uniform_floor(
        operator: Array2<f64>,
        eigenvalues: Array1<f64>,
        eigenvectors: Array2<f64>,
        floor: f64,
    ) -> ExactHessianSpectralBlock {
        let dimension = eigenvalues.len();
        let spectral_norm = eigenvalues
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);
        let block = ExactHessianSpectralBlock {
            operator,
            eigenvalues,
            eigenvectors,
            metric_scale: Array1::from_elem(
                dimension,
                floor / super::sae_exact_a_identifiability_floor(),
            ),
            spectral_norm,
        };
        for index in 0..dimension {
            let realised = block.rank_floor(index);
            assert!(
                (realised - floor).abs() <= 1.0e-12 * floor,
                "#2673: the synthetic block must realise the requested band \
                 (direction {index}: asked {floor:.6e}, got {realised:.6e}); the arithmetic \
                 floor dim·ε·‖A‖₂ = {:.6e} may be binding instead",
                (dimension as f64) * f64::EPSILON * spectral_norm
            );
        }
        block
    }

    fn ambient_residual_merit(residual: &SaeArrowVector) -> f64 {
        0.5 * (residual.t.dot(&residual.t) + residual.beta.dot(&residual.beta))
    }

    /// #2653: the dense owner is a signed Moore--Penrose solve, not an SPD
    /// inverse and not a projected-residual proxy. A resolved negative mode is
    /// retained, the MEASURED spectral null band is removed (#2674 — nothing is
    /// removed by declaration), and the physical operator certifies the
    /// resulting response.
    #[test]
    fn dense_exact_stationarity_pseudoinverse_keeps_signed_range_and_drops_null_2653() {
        let eigenvalues = Array1::from_vec(vec![4.0_f64, 1.0e-12, -2.0]);
        let geometry = spectral_block_with_uniform_floor(
                Array2::from_diag(&eigenvalues),
                eigenvalues,
                Array2::from_diag(&Array1::ones(3)),
                1.0e-9,
            );
        let rhs = SaeArrowVector {
            t: Array1::from_vec(vec![8.0, 3.0]),
            beta: Array1::from_vec(vec![6.0]),
        };
        let solved = geometry
            .solve_stationarity(&rhs)
            .expect("rank-revealing dense exact-stationarity solve");
        assert_abs_diff_eq!(solved.t[0], 2.0, epsilon = 1.0e-14);
        assert_abs_diff_eq!(solved.t[1], 0.0, epsilon = 1.0e-14);
        assert_abs_diff_eq!(solved.beta[0], -3.0, epsilon = 1.0e-14);
    }

    /// #2762 — `ν = 0` on the damped path IS the pseudoinverse step, including
    /// the null-band classification.
    ///
    /// This is what lets the polish keep its quadratic tail while gaining a
    /// trust region: its first trial at every step is the step it has always
    /// taken, and a state that never needed damping never pays for one. Same
    /// block, same rhs, same expected answer as
    /// `dense_exact_stationarity_pseudoinverse_keeps_signed_range_and_drops_null_2653`
    /// directly above — the two must not be allowed to drift apart.
    #[test]
    fn damped_residual_step_at_zero_damping_is_the_pseudoinverse_step_2762() {
        let eigenvalues = Array1::from_vec(vec![4.0_f64, 1.0e-12, -2.0]);
        let geometry = spectral_block_with_uniform_floor(
                Array2::from_diag(&eigenvalues),
                eigenvalues,
                Array2::from_diag(&Array1::ones(3)),
                1.0e-9,
            );
        // The damped path is stated in the RESIDUAL `g`; the pseudoinverse
        // route is stated in `rhs = −g`.
        let residual = SaeArrowVector {
            t: Array1::from_vec(vec![-8.0, -3.0]),
            beta: Array1::from_vec(vec![-6.0]),
        };
        let rhs = SaeArrowVector {
            t: Array1::from_vec(vec![8.0, 3.0]),
            beta: Array1::from_vec(vec![6.0]),
        };
        let pseudoinverse = geometry
            .solve_stationarity(&rhs)
            .expect("rank-revealing dense exact-stationarity solve");
        let damped = geometry
            .damped_residual_step(&residual, 0.0)
            .expect("zero damping is the pseudoinverse point of the path");
        assert_abs_diff_eq!(damped.step.t[0], pseudoinverse.t[0], epsilon = 1.0e-15);
        assert_abs_diff_eq!(damped.step.t[1], pseudoinverse.t[1], epsilon = 1.0e-15);
        assert_abs_diff_eq!(damped.step.beta[0], pseudoinverse.beta[0], epsilon = 1.0e-15);
        // The `1e-12` direction is inside the null band, so its whole
        // coefficient survives into the model residual and nothing else does:
        // `½·3² = 4.5`.
        let ambient_model_merit = ambient_residual_merit(&damped.model_residual);
        assert_abs_diff_eq!(ambient_model_merit, 4.5, epsilon = 1.0e-14);
        assert_eq!(damped.retained_rank, 2);
    }

    /// #2762 — the modeled residual the damped path reports is the EXACT linear
    /// residual `g + AΔ(ν)`, on a non-diagonal operator, at every damping.
    ///
    /// The polish's acceptance test measures an achieved reduction against this
    /// number. If it were an approximation, the trust ratio would be measuring
    /// the approximation rather than the state, so this is asserted against an
    /// independent dense `A·Δ` — the operator the block carries — rather than
    /// against the spectral algebra that produced it.
    #[test]
    fn damped_residual_step_model_residual_is_exact_2762() {
        // A symmetric operator with a genuinely rotated eigenbasis, so the test
        // cannot pass by coincidence of a diagonal layout.
        let dim = 4usize;
        let mut basis = Array2::<f64>::zeros((dim, dim));
        let v = Array1::from_vec(vec![0.5_f64, -0.5, 0.5, -0.5]);
        for row in 0..dim {
            for column in 0..dim {
                basis[[row, column]] =
                    if row == column { 1.0 } else { 0.0 } - 2.0 * v[row] * v[column];
            }
        }
        let eigenvalues = Array1::from_vec(vec![3.0_f64, -0.75, 1.0e-5, 0.25]);
        let operator = basis.dot(&Array2::from_diag(&eigenvalues)).dot(&basis.t());
        let geometry = spectral_block_with_uniform_floor(
                operator.clone(),
                eigenvalues,
                basis,
                1.0e-12,
            );
        let residual = SaeArrowVector {
            t: Array1::from_vec(vec![0.7_f64, -1.3, 0.2]),
            beta: Array1::from_vec(vec![0.9]),
        };
        let mut flat_residual = Array1::<f64>::zeros(dim);
        flat_residual
            .slice_mut(s![..3])
            .assign(&residual.t);
        flat_residual[3] = residual.beta[0];
        let mut previous_reduction = f64::INFINITY;
        for nu in [0.0_f64, 1.0e-10, 1.0e-6, 1.0e-2, 1.0, 1.0e3] {
            let damped = geometry
                .damped_residual_step(&residual, nu)
                .expect("damped step on a well-posed block");
            let mut flat_step = Array1::<f64>::zeros(dim);
            flat_step.slice_mut(s![..3]).assign(&damped.step.t);
            flat_step[3] = damped.step.beta[0];
            let linear_residual = &flat_residual + &operator.dot(&flat_step);
            let independent = 0.5 * linear_residual.dot(&linear_residual);
            let ambient_model_merit = ambient_residual_merit(&damped.model_residual);
            assert_abs_diff_eq!(
                ambient_model_merit,
                independent,
                epsilon = 1.0e-12 * independent.max(1.0)
            );
            // The ladder's termination proof: the model's predicted reduction is
            // monotonically decreasing in the damping, so a rung that fails the
            // round-off floor proves every later rung fails it too.
            let reduction = 0.5 * flat_residual.dot(&flat_residual) - ambient_model_merit;
            assert!(
                reduction <= previous_reduction + 1.0e-15,
                "predicted reduction rose from {previous_reduction:.6e} to {reduction:.6e} at \
                 ν={nu:.6e}"
            );
            previous_reduction = reduction;
        }
    }

    /// #2762 — the property the whole fix rests on: damping separates a
    /// near-null direction from a resolved one, and a scalar step length cannot.
    ///
    /// `A = diag(1, 1e-6)` with `g = (1, 1)`. The undamped step is `(-1, -1e6)`:
    /// its LENGTH is entirely the flat direction, and any `α` small enough to
    /// keep that component inside a local model shrinks the resolved component
    /// by the same factor — which is exactly the measured `#2015` witness, where
    /// `‖Δ‖ = 0.44` at `‖g‖ = 1.2e-4` and Armijo on `½‖g‖²` first passed at
    /// `α = 4.9e-4` for a 0.03% reduction. At `ν = λ_flat·λ_resolved` the damped
    /// step is `O(1)` in BOTH coordinates, kills the resolved direction's
    /// residual entirely, and leaves the flat one — which is what a step is
    /// allowed to do.
    #[test]
    fn damping_separates_a_flat_direction_from_a_resolved_one_2762() {
        let eigenvalues = Array1::from_vec(vec![1.0_f64, 1.0e-6]);
        let geometry = spectral_block_with_uniform_floor(
                Array2::from_diag(&eigenvalues),
                eigenvalues,
                Array2::from_diag(&Array1::ones(2)),
                1.0e-14,
            );
        let residual = SaeArrowVector {
            t: Array1::from_vec(vec![1.0_f64]),
            beta: Array1::from_vec(vec![1.0]),
        };
        let undamped = geometry
            .damped_residual_step(&residual, 0.0)
            .expect("undamped step");
        let damped = geometry
            .damped_residual_step(&residual, 1.0e-6)
            .expect("damped step");
        assert!(
            undamped.step_norm_sq.sqrt() > 1.0e5 * damped.step_norm_sq.sqrt(),
            "the undamped step must be dominated by the flat direction: ‖Δ(0)‖={:.6e} vs \
             ‖Δ(ν)‖={:.6e}",
            undamped.step_norm_sq.sqrt(),
            damped.step_norm_sq.sqrt(),
        );
        assert!(
            damped.step.t[0].abs() < 2.0 && damped.step.beta[0].abs() < 2.0,
            "the damped step must be O(1) in both coordinates, got ({:.6e}, {:.6e})",
            damped.step.t[0],
            damped.step.beta[0],
        );
        // Model residual per direction: `c_i ν/(λ_i² + ν)`. The resolved
        // direction is solved to `1e-6`; the flat one keeps essentially its
        // whole coefficient. So `½‖g‖² = 1` falls to `½`, and it falls entirely
        // in the direction that could move.
        let resolved_leftover = 1.0e-6 / (1.0 + 1.0e-6);
        let flat_leftover = 1.0e-6 / (1.0e-12 + 1.0e-6);
        let ambient_model_merit = ambient_residual_merit(&damped.model_residual);
        assert_abs_diff_eq!(
            ambient_model_merit,
            0.5 * (resolved_leftover * resolved_leftover + flat_leftover * flat_leftover),
            epsilon = 1.0e-14
        );
        assert!(resolved_leftover < 1.0e-5 && flat_leftover > 0.999);
    }

    /// #2762 — the polish may not leave the state with a LARGER KKT residual
    /// than it found. Ever, at any budget.
    ///
    /// This is the property the shipped acceptance test could not enforce and
    /// measurably violated: on both #2762 witnesses EVERY step was accepted
    /// while the raw KKT gradient rose 15x and 107x, because acceptance was
    /// carried by a comparison between the trial state's decrement in the
    /// MAJORIZER metric and the pre-state's decrement in the EXACT-Hessian
    /// metric. This drives the phase directly with a tolerance no state can
    /// meet, so it must step rather than return at its own gate, and asserts the
    /// merit it now descends — `½‖g‖²` — is monotone across the whole budget.
    ///
    /// The end-to-end witnesses (`planted_1e4_column_spread…`,
    /// `reactive_entry_reseeds…`) pin the CONVERGENCE this buys; this pins the
    /// safety property, which holds on states where nothing converges at all.
    #[test]
    fn terminal_polish_never_raises_the_kkt_residual_2762() {
        let (mut term, target, rho, _cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let lambda_smooth = rho.lambda_smooth_vec().expect("smoothness strengths");
        let options = ArrowSolveOptions::direct()
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let residual_norm = |term: &mut SaeManifoldTerm| -> f64 {
            let system = term
                .assemble_arrow_schur(target.view(), &rho, None)
                .expect("arrow-Schur assembly at the polish entry state");
            SaeManifoldTerm::system_grad_norm_sq(&system).sqrt()
        };
        let objective_scale = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("penalized objective")
            .abs()
            + 1.0;
        let before = residual_norm(&mut term);
        let mut best_seen = None;
        // Tolerance `0`: `quasi_laplace_kkt_stationary` cannot fire, so the
        // phase runs its budget instead of handing straight back.
        let moved = term
            .terminal_exact_newton_polish(
                target.view(),
                &rho,
                None,
                &lambda_smooth,
                0.0,
                objective_scale,
                &options,
                8,
                &mut best_seen,
            )
            .expect("the polish degrades every internal failure to Ok(false)");
        let after = residual_norm(&mut term);
        assert!(
            after <= before,
            "the polish raised the KKT residual it is judged on: {before:.6e} -> {after:.6e} \
             (moved={moved})"
        );
        // Non-vacuity: an unreachable tolerance on a state with a live residual
        // must make this phase actually step, or the assertion above is testing
        // an early return.
        assert!(
            moved && after < before,
            "the phase must commit at least one step at tolerance 0 on a state with a live \
             residual: {before:.6e} -> {after:.6e} (moved={moved})"
        );
    }

    /// #2762 — `retained_curvature_extremes` is the DERIVED span of the damping
    /// ladder, and it reads the retained band only.
    #[test]
    fn retained_curvature_extremes_span_the_resolved_band_only_2762() {
        let eigenvalues = Array1::from_vec(vec![-7.0_f64, 1.0e-12, 0.5, 2.0]);
        let geometry = spectral_block_with_uniform_floor(
                Array2::from_diag(&eigenvalues),
                eigenvalues,
                Array2::from_diag(&Array1::ones(4)),
                1.0e-9,
            );
        let (smallest, largest) = geometry
            .retained_curvature_extremes()
            .expect("three directions clear the null band");
        assert_abs_diff_eq!(smallest, 0.5, epsilon = 0.0);
        assert_abs_diff_eq!(largest, 7.0, epsilon = 0.0);

        // A block that is entirely inside its own null band has no ladder, and
        // must say so rather than hand back a degenerate span.
        let null_eigenvalues = Array1::from_vec(vec![1.0e-12_f64, -2.0e-12]);
        let null_geometry = spectral_block_with_uniform_floor(
                Array2::from_diag(&null_eigenvalues),
                null_eigenvalues,
                Array2::from_diag(&Array1::ones(2)),
                1.0e-9,
            );
        assert!(null_geometry.retained_curvature_extremes().is_none());
    }

    /// `solve_exact_stationarity` returns the EXACT solve of `A x = rhs` (small
    /// `A`-residual), AND the surrogate solve `x_B = B⁻¹ rhs` leaves a LARGE
    /// `A`-residual — so the certificate is non-vacuous (`A ≠ B`) and the IFT
    /// step genuinely inverts `A`. The surrogate solve `x_B = B⁻¹ rhs` leaves
    /// the large `A`-residual asserted below, so this test passes only when the
    /// implicit solve targets the exact stationarity Jacobian.
    #[test]
    fn solve_exact_stationarity_inverts_a_not_b_1418() {
        let (term, target, rho, cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
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
            .solve_exact_stationarity(&rho, target.view(), &cache, &rhs)
            .expect("exact stationarity solve");
        let exact_resid = a_residual_norm(&term, &rho, target.view(), &cache, &x, &rhs);

        // Surrogate solve x_B = B⁻¹ rhs (the pre-#1418 implicit step).
        let x_b = solver
            .solve(rhs.t.view(), rhs.beta.view())
            .expect("B inverse");
        let surrogate_resid = a_residual_norm(&term, &rho, target.view(), &cache, &x_b, &rhs);

        // 1) The exact solve drives the FULL ambient residual `Ax-rhs` to ~0.
        //    #2674 — this used to be asserted on the chart-gauge quotient of the
        //    residual, because the solve deleted that orbit and could not reduce
        //    an arbitrary RHS's component along it. It no longer deletes it, so
        //    the whole residual is now in scope and this bar is strictly harder.
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
            (term.beta_offsets(), Box::new(move |_: usize| p))
        }
    }

    #[test]
    fn hutchinson_smoothness_dof_matches_exact_and_is_deterministic() {
        // #2253: small_two_atom_periodic_term is p=1 output, where two decoders
        // are trivially collinear (the scalar output-Gram makes the barrier
        // coherence O identically 1) and K=2 is non-identifiable, so the fit
        // co-collapses. Rank the p=3 REACHABLE gamma_fd_tiny two-atom fixture into
        // its converging PD basin instead; the per-atom smoothness-DOF split this
        // test pins (Hutchinson vs exact column-solve, cross-atom coupling) is
        // exercised identically on two identifiable atoms.
        let (mut term, target, mut rho) =
            crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.0;
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for v in axis.iter_mut() {
                *v = -1.0;
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
        // #2253: the historical OBB-gate-on-a-softmax-built-target left a
        // residual but is model-UNREACHABLE, so after #2330 Phase-2a the joint
        // fit co-collapses (reconstruction EV=-4.06, decoders cannot anchor K=2).
        // Keep the REACHABLE softmax gamma_fd_tiny (p=3) but rank it into a
        // moderate-penalty basin (all log-lambda = -1 except a mild sparse), where
        // the regularized fit carries a genuine residual (positive dispersion,
        // non-degenerate per-output-channel bands) and converges. Both the direct
        // and the recompute paths fit this COLD state identically, so the
        // band-reproduction invariance this test pins holds exactly; the mode
        // switch was only a residual-creation trick, not an OBB-recompute assertion.
        let (mut term, target, mut rho) =
            crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.0;
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = -1.0;
        }
        for axis in rho.log_ard.iter_mut() {
            for v in axis.iter_mut() {
                *v = -1.0;
            }
        }

        // Reference joint bands via the direct Schur path.
        let (_c, loss, cache) = term
            .penalized_quasi_laplace_criterion_with_cache(
                target.view(),
                &rho,
                None,
                40,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("converged joint cache");
        // #2253: price the reference band from the SAME dispersion source the
        // production recompute_joint_shape_uncertainty uses -- the explicit
        // whitened reconstruction residual. Passing None reads 2*data_fit (the
        // deviance RSS), which diverges from the whitened residual RSS under a
        // whitening row-metric, so the reference (not recompute) was stale.
        let residual = term
            .reconstruction_residual(target.view(), &rho)
            .expect("reconstruction residual");
        let dispersion = term
            .reconstruction_dispersion(&loss, &cache, &rho, Some(residual.view()))
            .expect("dispersion");
        assert!(dispersion > 0.0, "a real residual ⇒ positive dispersion");
        let joint = term
            .assemble_shape_uncertainty(&cache, dispersion)
            .expect("direct joint bands");

        // Property 1: the final-state recompute reproduces the joint path (it IS
        // the joint path, rebuilt from the term + ρ rather than a cached factor).
        let recomputed = term
            .recompute_joint_shape_uncertainty(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
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
