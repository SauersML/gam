use crate::manifold::tests::gamma_fd_tiny_fixture;

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

        let (mut term, mut target, mut rho) = gamma_fd_tiny_fixture();
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
            .expect("off-manifold fixture must converge with both atoms alive");
        (term, target, rho, cache)
    }
}

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
                term.atoms[atom_idx].latent_dim(),
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
        use ndarray::{Array1, array};
        // Construct θ̂ through the shared, independently exercised PD-basin
        // authority.  This is deliberately distinct from the lifted evaluation ρ
        // below: fitting at that point drives this tiny decoder to co-collapse.
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();

        // Evaluation ρ: lift ρ_smooth / ρ_ard off the −6 floor so the Daleckii–Krein
        // CROSS terms — which scale as O(λ²) and O(α²) — sit well above FD noise. At
        // the floor (λ = α ≈ 2.5e-3) the cross term is ~6e-6, under the FD tolerance,
        // and the gate would pass on the δ self-term alone while the D-K machinery
        // went entirely unchecked. The fixed-stratum Hessian is exact at ANY frozen
        // θ̂ — it does not require θ̂ to be stationary for this ρ — and a ZERO inner
        // budget assembles H(ρ) = H_data(θ̂) + penalty(ρ) without re-running the fit,
        // so no co-collapse guard is tripped. Extra penalty only makes H more PD.
        // A ρ perturbation scales only α, never the frozen circle coordinate t, so
        // the max(·,0) majorizer active set is invariant — no subgradient ambiguity.
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
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
            if let Some(si) = r.sparse_flat_index() {
                let joint = t
                    .assignment_log_strength_hessian_trace(&r, &cache, &solver)
                    .expect("sparse joint logdet trace");
                let coord = t
                    .coordinate_block_assignment_log_strength_hessian_trace(&r, &cache)
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

    /// PATH C (#2253) CH5 — the third-order forward-sensitivity Hessian block
    /// `H3[i,j] = ∂g3[j]/∂ρ_i`, `g3 = −½Γ_effᵀθ̂_ρ` (the gradient's
    /// `third_order_correction`), must equal a central finite difference of that
    /// EXACT analytic gradient component at a frozen inner state. The cache is
    /// rebuilt at each ρ±h so the penalty part of `H` moves with ρ while θ̂ stays
    /// frozen (a frozen cache false-greens the twist). The lifted ρ keeps the
    /// periodic ARD majorizer well inside its positive branch (`|cos κt| ≳ 0.1`),
    /// where the `max(·,0)` active set is invariant and the FD is valid.
    #[test]
    fn third_order_forward_sensitivity_hessian_matches_finite_difference_2253() {
        use crate::manifold::arrow_solver::DeflatedArrowSolver;
        use ndarray::{Array1, array};
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
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
            .expect("fixed-theta base cache");
        let analytic = term
            .third_order_forward_sensitivity_hessian(target.view(), &rho, &loss, &cache)
            .expect("CH5 third-order block assembles");

        // Self-check the dense θ-adjoint reproduction against the trusted
        // production builder BEFORE the FD comparison: if this reds, the dense
        // `dh` + Daleckii–Krein reconstruction is wrong (not the twist / rank-charge
        // assembly), which localizes the bug in one read.
        let (dj, dt) = term
            .ch5_dense_theta_adjoint_selfcheck(&rho, &cache)
            .expect("dense θ-adjoint self-check runs");
        eprintln!("CH5 dense θ-adjoint self-check: max|dense−prod| joint={dj:.3e} tt={dt:.3e}");
        assert!(
            dj < 1.0e-7 && dt < 1.0e-7,
            "CH5 dense θ-adjoint reconstruction diverges from production \
             (joint={dj:.3e}, tt={dt:.3e}): the dense dh/DK reproduction is wrong — \
             fix the dense builder before the twist"
        );

        let base = rho.to_flat();
        let h = 1.0e-5;
        // `g3[j]` in isolation at a REBUILT fixed-θ̂ cache (the frozen-cache trap).
        let g3_at = |sign: f64, coord: usize| -> Array1<f64> {
            let mut flat = base.clone();
            flat[coord] += sign * h;
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
            t.analytic_outer_rho_gradient_components(target.view(), &r, &loss, &cache, &solver)
                .expect("analytic gradient components")
                .third_order_correction
        };

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

        // Non-vacuity: the block must carry real curvature (else the gate would
        // pass on an all-zero block), and its OFF-diagonal (the twist / mixed
        // cross terms with no diagonal self-term) must be materially exercised.
        let max_abs = coords
            .iter()
            .flat_map(|&i| coords.iter().map(move |&j| (i, j)))
            .map(|(i, j)| analytic[[i, j]].abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs > 1.0e-6,
            "CH5 third-order block must carry real curvature: max|H3|={max_abs}"
        );

        // Channel label for a flat coordinate, so a failing grid reads as a
        // sub-channel pattern (a full row i = a `dΓ/dρ_i` bug; a full column j =
        // a `g_ρ,j` / `b_j` bug; scattered = FD kink noise).
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
        // Kink diagnostic: the periodic ARD majorizer's active set flips at
        // cos(κt) = 0 (period 1 ⇒ κ = 2π), where a row deflates/undeflates across
        // ρ±h and the ARD-coordinate FD is invalid. Print any near-kink row so
        // scattered ARD-row failures can be attributed.
        for (atom, coord) in term.assignment.coords.iter().enumerate() {
            for row in 0..coord.n_obs() {
                let t = coord.row(row)[0];
                let c = (std::f64::consts::TAU * t).cos();
                if c.abs() < 0.2 {
                    eprintln!(
                        "KINK atom{atom} row{row}: t={t:.4} cos(2πt)={c:.4} (ARD FD invalid here)"
                    );
                }
            }
        }

        let tol = |a: f64| 1.0e-5 + 1.0e-4 * a.abs();
        let mut worst: Option<(usize, usize, f64, f64, f64)> = None;
        let mut n_fail = 0usize;
        for &i in &coords {
            // `H3[i,j] = ∂g3[j]/∂ρ_i`: perturb ρ_i, read the j-th component.
            let fd_row = (g3_at(1.0, i) - g3_at(-1.0, i)) / (2.0 * h);
            for &j in &coords {
                let a_ij = analytic[[i, j]];
                let fd_ij = fd_row[j];
                let abs_err = (a_ij - fd_ij).abs();
                let rel = if a_ij.abs() > 0.0 {
                    abs_err / a_ij.abs()
                } else {
                    abs_err
                };
                let fail = abs_err >= tol(a_ij);
                eprintln!(
                    "CH5 H3[{i}={},{j}={}] analytic={a_ij:.9e} fd={fd_ij:.9e} abs={abs_err:.3e} rel={rel:.3e}{}",
                    label(i),
                    label(j),
                    if fail { "  <== FAIL" } else { "" }
                );
                if fail {
                    n_fail += 1;
                    if worst.is_none_or(|(_, _, _, _, wr)| rel > wr) {
                        worst = Some((i, j, a_ij, fd_ij, rel));
                    }
                }
            }
        }
        assert!(
            n_fail == 0,
            "CH5 third-order Hessian: {n_fail} pair(s) mismatch (full grid above; row i = perturbed ρ_i, col j = gradient component). Worst: {worst:?}"
        );
    }

    /// PATH C (#2253) — the FULLY assembled exact fixed-stratum outer Hessian
    /// (`exact_fixed_stratum_outer_hessian`: ch1 explicit + ch2 rank-charge + ch4
    /// log-determinant + ch5 third-order) must equal a central finite difference
    /// of the COMPLETE analytic outer gradient (`.gradient()`). This is the gate
    /// that catches a silently-dropped coordinate or channel: it exercises the
    /// whole channel set as one, in the symmetric orientation `H[i,j] =
    /// ∂g[i]/∂ρ_j`, so an asymmetric CH5 sign error also trips it.
    #[test]
    fn full_gradient_hessian_channel_set_matches_finite_difference_2253() {
        use crate::manifold::arrow_solver::DeflatedArrowSolver;
        use ndarray::{Array1, array};
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
        }
        rho_eval.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let rho = rho_eval;
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
            .expect("fixed-theta base cache");
        // COMMIT 1 — the public `exact_fixed_stratum_outer_hessian` intentionally
        // returns `Err` (production stays on BFGS); validate the assembly path
        // directly via the assembler that returns the block.
        let analytic = term
            .assemble_exact_fixed_stratum_outer_hessian(target.view(), &rho, &loss, &cache)
            .expect("full exact fixed-stratum outer Hessian assembles");

        let base = rho.to_flat();
        let h = 1.0e-5;
        let gradient_at = |sign: f64, coord: usize| -> Array1<f64> {
            let mut flat = base.clone();
            flat[coord] += sign * h;
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
            t.analytic_outer_rho_gradient_components(target.view(), &r, &loss, &cache, &solver)
                .expect("analytic gradient components")
                .gradient()
        };

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
        let tol = |a: f64| 1.0e-5 + 1.0e-4 * a.abs();
        let mut worst: Option<(usize, usize, f64, f64, f64)> = None;
        let mut n_fail = 0usize;
        for &j in &coords {
            // `H[i,j] = ∂g[i]/∂ρ_j`: perturb ρ_j, read the i-th component.
            let fd_col = (gradient_at(1.0, j) - gradient_at(-1.0, j)) / (2.0 * h);
            for &i in &coords {
                let a_ij = analytic[[i, j]];
                let fd_ij = fd_col[i];
                let abs_err = (a_ij - fd_ij).abs();
                let rel = if a_ij.abs() > 0.0 {
                    abs_err / a_ij.abs()
                } else {
                    abs_err
                };
                let fail = abs_err >= tol(a_ij);
                eprintln!(
                    "full H[{i}={},{j}={}] analytic={a_ij:.9e} fd={fd_ij:.9e} abs={abs_err:.3e} rel={rel:.3e}{}",
                    label(i),
                    label(j),
                    if fail { "  <== FAIL" } else { "" }
                );
                if fail {
                    n_fail += 1;
                    if worst.is_none_or(|(_, _, _, _, wr)| rel > wr) {
                        worst = Some((i, j, a_ij, fd_ij, rel));
                    }
                }
            }
        }
        assert!(
            n_fail == 0,
            "full exact fixed-stratum Hessian: {n_fail} pair(s) mismatch (full grid above; H[i,j]=∂g[i]/∂ρ_j). Worst: {worst:?}"
        );
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
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
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
                    .coordinate_block_logdet_theta_adjoint(&r, &cache, &solver)
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
                .solve_exact_stationarity(&r, target.view(), &cache, &solver, &gamma)
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
            DeflatedArrowSolver, SaeArrowVector, apply_cached_arrow_hessian,
        };
        use ndarray::array;
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
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
        let solver = DeflatedArrowSolver::plain(&cache);
        let norm = |v: &SaeArrowVector| (v.t.dot(&v.t) + v.beta.dot(&v.beta)).sqrt();
        let smooth0 = rho.smooth_flat_index(0);
        let ard0 = rho.ard_flat_index(0, 0);
        for (name, j) in [("smooth0", smooth0), ("ard0", ard0)] {
            let g_rho = term
                .outer_rho_gradient_ift_rhs(&rho, j, &cache)
                .expect("ift rhs");
            let x = term
                .solve_exact_stationarity(&rho, target.view(), &cache, &solver, &g_rho)
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
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
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
                .coordinate_block_logdet_theta_adjoint(&rho, &cache, &solver)
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
            .solve_exact_stationarity(&rho, target.view(), &cache, &solver, &gamma_eff)
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
                .solve_exact_stationarity(&rho, target.view(), &cache, &solver, &g_rho)
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
                    .coordinate_block_logdet_theta_adjoint(&r, &cache, &solver)
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
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
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
        }
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
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
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
        for (name, (ij, ji)) in names.iter().zip(legs.iter()) {
            eprintln!(
                "LEG {name}: <leg_smooth,b_ard>={ij:.6e} <leg_ard,b_smooth>={ji:.6e} \
                 asym={:.3e}",
                (ij - ji).abs()
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
        use super::{FaerEigh, SaeArrowVector, SaeCriterionError, Side};
        let (mut term, target, rho, _stationary_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho_eval = rho.clone();
        rho_eval.log_lambda_sparse = 0.5;
        for v in rho_eval.log_lambda_smooth.iter_mut() {
            *v = -2.0;
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
        let (eigs, _vecs) = sym.eigh(Side::Lower).expect("A eigendecomposition");
        let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
        let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let n_nonpos = eigs.iter().filter(|&&l| l <= 0.0).count();
        eprintln!(
            "A spectrum: min_eig={min_eig:.6e} max_eig={max_eig:.6e} n_nonpos={n_nonpos}/{dim}"
        );
        let result = term.exact_observed_information_log_dets(&rho, target.view(), &cache);
        let pd_floor = 1.0e-9 * max_eig.max(1.0);
        if min_eig > pd_floor {
            // A ≻ 0: a genuine exact-Laplace maximum. Refusal must NOT fire, and
            // the strict-Cholesky logdet must equal Σ ln λ (both joint and tt).
            let (log_a, log_a_tt) =
                result.expect("A is positive definite so the log-dets must be Ok");
            let eig_log_det: f64 = eigs.iter().map(|l| l.ln()).sum();
            assert!(
                (log_a - eig_log_det).abs() <= 1.0e-9 * (1.0 + eig_log_det.abs()),
                "log|A| strict-Cholesky path {log_a} != eigendecomposition oracle {eig_log_det}"
            );
            let a_tt = sym.slice(s![..total_t, ..total_t]).to_owned();
            let (eigs_tt, _) = a_tt.eigh(Side::Lower).expect("A_tt eigendecomposition");
            let eig_log_det_tt: f64 = eigs_tt.iter().map(|l| l.ln()).sum();
            assert!(
                (log_a_tt - eig_log_det_tt).abs() <= 1.0e-9 * (1.0 + eig_log_det_tt.abs()),
                "log|A_tt| strict-Cholesky path {log_a_tt} != eigendecomposition oracle \
                 {eig_log_det_tt}"
            );
        } else {
            // A is non-PD: the typed refusal must fire on the joint block, and the
            // eigen oracle independently confirms the non-positive spectrum. This
            // is a complete parity test (refusal ⟺ indefinite), not a skipped one.
            match result {
                Err(SaeCriterionError::IndefiniteObservedInformation { block }) => {
                    assert_eq!(block, "joint", "refusal fired on the wrong block: {block}");
                }
                other => panic!(
                    "A is non-PD (min_eig={min_eig:.3e}) but exact_observed_information_log_dets \
                     did not refuse: {other:?}"
                ),
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
    use crate::manifold::tests::diagonal_latent_cache;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

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
