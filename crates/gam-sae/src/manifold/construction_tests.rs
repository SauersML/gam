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

    /// #2080/#2228 — the polish tests' entry: the PD-basin fixture's term and `rho`
    /// at a state that still carries a live residual.
    ///
    /// `converged_state_with_residual` prices through the criterion, and the criterion
    /// carries an accepted state to its root before pricing it (`refine_accepted_root`,
    /// ae0d368e20). At that root the gradient sits inside the KKT tolerance and no
    /// polish step resolves a decrease: bis2 (job 1162220) read the entry `‖g‖` as
    /// 4.67e-16 at ae0d368e20 against 1.01e-4 at its parent, with byte-identical test
    /// bodies. A polish test built on it measures an early return.
    ///
    /// This state is the majorized evidence inner loop's own fixed point instead: the
    /// same term and `rho` re-enter `run_joint_fit_arrow_schur_for_quasi_laplace` until
    /// a whole re-entry finds no strict decrease. That drive has no acceptance,
    /// refinement or polish site, so nothing on its path carries the state to the
    /// root, and the exact-A step the polish takes still resolves descent there. The
    /// premise is a property of the state, asserted with the criterion's own KKT gate
    /// (`quasi_laplace_kkt_stationary` at `SAE_MANIFOLD_INNER_GRAD_REL_TOL ·
    /// inner_iterate_scale`), and printed before it is asserted.
    pub(super) fn majorized_fixed_point_with_residual()
    -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        use crate::manifold::term::SAE_MANIFOLD_INNER_GRAD_REL_TOL;
        use crate::manifold::tests::gamma_fd_tiny_fixture;

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
        // Each re-entry that is not a fixed point committed a strict decrease of the
        // penalized objective under its own gates, so the loop ends where the drive
        // itself stops moving the state.
        let mut reentries = 0usize;
        loop {
            let outcome = term
                .run_joint_fit_arrow_schur_for_quasi_laplace(
                    target.view(),
                    &mut rho,
                    None,
                    40,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                )
                .expect("the majorized evidence drive must run on the PD-basin fixture");
            reentries += 1;
            if outcome.fixed_point {
                break;
            }
        }
        let system = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("arrow-Schur assembly at the majorized fixed point");
        let grad_norm_sq = SaeManifoldTerm::system_grad_norm_sq(&system);
        let lambda_smooth = rho.lambda_smooth_vec().expect("smoothness strengths");
        let quotient_grad_norm =
            term.quotient_gradient_norm_from_system(&system, grad_norm_sq, &lambda_smooth);
        let tolerance = SAE_MANIFOLD_INNER_GRAD_REL_TOL * term.inner_iterate_scale();
        eprintln!(
            "[#2080 polish fixture] majorized fixed point after {reentries} re-entries: \
             ‖g‖={:.6e} ‖Π⊥null g‖={quotient_grad_norm:.6e} tol={tolerance:.6e}",
            grad_norm_sq.sqrt(),
        );
        assert!(
            !SaeManifoldTerm::quasi_laplace_kkt_stationary(
                grad_norm_sq.sqrt(),
                quotient_grad_norm,
                tolerance,
            ),
            "#2080 polish fixture premise: the majorized fixed point must carry a live \
             residual the criterion's KKT gate refuses: ‖g‖={:.6e} ‖Π⊥null g‖=\
             {quotient_grad_norm:.6e} tol={tolerance:.6e}",
            grad_norm_sq.sqrt(),
        );
        (term, target, rho)
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
    use crate::manifold::tests::small_two_atom_periodic_term;

    /// The exact observed information `A` materialized by columns of the production apply,
    /// symmetrized, for the pencil oracles below.
    fn dense_exact_hessian(
        term: &super::SaeManifoldTerm,
        rho: &super::SaeManifoldRho,
        target: ndarray::ArrayView2<'_, f64>,
        cache: &super::ArrowFactorCache,
    ) -> ndarray::Array2<f64> {
        use super::SaeArrowVector;
        use ndarray::{Array1, Array2};
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
                .apply_exact_hessian(rho, target, cache, &unit)
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
        (&a + &a.t()) * 0.5
    }

    /// #2330 / #2933 F07 — at a root whose pencil has no resolved negative direction, the
    /// exact observed-information Laplace log-det equals the independent pencil oracle
    /// `log|A| = log|Φ| + Σ_retained ln μ`, with `A w = μΦw` and `Φ` the evidence factor.
    /// This pins the `log|A|` VALUE the dense route ranks against the exact observed
    /// information, not the majorized surrogate `B`.
    ///
    /// Evaluated at the fixture's own converged root and ρ. The oracle keeps its own
    /// operands (dense `A` from column applies, dense `Φ` from metric applies, a symmetric
    /// square-root reduction) and shares only the scalar band rule. The refusal half of the
    /// contract is `exact_observed_information_refuses_a_genuinely_indefinite_frozen_state_2330`.
    #[test]
    fn exact_observed_information_log_det_matches_the_pencil_oracle_at_a_pd_root_2330() {
        let (term, target, rho, cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let sym = dense_exact_hessian(&term, &rho, target.view(), &cache);
        let oracle = crate::manifold::tests::PencilOracle::new(&sym, &cache);
        let (mu, floors) = (&oracle.values, &oracle.floors);
        let most_negative = (0..mu.len())
            .min_by(|&a, &b| mu[a].total_cmp(&mu[b]))
            .expect("the pencil has directions");
        eprintln!(
            "PD root pencil: dim={} min μ={:.6e} edge={:.6e} retained={} in_band={} log|Φ|={:.6e}",
            mu.len(),
            mu[most_negative],
            floors[most_negative],
            oracle.retained().len(),
            oracle.in_band(),
            oracle.metric_log_det,
        );
        assert!(
            oracle.negative().is_empty(),
            "fixture: the root must carry no resolved negative pencil direction \
             (min μ={:.6e}, edge {:.6e})",
            mu[most_negative],
            floors[most_negative],
        );
        let retained = oracle.retained();
        let oracle_log_a = oracle.metric_log_det
            + retained.iter().map(|&index| mu[index].ln()).sum::<f64>();
        let log_a = term
            .exact_observed_information_log_dets(&rho, target.view(), &cache)
            .expect("the pencil has no resolved negative direction, so the log-det must be Ok");
        let gap = (log_a - oracle_log_a).abs();
        eprintln!("PD root log|A|: production={log_a:.12e} oracle={oracle_log_a:.12e} gap={gap:.3e}");
        assert!(
            gap <= 1.0e-9 * (1.0 + oracle_log_a.abs()),
            "log|A| {log_a:.12e} != pencil oracle {oracle_log_a:.12e} (gap {gap:.3e})"
        );
    }

    /// #2330 / #2336 / #2933 F07 — a frozen state whose exact observed information carries a
    /// genuinely indefinite basin refuses typed on the joint block.
    ///
    /// The fixture's root is frozen and re-assembled at ρ_eval = (sparse −0.5, smooth −1.5,
    /// ARD [−1.2, −1.0]). There `A` has one resolved negative direction that the bounded ARD
    /// concave clamp does not explain. Probe job 1157737 (pre-F07 Euclidean classifier) read
    /// λ = −5.010035e-2 against edge 3.846366e-9 and basin curvature −5.034747e-2 with a zero
    /// clamp diagonal. The cache is factored through the production freeze lane
    /// (`inner_max_iter == 0`), which neither converges nor prices, so the verdict is read from
    /// `exact_observed_information_log_dets` itself. The criterion prices `½log|A|` while it
    /// builds its cache, so a cache taken from the criterion at this state refuses before any
    /// oracle can run.
    #[test]
    fn exact_observed_information_refuses_a_genuinely_indefinite_frozen_state_2330() {
        use super::{FaerEigh, SaeCriterionError, Side, sae_exact_a_band_edge, sae_exact_a_pencil_resolution};
        use ndarray::{Array2, array};
        let (mut term, target, root_rho, _root_cache) =
            super::exact_hessian_fixture_tests::converged_state_with_residual();
        let mut rho = root_rho.clone();
        rho.log_lambda_sparse = -0.5;
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = -1.5;
        }
        rho.log_ard = vec![array![-1.2_f64], array![-1.0_f64]];
        let mut rho_fixed = rho.clone();
        let refresh = term
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                target.view(),
                &mut rho_fixed,
                None,
                crate::manifold::tests::FROZEN_INNER_STATE,
                0.4,
                1.0e-6,
                1.0e-6,
            )
            .expect("freeze-lane refresh at the frozen root");
        let mut loss = refresh.loss;
        let mut fixed_point = refresh.fixed_point;
        let options = term.evidence_factor_options();
        let cache = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                &rho,
                &mut rho_fixed,
                None,
                crate::manifold::tests::FROZEN_INNER_STATE,
                0.4,
                1.0e-6,
                1.0e-6,
                &mut loss,
                &mut fixed_point,
                &options,
                true,
            )
            .expect("freeze-lane factorization at the frozen root");
        let total_t = cache.delta_t_len();
        let dim = total_t + cache.k;
        let sym = dense_exact_hessian(&term, &rho, target.view(), &cache);
        let oracle = crate::manifold::tests::PencilOracle::new(&sym, &cache);
        let (mu, vectors) = (&oracle.values, &oracle.vectors);
        let negative = oracle.negative();
        let most_negative = (0..mu.len())
            .min_by(|&a, &b| mu[a].total_cmp(&mu[b]))
            .expect("the pencil has directions");
        eprintln!(
            "frozen state pencil: dim={dim} min μ={:.6e} edge={:.6e} negative={}",
            mu[most_negative],
            oracle.floors[most_negative],
            negative.len(),
        );
        assert!(
            !negative.is_empty(),
            "fixture: the frozen state must carry a resolved negative pencil direction \
             (min μ={:.6e}, edge {:.6e})",
            mu[most_negative],
            oracle.floors[most_negative],
        );
        // #2336 — the negative subspace is priced at its basin curvature
        // `C = W_NᵀAW_N + W_NᵀEW_N`; only a genuinely indefinite `C` refuses.
        let e_diag = term
            .materialize_ard_concave_clamp_diagonal(&rho, &cache)
            .expect("ARD concave-clamp diagonal");
        let e_beta = term
            .decoder_prior_majorizer_gap_border(&cache)
            .expect("decoder-prior majorization gap");
        let q = negative.len();
        let mut basin = Array2::<f64>::zeros((q, q));
        for (i, &ni) in negative.iter().enumerate() {
            for (j, &nj) in negative.iter().enumerate() {
                let (wi, wj) = (vectors.column(ni), vectors.column(nj));
                let mut value = (0..total_t)
                    .map(|row| e_diag[row] * wi[row] * wj[row])
                    .sum::<f64>();
                if let Some(gap) = e_beta.as_ref() {
                    value += wi
                        .slice(ndarray::s![total_t..])
                        .dot(&gap.dot(&wj.slice(ndarray::s![total_t..])));
                }
                if i == j {
                    value += mu[ni];
                }
                basin[[i, j]] = value;
            }
        }
        let basin = (&basin + &basin.t()) * 0.5;
        let (kappa, basin_rotation) = basin.eigh(Side::Lower).expect("basin eigendecomposition");
        let negative_basis =
            Array2::from_shape_fn((dim, q), |(row, col)| vectors[[row, negative[col]]]);
        let basin_vectors = negative_basis.dot(&basin_rotation);
        let e_frobenius = (e_diag.iter().take(total_t).map(|x| x * x).sum::<f64>()
            + e_beta.as_ref().map_or(0.0, |gap| gap.iter().map(|x| x * x).sum::<f64>()))
        .sqrt();
        let mut genuinely_indefinite = 0usize;
        for j in 0..q {
            let direction = basin_vectors.column(j);
            let resolution = sae_exact_a_pencil_resolution(
                dim,
                direction.dot(&direction),
                oracle.operator_frobenius + e_frobenius,
                oracle.metric_frobenius,
                kappa[j],
            );
            let floor = sae_exact_a_band_edge(kappa[j], resolution, 0.0);
            eprintln!(
                "frozen state basin mode {j}: κ={:.6e} floor={floor:.6e} max|E diag|={:.6e}",
                kappa[j],
                e_diag.iter().take(total_t).map(|x| x.abs()).fold(0.0_f64, f64::max),
            );
            if kappa[j] < -floor {
                genuinely_indefinite += 1;
            }
        }
        assert!(
            genuinely_indefinite > 0,
            "fixture: the frozen state's basin must carry a curvature below its floor that the \
             clamp does not explain; every one of its {q} basin curvatures is attributable"
        );
        match term.exact_observed_information_log_dets(&rho, target.view(), &cache) {
            Err(SaeCriterionError::IndefiniteObservedInformation { block }) => {
                assert_eq!(block, "joint", "refusal fired on the wrong block: {block}");
            }
            other => panic!(
                "the pencil has {genuinely_indefinite} genuinely indefinite basin direction(s), \
                 not clamp-attributable, but exact_observed_information_log_dets did not refuse: \
                 {other:?}"
            ),
        }
    }

    /// #2330/#2336 PRICING arbiter (reachable companion to the PD parity test
    /// above). MEASURED correction to the earlier premise: the a_saddle specimen's
    /// two exact-A negatives are FULLY attributable to the bounded ARD periodic
    /// concave-clamp wrinkle (`λ+e_v(ARD)=+0.026`, verified in
    /// `zz_measure_e_attributability_2336`) — NOT a residual-rooted genuine saddle.
    /// So under the value-side E-attributability semantics (#2336) the criterion
    /// PRICES it at the basin curvature and returns a FINITE value. This is the
    /// price half of the (price ⟺ E-attributable, refuse ⟺ genuine) contract; no
    /// specimen pins the refuse half, since the criterion descends a refused exact-A
    /// saddle before it refuses (#2080).
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
            result
                .as_ref()
                .map(|(value, _, _)| *value)
                .map_err(|e| format!("{e:?}"))
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
    //! (`psd_majorizer_abs_row_sums`, `row_psd_majorizer`)
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
    /// constant (#2673, #2933 F07).
    ///
    /// The band is a threshold on the pencil curvature `μ` against the floor `√ε`. In the
    /// metric `Φ = c·I` with `c = floor/√ε` the pencil curvature of an eigenvector of the
    /// operator is `μ = λ/c`, so `|λ| ≤ floor` is exactly the band. The solves, steps and
    /// model residuals the fixtures below assert are the same physical vectors in any
    /// metric `c·I`; only curvatures and dampings read in `μ` units. That the block's own
    /// `rank_floor` realises the requested band is asserted here rather than assumed.
    fn spectral_block_with_uniform_floor(
        operator: Array2<f64>,
        eigenvalues: Array1<f64>,
        eigenvectors: Array2<f64>,
        floor: f64,
    ) -> ExactHessianSpectralBlock {
        let dimension = eigenvalues.len();
        let scale = floor / super::sae_exact_a_pencil_floor();
        let operator_frobenius = operator.iter().map(|value| value * value).sum::<f64>().sqrt();
        let vectors = eigenvectors.mapv(|value| value / scale.sqrt());
        let band: Vec<usize> = (0..dimension)
            .filter(|&index| eigenvalues[index].abs() <= floor)
            .collect();
        let band_metric_images = Array2::from_shape_fn((dimension, band.len()), |(row, col)| {
            scale * vectors[[row, band[col]]]
        });
        let block = ExactHessianSpectralBlock {
            operator,
            eigenvalues: eigenvalues.mapv(|value| value / scale),
            eigenvectors: vectors,
            substituted_stiffness: Array1::zeros(dimension),
            resolution: Array1::zeros(dimension),
            metric_log_det: dimension as f64 * scale.ln(),
            operator_frobenius,
            metric_frobenius: scale * (dimension as f64).sqrt(),
            band,
            band_metric_images,
            orbit: None,
        };
        for index in 0..dimension {
            let realised = block.rank_floor(index) * scale;
            assert!(
                (realised - floor).abs() <= 1.0e-12 * floor,
                "#2673: the synthetic block must realise the requested band \
                 (direction {index}: asked {floor:.6e}, got {realised:.6e})"
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
            .expect("rank-revealing dense exact-stationarity solve")
            .step;
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
            .expect("rank-revealing dense exact-stationarity solve")
            .step;
        let damped = geometry
            .damped_residual_step(&residual, 0.0)
            .expect("zero damping is the pseudoinverse point of the path");
        assert_abs_diff_eq!(damped.step.t[0], pseudoinverse.t[0], epsilon = 1.0e-15);
        assert_abs_diff_eq!(damped.step.t[1], pseudoinverse.t[1], epsilon = 1.0e-15);
        assert_abs_diff_eq!(
            damped.step.beta[0],
            pseudoinverse.beta[0],
            epsilon = 1.0e-15
        );
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
        let geometry =
            spectral_block_with_uniform_floor(operator.clone(), eigenvalues, basis, 1.0e-12);
        let residual = SaeArrowVector {
            t: Array1::from_vec(vec![0.7_f64, -1.3, 0.2]),
            beta: Array1::from_vec(vec![0.9]),
        };
        let mut flat_residual = Array1::<f64>::zeros(dim);
        flat_residual.slice_mut(s![..3]).assign(&residual.t);
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
        // The block reads curvatures in pencil units `μ = λ/c`, so the damping `ν = 1e-6`
        // in `λ²` units is `1e-6/c²` there.
        let scale = 1.0e-14 / super::sae_exact_a_pencil_floor();
        let undamped = geometry
            .damped_residual_step(&residual, 0.0)
            .expect("undamped step");
        let damped = geometry
            .damped_residual_step(&residual, 1.0e-6 / (scale * scale))
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

    /// #2080/#2228 — residual minimization and objective minimization point in
    /// opposite directions on a negative-curvature mode.  The terminal polish
    /// must use the latter: otherwise it rejects precisely the direction which
    /// can leave the non-stationary inner saddle and every outer rho probe is
    /// reported as infeasible.
    #[test]
    fn terminal_objective_step_descends_resolved_negative_curvature_2080() {
        let eigenvalues = Array1::from_vec(vec![-2.0_f64, 4.0]);
        let geometry = spectral_block_with_uniform_floor(
            Array2::from_diag(&eigenvalues),
            eigenvalues,
            Array2::from_diag(&Array1::ones(2)),
            1.0e-12,
        );
        let gradient = SaeArrowVector {
            t: Array1::from_vec(vec![3.0]),
            beta: Array1::from_vec(vec![5.0]),
        };
        let residual_step = geometry
            .damped_residual_step(&gradient, 0.0)
            .expect("residual Gauss-Newton step");
        let objective_step = geometry
            .damped_objective_step(&gradient, 0.0)
            .expect("objective trust-region step");
        assert!(
            gradient.t.dot(&residual_step.step.t) > 0.0,
            "the negative-mode residual step must expose the old objective-ascent defect"
        );
        let directional_derivative =
            gradient.t.dot(&objective_step.step.t) + gradient.beta.dot(&objective_step.step.beta);
        assert!(
            directional_derivative < 0.0,
            "the objective step must be descent across both signs of curvature, got g dot d = {directional_derivative:.6e}"
        );
        assert_abs_diff_eq!(objective_step.step.t[0], -1.5, epsilon = 1.0e-14);
        assert_abs_diff_eq!(objective_step.step.beta[0], -1.25, epsilon = 1.0e-14);
    }

    /// #2080/#2228 — the polish may not raise the penalized objective.  The
    /// former residual-monotonicity assertion was itself wrong: objective descent
    /// along resolved negative curvature necessarily increases `||g||`.  Driving
    /// with an unreachable tolerance forces a real step and pins the scalar
    /// currency on which terminal globalization is now accepted. The entry is the
    /// majorized fixed point, whose live residual the fixture asserts.
    #[test]
    fn terminal_polish_never_raises_the_penalized_objective_2080() {
        let (mut term, target, rho) =
            super::exact_hessian_fixture_tests::majorized_fixed_point_with_residual();
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
        let objective_before = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective before terminal polish");
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
        let objective_after = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective after terminal polish");
        assert!(
            objective_after <= objective_before,
            "the polish raised the scalar objective: {objective_before:.6e} -> \
             {objective_after:.6e} (residual {before:.6e} -> {after:.6e}, moved={moved})"
        );
        // Non-vacuity: an unreachable tolerance on a state with a live residual
        // must make this phase actually step, or the assertion above is testing
        // an early return.
        assert!(
            moved && objective_after < objective_before,
            "the phase must commit objective descent at tolerance 0: objective \
             {objective_before:.6e} -> {objective_after:.6e}, residual \
             {before:.6e} -> {after:.6e} (moved={moved})"
        );
    }

    /// #2283 — the step a declined dense geometry falls back to. On the 2080
    /// fixture's live residual, the shifted Newton step on the arrow exact-A system
    /// predicts a positive decrease, commits an Armijo decrease, and reports both
    /// objectives exactly as an independent evaluation at the entry and committed
    /// states reads them. The entry is the majorized fixed point, whose live residual
    /// the fixture asserts.
    #[test]
    fn arrow_exact_a_polish_step_commits_objective_descent_2283() {
        let (mut term, target, rho) =
            super::exact_hessian_fixture_tests::majorized_fixed_point_with_residual();
        let options = term.evidence_factor_options();
        let majorizer = term
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("arrow-Schur assembly at the polish entry state");
        let objective_before = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective before the arrow exact-A step");
        let committed = term
            .shifted_exact_newton_polish_trials(target.view(), &rho, None, &options, &majorizer, 0.0)
            .expect("the arrow exact-A step degrades every internal failure to Ok(None)")
            .expect("a state with a live residual must buy an Armijo decrease on some rung");
        let objective_after = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective after the arrow exact-A step");
        assert!(
            committed.predicted_objective_decrease > 0.0
                && committed.curvature_along_step > 0.0
                && committed.trials >= 1,
            "the committed arrow exact-A step must be descent with positive model curvature: \
             predicted {:.6e}, curvature {:.6e}, trials {}",
            committed.predicted_objective_decrease,
            committed.curvature_along_step,
            committed.trials,
        );
        assert!(
            committed.committed_objective < committed.pre_objective,
            "the arrow exact-A step must lower the penalized objective: {:.6e} -> {:.6e}",
            committed.pre_objective,
            committed.committed_objective,
        );
        assert_abs_diff_eq!(committed.pre_objective, objective_before, epsilon = 0.0);
        assert_abs_diff_eq!(committed.committed_objective, objective_after, epsilon = 0.0);
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
        // In pencil units `μ = λ/c` of the block's metric `c·I`.
        let scale = 1.0e-9 / super::sae_exact_a_pencil_floor();
        assert_abs_diff_eq!(smallest * scale, 0.5, epsilon = 1.0e-12);
        assert_abs_diff_eq!(largest * scale, 7.0, epsilon = 1.0e-12);

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

        // The RHS component on the declared null band, read off the production
        // geometry at this state. #2267 — the band includes every positive direction
        // only the evidence factor's substituted stiffness resolves, and the
        // pseudoinverse leaves exactly this component unsolved, so it is removed
        // from the bar below and nothing else is.
        let geometry = term
            .materialize_exact_stationarity_geometry(&rho, target.view(), &cache)
            .expect("exact stationarity geometry");
        let rhs_flat = Array1::from_iter(rhs.t.iter().chain(rhs.beta.iter()).copied());
        // #2933 F07 — the pseudoinverse removes the band's dual components `ΦW_Z W_Zᵀ rhs`.
        let band_directions = geometry.band.len();
        let band_coefficients = Array1::from_iter(
            geometry
                .band
                .iter()
                .map(|&index| geometry.eigenvectors.column(index).dot(&rhs_flat)),
        );
        let band = geometry.band_metric_images.dot(&band_coefficients);
        let band_norm = band.dot(&band).sqrt();
        let rhs_range = SaeArrowVector {
            t: &rhs.t - &band.slice(s![..total_t]),
            beta: &rhs.beta - &band.slice(s![total_t..]),
        };
        let range_resid = a_residual_norm(&term, &rho, target.view(), &cache, &x, &rhs_range);

        // Surrogate solve x_B = B⁻¹ rhs (the pre-#1418 implicit step).
        let x_b = solver
            .solve(rhs.t.view(), rhs.beta.view())
            .expect("B inverse");
        let surrogate_resid = a_residual_norm(&term, &rho, target.view(), &cache, &x_b, &rhs);

        // 1) The exact solve drives the residual on the retained range to ~0.
        //    #2674 — this used to be asserted on the chart-gauge quotient of the
        //    residual, because the solve deleted that orbit before inverting; that
        //    deletion is gone, so no direction is excluded by declaration. What
        //    is excluded is the spectral band `|λ| ≤ rank_floor`, the one null
        //    predicate the value, the differential and this solve share, so the
        //    bar is `‖A x − (rhs − P_band rhs)‖` at the full ambient strength.
        assert!(
            range_resid <= 1.0e-6 * rhs_norm,
            "solve_exact_stationarity must invert the EXACT A on its retained range: \
             ‖A x − (rhs − P_band rhs)‖/‖rhs‖ = {:.3e} (rhs_norm={rhs_norm:.3e}, band \
             directions={band_directions}, ‖P_band rhs‖={band_norm:.3e}, full residual \
             {exact_resid:.3e}) — the IFT step is not solving A x = rhs (#1418)",
            range_resid / rhs_norm
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

        // 3) The exact solve is a strict, large improvement over the surrogate on
        //    the range it inverts.
        assert!(
            range_resid < 1.0e-3 * surrogate_resid,
            "exact A-solve range residual {range_resid:.3e} must be far below surrogate \
             {surrogate_resid:.3e}"
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
            .reconstruction_dispersion(&loss, &cache, &rho, residual.view())
            .expect("dispersion");
        assert!(
            dispersion.posterior_covariance_scale() > 0.0,
            "a real residual ⇒ positive dispersion"
        );
        let geometry = term
            .materialize_exact_stationarity_geometry(&rho, target.view(), &cache)
            .expect("exact stationarity geometry at the converged state");
        let information = term
            .exact_observed_information_shape_covariance(&geometry, &rho, target.view(), &cache)
            .expect("exact observed information at the converged state");
        let joint = term
            .assemble_shape_uncertainty(&information, dispersion)
            .expect("direct joint bands");
        assert!(
            matches!(
                joint.operator,
                crate::manifold::SaeShapeCovarianceOperator::ObservedInformation { .. }
            ),
            "the converged PD basin must report an observed-information covariance; got {:?}",
            joint.operator
        );

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

    /// Joint shape bands of the tiny fixture with every observation unit scaled
    /// by `unit`, optionally under the known whitening covariance
    /// `Σ = unit²·diag(sd²)` (`M_n = Σ⁻¹`, installed as `U_n = Σ^{-1/2}`).
    ///
    /// Scaling the target and decoders by `unit`, `Σ` by `unit²` and
    /// `λ_smooth` by `unit⁻²` leaves the whitened inner objective unchanged, so
    /// the coordinates and gates are the same state and only the decoder moves
    /// to `unit·B`. The transformation law of a covariance is therefore
    /// `Cov(unit·B) = unit²·Cov(B)` exactly.
    fn scaled_whitened_shape_uncertainty_2933_f34(
        unit: f64,
        sd: Option<&[f64]>,
    ) -> crate::manifold::SaeShapeUncertainty {
        let (mut term, target, mut rho) =
            crate::manifold::tests_recovery_split_780::gamma_fd_tiny_fixture();
        rho.log_lambda_sparse = 0.0;
        for v in rho.log_lambda_smooth.iter_mut() {
            *v = -1.0 - 2.0 * unit.ln();
        }
        for axis in rho.log_ard.iter_mut() {
            for v in axis.iter_mut() {
                *v = -1.0;
            }
        }
        for atom in term.atoms.iter_mut() {
            atom.decoder_coefficients_mut().mapv_inplace(|v| unit * v);
        }
        let target = target.mapv(|v| unit * v);
        let (n, p) = target.dim();
        if let Some(sd) = sd {
            let factors = ndarray::Array2::<f64>::from_shape_fn((n, p * p), |(_, flat)| {
                let (i, k) = (flat / p, flat % p);
                if i == k { 1.0 / (unit * sd[i]) } else { 0.0 }
            });
            let metric = gam_problem::RowMetric::whitened_structured(
                std::sync::Arc::new(factors),
                p,
                p,
            )
            .expect("diagonal whitening factors");
            term.set_row_metric(metric).expect("conformable whitening metric");
        }
        term.recompute_joint_shape_uncertainty(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .expect("joint shape uncertainty")
    }

    /// `scaled` must be the covariance of `factor·B` when `base` is that of `B`:
    /// every band sd times `factor`, every decoder covariance times `factor²`.
    fn assert_shape_covariance_law_2933_f34(
        base: &crate::manifold::SaeShapeUncertainty,
        scaled: &crate::manifold::SaeShapeUncertainty,
        factor: f64,
        label: &str,
    ) {
        assert_eq!(base.atoms.len(), scaled.atoms.len(), "{label}: atom count");
        let mut max_base_sd = 0.0_f64;
        for (k, (a, b)) in base.atoms.iter().zip(scaled.atoms.iter()).enumerate() {
            let a_sd = a.band_sd.as_ref().expect("base band");
            let b_sd = b.band_sd.as_ref().expect("scaled band");
            assert_eq!(a_sd.dim(), b_sd.dim(), "{label}: atom {k} band shape");
            for (x, y) in a_sd.iter().zip(b_sd.iter()) {
                max_base_sd = max_base_sd.max(*x);
                assert!(
                    (y - factor * x).abs() <= 1.0e-3 * factor * x.abs() + 1.0e-12,
                    "{label}: atom {k} band sd {y:.6e} must be {factor}×{x:.6e} = {:.6e}",
                    factor * x
                );
            }
            let a_cov = a.decoder_covariance.as_ref().expect("base covariance");
            let b_cov = b.decoder_covariance.as_ref().expect("scaled covariance");
            let norm = a_cov.mapv(|v| v * v).sum().sqrt();
            let miss = (b_cov - &a_cov.mapv(|v| v * factor * factor))
                .mapv(|v| v * v)
                .sum()
                .sqrt();
            assert!(
                miss <= 1.0e-3 * factor * factor * norm,
                "{label}: atom {k} covariance must scale by {}; ‖Cov′ − {}·Cov‖_F = {miss:.3e} \
                 against ‖Cov‖_F = {norm:.3e}",
                factor * factor,
                factor * factor
            );
        }
        assert!(
            max_base_sd > 1.0e-6,
            "{label}: the base band must be materially nonzero (max sd {max_base_sd:.3e})"
        );
    }

    /// #2933 F34 — with `M = σ⁻²I` the joint Hessian is `H = XᵀX/σ² + …`, so
    /// `H⁻¹` already carries `σ²` and the covariance multiplier must be the
    /// dimensionless whitened dispersion. The isotropic fit on `z` and the
    /// `Σ = 4I` fit on `2z` describe the same model in units differing by 2, so
    /// the covariance must grow by exactly 4. Multiplying the whitened inverse
    /// by the raw residual variance (≈ 4× the unit-scale one) gave 16.
    #[test]
    fn isotropic_known_whitening_counts_the_noise_variance_once_2933_f34() {
        let euclidean = scaled_whitened_shape_uncertainty_2933_f34(1.0, None);
        let unit_whitened = scaled_whitened_shape_uncertainty_2933_f34(1.0, Some(&[1.0; 3]));
        let doubled_whitened = scaled_whitened_shape_uncertainty_2933_f34(2.0, Some(&[1.0; 3]));

        // `M = I` is the isotropic likelihood routed through the whitening seam.
        assert_shape_covariance_law_2933_f34(&euclidean, &unit_whitened, 1.0, "Σ = I vs isotropic");
        assert_shape_covariance_law_2933_f34(&euclidean, &doubled_whitened, 2.0, "Σ = 4I on 2z");

        assert_eq!(
            euclidean.dispersion.likelihood_frame,
            crate::manifold::SaeLikelihoodFrame::RawOutput
        );
        assert_eq!(
            doubled_whitened.dispersion.likelihood_frame,
            crate::manifold::SaeLikelihoodFrame::Whitened { metric_rank: 3 }
        );
        let raw = euclidean.dispersion.raw_output_noise_variance;
        assert!(raw > 0.0, "the fixture carries a genuine residual");
        assert!(
            (euclidean.dispersion.likelihood_dispersion - raw).abs() <= 1.0e-12 * raw,
            "on the isotropic frame the two scales coincide"
        );
        assert!(
            (doubled_whitened.dispersion.raw_output_noise_variance - 4.0 * raw).abs()
                <= 1.0e-3 * 4.0 * raw,
            "the raw noise variance is in squared output units: {} vs 4·{raw}",
            doubled_whitened.dispersion.raw_output_noise_variance
        );
        assert!(
            (doubled_whitened.dispersion.likelihood_dispersion - raw).abs() <= 1.0e-3 * raw,
            "the whitened dispersion is dimensionless: {} vs {raw}",
            doubled_whitened.dispersion.likelihood_dispersion
        );
    }

    /// #2933 F34 — rescaling every observation unit by 3 under a known
    /// ANISOTROPIC noise covariance `Σ = diag(1, 4, 1/4)` must scale each band
    /// sd by 3 and every decoder covariance by 9, while the whitened dispersion
    /// is unchanged and the raw noise variance scales by 9. Double counting
    /// the scale gives band ratio 9 and covariance ratio 81.
    #[test]
    fn whitened_shape_covariance_obeys_the_observation_unit_law_2933_f34() {
        let sd = [1.0, 2.0, 0.5];
        let base = scaled_whitened_shape_uncertainty_2933_f34(1.0, Some(&sd));
        let tripled = scaled_whitened_shape_uncertainty_2933_f34(3.0, Some(&sd));
        assert_shape_covariance_law_2933_f34(&base, &tripled, 3.0, "Σ = 9·diag(1, 4, 1/4) on 3z");
        let base_likelihood = base.dispersion.likelihood_dispersion;
        let base_raw = base.dispersion.raw_output_noise_variance;
        assert!(
            (tripled.dispersion.likelihood_dispersion - base_likelihood).abs()
                <= 1.0e-3 * base_likelihood,
            "whitened dispersion must be unit-free: {} vs {base_likelihood}",
            tripled.dispersion.likelihood_dispersion
        );
        assert!(
            (tripled.dispersion.raw_output_noise_variance - 9.0 * base_raw).abs()
                <= 1.0e-3 * 9.0 * base_raw,
            "raw noise variance must scale by 9: {} vs 9·{base_raw}",
            tripled.dispersion.raw_output_noise_variance
        );
    }
}

#[cfg(test)]
mod learned_frame_shape_covariance_2933_f35_tests {
    use crate::basis::SaeBasisEvaluator;
    use crate::manifold::{
        FaerCholesky, SaeFrameConditioning, SaeManifoldRho, SaeManifoldTerm,
        SaeShapeCovarianceOperator, SaeShapeInformation, SaeShapeUncertainty, Side,
    };
    use ndarray::{Array1, Array2};
    use std::sync::Arc;

    /// One periodic atom `m(t) = B·[1, sin 2πt, cos 2πt]` in `p = 12` outputs.
    /// The decoder and the target lie in the span of the first `rank` output
    /// axes, so the fit activates a rank-`rank` Grassmann frame on exactly that
    /// span, and the residual never leaves it.
    fn framed_circle_2933_f35(rank: usize) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
        let (n, p, m) = (24usize, 12usize, 3usize);
        let evaluator =
            Arc::new(crate::basis::PeriodicHarmonicEvaluator::new(m).expect("periodic basis"));
        let coords = Array2::from_shape_fn((n, 1), |(row, _)| (row as f64 + 0.25) / n as f64);
        let (phi, jet) = evaluator.evaluate(coords.view()).expect("periodic jets");
        let mut decoder = Array2::<f64>::zeros((m, p));
        decoder[[1, 0]] = 0.9;
        decoder[[1, 1]] = 0.2;
        decoder[[2, 0]] = -0.1;
        decoder[[2, 1]] = 0.8;
        if rank == 3 {
            decoder[[0, 2]] = 0.35;
        }
        let mut target = phi.dot(&decoder);
        for row in 0..n {
            let x = row as f64;
            target[[row, 0]] += 0.02 * (1.7 * x).sin();
            target[[row, 1]] += 0.02 * (1.3 * x).cos();
            if rank == 3 {
                target[[row, 2]] += 0.02 * (0.9 * x).sin();
            }
        }
        let atom = crate::manifold::SaeManifoldAtom::new_with_provided_function_gram(
            "framed_circle",
            crate::manifold::SaeAtomBasisKind::Periodic,
            1,
            phi,
            jet,
            decoder,
            Array2::<f64>::eye(m),
        )
        .expect("atom shapes agree")
        .with_basis_second_jet(evaluator);
        let assignment = crate::assignment::SaeAssignment::from_blocks_with_mode_and_manifolds(
            Array2::<f64>::zeros((n, 1)),
            vec![coords],
            vec![gam_terms::latent::LatentManifold::Circle { period: 1.0 }],
            crate::assignment::AssignmentMode::softmax(1.0),
        )
        .expect("assignment shapes agree");
        let term = SaeManifoldTerm::new(vec![atom], assignment).expect("term");
        let rho = SaeManifoldRho::new(
            0.0,
            0.8_f64.ln(),
            vec![Array1::from_vec(vec![250.0_f64.ln()])],
        );
        (term, target, rho)
    }

    /// Fit the fixture, check the frame activated at the decoder's rank, and
    /// return the fitted term with its joint shape uncertainty.
    fn fitted_framed_circle_2933_f35(
        rank: usize,
    ) -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho, SaeShapeUncertainty) {
        let (mut term, target, rho) = framed_circle_2933_f35(rank);
        let shape = term
            .recompute_joint_shape_uncertainty(target.view(), &rho, None, 40, 0.4, 1.0e-6, 1.0e-6)
            .expect("joint shape uncertainty");
        let frame = term.atoms[0]
            .decoder_frame
            .as_ref()
            .expect("the fit must activate a Grassmann frame at p = 12");
        assert_eq!(frame.rank(), rank, "the frame must carry the decoder's rank");
        (term, target, rho, shape)
    }

    /// Checked after the values, so a fixed-frame covariance fails on the number.
    fn assert_integrates_learned_frames_2933_f35(shape: &SaeShapeUncertainty) {
        assert!(
            matches!(
                shape.operator,
                SaeShapeCovarianceOperator::ObservedInformation {
                    frame_conditioning: SaeFrameConditioning::MarginalOverLearnedFrames,
                    ..
                }
            ),
            "a small framed fit must integrate its frame: {:?}",
            shape.operator
        );
    }

    /// #2933 F35 — the audit's rank-one counterexample in general form. A framed
    /// decoder moves along `δB = δC·Uᵀ + C·δUᵀ`, and the lift
    /// `(I ⊗ U)·Cov(vec C)·(I ⊗ U)ᵀ` kept the first term only: every output axis
    /// outside the frame span reported zero variance, where the frame rotation
    /// contributes `C²·Var(δφ)` (0.09 for `C = 3`, `Var(δφ) = 0.01`).
    ///
    /// Here the decoder has rank 2 on a 3-column basis, so the rank constraint
    /// binds. The target lies in the frame span, so the residual, the decoder
    /// gradient and every coordinate coupling vanish off it. Along a transverse
    /// axis the observed information is exactly `(G + λS) ⊗ P_⊥` with
    /// `G = Σ_i φ_i φ_iᵀ`, and in the tangent coordinates `δU = U_⊥ W` the band
    /// variance there is `φ̂·ĉ(t)ᵀ (Cᵀ(G + λS)C)⁻¹ ĉ(t)` with `ĉ(t) = Cᵀφ(t)`.
    #[test]
    fn learned_frame_band_carries_transverse_orientation_variance_2933_f35() {
        let (term, _target, rho, shape) = fitted_framed_circle_2933_f35(2);
        let atom = &term.atoms[0];
        let frame = atom.decoder_frame.as_ref().expect("frame");
        let u = frame.frame();
        let r = u.ncols();
        let c_coords = atom.decoder_coefficients().dot(&u);
        let basis = &atom.basis_values;
        let precision =
            basis.t().dot(basis) + &(atom.smooth_penalty() * rho.log_lambda_smooth[0].exp());
        let projected = c_coords.t().dot(&precision).dot(&c_coords);
        let projected_inverse = projected
            .cholesky(Side::Lower)
            .expect("the frame coordinates carry full rank")
            .solve_mat(&Array2::<f64>::eye(r));
        let scale = shape.dispersion.posterior_covariance_scale();
        let band = shape.atoms[0].band_sd.as_ref().expect("model-based band");
        assert_eq!(band.nrows(), term.n_obs(), "every row is a band point at n = 24");
        let mut in_frame_variance = 0.0_f64;
        let mut transverse_variance = 0.0_f64;
        for row in 0..term.n_obs() {
            let c_hat = c_coords.t().dot(&basis.row(row));
            let expected = scale * c_hat.dot(&projected_inverse.dot(&c_hat));
            for c in 0..term.output_dim() {
                let frame_loading: f64 = (0..r).map(|j| u[[c, j]] * u[[c, j]]).sum();
                let got = band[[row, c]] * band[[row, c]];
                if frame_loading > 1.0e-12 {
                    in_frame_variance = in_frame_variance.max(got);
                    continue;
                }
                transverse_variance = transverse_variance.max(expected);
                assert!(
                    (got - expected).abs() <= 1.0e-6 * expected + 1.0e-15,
                    "row {row}, transverse channel {c}: band variance {got:.6e} must be the \
                     frame-rotation variance {expected:.6e}"
                );
            }
        }
        assert!(
            transverse_variance > 1.0e-3 * in_frame_variance,
            "the transverse orientation variance {transverse_variance:.3e} must be material \
             against the in-frame variance {in_frame_variance:.3e}"
        );
        assert_integrates_learned_frames_2933_f35(&shape);
    }

    /// #2933 F35 — at frame rank equal to the basis width there is no rank
    /// constraint: every decoder is a rank-`M` matrix, so `B = C·Uᵀ` is a pure
    /// factorization gauge of the unframed model at the same state. The decoder
    /// covariance integrated over the frame must equal the unframed observed
    /// information's `φ̂·[A⁺]_ββ`, formed here directly in `vec B` coordinates. The
    /// fixed-frame lift was zero outside the frame span.
    #[test]
    fn framed_covariance_equals_unframed_observed_information_at_full_rank_2933_f35() {
        let (term, target, rho, shape) = fitted_framed_circle_2933_f35(3);
        let mut unframed = term.clone();
        for atom in unframed.atoms.iter_mut() {
            atom.deactivate_decoder_frame();
        }
        let mut sys = unframed
            .assemble_arrow_schur(target.view(), &rho, None)
            .expect("unframed assembly at the fitted state");
        SaeManifoldTerm::ensure_row_gauge_deflation_for_quasi_laplace(&mut sys);
        let (_delta_t, _delta_beta, cache) = crate::manifold::solve_arrow_newton_step_with_options(
            &sys,
            0.0,
            0.0,
            &unframed.evidence_factor_options(),
        )
        .expect("frozen unframed evidence factor");
        let geometry = unframed
            .materialize_exact_stationarity_geometry(&rho, target.view(), &cache)
            .expect("unframed exact stationarity geometry");
        let covariance = match unframed
            .exact_observed_information_shape_covariance(&geometry, &rho, target.view(), &cache)
            .expect("unframed observed information")
        {
            SaeShapeInformation::ObservedInformation(covariance) => covariance,
            other => panic!("the unframed state must be a mode: {other:?}"),
        };
        let scale = shape.dispersion.posterior_covariance_scale();
        let expected = covariance.blocks[0].mapv(|v| v * scale);
        let got = shape.atoms[0]
            .decoder_covariance
            .as_ref()
            .expect("dense decoder covariance");
        assert_eq!(got.dim(), expected.dim(), "decoder covariance layout");
        let norm = expected.mapv(|v| v * v).sum().sqrt();
        let miss = (got - &expected).mapv(|v| v * v).sum().sqrt();
        assert!(norm > 0.0, "the unframed covariance must be nonzero");
        assert!(
            miss <= 1.0e-6 * norm,
            "framed covariance must equal the unframed observed information: \
             ‖Cov_framed − Cov_unframed‖_F = {miss:.3e} against {norm:.3e}"
        );
        let p = term.output_dim();
        let m = term.atoms[0].basis_size();
        let largest = (0..m * p).map(|i| expected[[i, i]]).fold(0.0_f64, f64::max);
        let transverse = (0..m)
            .flat_map(|b| (3..p).map(move |c| b * p + c))
            .map(|i| expected[[i, i]])
            .fold(0.0_f64, f64::max);
        assert!(
            transverse > 1.0e-3 * largest,
            "outside the frame span the unframed variance {transverse:.3e} must be material \
             against {largest:.3e}"
        );
        assert_integrates_learned_frames_2933_f35(&shape);
    }
}

mod tests_zero_decoder_entry_2822 {
    use crate::manifold::tests::trivial_k1_euclidean_term;

    /// #2822 — an entry refuses an identically zero decoder and names the atom.
    #[test]
    fn identically_zero_decoder_is_refused_at_entry_naming_the_atom_2822() {
        let mut term = trivial_k1_euclidean_term();
        let refusal = term.prepare_entry_stages();
        assert!(
            matches!(&refusal, Err(message) if message.contains("atom 0 'atom0'")
                && message.contains("identically zero decoder")),
            "an identically zero decoder must be refused at entry, naming the atom: {refusal:?}"
        );
    }

    /// #2822 — only exact zero is refused: a nonzero decoder, however small, spans a direction.
    #[test]
    fn nonzero_near_zero_decoder_is_admitted_at_entry_2822() {
        let mut term = trivial_k1_euclidean_term();
        term.atoms[0].decoder_coefficients_mut().fill(1.0e-300);
        let admitted = term.prepare_entry_stages();
        assert!(
            admitted.is_ok(),
            "a 1e-300 decoder is not identically zero, so the entry must admit it: {admitted:?}"
        );
    }
}
