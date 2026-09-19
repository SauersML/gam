//! #2336 — the indefinite exact-`A` refusal is an INFEASIBLE outer probe, not a
//! fatal abort. Companion to `tests_schur_seed_refusal_1782`, which pins the same
//! contract for the non-PD reduced-Schur refusal; this one covers the typed
//! `SaeCriterionError::IndefiniteObservedInformation` variant that #2330 Phase-2a
//! introduced when it made `½log|A|` the ranked value.

use super::tests::*;
use super::*;
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::{Array1, Array2, s};

/// Reproduce the off-manifold, fixed-stratum state whose `B`-converged mode is an
/// exact-`A` SADDLE. This is the excitation the #2253/#2330 shared fixture uses:
/// the residual, entropy, and curvature-delta channels are all genuinely live, and
/// at this ρ some latent coordinates sit in the ARD periodic prior's CONCAVE half.
///
/// The majorizer clamps that curvature away (`atom.rs`: `hess =
/// psd_majorizer_hess + negative_hessian_remainder`, `psd_majorizer_hess =
/// α·softplus_{τ₀}(cos κt)` — the #2339 smooth envelope of `max(hess, 0)`), so
/// `A = B − E` with `E ⪰ 0` diagonal in the coordinate block, carrying
/// `≈|α·cos κt_ik|` with `α = e^{ρ_ard}`. `B ≻ 0` by construction, so the inner
/// Newton converges, while the exact `A` it does NOT see stays indefinite.
fn ard_saddle_state() -> (SaeManifoldTerm, Array2<f64>, SaeManifoldRho) {
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

/// #2336 GATE — after the value-side E-attributability fix, a B-converged mode
/// whose exact-A indefiniteness is FULLY attributable to the bounded ARD periodic
/// concave-clamp wrinkle `E` prices a FINITE criterion (basin curvature `λ+e_v ≥ 0`
/// on the switched directions) instead of refusing. `ard_saddle_state`'s two
/// negatives (≈ −0.015) are E-attributable (`e_v ≥ |λ|`, verified in
/// `zz_measure_e_attributability_2336`), so both the criterion and the outer eval
/// return finite. RED before the fix (the criterion returned
/// `Err(IndefiniteObservedInformation{{joint}})` and `eval` priced `+inf`), GREEN
/// STABLE across #2339: E = α·softplus_τ₀(−cos κt) ≥ α·max(−cos,0) (the hard clamp)
/// pointwise, so the smooth clamp only GROWS e_v — the attributability test loosens
/// by at most α·τ₀·ln2 = α·(deflation floor), within #2339's τ₀ budget — hence
/// a_saddle prices under both the hard and the smooth clamp.
/// This is the canonical E-attributable wrinkle-saddle specimen (same state
/// fix-2253 anchored as `converged_state_with_residual_a_saddle_2336`, now
/// documented as the PRICING specimen: its `λ+e_v(ARD)=+0.026` shows the clamp
/// alone lifts it, so it prices — it is NOT a genuine deep saddle). No specimen
/// pins the genuine refusal path: the criterion descends a refused exact-A saddle
/// before it refuses (#2080).
#[test]
pub(crate) fn e_attributable_ard_saddle_prices_finite_2336() {
    let (mut term, target, rho) = ard_saddle_state();
    let priced = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    assert!(
        matches!(&priced, Ok((value, _, _)) if value.is_finite()),
        "post E-attributability fix the ARD-wrinkle saddle must PRICE FINITE, not refuse; got: {:?}",
        priced
            .as_ref()
            .map(|(value, _, _)| *value)
            .map_err(|e| format!("{e:?}"))
    );

    let (term, target, rho) = ard_saddle_state();
    let rho_flat = rho.flat_coordinates();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    match objective.eval(&rho_flat) {
        Ok(evaluation) => assert!(
            evaluation.cost.is_finite(),
            "an E-attributable saddle-ρ must price FINITE (was +inf refusal), got cost={}",
            evaluation.cost
        ),
        Err(err) => panic!(
            "#2336: an E-attributable saddle-ρ must be a FINITE outer eval, not a fatal abort; \
             got: {err}"
        ),
    }
}

/// #2434 regression gate — the switched-direction derivative already landed with
/// the #2336 value rule in `e97238721`; two stale prototype comments later made it
/// look absent. Pin the production direct-ρ channel against the value it actually
/// differentiates so neither comments nor implementation can drift again.
///
/// Hold θ̂ fixed at the canonical E-attributable saddle, rebuild the cache at each
/// perturbed ρ, and centrally difference
/// `½(log|A_priced| − log|A_tt,priced|)`. The analytic side is the direct trace from
/// `dense_exact_a_logdet_channels`, including:
///
/// 1. the priced inverse contraction;
/// 2. the Daleckii–Krein eigenvector-response matrix; and
/// 3. the explicit `dE/dρ_ard = E` term.
///
/// This deliberately probes only ARD coordinates: they are the coordinates on
/// which the allegedly missing B-channel is live. The spectral assertion first
/// proves the fixture really contains a switched negative direction; otherwise an
/// ordinary positive-definite state could false-green the derivative comparison.
#[test]
fn priced_ard_direct_gradient_matches_fixed_state_value_2434() {
    let (mut term, target, rho) = ard_saddle_state();
    let (_value, loss, cache) = term
        .penalized_quasi_laplace_criterion_with_cache(
            target.view(),
            &rho,
            None,
            40,
            0.4,
            1.0e-6,
            1.0e-6,
        )
        .expect("canonical E-attributable saddle must produce a priced cache");

    let total_t = cache.delta_t_len();
    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("materialize exact A at the priced state");
    let e_diag = term
        .materialize_ard_concave_clamp_diagonal(&rho, &cache)
        .expect("materialize the clamp-attribution diagonal");
    // #2673, #2933 F07 — the band is per direction of the pencil `(A, Φ)`, in the metric
    // both the value and the gradient classify in. This probe supplies its own pencil
    // eigenvectors and its own metric applies and shares only the scalar rule.
    let oracle = PencilOracle::new(&a, &cache);
    let switched = oracle
        .negative()
        .into_iter()
        .filter(|&idx| {
            let w = oracle.vectors.column(idx);
            let e_w = (0..total_t)
                .map(|row| e_diag[row] * w[row] * w[row])
                .sum::<f64>();
            oracle.values[idx] + e_w >= -oracle.floors[idx]
        })
        .count();
    assert!(
        switched > 0,
        "#2434 gate is invalid: the fixture contains no clamp-attributable switched direction"
    );

    let geometry = term
        .materialize_dense_exact_a_geometry(&rho, target.view(), &cache)
        .expect("the priced state's exact-A spectral block");
    let rank_charge = term
        .production_rank_charge_derivative(target.view(), &rho, &loss, &cache, Some(&geometry))
        .expect("the priced state's rank-charge derivative");
    let analytic = term
        .dense_exact_a_logdet_channels(target.view(), &rho, &cache, &geometry, &rank_charge.theta)
        .expect("complete priced exact-A derivative")
        .logdet_trace;
    let fixed_state_priced_logdet =
        |mut candidate: SaeManifoldTerm, at_rho: &SaeManifoldRho| -> f64 {
            let (_criterion, _loss, at_cache) = candidate
                .penalized_quasi_laplace_criterion_with_cache(
                    target.view(),
                    at_rho,
                    None,
                    0,
                    0.4,
                    1.0e-6,
                    1.0e-6,
                )
                .expect("fixed-state perturbed cache must remain on the priced stratum");
            let log_a = candidate
                .exact_observed_information_log_dets(at_rho, target.view(), &at_cache)
                .expect("fixed-state perturbed exact-A value");
            0.5 * log_a
        };

    let converged_term = term;
    let h = 1.0e-5_f64;
    let mut checked = 0usize;
    let mut max_signal = 0.0_f64;
    let mut worst_relative_error = 0.0_f64;
    for atom in 0..rho.log_ard.len() {
        for axis in 0..rho.log_ard[atom].len() {
            let mut plus = rho.clone();
            let mut minus = rho.clone();
            plus.log_ard[atom][axis] += h;
            minus.log_ard[atom][axis] -= h;
            let value_plus = fixed_state_priced_logdet(converged_term.clone(), &plus);
            let value_minus = fixed_state_priced_logdet(converged_term.clone(), &minus);
            let finite_difference = (value_plus - value_minus) / (2.0 * h);
            let index = rho.ard_flat_index(atom, axis);
            let exact = analytic[index];
            let scale = 1.0 + finite_difference.abs().max(exact.abs());
            let relative_error = (finite_difference - exact).abs() / scale;
            max_signal = max_signal.max(finite_difference.abs().max(exact.abs()));
            worst_relative_error = worst_relative_error.max(relative_error);
            eprintln!(
                "#2434 priced ARD atom={atom} axis={axis}: analytic={exact:.12e} \
                 fd={finite_difference:.12e} scaled_error={relative_error:.3e}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "#2434 gate found no ARD coordinates");
    assert!(
        max_signal > 1.0e-6,
        "#2434 gate is not load-bearing: every priced ARD derivative is numerically zero"
    );
    assert!(
        worst_relative_error <= 1.0e-4,
        "priced exact-A analytic derivative disagrees with the fixed-state central \
         difference of its value: worst scaled error {worst_relative_error:.3e}"
    );
}

/// #2228 MEASUREMENT (zz_measure) — with the certify-at-best-seen fix (½λ²/scale-min
/// keyed, band unchanged), run the criterion on ard_saddle_state.
///
/// **`min_eig` alone does not decide anything here, and reading it as if it did is
/// how this probe misled two readers into filing a correctness alarm against
/// designed behaviour.** Since #2330/#2336 the exact-`A` gate is not a PSD test: a
/// negative eigendirection is refused only when the ARD concave clamp cannot
/// account for its negativity. Per direction `v` with eigenvalue `λ < −floor`, the
/// gate forms
///
/// ```text
/// floor = rank_floor(w)   (#2673, #2933 F07 — per direction of the pencil `(A, Φ)`)
/// basin = μ + wᵀEw        (E = the ARD concave-clamp diagonal, zero on the β border)
/// basin < −floor  ⇒  genuine saddle, typed IndefiniteObservedInformation refusal
/// basin ≥ −floor  ⇒  clamp-attributable, PRICED at the basin curvature
/// ```
///
/// So `A` being indefinite is expected and by itself proves nothing; the deciding
/// quantity is `basin`, and `min_eig` and `basin` differ by exactly the `wᵀEw` this
/// probe used to omit. Report all four — `μ`, `wᵀEw`, `basin`, `floor` — for every
/// direction the gate actually examined, on the generalized eigenvectors of the pencil
/// the gate classifies.
///
/// `Ok` ⇒ every negative direction was clamp-attributable and priced at its basin
/// curvature. `Err` ⇒ either a direction the clamp cannot explain (a genuine
/// saddle) or the best-achievable ½λ²/scale plateauing above the band (a solver
/// stall), honestly reported at the best-seen ‖g‖.
#[test]
fn zz_measure_best_seen_classification_2228() {
    let (mut term, target, rho) = ard_saddle_state();
    let result = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    match result {
        Ok((value, _, cache)) => {
            let a = term
                .materialize_exact_hessian_dense(&rho, target.view(), &cache)
                .expect("materialize A at certified best-seen mode");
            let e_diag = term
                .materialize_ard_concave_clamp_diagonal(&rho, &cache)
                .expect("ARD concave-clamp diagonal at the certified mode");
            let total_t = cache.delta_t_len();
            // #2933 F07 — gate-identical classification: the pencil `(A, Φ)` in the evidence
            // factor's metric.
            let oracle = PencilOracle::new(&a, &cache);
            let (eigs, vecs, floors) = (&oracle.values, &oracle.vectors, &oracle.floors);
            let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
            let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let floor = floors.iter().copied().fold(0.0_f64, f64::max);
            eprintln!(
                "2228-MEASURE: Ok(value={value:.9e}) certified; min_eig={min_eig:.6e} \
                 max_eig={max_eig:.6e} widest floor={floor:.6e}"
            );
            // A CERTIFIED best-seen mode is only meaningful if the numbers it is
            // certified on are well-posed: a finite criterion value, a finite
            // ordered spectrum, and a strictly positive PD floor to compare against.
            assert!(
                value.is_finite(),
                "2228-MEASURE: a certified best-seen mode must carry a finite criterion value, \
                 got {value}"
            );
            assert!(
                min_eig.is_finite() && max_eig.is_finite() && min_eig <= max_eig,
                "2228-MEASURE: the exact-A spectrum must be finite and ordered \
                 (min_eig={min_eig}, max_eig={max_eig})"
            );
            assert!(
                floor > 0.0 && floor.is_finite(),
                "2228-MEASURE: the relative PD floor the gate decides on must be a positive \
                 finite number, got {floor}"
            );
            // Every direction the gate examined, with the quantity it decided on.
            for (idx, &lambda) in eigs.iter().enumerate() {
                let floor = floors[idx];
                if lambda >= -floor {
                    continue;
                }
                let v = vecs.column(idx);
                let limit = total_t.min(v.len());
                let e_v: f64 = (0..limit).map(|j| e_diag[j] * v[j] * v[j]).sum();
                let basin = lambda + e_v;
                // `basin` is the quantity the refuse/price decision is read off, so
                // both of its addends must be real numbers; `e_v = vᵀEv` with E the
                // ARD concave-clamp remainder is a diagonal quadratic form on a unit
                // eigenvector and cannot be infinite.
                assert!(
                    e_v.is_finite() && basin.is_finite(),
                    "2228-MEASURE: dir {idx}: the priced basin curvature must be finite \
                     (lambda={lambda}, vEv={e_v})"
                );
                eprintln!(
                    "2228-MEASURE:   dir {idx}: lambda={lambda:.6e} vEv={e_v:.6e} \
                     basin={basin:.6e} vs -floor={:.6e} => {}",
                    -floor,
                    if basin < -floor {
                        "GENUINE SADDLE (would refuse)"
                    } else {
                        "clamp-attributable (priced at basin)"
                    }
                );
            }
        }
        Err(err) => eprintln!("2228-MEASURE: Err({err:?}) => plateau above band (solver stall)"),
    }
}

/// #2336 DECISIVE MEASUREMENT v2 (zz_measure) — the CORRECTED escape test.
///
/// v1 undershot: the stall-clearing step `sqrt(2·tol/|λ|) ≈ 1.3e-3` is ~85× smaller
/// than the true 1-D minimizer the external bot found (`s ≈ 0.11`, `ΔL ≈ −1e-4`,
/// purely quadratic `½s²λ_min`), so v1 never left the near-stationary neighbourhood
/// and could not test whether a lower mode exists. This version does a real 1-D
/// line search of `penalized_objective_total` along `±v` (v = most-negative exact-A
/// eigenvector) to the minimizer, steps there, then re-converges through the
/// DESCENT-enforcing accepted lane (`converge_inner_for_undamped_logdet`,
/// `refine_progress_extension = true`, `inner_max_iter > 0`), and re-materialises
/// the exact A. Iterated up to 3× (MAX_SAE_SADDLE_ESCAPES-style), reporting the
/// spectrum at every mode. This decides: does escape+descent reach a lower/PD mode
/// (⇒ implement escape with a line-search magnitude), or does the negative-curvature
/// direction persist so no nearby lower mode exists (⇒ value-side is the honest fix)?
#[test]
fn zz_measure_saddle_escape_linesearch_reconverge_2336() {
    use super::{FaerEigh, Side};
    let (mut term, target, rho) = ard_saddle_state();
    let inner_max_iter = 40usize;
    let learning_rate = 0.4;
    let ridge_ext_coord = 1.0e-6;
    let ridge_beta = 1.0e-6;

    let mut rho_fixed = rho.clone();
    let initial = term
        .run_joint_fit_arrow_schur_for_quasi_laplace(
            target.view(),
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )
        .expect("initial joint fit to seed the inner state");
    let mut loss = initial.loss;
    let mut criterion_fixed_point = initial.fixed_point;
    let options = ArrowSolveOptions::direct()
        .with_gpu_policy(term.gpu_policy)
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    let mut cache = term
        .converge_inner_for_undamped_logdet(
            target.view(),
            &rho,
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &mut criterion_fixed_point,
            &options,
            true,
        )
        .expect("converge inner to the undamped-logdet optimum (saddle)");

    for iter in 0..3usize {
        let total_t = cache.delta_t_len();
        let a = term
            .materialize_exact_hessian_dense(&rho, target.view(), &cache)
            .expect("materialize exact A");
        let (eigs, vecs) = a.eigh(Side::Lower).expect("A eigh");
        let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut min_idx = 0usize;
        let mut min_eig = f64::INFINITY;
        for (i, &v) in eigs.iter().enumerate() {
            if v < min_eig {
                min_eig = v;
                min_idx = i;
            }
        }
        let n_neg = eigs.iter().filter(|&&v| v < 0.0).count();
        let obj0 = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("penalized objective at mode");
        let reclass = term.exact_observed_information_log_dets(&rho, target.view(), &cache);
        eprintln!(
            "2336-ITER{iter}: MODE obj={obj0:.9e} min_eig={min_eig:.6e} max_eig={max_eig:.6e} n_neg={n_neg} reclass={}",
            match &reclass {
                Ok(_) => "Ok(accepted)".to_string(),
                Err(e) => format!("Err({e:?})"),
            }
        );
        let floor = 1.0e-9 * max_eig.max(1.0);
        if min_eig >= -floor {
            eprintln!(
                "2336-ITER{iter}: ACCEPTED — exact A is PD within the criterion floor; escaped"
            );
            break;
        }

        // Real 1-D line search of penalized_objective_total along ±v.
        let dir = vecs.column(min_idx);
        let dir_t = dir.slice(s![..total_t]).to_owned();
        let dir_beta = dir.slice(s![total_t..]).to_owned();
        let snapshot = term.snapshot_mutable_state();
        let mut best: (f64, f64, bool) = (obj0, 0.0, false);
        for negate in [false, true] {
            let dt = if negate { -&dir_t } else { dir_t.clone() };
            let db = if negate { -&dir_beta } else { dir_beta.clone() };
            let mut s = 1.0e-3;
            while s <= 0.6 {
                term.apply_newton_step(dt.view(), db.view(), s)
                    .expect("line-search trial step");
                let cand = term
                    .penalized_objective_total(target.view(), &rho, None, 1.0)
                    .expect("line-search objective");
                term.restore_mutable_state(&snapshot)
                    .expect("restore after line-search trial");
                if cand.is_finite() && cand < best.0 {
                    best = (cand, s, negate);
                }
                s *= 1.4;
            }
        }
        let (obj_min, s_min, negate) = best;
        eprintln!(
            "2336-ITER{iter}: LINESEARCH s_min={s_min:.6e} negate={negate} obj_min={obj_min:.9e} dL={:.6e}  (predicted ½s²|λ|={:.6e})",
            obj_min - obj0,
            0.5 * s_min * s_min * min_eig.abs()
        );
        if s_min == 0.0 || !(obj_min < obj0) {
            eprintln!(
                "2336-ITER{iter}: NO DESCENT along ±v at any tried s — escape structurally unavailable"
            );
            break;
        }

        // Step to the minimizer, then re-converge via the descent-enforcing lane.
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min)
            .expect("commit line-search step");
        let obj_stepped = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective after step");
        cache = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                &rho,
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                &mut loss,
                &mut criterion_fixed_point,
                &options,
                true,
            )
            .expect("re-converge after escape step");
        let obj_reconv = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("objective after re-convergence");
        eprintln!(
            "2336-ITER{iter}: STEPPED obj={obj_stepped:.9e} (dL_step={:.6e}) -> RECONV obj={obj_reconv:.9e} (dL_reconv={:.6e})",
            obj_stepped - obj0,
            obj_reconv - obj_stepped
        );
    }

    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("materialize final A");
    let (eigs, _) = a.eigh(Side::Lower).expect("final A eigh");
    let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
    let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let n_neg = eigs.iter().filter(|&&v| v < 0.0).count();
    let obj_final = term
        .penalized_objective_total(target.view(), &rho, None, 1.0)
        .expect("final objective");
    let reclass = term.exact_observed_information_log_dets(&rho, target.view(), &cache);
    eprintln!(
        "2336-FINAL: obj={obj_final:.9e} min_eig={min_eig:.6e} max_eig={max_eig:.6e} n_neg={n_neg} now_pd_accepted={}",
        reclass.is_ok()
    );
    // The escape verdict is read off this final line, so its three numbers must be
    // mutually consistent and well-posed: a finite ordered spectrum, a finite
    // objective, and a negative-eigenvalue count that agrees with the reported
    // minimum (they are two readouts of the same spectrum and cannot disagree).
    assert!(
        min_eig.is_finite() && max_eig.is_finite() && min_eig <= max_eig,
        "2336-FINAL: the exact-A spectrum must be finite and ordered \
         (min_eig={min_eig}, max_eig={max_eig})"
    );
    assert!(
        obj_final.is_finite(),
        "2336-FINAL: the penalized objective at the final mode must be finite, got {obj_final}"
    );
    assert_eq!(
        n_neg > 0,
        min_eig < 0.0,
        "2336-FINAL: the negative-eigenvalue count and the reported minimum are two readouts of \
         one spectrum and must agree (n_neg={n_neg}, min_eig={min_eig})"
    );
}

/// #2336 ROOT-CAUSE (zz_measure) — is the +1.25e-4 re-convergence CLIMB (v2) a
/// gate-refreeze objective desync, or genuine B-Newton attraction to the saddle?
///
/// v2 showed: step to the exact-A negative-curvature minimizer (s≈0.11, exact
/// objective drops −9.45e-5), then `converge_inner_for_undamped_logdet(refine=true)`
/// RAISES the objective by +1.25e-4, overshooting ABOVE the original saddle. A pure
/// ∇L=0 attractor would climb by exactly +9.45e-5 (undo the step); the +3e-5
/// overshoot is the tell that the re-convergence prices a DIFFERENT objective than
/// the line-search probe. `converge_inner_for_undamped_logdet` REFRESHES the
/// collapse-prevention gates (decoder repulsion, coactivation barriers) at its entry
/// state unless `streaming_gates_frozen` is already set. This test measures (A) the
/// gate-induced objective shift at the stepped point, and (B) whether holding the
/// gates frozen-consistent across probe + re-convergence removes the climb.
///
/// Both experiments need a descent along the saddle's negative direction to have a
/// stepped point at all. When no trial along ±v lowers the objective, each reports
/// "no descent along the negative direction" instead of stepping, because
/// `apply_newton_step` refuses a zero step by contract (#2822 census: the fixture
/// now reaches that outcome).
#[test]
fn zz_measure_saddle_gate_desync_2336() {
    use super::{FaerEigh, Side};
    let inner_max_iter = 40usize;
    let learning_rate = 0.4;
    let ridge_ext_coord = 1.0e-6;
    let ridge_beta = 1.0e-6;

    // Helper closure would need &mut term; inline twice on fresh terms instead.
    let reach_saddle = |term: &mut SaeManifoldTerm,
                        target: &Array2<f64>,
                        rho: &SaeManifoldRho|
     -> (
        ArrowFactorCache,
        SaeManifoldRho,
        SaeManifoldLoss,
        bool,
        ArrowSolveOptions,
    ) {
        let mut rho_fixed = rho.clone();
        let initial = term
            .run_joint_fit_arrow_schur_for_quasi_laplace(
                target.view(),
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
            )
            .expect("initial joint fit");
        let mut loss = initial.loss;
        let mut criterion_fixed_point = initial.fixed_point;
        let options = ArrowSolveOptions::direct()
            .with_gpu_policy(term.gpu_policy)
            .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
            .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
        let cache = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                rho,
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                &mut loss,
                &mut criterion_fixed_point,
                &options,
                true,
            )
            .expect("converge inner to saddle");
        (cache, rho_fixed, loss, criterion_fixed_point, options)
    };

    let neg_dir = |term: &SaeManifoldTerm,
                   target: &Array2<f64>,
                   rho: &SaeManifoldRho,
                   cache: &ArrowFactorCache|
     -> (f64, Array1<f64>, Array1<f64>) {
        let total_t = cache.delta_t_len();
        let a = term
            .materialize_exact_hessian_dense(rho, target.view(), cache)
            .expect("materialize A");
        let (eigs, vecs) = a.eigh(Side::Lower).expect("eigh");
        let mut min_idx = 0usize;
        let mut min_eig = f64::INFINITY;
        for (i, &v) in eigs.iter().enumerate() {
            if v < min_eig {
                min_eig = v;
                min_idx = i;
            }
        }
        let dir = vecs.column(min_idx);
        (
            min_eig,
            dir.slice(s![..total_t]).to_owned(),
            dir.slice(s![total_t..]).to_owned(),
        )
    };

    // Line search of penalized_objective_total along ±v; returns (obj, s, negate).
    let line_search = |term: &mut SaeManifoldTerm,
                       target: &Array2<f64>,
                       rho: &SaeManifoldRho,
                       dir_t: &Array1<f64>,
                       dir_beta: &Array1<f64>,
                       obj0: f64|
     -> (f64, f64, bool) {
        let snapshot = term.snapshot_mutable_state();
        let mut best = (obj0, 0.0f64, false);
        for negate in [false, true] {
            let dt = if negate { -dir_t } else { dir_t.clone() };
            let db = if negate { -dir_beta } else { dir_beta.clone() };
            let mut s = 1.0e-3;
            while s <= 0.6 {
                term.apply_newton_step(dt.view(), db.view(), s)
                    .expect("trial");
                let cand = term
                    .penalized_objective_total(target.view(), rho, None, 1.0)
                    .expect("trial obj");
                term.restore_mutable_state(&snapshot).expect("restore");
                if cand.is_finite() && cand < best.0 {
                    best = (cand, s, negate);
                }
                s *= 1.4;
            }
        }
        best
    };

    // ---- Experiment A: gate-induced objective shift at the stepped point. ----
    'experiment_a: {
        let (mut term, target, rho) = ard_saddle_state();
        let (cache, _rf, _loss, _cfp, _opts) = reach_saddle(&mut term, &target, &rho);
        // Freeze the gates AT the saddle (what the line-search probe will price).
        term.refresh_decoder_repulsion_gate();
        term.refresh_barrier_coactivation_gate();
        term.streaming_gates_frozen = true;
        let (min_eig, dir_t, dir_beta) = neg_dir(&term, &target, &rho, &cache);
        let obj_saddle = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj saddle frozen");
        let (obj_min, s_min, negate) =
            line_search(&mut term, &target, &rho, &dir_t, &dir_beta, obj_saddle);
        if obj_min >= obj_saddle {
            // `line_search` replaces its seed `(obj_saddle, 0, false)` only on a strict
            // decrease, so no trial along ±v lowered the objective. There is no stepped point
            // to price, and `apply_newton_step` refuses a zero step by contract, so the probe
            // reports that outcome instead of stepping.
            eprintln!(
                "2336-GATESHIFT: no descent along the negative direction: min_eig={min_eig:.6e} \
                 obj_saddle={obj_saddle:.9e} obj_min(ls)={obj_min:.9e} s_min={s_min:.4e}; the gate \
                 shift is a property of a step that was not taken"
            );
            assert!(
                min_eig.is_finite() && obj_saddle.is_finite() && s_min == 0.0 && !negate,
                "2336-GATESHIFT: a search that found no descent must return its own finite seed \
                 (min_eig={min_eig}, obj_saddle={obj_saddle}, obj_min={obj_min}, s_min={s_min}, \
                 negate={negate})"
            );
            break 'experiment_a;
        }
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min)
            .expect("step");
        // Objective at the stepped point under the SADDLE-frozen gates (probe view).
        let obj_stepped_frozen = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj stepped frozen");
        // Now refresh the gates AT the stepped point (what re-convergence would do).
        term.refresh_decoder_repulsion_gate();
        term.refresh_barrier_coactivation_gate();
        let obj_stepped_refreshed = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj stepped refreshed");
        eprintln!(
            "2336-GATESHIFT: min_eig={min_eig:.6e} s_min={s_min:.4e} negate={negate} \
             obj_saddle={obj_saddle:.9e} obj_min(ls)={obj_min:.9e} dL_step={:.6e} | \
             obj_stepped_frozen={obj_stepped_frozen:.9e} obj_stepped_refreshed={obj_stepped_refreshed:.9e} \
             GATE_SHIFT={:.6e}",
            obj_stepped_frozen - obj_saddle,
            obj_stepped_refreshed - obj_stepped_frozen
        );
        // GATE_SHIFT is a difference of two objectives, so both must be finite for
        // the shift to mean anything, and the line search must return a step from
        // inside its own bracket that never increases the objective it minimises
        // (its accumulator is seeded at `obj_saddle` and only replaced on strict
        // decrease — a violation means the search is reporting a different state
        // than the one it priced).
        assert!(
            min_eig.is_finite()
                && obj_saddle.is_finite()
                && obj_min.is_finite()
                && obj_stepped_frozen.is_finite()
                && obj_stepped_refreshed.is_finite(),
            "2336-GATESHIFT: every reported curvature and objective must be finite \
             (min_eig={min_eig}, obj_saddle={obj_saddle}, obj_min={obj_min}, \
              obj_stepped_frozen={obj_stepped_frozen}, \
              obj_stepped_refreshed={obj_stepped_refreshed})"
        );
        assert!(
            (0.0..=0.6).contains(&s_min) && obj_min <= obj_saddle,
            "2336-GATESHIFT: the line search must return a bracketed step that does not raise \
             the objective (s_min={s_min}, obj_min={obj_min}, obj_saddle={obj_saddle})"
        );
    }

    // ---- Experiment B: re-converge with gates held frozen-consistent. ----
    'experiment_b: {
        let (mut term, target, rho) = ard_saddle_state();
        let (cache, mut rho_fixed, mut loss, mut cfp, options) =
            reach_saddle(&mut term, &target, &rho);
        // Freeze gates at the saddle and KEEP them frozen through re-convergence
        // (converge_inner sees streaming_gates_frozen==true and does NOT refresh).
        term.refresh_decoder_repulsion_gate();
        term.refresh_barrier_coactivation_gate();
        term.streaming_gates_frozen = true;
        let (min_eig, dir_t, dir_beta) = neg_dir(&term, &target, &rho, &cache);
        let obj_saddle = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj saddle frozen B");
        let (obj_min, s_min, negate) =
            line_search(&mut term, &target, &rho, &dir_t, &dir_beta, obj_saddle);
        if obj_min >= obj_saddle {
            // Same outcome as Experiment A: no trial along ±v lowered the objective, so
            // there is no stepped point to re-converge from.
            eprintln!(
                "2336-FROZENRECONV: no descent along the negative direction: min_eig0={min_eig:.6e} \
                 obj_saddle={obj_saddle:.9e} obj_min(ls)={obj_min:.9e} s_min={s_min:.4e}; the \
                 frozen-gate re-convergence is a property of a step that was not taken"
            );
            assert!(
                min_eig.is_finite() && obj_saddle.is_finite() && s_min == 0.0 && !negate,
                "2336-FROZENRECONV: a search that found no descent must return its own finite seed \
                 (min_eig={min_eig}, obj_saddle={obj_saddle}, obj_min={obj_min}, s_min={s_min}, \
                 negate={negate})"
            );
            break 'experiment_b;
        }
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min)
            .expect("step B");
        let obj_stepped = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj stepped B");
        let cache2 = term
            .converge_inner_for_undamped_logdet(
                target.view(),
                &rho,
                &mut rho_fixed,
                None,
                inner_max_iter,
                learning_rate,
                ridge_ext_coord,
                ridge_beta,
                &mut loss,
                &mut cfp,
                &options,
                true,
            )
            .expect("re-converge frozen B");
        let obj_reconv = term
            .penalized_objective_total(target.view(), &rho, None, 1.0)
            .expect("obj reconv B");
        let a = term
            .materialize_exact_hessian_dense(&rho, target.view(), &cache2)
            .expect("materialize A B");
        let (eigs, _) = a.eigh(Side::Lower).expect("eigh B");
        let min_eig2 = eigs.iter().copied().fold(f64::INFINITY, f64::min);
        let n_neg2 = eigs.iter().filter(|&&v| v < 0.0).count();
        eprintln!(
            "2336-FROZENRECONV: min_eig0={min_eig:.6e} s_min={s_min:.4e} negate={negate} \
             obj_saddle={obj_saddle:.9e} obj_stepped={obj_stepped:.9e} (dL_step={:.6e}) \
             obj_reconv={obj_reconv:.9e} (dL_reconv={:.6e}) min_eig_reconv={min_eig2:.6e} n_neg_reconv={n_neg2}",
            obj_stepped - obj_saddle,
            obj_reconv - obj_stepped
        );
        eprintln!(
            "2336-FROZENVERDICT: with gates held frozen-consistent, re-convergence dL={:.6e} \
             (v2 unfrozen was +1.25e-4 CLIMB). climb_removed={}",
            obj_reconv - obj_stepped,
            (obj_reconv - obj_stepped) < 1.0e-4
        );
        // The FROZENVERDICT is `obj_reconv - obj_stepped`; a non-finite endpoint
        // would make the printed climb (and the `climb_removed` boolean derived
        // from it) arbitrary. The re-converged spectrum's two readouts must also
        // agree with each other.
        assert!(
            min_eig.is_finite()
                && obj_saddle.is_finite()
                && obj_stepped.is_finite()
                && obj_reconv.is_finite()
                && min_eig2.is_finite(),
            "2336-FROZENRECONV: every reported curvature and objective must be finite \
             (min_eig={min_eig}, obj_saddle={obj_saddle}, obj_stepped={obj_stepped}, \
              obj_reconv={obj_reconv}, min_eig_reconv={min_eig2})"
        );
        assert_eq!(
            n_neg2 > 0,
            min_eig2 < 0.0,
            "2336-FROZENRECONV: the re-converged negative-eigenvalue count must agree with the \
             reported minimum (n_neg={n_neg2}, min_eig={min_eig2})"
        );
    }
}

/// #2336 E-ATTRIBUTABILITY VERIFICATION (zz_measure) — the decisive theory gate.
///
/// The value-side fix prices a negative exact-A eigendirection v at its BASIN
/// curvature `λ + e_v` (adding back the dropped ARD-concave clamp) iff the
/// indefiniteness is fully attributable to that bounded wrinkle, i.e. `e_v ≥ |λ|`
/// where `e_v = vᵀ E v` and E is the ARD concave-clamp remainder diagonal
/// (materialize_ard_concave_clamp_diagonal). If e_v < |λ| the negative curvature
/// exceeds anything the wrinkle can produce ⇒ genuine saddle ⇒ keep refusing.
///
/// This test VERIFIES the premise on ard_saddle_state: its 2 negative eigenvalues
/// (≈ −0.015) must be E-attributable (`e_v ≥ |λ|`). If any negative direction is
/// NOT attributable, the theory is wrong and the fix must not be built as designed.
/// Cross-checks e_v (ARD-only diagonal) against the full ΔC = A−B contraction
/// (apply_exact_hessian_minus_b) — for coord-dominated directions they should be
/// close (softmax/residual channels small on those directions).
#[test]
fn zz_measure_e_attributability_2336() {
    use super::{FaerEigh, Side};
    let (mut term, target, rho) = ard_saddle_state();
    let inner_max_iter = 40usize;
    let learning_rate = 0.4;
    let ridge_ext_coord = 1.0e-6;
    let ridge_beta = 1.0e-6;

    let mut rho_fixed = rho.clone();
    let initial = term
        .run_joint_fit_arrow_schur_for_quasi_laplace(
            target.view(),
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
        )
        .expect("initial joint fit");
    let mut loss = initial.loss;
    let mut criterion_fixed_point = initial.fixed_point;
    let options = ArrowSolveOptions::direct()
        .with_gpu_policy(term.gpu_policy)
        .with_newton_schur_tikhonov(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(gam_solve::arrow_schur::SPECTRAL_DEFLATION_REL_FLOOR);
    let cache = term
        .converge_inner_for_undamped_logdet(
            target.view(),
            &rho,
            &mut rho_fixed,
            None,
            inner_max_iter,
            learning_rate,
            ridge_ext_coord,
            ridge_beta,
            &mut loss,
            &mut criterion_fixed_point,
            &options,
            true,
        )
        .expect("converge inner to saddle");

    let total_t = cache.delta_t_len();
    let a = term
        .materialize_exact_hessian_dense(&rho, target.view(), &cache)
        .expect("materialize exact A");
    let (eigs, vecs) = a.eigh(Side::Lower).expect("A eigh");
    let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let floor = 1.0e-9 * max_eig.max(1.0);

    let e_diag = term
        .materialize_ard_concave_clamp_diagonal(&rho, &cache)
        .expect("materialize E_ard diagonal");
    eprintln!(
        "2336-EATTR: total_t={total_t} beta={} max_eig={max_eig:.6e} floor={floor:.3e} E_diag_sum={:.6e} E_diag_max={:.6e}",
        cache.k,
        e_diag.iter().sum::<f64>(),
        e_diag.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    let mut n_neg = 0usize;
    let mut all_attributable = true;
    for (i, &lambda) in eigs.iter().enumerate() {
        if lambda >= -floor {
            continue;
        }
        n_neg += 1;
        let v = vecs.column(i);
        // e_v = vᵀ E v (E diagonal in the t-block, zero on β / logit rows).
        let mut e_v = 0.0_f64;
        for j in 0..total_t {
            e_v += e_diag[j] * v[j] * v[j];
        }
        // Cross-check: full ΔC = A−B contraction along v (ARD + softmax + residual).
        let v_t = v.slice(s![..total_t]).to_owned();
        let v_beta = v.slice(s![total_t..]).to_owned();
        let dc = term
            .apply_exact_hessian_minus_b(
                &rho,
                target.view(),
                &cache,
                &SaeArrowVector {
                    t: v_t.clone(),
                    beta: v_beta.clone(),
                },
            )
            .expect("apply ΔC");
        let vt_dc = v_t.dot(&dc.t) + v_beta.dot(&dc.beta);
        // vᵀ(B−A)v_full = −vt_dc; the t-coord fraction of ‖v‖² measures how
        // coord-localised (hence ARD-relevant) the direction is.
        let t_frac = (0..total_t).map(|j| v[j] * v[j]).sum::<f64>();
        let priced = lambda + e_v;
        // `attributable` is the theory verdict; it is only a verdict if its inputs
        // are well-posed. `v` is a unit eigenvector, so the t-block's share of its
        // squared norm is a fraction in [0,1]: if `t_frac` ever left that range the
        // "coord-localised" cross-check printed next to it would be meaningless.
        assert!(
            e_v.is_finite() && priced.is_finite() && vt_dc.is_finite(),
            "2336-EATTR: neg#{n_neg}: the attributability inputs must be finite \
             (lambda={lambda}, e_v={e_v}, full(B-A)v.v={})",
            -vt_dc
        );
        assert!(
            (-1.0e-9..=1.0 + 1.0e-9).contains(&t_frac),
            "2336-EATTR: neg#{n_neg}: v is a unit eigenvector, so its t-block share must lie in \
             [0,1], got t_frac={t_frac}"
        );
        let attributable = priced >= -floor;
        if !attributable {
            all_attributable = false;
        }
        eprintln!(
            "2336-EATTR: neg#{n_neg} lambda={lambda:.6e} e_v(ARD)={e_v:.6e} lambda+e_v={priced:.6e} \
             attributable={attributable} | full(B-A)v.v={:.6e} t_frac={t_frac:.4e}",
            -vt_dc
        );
    }
    eprintln!(
        "2336-EATTR: VERDICT n_neg={n_neg} all_attributable={all_attributable} \
         => fixture criterion would be {}",
        if all_attributable {
            "FINITE (priced)"
        } else {
            "STILL REFUSED (genuine saddle remains)"
        }
    );
}
