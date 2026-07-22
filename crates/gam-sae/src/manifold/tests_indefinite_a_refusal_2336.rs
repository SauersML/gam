//! #2336 — the indefinite exact-`A` refusal is an INFEASIBLE outer probe, not a
//! fatal abort. Companion to `tests_schur_seed_refusal_1782`, which pins the same
//! contract for the non-PD reduced-Schur refusal; this one covers the typed
//! `SaeCriterionError::IndefiniteObservedInformation` variant that #2330 Phase-2a
//! introduced when it made `½log|A|` the ranked value.

use super::tests::*;
use super::*;
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::{s, Array1, Array2};

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
/// after. This is the canonical E-attributable wrinkle-saddle specimen (same state
/// fix-2253 anchored as `converged_state_with_residual_a_saddle_2336`, now
/// documented as the PRICING specimen: its `λ+e_v(ARD)=+0.026` shows the clamp
/// alone lifts it, so it prices — it is NOT a genuine deep saddle). The genuine
/// refusal path is exercised by `genuine_saddle_is_infeasible_probe_not_fatal_2336`.
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
        priced.as_ref().map(|(value, _, _)| *value).map_err(|e| format!("{e:?}"))
    );

    let (term, target, rho) = ard_saddle_state();
    let rho_flat = rho.to_flat();
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

/// #2336 refusal companion — a GENUINE saddle (indefiniteness NOT attributable to
/// the bounded ARD concave-clamp: `λ+e_v < −floor`) must STILL return the typed
/// `IndefiniteObservedInformation` refusal, and the outer eval must price it as
/// `+inf` infeasible (not a fatal abort). The specimen is `obb_patchd_fixture`
/// at a window-scan saddle scale — verified genuine (E-non-attributable) by the
/// #2336 spectral scan (scale 0.02: `n_neg=1, attributable=0`). Guards the
/// "refuse ⟺ genuinely-indefinite" half of the value-side contract; the price
/// half is `e_attributable_ard_saddle_prices_finite_2336`.
#[test]
fn genuine_saddle_is_infeasible_probe_not_fatal_2336() {
    let (mut term, target, rho) =
        super::tests_logdet_adjoint_780::obb_patchd_fixture(0.02, -6.0);
    let refusal = term.penalized_quasi_laplace_criterion_with_cache(
        target.view(),
        &rho,
        None,
        40,
        0.4,
        1.0e-6,
        1.0e-6,
    );
    assert!(
        matches!(
            refusal,
            Err(SaeCriterionError::IndefiniteObservedInformation { block }) if block == "joint"
        ),
        "the genuine (non-E-attributable) saddle specimen must refuse on the joint block; got: {:?}",
        refusal.map(|(value, _, _)| value)
    );

    let (term, target, rho) =
        super::tests_logdet_adjoint_780::obb_patchd_fixture(0.02, -6.0);
    let rho_flat = rho.to_flat();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    match objective.eval(&rho_flat) {
        Ok(evaluation) => assert!(
            evaluation.cost.is_infinite() && evaluation.cost.is_sign_positive(),
            "a genuine saddle-ρ must price as +inf infeasible, got cost={}",
            evaluation.cost
        ),
        Err(err) => panic!(
            "#2336: a genuine indefinite exact A must be an INFEASIBLE probe the outer solver \
             can backtrack from, not a fatal abort; got: {err}"
        ),
    }
}

/// #2228 MEASUREMENT (zz_measure) — with the certify-at-best-seen fix (½λ²/scale-min
/// keyed, band unchanged), run the criterion on ard_saddle_state. Ok ⇒ the best-seen
/// certificate cleared the 1e-8 band and A is materialized at that certified mode
/// (report min_eig — PD ⇒ genuine convergence, retires the #2336 escape). Err ⇒ the
/// best-achievable ½λ²/scale plateaus above the band (a solver stall at a saddle-
/// adjacent mode), honestly reported at the best-seen ‖g‖.
#[test]
fn zz_measure_best_seen_classification_2228() {
    use super::{FaerEigh, Side};
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
            let (eigs, _) = a.eigh(Side::Lower).expect("A eigendecomposition");
            let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
            let max_eig = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "2228-MEASURE: Ok(value={value:.9e}) certified; min_eig={min_eig:.6e} max_eig={max_eig:.6e}"
            );
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
            eprintln!("2336-ITER{iter}: ACCEPTED — exact A is PD within the criterion floor; escaped");
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
     -> (ArrowFactorCache, SaeManifoldRho, SaeManifoldLoss, bool, ArrowSolveOptions) {
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
                term.apply_newton_step(dt.view(), db.view(), s).expect("trial");
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
    {
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
        let (obj_min, s_min, negate) = line_search(&mut term, &target, &rho, &dir_t, &dir_beta, obj_saddle);
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min).expect("step");
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
    }

    // ---- Experiment B: re-converge with gates held frozen-consistent. ----
    {
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
        let (_om, s_min, negate) = line_search(&mut term, &target, &rho, &dir_t, &dir_beta, obj_saddle);
        let dt = if negate { -&dir_t } else { dir_t.clone() };
        let db = if negate { -&dir_beta } else { dir_beta.clone() };
        term.apply_newton_step(dt.view(), db.view(), s_min).expect("step B");
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
                &SaeArrowVector { t: v_t.clone(), beta: v_beta.clone() },
            )
            .expect("apply ΔC");
        let vt_dc = v_t.dot(&dc.t) + v_beta.dot(&dc.beta);
        // vᵀ(B−A)v_full = −vt_dc; the t-coord fraction of ‖v‖² measures how
        // coord-localised (hence ARD-relevant) the direction is.
        let t_frac = (0..total_t).map(|j| v[j] * v[j]).sum::<f64>();
        let priced = lambda + e_v;
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
        if all_attributable { "FINITE (priced)" } else { "STILL REFUSED (genuine saddle remains)" }
    );
}
