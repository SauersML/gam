//! #2336 — the indefinite exact-`A` refusal is an INFEASIBLE outer probe, not a
//! fatal abort. Companion to `tests_schur_seed_refusal_1782`, which pins the same
//! contract for the non-PD reduced-Schur refusal; this one covers the typed
//! `SaeCriterionError::IndefiniteObservedInformation` variant that #2330 Phase-2a
//! introduced when it made `½log|A|` the ranked value.

use super::tests::*;
use super::*;
use gam_solve::rho_optimizer::OuterObjective;
use ndarray::Array2;

/// Reproduce the off-manifold, fixed-stratum state whose `B`-converged mode is an
/// exact-`A` SADDLE. This is the excitation the #2253/#2330 shared fixture uses:
/// the residual, entropy, and curvature-delta channels are all genuinely live, and
/// at this ρ some latent coordinates sit in the ARD periodic prior's CONCAVE half.
///
/// The majorizer clamps that curvature away (`atom.rs`: `hess =
/// psd_majorizer_hess + negative_hessian_remainder`, `psd_majorizer_hess =
/// max(hess, 0)`), so `A = B − E` with `E ⪰ 0` diagonal in the coordinate block,
/// carrying `|α·cos κt_ik|` with `α = e^{ρ_ard}`. `B ≻ 0` by construction, so the
/// inner Newton converges, while the exact `A` it does NOT see stays indefinite.
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

/// #2336 regression — an outer evaluation at a ρ whose exact observed information
/// is INDEFINITE must return the conventional `+inf` infeasible evaluation so the
/// optimizer backtracks, NOT a fatal error that kills the whole fit.
///
/// Before the fix, all four `IndefiniteObservedInformation` arms in
/// `outer_objective.rs` returned `Err`. Those arms were added as exhaustiveness
/// patches for the new enum variant (`223715e9b`, `40855deec`), not as a lane
/// policy — and they contradict both the criterion's own documented contract
/// ("makes saddle-ρ probe-infeasible (+inf) and steers the outer away") and the
/// doctrine `is_recoverable_value_probe_refusal` states for exactly this class:
/// the indefinite basin is adjacent to the PD optimum, so the outer solver must
/// read `+∞` and steer back rather than abort. RED before the fix (the `eval`
/// below returns `Err`), GREEN after.
#[test]
pub(crate) fn indefinite_exact_a_is_an_infeasible_probe_not_a_fatal_abort_2336() {
    let (mut term, target, rho) = ard_saddle_state();

    // The fixture must actually exercise the defect: at this ρ the criterion's
    // exact-A classification refuses on the joint block. If this ever stops
    // holding, the test below would pass vacuously, so assert it explicitly.
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
        "fixture must exercise the indefinite exact-A refusal on the joint block; got: {:?}",
        refusal.map(|(value, _, _)| value)
    );

    // The load-bearing assertion: the SAME ρ, driven through the outer objective's
    // evaluation lane, is an infeasible probe rather than a fatal error.
    let (term, target, rho) = ard_saddle_state();
    let rho_flat = rho.to_flat();
    let mut objective =
        SaeManifoldOuterObjective::new(term, target, None, rho, 40, 0.4, 1.0e-6, 1.0e-6);
    match objective.eval(&rho_flat) {
        Ok(evaluation) => assert!(
            evaluation.cost.is_infinite() && evaluation.cost.is_sign_positive(),
            "a saddle-ρ must price as +inf infeasible, got cost={}",
            evaluation.cost
        ),
        Err(err) => panic!(
            "#2336: an indefinite exact A must be an INFEASIBLE probe the outer solver can \
             backtrack from, not a fatal abort; got: {err}"
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
