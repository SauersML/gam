//! Finite-difference contract for the ALO-stabilization rho-gradient (#813/#821).
//!
//! Background. On a `te()` tensor basis at small n the ALO stabilization gate
//! fires (max hat-diagonal >= 0.80) and augments the REML cost+gradient. Before
//! the fix the augmented analytic outer gradient did NOT vanish where the
//! augmented cost is minimized (observed: cost flat while |g| stayed pinned /
//! grew), so the ARC outer optimizer never reached tolerance and ran the full
//! `max_iter=200` budget twice without converging. Root cause: the PSIS reweight
//! `influence_scale` and the dispersion `phi` are rho-dependent in the cost but
//! were held constant in the gradient. The fix freezes both per fit
//! (`ALO_FROZEN_NUISANCE` in src/solver/reml/runtime.rs), making the augmented
//! objective a fixed function of rho within the fit so the existing analytic
//! gradient is exactly its gradient.
//!
//! This test asserts that contract: at an ACTIVE rho, the analytic ALO gradient
//! must equal a central finite difference of the ALO cost to < 1e-5 relative,
//! using ONE shared `RemlState` for all three evaluations (analytic, +h, -h) so
//! `influence_scale`/`phi` are frozen once at `rho_active` and reused at
//! `rho +/- h` -- exactly what the optimizer sees.
//!
//! Wiring note for whoever lands this. `alo_stabilization_eval` is private to
//! `solver::reml::runtime`; the crate-internal shim
//! `RemlState::alo_stabilization_eval_for_test` (added under `#[cfg(test)]` in
//! that module) exposes it. Build the `RemlState` for `y ~ te(x1, x2)` (n = 30,
//! seed 0) the same way the in-crate REML tests do, obtain an `EvalShared`
//! bundle via `obtain_eval_bundle`, and fill in the body below. `rho_active`
//! must be a point where the gate fires (assert the eval returns `Some`); for
//! te(x1,x2) n=30 a wiggly point such as `[-2.0, -2.0]` activates it.
//!
//! Pseudocode (drop-in once the RemlState builder is wired):
//!
//!   let state = build_reml_state_te_x1_x2(/*n=*/30, /*seed=*/0);
//!   let rho_active = ndarray::array![-2.0_f64, -2.0_f64];
//!   let bundle = state.obtain_eval_bundle(&rho_active).unwrap();
//!   let analytic = state
//!       .alo_stabilization_eval_for_test(&rho_active, &bundle, true)
//!       .unwrap()
//!       .expect("ALO gate must be active at rho_active")
//!       .gradient
//!       .expect("analytic ALO gradient must be available");
//!   let h = 1e-6;
//!   for k in 0..rho_active.len() {
//!       let mut rp = rho_active.clone(); rp[k] += h;
//!       let mut rm = rho_active.clone(); rm[k] -= h;
//!       let bp = state.obtain_eval_bundle(&rp).unwrap();
//!       let bm = state.obtain_eval_bundle(&rm).unwrap();
//!       let cp = state.alo_stabilization_eval_for_test(&rp, &bp, false)
//!           .unwrap().expect("active").cost;
//!       let cm = state.alo_stabilization_eval_for_test(&rm, &bm, false)
//!           .unwrap().expect("active").cost;
//!       let fd = (cp - cm) / (2.0 * h);
//!       let rel = (analytic[k] - fd).abs() / fd.abs().max(1.0);
//!       assert!(
//!           rel < 1e-5,
//!           "ALO gradient[{k}] = {} disagrees with FD {} (rel {:.2e})",
//!           analytic[k], fd, rel
//!       );
//!   }

#[test]
#[ignore = "scaffold: wire the RemlState builder (see module doc) then remove #[ignore]"]
fn alo_stabilization_gradient_matches_central_fd_at_active_rho() {
    // Body intentionally empty until the crate-internal RemlState builder is
    // wired per the module doc above; marked #[ignore] so the scaffold compiles
    // and is discoverable via `cargo test -- --ignored` without failing CI.
}
