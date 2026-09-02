#![cfg(test)]
//! #2629 scope item 2 — settle the custom-family engine's row of the objective
//! table by MEASUREMENT.
//!
//! #2629 lists seven outer-objective families and asks which of them carry the
//! soft ρ-guard barrier that #2545 taught the certificate to subtract. Three of
//! those rows — `gamlss mean-wiggle`, `spatial-adaptive`, and `custom family` —
//! are the same evaluator seen from three call sites:
//! [`evaluate_custom_family_joint_hyper_owned`]. So one measurement settles
//! three rows, and it is this one.
//!
//! The issue's own evidence for those rows was a call-graph argument:
//! `RemlState::build_prior` is the only site that adds
//! `soft_rho_guard_prior_atom`'s gradient to a criterion, its only callers are
//! `RemlState` methods, and this engine holds no `RemlState`. That argument is
//! correct, and it is still an argument about code rather than about numbers.
//! The issue said what would settle it — *"evaluate each path's ρ-gradient at a
//! saturated ρ and look for the 1.3333e-7 floor"* — and
//! [`gam_solve::rho_optimizer::soft_rho_guard_floor`] is that check, with a
//! positive control (`the_floor_classifier_reports_carried_on_the_live_mixture_sas_criterion`,
//! gam-solve) proving it can see a floor when one is there.
//!
//! What a "carried" verdict here would have meant: every railed coordinate of
//! every gamlss, spatial-adaptive, and custom-family fit carrying a standing
//! `|Pg| ≥ w·a = 1.3333e-7` that no amount of convergence clears, and three more
//! objectives owing the seam a publication.
//!
//! [`evaluate_custom_family_joint_hyper_owned`]: crate::psi_hyper::evaluate_custom_family_joint_hyper_owned

