//! #1023 task-2 radial-bias diagnostic: a K=1 ordered independent Beta--Bernoulli planted-circle fit through
//! the production outer engine, measuring the three discriminating numbers
//! (mean gate ζ, fitted_radius/data_radius, and λ_smooth vs the empirical-Bayes
//! optimum). The historical user-facing default `sae_manifold_fit(assignment="ordered_beta_bernoulli")`
//! routed K=1 through this gate path with a cold logit seed of 0 (the EM
//! residual seed is gated on K>1), so ζ started at σ(0)=0.5 and the joint fit had
//! to drive it back toward 1. A uniform radial contraction whose size tracks mean
//! ζ (with no harmonic-spectrum signature) is the gate defect (cause B); a
//! contraction that also suppresses higher harmonics is λ over-smoothing (cause
//! A). This test prints all three numbers and gates the production fixed seed at
//! 1% radius bias.

