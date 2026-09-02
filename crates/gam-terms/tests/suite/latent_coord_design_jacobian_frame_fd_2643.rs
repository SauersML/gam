//! FD gate for #2643: the latent-coordinate design Jacobian must be the
//! derivative of the design the fit actually builds.
//!
//! `LatentCoordDesignDerivative::{new_matern,new_duchon}` are handed the
//! metadata's `centers` (STANDARDIZED, `x / input_scale`) together with its
//! `length_scale` (ORIGINAL units), and evaluate both against the RAW latent
//! coordinates. Three frames meet in one kernel evaluation, and the result is
//! the analytic `∂X/∂t` that `LatentCoordDerivativeOp` uses to steer the joint
//! `[rho, latent]` REML directions.
//!
//! The ground truth here is not a hand-written formula — it is
//! `build_term_collection_design` itself, which is literally what the
//! latent-coordinate driver re-runs on every θ (`spatial_optimization.rs`
//! `ensure_theta` writes the raw latent values into `data` and rebuilds through
//! the frozen spec). Central-differencing that rebuild is therefore the
//! definition of the quantity the operator claims to supply.
//!
//! Why this test did not exist: nothing pinned `local_design_jacobian_row` at
//! all, and — this is the trap — the defect is INVISIBLE at `input_scale == 1`,
//! where the standardized and original frames coincide. Any fixture built on
//! unit-spread latents passes while the shipped code is wrong. The
//! `input_scale == 1` arm below is kept precisely so a future reader can see
//! that the sensitivity is to σ and to nothing else.

