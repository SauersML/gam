//! F4 reviewer acceptance — the LayerNorm-sphere ambient path removes the
//! spurious curvature a flat-space circle fit invents from norm variation.
//!
//! The residual stream post-LayerNorm is a sphere × scale product: the model
//! reads only the direction `u = x/‖x‖`, and the per-token norm is a nuisance the
//! LayerNorm discards. Reviewer F4's charge: a curved atom fitted to reconstruct
//! the *flat* activation `x` absorbs that norm variation into its decoder as
//! spurious higher-harmonic curvature, and its reconstruction certificate — which
//! assumes an additive isotropic-Gaussian residual in the flat metric — is
//! conditional on an assumption the radial (norm-direction) residual violates.
//!
//! These tests plant data with a KNOWN pure-harmonic-1 direction on the sphere
//! and an INDEPENDENT lognormal norm, then run the REAL K=1 circle fit
//! ([`SaeManifoldTerm::run_joint_fit_arrow_schur`]) two ways:
//!   * FLAT   — reconstruct `x` directly (ambient Euclidean);
//!   * SPHERE — reconstruct [`ln_sphere_project`]`(x)` (atom as an LN-sphere
//!              submanifold).
//! and show that (1) the flat decoder carries large higher-harmonic (spurious
//! curvature) energy that GROWS with the norm variation, while the sphere decoder
//! stays pure harmonic-1; and (2) the flat residual is dominated by the radial
//! norm-direction term, while the sphere residual is not. The true generator has
//! exactly zero of both, so anything the flat fit reports is an artifact of the
//! wrong ambient metric.

