//! #2027 — deterministic repro for the K≥2 whitened dictionary CO-COLLAPSE, and
//! the regression guard for the disjoint-subspace / ownership-anchor / reseed-
//! hysteresis fix.
//!
//! Two planted circles live in DISJOINT 2-planes of an ambient `p`-dim cloud; the
//! per-column-standardized ("whitened") target is their sum, so a faithful K=2
//! reconstruction REQUIRES both atoms to carry signal on different subspaces.
//! Before the fix the joint decoder refit at the co-collapse reseed re-spread one
//! residual direction across both atoms and the gate let them trade rows, so the
//! dictionary re-symmetrised into a single shared basin and the reconstruction EV
//! collapsed to the no-signal level. With the greedy disjoint-subspace decoder
//! refit + soft row-ownership anchor + reseed cooldown the two atoms hold distinct
//! territories and the fit recovers a materially positive EV.

