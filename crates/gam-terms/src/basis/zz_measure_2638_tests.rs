//! #2638 follow-up: gates on [`duchon_resolve_chart`], the seam that lets the
//! ψ-derivative entry point ANSWER for a cold spec instead of refusing it.
//!
//! `1f9171850` closed #2638 by routing the three ψ-derivative builders through
//! `duchon_frozen_radial_chart`, which refuses a spec carrying no frozen `V`
//! while the constrained kernel block is non-empty. That is the right answer for
//! those three builders — they receive `centers`, not `data`, so they *cannot*
//! solve the `Ω_c v = μ G_c v` eigenproblem that decides `V`.
//!
//! `build_duchon_basis_log_kappa_derivatives(data, spec)` does receive the data,
//! and its documented job is to return the ψ-jet of `build_duchon_basis(data,
//! spec)`. For a cold spec that forward build is perfectly well defined — it
//! adopts a fresh `V` — so the derivative entry point has no reason to refuse:
//! it resolves the same chart and differentiates in it.
//!
//! What is gated here:
//!
//!  1. `duchon_resolve_chart` is behaviour-preserving — `build_duchon_basis` on
//!     the RESOLVED spec reproduces the cold build bit-for-bit. That pins all
//!     five resolved decisions at once (centers, effective null-space order,
//!     seeded anisotropy, adopted `V`, identifiability transform `T`).
//!  2. `T` is read off the `V`-ROTATED design. The pre-#2638 derivative context
//!     derived it from the un-rotated one, which constrains a different function
//!     space.
//!  3. The cold-spec ψ-jet matches a finite difference of the forward taken at
//!     the resolved chart, on EVERY penalty block. Before the entry point
//!     resolved, this same call returned the raw-`Z`-chart jet: 32× the true
//!     Primary jet and 242× too small on OperatorMass (measured; see the table
//!     in the commit that added this file).
//!  4. The chart-motion decomposition itself, printed rather than asserted —
//!     the evidence that the residual #2638 reported is `|FD_cold − FD_frozen|`
//!     and not a dropped term.

#![cfg(test)]

