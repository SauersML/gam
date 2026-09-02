//! #996 — local refinement around the mixture-ladder winner.
//!
//! The coarse [`MIXTURE_K_LADDER`] = [1, 2, 3, 5, 7, 9] cannot *name* a
//! planted k = 4, 6, or 8 truth; before refinement the rung returned the
//! nearest rung instead of the true order, and the order is part of the
//! scientific claim ("7 clusters" vs "circle"). These planted tests assert:
//!
//! * an off-ladder truth (k = 4 and k = 6 blobs at the same matched SNR as
//!   the existing planted races) is recovered as EXACTLY its planted order;
//! * an in-ladder truth (k = 7) still wins, with the refinement proving its
//!   bracket (both immediate neighbours fitted and worse) rather than
//!   creeping the order;
//! * on circle truth (no cluster structure) the refinement terminates with a
//!   bracketed small-order winner instead of walking the ladder upward.

// ---------------------------------------------------------------------------
// Deterministic RNG (fixed integer seed, no clock).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Planted generators at the same matched SNR as tests/topology_mixture_rung.rs.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

