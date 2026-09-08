//! zz_measure diagnostic (2026-07-10, #2234 blocker): every Python-entry
//! `sae_manifold_fit` on HEAD refuses to mint — the outer search descends,
//! then freezes with a bit-stable projected gradient (synthetic planted
//! circle: 63 iterations, objective 1.216074e2, |g_proj| = 1.381e0) and the
//! #2241 certified-termination contract refuses. Lane-consistency (#1224) and
//! every FD gate PASS, so this drives the SAME planted circle through the
//! plain RUST engine (seed builders + `OuterProblem`) to split the fault:
//!
//! - engine converges here  ⇒ the stall lives in the pyffi orchestration
//!   above the engine (topology/promotion/alternation), not the optimizer;
//! - engine stalls here     ⇒ run per-coordinate central differences of
//!   `eval_cost` against `eval`'s analytic gradient AT the stalled ρ and
//!   print both (the desync, if any, named coordinate by coordinate).

// `manifold/mod.rs` declares this module as
// `#[cfg(test)] mod tests_stall_diagnostic_2234;` — its single declaration. Saying so in-file
// makes the test scope a claim the compiler enforces rather than one the
// filename merely implies, which is what puts the fixture helpers below in
// the same scope as the `#[test]` fns they serve.
#![cfg(test)]

