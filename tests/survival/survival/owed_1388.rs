//! Owed-work regression gate for GitHub issue #1388 (survival marginal-slope).
//!
//! ISSUE — two failures on the tiny `cirrhosis_survival` / `heart_failure_survival`
//! benchmarks, both rooted in the SAME degenerate geometry: once every
//! categorical level expands into its own column the joint marginal-slope design
//! becomes UNDER-DETERMINED (`p_joint > n`), so the unpenalized joint Jacobian
//! rank is capped at `min(n, p_joint) < p_joint`. In that regime:
//!
//!   1. "post-T construction rank invariant violated: rank(T)=53 != dim=31" — the
//!      post-T verifier ranked the BARE data Gram (no penalty augmentation, and a
//!      bespoke eigenvalue cutoff) while the channel-aware audit decides kept rank
//!      from the PENALTY-AUGMENTED joint Gram with its own `rank_of_gram`
//!      convention. A penalty-anchored direction the audit legitimately keeps was
//!      demoted by the verifier → a FALSE "post-T rank invariant violated" abort.
//!
//!   2. Stack overflow in `rust_gamlss_survival` right after `[STAGE] runtime
//!      threads` — the downstream canonicalisation fan-out over the many
//!      penalty/level blocks of an under-determined joint.
//!
//! FIX (landed on `origin/main` in `src/identifiability/canonical.rs`):
//!   * `1573681d3` / `2e2871e75` — the post-T rank invariant now ranks the
//!     PENALTY-AUGMENTED reduced design `J_can` with the audit's OWN
//!     `rank_of_gram` convention and certifies `rank(J_can) == p_total_red`
//!     (== audit_kept_rank), so an under-determined-but-penalty-identifiable
//!     `p_joint > n` joint no longer trips the invariant.
//!   * `eb51fa190` (#1391) — convention-robust post-T invariant
//!     `rank(J_can) == min(rank(J_pre), p_total_red)`.
//!   * `15ae5d389` (#1388) — preflight WARN surfacing the `p_joint > n`
//!     under-determination so the failure is diagnosable from the log.
//!
//! CERTIFICATE (public API only — `canonicalize_for_identifiability`):
//! reconstruct the issue's geometry at unit scale — a channel-aware survival
//! marginal-slope joint whose total column count EXCEEDS the row count
//! (`p_joint > n`) via categorical level expansion in a lower-priority block,
//! anchored by a geometry-owning `stacked_design` time block. This is the exact
//! regime that produced both the "post-T rank invariant violated" abort and the
//! canonicalisation fan-out overflow.
//!
//! The whole canonicalisation is driven on a DELIBERATELY BOUNDED-STACK worker
//! thread (1 MiB). A genuine UNBOUNDED recursion / data-proportional descent in
//! the fan-out would overflow that worker and abort the process (the `join`
//! below observes a panic-free completion), so the bounded stack is the active
//! guard for failure #2; the `Ok(..)` + post-T assertions are the guard for #1.
//! Were the rank-invariant fix reverted, `canonicalize_for_identifiability`
//! would return `Err(DimensionMismatch { reason: "... post-T rank invariant
//! violated ..." })` and the test would fail.

