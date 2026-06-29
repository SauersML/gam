# Batch triage notes — agent gam-closed-598-777 (issues #598–#777)

## Confirmed improperly-closed (reopen + fix)
- **#759** — `trace_product_sparse` parallelization (commit b7879667b) was REVERTED to a
  serial loop by the gam-linalg crate-extraction refactor (a80fe6943). The fix was lost.
  Current `crates/gam-linalg/src/sparse_exact.rs:1078` is a serial scan. `get()`'s
  exact-column cache is `Mutex`-guarded so the rayon reduction is sound. FIXING NOW.

## Other strong candidates (under investigation)
- #638 — PIRLS adaptive early-exit: claimed but EMA/predicate not found in reweight.rs.
- #667 — multinomial via fit(family='multinomial'): formula entry point still gated w/ error.
- #650 — README quickstart still has n=4 rows (doc never updated).

## Triage method
Parallel Explore agents over clusters; verify claimed fix against current source, not the
closing comment. Most closes in this range are SOLID (real root-cause fixes w/ tests).
