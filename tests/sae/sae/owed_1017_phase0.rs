//! Owed-work regression gate for #1017 Phase 0 (process-parallel candidates).
//!
//! The SAE driver fits several INDEPENDENT candidates — topology candidates,
//! layer-ladder charts, checkpoint trajectories — but historically walked them
//! sequentially, leaving a multi-core box idle. Phase 0 is the zero-engine-code
//! win: fan the independent candidate fits out over rayon at the driver level,
//! capped so the heavy per-candidate solves don't oversubscribe. (The separate
//! device-residency leg of #1017 — the GPU routing seam — is pinned by
//! `tests/owed_1017.rs`; this file owns the CPU driver-parallelism leg.)
//!
//! This gate owns the LAYER-LADDER candidate loop:
//! `atom_transport_ladder_reports` (src/terms/sae/identifiability.rs). Each
//! atom's #1096 transport ladder is an independent, pure fit (read the shared
//! model by index, build that atom's canonical per-layer coordinates, run the
//! pure `transport_ladder` solve), so the reports now fan across rayon. The
//! parallelization must preserve results bit-identically to the sequential
//! walk. Runtime is measured in the benchmark harnesses, never used as a test
//! oracle or a fit deadline.
//!
//! Two properties are pinned:
//!
//! 1. PARITY. The parallel dispatch (many inputs, called from outside any rayon
//!    worker) and the sequential dispatch (the SAME call forced down the
//!    sequential body by invoking it from INSIDE a rayon worker, where the
//!    function's `current_thread_index().is_none()` nested-rayon guard keeps it
//!    sequential) must produce byte-identical reports, in input order. The
//!    transport solve is deterministic, so the two `Debug` renderings must match
//!    exactly — a regression that reordered results or raced shared state would
//!    diverge here. First-by-index error recovery is also pinned: an invalid
//!    input at a known position surfaces the SAME error string from both paths,
//!    even though the parallel path evaluates later inputs that the sequential
//!    `?`-walk would have skipped (the per-atom body is pure and cannot panic on
//!    shape-valid input, so running them is harmless).
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

