// Split from the original oversized module; keep included in order.
include!("split_parts/part_000.rs");

// ── Submodule split ─────────────────────────────────────────────────────────
mod convergence;

mod damping;

mod edf;

mod gpu_dispatch;

mod log_link_working_state;

mod loop_driver;

mod penalty;

mod pls_solver;

mod reweight;

mod state;

include!("split_parts/part_001.rs");
include!("split_parts/part_002.rs");
