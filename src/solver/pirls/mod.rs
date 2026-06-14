// Concern-named fragments inlined into this module's namespace; keep in order.
include!("imports.rs");

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

include!("working_model_core.rs");
include!("tests.rs");
