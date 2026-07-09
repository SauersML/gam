//! Diagnostic capture of the first few flexible-link (SAS/mixture) OUTER
//! evaluations, for measurement tests that need the ANALYTIC θ-gradient the
//! outer optimizer actually received at its opening iterates (#1876).
//!
//! The Rust test harness installs no `log` subscriber, so the production
//! `[OUTER-FD-AUDIT]` / `[EXT-GRAD]` telemetry is dropped silently. Rather than
//! reconstruct a `RemlState` + `EvalShared` bundle from an external test (both
//! crate-internal), a measurement test can `enable()` this sink, run the fit
//! through the public `fit_gam` API, and `take()` the recorded
//! `(theta, cost, gradient)` tuples of the opening evals — directly reading the
//! ε/log_δ gradient component the optimizer saw at the ε=0 init.
//!
//! Disabled by default and gated behind a single relaxed atomic load, so the
//! optimizer hot path is unaffected in production.

use ndarray::Array1;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// One captured outer evaluation: the outer coordinate `theta = (ρ ‖ link)`, the
/// scalar cost, and the analytic outer gradient in the same layout.
#[derive(Clone, Debug)]
pub struct OuterEvalRecord {
    pub theta: Array1<f64>,
    pub cost: f64,
    pub gradient: Array1<f64>,
}

/// Maximum evaluations retained per capture window (opening iterates only).
const MAX_CAPTURED: usize = 8;

static ENABLED: AtomicBool = AtomicBool::new(false);

fn buffer() -> &'static Mutex<Vec<OuterEvalRecord>> {
    static BUFFER: OnceLock<Mutex<Vec<OuterEvalRecord>>> = OnceLock::new();
    BUFFER.get_or_init(|| Mutex::new(Vec::new()))
}

/// Start capturing outer evaluations, clearing any prior window.
pub fn enable_outer_eval_capture() {
    buffer().lock().expect("outer-eval capture buffer").clear();
    ENABLED.store(true, Ordering::Relaxed);
}

/// Stop capturing and drain the recorded opening evaluations (in eval order).
pub fn take_outer_eval_capture() -> Vec<OuterEvalRecord> {
    ENABLED.store(false, Ordering::Relaxed);
    std::mem::take(&mut *buffer().lock().expect("outer-eval capture buffer"))
}

/// Record one outer evaluation when capture is enabled (no-op otherwise). Only
/// the first [`MAX_CAPTURED`] evaluations of a window are retained.
pub(crate) fn record_outer_eval(theta: &Array1<f64>, cost: f64, gradient: &Array1<f64>) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let mut b = buffer().lock().expect("outer-eval capture buffer");
    if b.len() < MAX_CAPTURED {
        b.push(OuterEvalRecord {
            theta: theta.clone(),
            cost,
            gradient: gradient.clone(),
        });
    }
}
