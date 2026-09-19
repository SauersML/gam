//! One-shot engagement reports for device routes.
//!
//! The #1551 "GPU 0%" class: an `Auto` run that silently declines the device
//! and falls back to the CPU otherwise leaves no trace of WHY. Every router
//! reports the first engagement and the first decline it sees, once per
//! process per route — the routes are per-minibatch or per-iterate, so an
//! unconditional line would flood the fit log with thousands of identical
//! entries. Routed through `log::debug!`, the repo's sanctioned diagnostics
//! path, so an initialised `log` backend lands it in the job logs.

use std::sync::Mutex;

/// Routes that have already reported, by `(route, engaged)`.
static REPORTED: Mutex<Vec<(&'static str, bool)>> = Mutex::new(Vec::new());

/// Report `route`'s first engagement (`engaged == true`) or first decline
/// (`engaged == false`, with `fallback` naming what runs instead — "falling
/// back to CPU", "CPU reference"); later calls with the same `(route,
/// engaged)` are silent.
pub fn note_route_engagement(
    route: &'static str,
    fallback: &'static str,
    engaged: bool,
    detail: &str,
) {
    let first = match REPORTED.lock() {
        Ok(mut reported) => {
            if reported.contains(&(route, engaged)) {
                false
            } else {
                reported.push((route, engaged));
                true
            }
        }
        // A poisoned registry only means a reporter panicked mid-push; losing
        // the once-only guarantee is preferable to losing the report.
        Err(_) => true,
    };
    if !first {
        return;
    }
    if engaged {
        log::debug!("[{route}] device ENGAGED: {detail}");
    } else {
        log::debug!("[{route}] device DECLINED - {fallback}: {detail}");
    }
}
