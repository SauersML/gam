//! Test-side backend for the diagnostics production already emits.
//!
//! Production instruments its solvers through the `log` facade — the BMS
//! intercept-solve counters, the GL-ladder rung histogram, the cell-moment cache
//! stats, the certificate-bound discriminator. Every one of those is a
//! `log::debug!`, and **the `log` facade drops every record until some binary
//! installs a backend**. No test binary in this workspace installed one, so all
//! of that instrumentation has been running and producing nothing, in unit tests
//! and integration tests alike. Two lanes independently spent hours re-deriving
//! facts these lines already carried (#2472's non-terminating marginal-slope
//! cluster; the `[CERTIFICATE-BOUND]` discriminator). This module is the missing
//! half.
//!
//! It lives in `gam-runtime` rather than `gam-test-support` for the reason that
//! crate's own header gives: a helper owning no model-layer type belongs in a
//! leaf, so depending on it does not drag the solver stack into an unrelated
//! test build. `gam-runtime` already owns the observability modules (`span`,
//! `process_monitor`, `loop_progress`) and already depends on `log`.
//! `gam-test-support` re-exports it for consumers that use that path.
//!
//! Following the workspace convention for `test_support` modules, this is a
//! plain always-compiled `pub mod`: a `cfg(test)` gate would make it invisible
//! to exactly the downstream integration binaries that need it.

use std::io::Write;
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Records that could not be written to stderr.
///
/// A logger that silently fails to emit is the exact defect this module exists
/// to remove, so the failures are counted rather than discarded — panicking
/// inside `log::Log::log` is not an option (it runs from arbitrary call sites,
/// including ones holding locks), but going quiet is what got us here.
/// [`diagnostic_write_failures`] lets a caller assert its diagnostics actually
/// reached the stream.
static WRITE_FAILURES: AtomicUsize = AtomicUsize::new(0);

/// Writes every record at or above `Info` straight to stderr, one line at a
/// time, flushing each.
///
/// Per-record flushing is the whole point rather than an inefficiency: the
/// diagnostics worth reading belong to runs that do not finish. A test killed at
/// the per-test cap, or a fit abandoned after hours, must leave its trace
/// behind — buffering it into a summary that never prints is how 17 core-hours
/// produced zero lines.
struct StderrDiagnosticLogger;

impl log::Log for StderrDiagnosticLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }

    fn log(&self, record: &log::Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        // `record.args()` is formatted through `Display`, never `Debug`: the
        // ban scanner rejects `{:?}` in `eprintln!`/`eprint!`, and a debug-format
        // diagnostic is unreadable anyway.
        let mut stderr = std::io::stderr().lock();
        let written = writeln!(stderr, "[{}] {}", record.level(), record.args());
        // Flush per record: the diagnostics worth reading belong to runs that do
        // not finish, so a buffered line is a lost line.
        let flushed = stderr.flush();
        if written.is_err() || flushed.is_err() {
            WRITE_FAILURES.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn flush(&self) {
        if std::io::stderr().flush().is_err() {
            WRITE_FAILURES.fetch_add(1, Ordering::Relaxed);
        }
    }
}

static DIAGNOSTIC_LOGGER: StderrDiagnosticLogger = StderrDiagnosticLogger;
static INSTALL_ONCE: Once = Once::new();

/// Install the stderr diagnostic backend for this process, once.
///
/// Call it from any test that wants to read production's `log::debug!` output.
/// It is safe to call from every test in a binary and from several binaries at
/// once: `log::set_logger` may only be called once per process, so the work is
/// behind a [`Once`], and a losing call is treated as success — some other
/// backend is already receiving the records, which is the outcome the caller
/// wanted.
///
/// The level is chosen in code (`Info`) and is deliberately not configurable
/// from the environment: the workspace bans environment-dependent behaviour, and
/// a diagnostic you have to know an env var to see is one nobody sees.
pub fn install_diagnostic_logger() {
    INSTALL_ONCE.call_once(|| {
        if log::set_logger(&DIAGNOSTIC_LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Debug);
        }
    });
}

/// How many records this backend failed to write.
///
/// Nonzero means diagnostics were lost, which for a run being read for its
/// instrumentation is a failed measurement rather than a cosmetic problem.
pub fn diagnostic_write_failures() -> usize {
    WRITE_FAILURES.load(Ordering::Relaxed)
}

/// Replay a cgroup-constrained memory environment without needing the machine
/// to be in it.
///
/// The two knobs are deliberately the two that #2684 conflated:
///
/// * `cgroup_limit_bytes` is the job's hard ceiling — the CAPACITY. It is a
///   property of how the process was launched and does not move while it runs.
/// * `cgroup_current_bytes` is how much of that ceiling is charged RIGHT NOW —
///   the load. It moves continuously, and on any cgroup that has done file I/O
///   it sits near the ceiling because page cache fills whatever is free.
///
/// Holding the first fixed while sweeping the second is what lets a test ask
/// "does this decision depend on ambient load?" as an assertion instead of an
/// argument. The returned observation is built by the same constructor the live
/// probe uses, so its derived `working_set`/`available` arithmetic is the
/// shipped arithmetic and not a second implementation that could agree with the
/// first only by luck.
///
/// The host numbers are taken as parameters too, because the host's own free
/// memory is the other half of the conflation: the incident this exists for had
/// hundreds of GB free on the host while the job's cgroup reported kilobytes.
pub fn simulated_cgroup_memory_environment(
    host_available_bytes: u64,
    host_total_bytes: u64,
    cgroup_limit_bytes: u64,
    cgroup_current_bytes: u64,
) -> crate::resource::MemoryAvailability {
    let cgroup = crate::resource::CgroupMemoryAvailability::from_consistent_counters(
        "/simulated/cgroup/memory",
        cgroup_limit_bytes,
        cgroup_current_bytes,
        0,
        6,
    )
    .expect("simulated cgroup counters must be internally consistent");
    crate::resource::MemoryAvailability::from_observation(
        host_available_bytes,
        host_total_bytes,
        crate::resource::CgroupMemoryObservation::V1Limited(cgroup),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_is_idempotent_and_enables_info() {
        install_diagnostic_logger();
        install_diagnostic_logger();
        // The facade only forwards records at or below the max level; if the
        // install silently did nothing, production's `log::debug!` diagnostics
        // stay invisible and every caller of this module is misled.
        assert!(
            log::max_level() >= log::LevelFilter::Debug,
            "diagnostic logger must leave Info records enabled, got {}",
            log::max_level()
        );
        assert!(log::log_enabled!(log::Level::Debug));
    }
}
