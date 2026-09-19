//! Forward the engine's [`log`] records to Python `logging` under the
//! `gamfit` logger.
//!
//! The engine logs from whichever thread is doing the work: the Python thread
//! inside a `#[pyfunction]` (holding the GIL) or a rayon worker (not holding
//! it, while the Python thread that started the fit waits on it with the GIL
//! held). A record therefore never calls into Python where it was produced —
//! acquiring the GIL from a worker would deadlock. Records are queued instead
//! and handed to `logging` with the GIL held, in the order they were produced,
//! by whichever comes first:
//!
//! * a pending call scheduled with `Py_AddPendingCall`, which the interpreter
//!   runs on its main thread as soon as that thread is back in Python code;
//! * [`sync_log_level_from_python`], which the Python wrapper calls before
//!   every engine call, so a fit on a worker Python thread delivers the records
//!   of the previous call even while the main thread is blocked.
//!
//! The level filter follows the `gamfit` logger's effective level, which the
//! same wrapper call reports: until someone lowers that level to `DEBUG`, the
//! engine's `debug!`/`trace!` records are filtered at the macro and cost
//! nothing.

use std::collections::VecDeque;
use std::ffi::{c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use log::{Level, LevelFilter, Log, Metadata, Record};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Name of the Python logger the records go to.
const PYTHON_LOGGER_NAME: &str = "gamfit";

/// Python's numeric logging levels (`logging.ERROR`, `WARNING`, `INFO`,
/// `DEBUG`). Python defines no trace level; half of `DEBUG` is the value the
/// logging documentation's custom-level examples and most libraries use.
const PY_ERROR: i32 = 40;
const PY_WARNING: i32 = 30;
const PY_INFO: i32 = 20;
const PY_DEBUG: i32 = 10;
const PY_TRACE: i32 = PY_DEBUG / 2;

static LOGGER: PythonLogger = PythonLogger;
static PENDING: PendingQueue = Mutex::new(VecDeque::new());
static DRAIN_SCHEDULED: AtomicBool = AtomicBool::new(false);

type PendingQueue = Mutex<VecDeque<PendingRecord>>;

struct PendingRecord {
    level: Level,
    target: String,
    message: String,
}

struct PythonLogger;

impl Log for PythonLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        lock_queue(&PENDING).push_back(PendingRecord {
            level: record.level(),
            target: record.target().to_owned(),
            message: record.args().to_string(),
        });
        schedule_drain();
    }

    fn flush(&self) {}
}

fn lock_queue(queue: &PendingQueue) -> std::sync::MutexGuard<'_, VecDeque<PendingRecord>> {
    queue.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn schedule_drain() {
    if DRAIN_SCHEDULED.swap(true, Ordering::AcqRel) {
        return;
    }
    let queue = std::ptr::from_ref(&PENDING).cast_mut().cast::<c_void>();
    // SAFETY: `Py_AddPendingCall` is documented as callable from any thread
    // without an attached thread state. The argument points at the `PENDING`
    // static, which outlives every pending call, and `drain_pending_call` only
    // reads it through a shared reference.
    let status = unsafe { pyo3::ffi::Py_AddPendingCall(Some(drain_pending_call), queue) };
    if status != 0 {
        // The interpreter's pending-call queue is full. The record stays
        // queued; the next record or the next engine call delivers it.
        DRAIN_SCHEDULED.store(false, Ordering::Release);
    }
}

extern "C" fn drain_pending_call(queue: *mut c_void) -> c_int {
    // SAFETY: `schedule_drain` is the only scheduler, and it passes a pointer
    // to the `PENDING` static.
    let queue = unsafe { &*queue.cast::<PendingQueue>().cast_const() };
    // The interpreter runs pending calls on its main thread with the GIL held;
    // attaching here only takes the already-held thread state.
    Python::try_attach(|py| drain_pending(py, queue));
    0
}

/// Hand every queued record to the `gamfit` logger. A failure inside Python
/// logging is reported through `sys.unraisablehook` rather than raised: this
/// runs from a pending call, where an exception would surface in unrelated
/// user code.
fn drain_pending(py: Python<'_>, queue: &PendingQueue) {
    // Clear the flag before taking the queue: a record pushed after the take
    // then schedules a fresh drain instead of waiting for the next one.
    DRAIN_SCHEDULED.store(false, Ordering::Release);
    let records = std::mem::take(&mut *lock_queue(queue));
    if records.is_empty() {
        return;
    }
    if let Err(err) = emit_records(py, records) {
        err.write_unraisable(py, None);
    }
}

fn emit_records(py: Python<'_>, records: VecDeque<PendingRecord>) -> PyResult<()> {
    let logger = py
        .import("logging")?
        .call_method1("getLogger", (PYTHON_LOGGER_NAME,))?;
    for record in records {
        let extra = PyDict::new(py);
        extra.set_item("rust_target", record.target)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("extra", extra)?;
        logger.call_method(
            "log",
            (python_level(record.level), "%s", record.message),
            Some(&kwargs),
        )?;
    }
    Ok(())
}

fn python_level(level: Level) -> i32 {
    match level {
        Level::Error => PY_ERROR,
        Level::Warn => PY_WARNING,
        Level::Info => PY_INFO,
        Level::Debug => PY_DEBUG,
        Level::Trace => PY_TRACE,
    }
}

/// The engine filter that shows exactly the records a Python logger at
/// `python_level` would handle.
fn level_filter_for_python(python_level: i32) -> LevelFilter {
    if python_level <= PY_TRACE {
        LevelFilter::Trace
    } else if python_level <= PY_DEBUG {
        LevelFilter::Debug
    } else if python_level <= PY_INFO {
        LevelFilter::Info
    } else if python_level <= PY_WARNING {
        LevelFilter::Warn
    } else if python_level <= PY_ERROR {
        LevelFilter::Error
    } else {
        LevelFilter::Off
    }
}

/// Install the forwarding backend. Called once from the module initializer;
/// the filter starts closed and opens on the first
/// [`sync_log_level_from_python`].
pub(crate) fn install() {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Off);
    }
}

/// Match the engine's log filter to the `gamfit` logger's effective level and
/// deliver any queued records. The Python wrapper calls this before every
/// engine call with `logging.getLogger("gamfit").getEffectiveLevel()`.
///
/// The process monitor writes only `debug!` heartbeats, so it starts the first
/// time the level reaches `DEBUG` and never before: importing `gamfit` starts
/// no thread.
#[pyfunction]
pub(crate) fn sync_log_level_from_python(py: Python<'_>, effective_level: i32) {
    let filter = level_filter_for_python(effective_level);
    log::set_max_level(filter);
    if filter >= LevelFilter::Debug {
        gam_runtime::process_monitor::start();
    }
    drain_pending(py, &PENDING);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_levels_round_trip_through_the_engine_filter() {
        for level in [Level::Error, Level::Warn, Level::Info, Level::Debug, Level::Trace] {
            let filter = level_filter_for_python(python_level(level));
            assert_eq!(filter, level.to_level_filter(), "{level}");
        }
    }

    #[test]
    fn python_default_and_extreme_levels_map_to_the_expected_filters() {
        // `logging.WARNING` is Python's default effective level: the engine
        // logs nothing above debug, so a default process forwards nothing.
        assert_eq!(level_filter_for_python(PY_WARNING), LevelFilter::Warn);
        // `logging.CRITICAL` has no engine counterpart.
        assert_eq!(level_filter_for_python(PY_ERROR + PY_DEBUG), LevelFilter::Off);
        // `logging.NOTSET` on the root logger means "everything".
        assert_eq!(level_filter_for_python(0), LevelFilter::Trace);
        // A level between two standard ones shows the stricter engine level.
        assert_eq!(level_filter_for_python(PY_DEBUG + PY_TRACE), LevelFilter::Info);
    }
}
