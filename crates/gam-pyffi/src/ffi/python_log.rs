//! Forward the engine's [`log`] records to Python `logging` under the
//! `gamfit` logger, while the engine call that produced them is still running.
//!
//! The engine logs from whichever thread is doing the work: the Python thread
//! inside a `#[pyfunction]` or a rayon worker. A record never calls into Python
//! where it was produced, because that thread may be a worker the GIL holder is
//! waiting on. Records are queued instead, and one delivery thread per process
//! hands them to `logging` in the order they were produced. It attaches to the
//! interpreter as soon as the GIL is free, which it is for the whole of every
//! engine call that detaches from Python (every fit does). So a fit that never
//! returns, for example one killed at a caller's time cap, has already
//! delivered everything it logged.
//!
//! [`sync_log_level_from_python`], which the Python wrapper calls before every
//! engine call, also delivers whatever is queued, so the records of one call
//! are with the logger before the next call starts.
//!
//! The level filter follows the `gamfit` logger's effective level, which the
//! same wrapper call reports: until someone lowers that level to `DEBUG`, the
//! engine's `debug!`/`trace!` records are filtered at the macro and cost
//! nothing.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};

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

/// No process: nothing is draining, or no delivery thread runs.
const NO_PROCESS: u32 = 0;

static LOGGER: PythonLogger = PythonLogger;
static PENDING: Mutex<Pending> = Mutex::new(Pending {
    records: VecDeque::new(),
    delivery_requested: false,
});
static DELIVERY_REQUESTED: Condvar = Condvar::new();
/// The process whose delivery thread is running. A forked child inherits the
/// parent's id but not its threads, so it starts its own.
static DELIVERY_PROCESS: AtomicU32 = AtomicU32::new(NO_PROCESS);
/// The process one of whose threads is handing records to `logging`. Only one
/// thread emits at a time, so records keep their order even when a handler
/// releases the GIL mid-batch (every stream write does). A value that is not
/// this process's id was inherited through a fork and is taken over.
static DRAINING_PROCESS: AtomicU32 = AtomicU32::new(NO_PROCESS);

struct Pending {
    records: VecDeque<PendingRecord>,
    /// Set by every push and cleared by the delivery thread when it wakes, so
    /// the thread sleeps while another thread is draining instead of spinning
    /// on a queue that thread is about to empty.
    delivery_requested: bool,
}

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
        let queued = PendingRecord {
            level: record.level(),
            target: record.target().to_owned(),
            message: record.args().to_string(),
        };
        {
            let mut pending = lock_pending();
            pending.records.push_back(queued);
            pending.delivery_requested = true;
        }
        DELIVERY_REQUESTED.notify_one();
        ensure_delivery_thread();
    }

    fn flush(&self) {}
}

fn lock_pending() -> MutexGuard<'static, Pending> {
    PENDING.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Start this process's delivery thread unless it is already running.
fn ensure_delivery_thread() {
    let process = std::process::id();
    let running = DELIVERY_PROCESS.load(Ordering::Acquire);
    if running == process
        || DELIVERY_PROCESS
            .compare_exchange(running, process, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
    {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("gamfit-log".to_owned())
        .spawn(deliver_until_the_interpreter_stops);
    if spawned.is_err() {
        // The records stay queued; the next record retries the spawn and the
        // next engine call delivers them either way.
        DELIVERY_PROCESS.store(NO_PROCESS, Ordering::Release);
    }
}

/// The delivery thread: sleep until a record is queued, then attach to the
/// interpreter (waiting for the GIL, never holding the queue lock while it
/// waits) and deliver. It ends when the interpreter can no longer be attached
/// to, which is at shutdown.
fn deliver_until_the_interpreter_stops() {
    loop {
        {
            let mut pending = lock_pending();
            while !pending.delivery_requested {
                pending = DELIVERY_REQUESTED
                    .wait(pending)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            pending.delivery_requested = false;
        }
        if Python::try_attach(drain_pending).is_none() {
            DELIVERY_PROCESS.store(NO_PROCESS, Ordering::Release);
            return;
        }
    }
}

/// Hand every queued record to the `gamfit` logger. A failure inside Python
/// logging is reported through `sys.unraisablehook` rather than raised: the
/// delivery thread has no caller to raise into, and an engine call must not
/// fail over a handler's error.
///
/// If another thread of this process is already emitting, this returns at
/// once: that thread empties the queue, and it looks again after releasing
/// its claim, so no record is left behind by the hand-over.
fn drain_pending(py: Python<'_>) {
    let process = std::process::id();
    loop {
        let draining = DRAINING_PROCESS.load(Ordering::Acquire);
        if draining == process
            || DRAINING_PROCESS
                .compare_exchange(draining, process, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return;
        }
        loop {
            let records = std::mem::take(&mut lock_pending().records);
            if records.is_empty() {
                break;
            }
            if let Err(err) = emit_records(py, records) {
                err.write_unraisable(py, None);
            }
        }
        DRAINING_PROCESS.store(NO_PROCESS, Ordering::Release);
        // A record queued between the last take and the release found the
        // claim still held and was left to this thread.
        if lock_pending().records.is_empty() {
            return;
        }
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
#[pyfunction]
pub(crate) fn sync_log_level_from_python(py: Python<'_>, effective_level: i32) {
    log::set_max_level(level_filter_for_python(effective_level));
    drain_pending(py);
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
