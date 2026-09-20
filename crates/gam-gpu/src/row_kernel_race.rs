//! Measured admission for row kernels (gam#3024).
//!
//! A row kernel runs one independent program per row, so both of its
//! executors cost `T(n) = a + b·n`: a per-call overhead `a` (probe, transfer,
//! launch on the device; scheduling on the CPU) and a per-row cost `b` that
//! depends on the kernel, the per-row widths and the CPU worker count. `auto`
//! selects the device exactly when `T_device(n) < T_cpu(n)` for this kernel on
//! this host, and no other kernel's crossover says anything about that: the
//! xtwx Gram's measured crossover, which the row kernels used to borrow, is a
//! cuBLAS number, while a row kernel is FP64 work bound by per-row polynomial
//! and transcendental evaluation.
//!
//! So each kernel measures its own two executors on the workload in front of
//! it. The first `auto` call for a shape the process has not timed races
//! them: the CPU executor once, the device executor twice with the warm second
//! call timed (the first pays compilation, allocation and first touch, once per
//! process), and the faster is recorded. That call returns the CPU result, the
//! one `gpu="off"` computes. Later calls read the record: an exact point at the
//! same row count, otherwise each executor's `a + b·n` fitted through the
//! points timed at two or more row counts for the same kernel, widths and
//! worker count; a shape neither answers is raced.
//!
//! The choice is a timing measurement, so near a crossover two runs can pick
//! different executors, and results can differ between runs at roundoff. The
//! two executors compute the same quantity. `gpu="off"` and `gpu="required"`
//! never race and are the deterministic choices.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::GpuKernel;

/// A row kernel's workload as its race is keyed: the kernel, the row count,
/// the per-row widths that set a row's work (each kernel names its own, unused
/// slots zero), and the worker count of the CPU executor, which sets its time.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RowKernelShape {
    pub kernel: GpuKernel,
    pub rows: usize,
    pub widths: [usize; 4],
    pub threads: usize,
}

impl RowKernelShape {
    /// The shapes whose timings share one `a + b·n` law: everything but the
    /// row count.
    fn law(&self) -> (GpuKernel, [usize; 4], usize) {
        (self.kernel, self.widths, self.threads)
    }
}

/// Which executor a measured shape selects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MeasuredExecutor {
    Cpu,
    Device,
}

/// One race: the row count and each executor's time at it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RaceTiming {
    rows: usize,
    cpu_seconds: f64,
    device_seconds: f64,
}

type Timings = HashMap<(GpuKernel, [usize; 4], usize), Vec<RaceTiming>>;

fn timings() -> &'static Mutex<Timings> {
    static TIMINGS: OnceLock<Mutex<Timings>> = OnceLock::new();
    TIMINGS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The recorded choice for `shape`, or `None` when it has to be raced.
pub(crate) fn measured_executor(shape: &RowKernelShape) -> Option<MeasuredExecutor> {
    let timings = timings()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    timings
        .get(&shape.law())
        .and_then(|points| executor_from_timings(points, shape.rows))
}

/// The executor the timed `points` select at `rows`: an exact point when one
/// was raced at `rows`, otherwise each executor's least-squares `a + b·n`
/// through the points, which needs two distinct row counts and a nonnegative
/// fitted per-row cost for both executors (a negative one is timing noise
/// between close row counts, and another race settles it).
fn executor_from_timings(points: &[RaceTiming], rows: usize) -> Option<MeasuredExecutor> {
    let faster = |cpu: f64, device: f64| {
        if device < cpu {
            MeasuredExecutor::Device
        } else {
            MeasuredExecutor::Cpu
        }
    };
    if let Some(point) = points.iter().find(|point| point.rows == rows) {
        return Some(faster(point.cpu_seconds, point.device_seconds));
    }
    let (cpu_overhead, cpu_per_row) = affine_fit(points, |point| point.cpu_seconds)?;
    let (device_overhead, device_per_row) = affine_fit(points, |point| point.device_seconds)?;
    if cpu_per_row < 0.0 || device_per_row < 0.0 {
        return None;
    }
    let n = rows as f64;
    Some(faster(
        cpu_overhead + cpu_per_row * n,
        device_overhead + device_per_row * n,
    ))
}

/// Least-squares `(a, b)` of `t = a + b·n` over `points`, or `None` when
/// they span fewer than two row counts.
fn affine_fit(points: &[RaceTiming], seconds: impl Fn(&RaceTiming) -> f64) -> Option<(f64, f64)> {
    let count = points.len() as f64;
    let mean_n = points.iter().map(|point| point.rows as f64).sum::<f64>() / count;
    let mean_t = points.iter().map(&seconds).sum::<f64>() / count;
    let spread = points
        .iter()
        .map(|point| (point.rows as f64 - mean_n).powi(2))
        .sum::<f64>();
    if spread <= 0.0 {
        return None;
    }
    let covariance = points
        .iter()
        .map(|point| (point.rows as f64 - mean_n) * (seconds(point) - mean_t))
        .sum::<f64>();
    let per_row = covariance / spread;
    Some((mean_t - per_row * mean_n, per_row))
}

/// Record one race's timings for `shape`, replacing an earlier race at the
/// same row count.
pub(crate) fn record(shape: &RowKernelShape, cpu_seconds: f64, device_seconds: f64) {
    let mut timings = timings()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let points = timings.entry(shape.law()).or_default();
    points.retain(|point| point.rows != shape.rows);
    points.push(RaceTiming {
        rows: shape.rows,
        cpu_seconds,
        device_seconds,
    });
}

/// Race `shape`'s two executors once and record the faster; return the CPU
/// executor's result, the quantity `gpu="off"` computes. `device` runs the
/// device executor and discards its output; it runs twice and its warm second
/// call is the one timed. A device error is the device path's fault and is
/// returned, as it is for a selected device.
pub fn race_row_kernel<T, E>(
    shape: RowKernelShape,
    cpu: impl FnOnce() -> Result<T, E>,
    mut device: impl FnMut() -> Result<(), E>,
) -> Result<T, E> {
    let started = Instant::now();
    let out = cpu()?;
    let cpu_seconds = started.elapsed().as_secs_f64();
    let cold = Instant::now();
    device()?;
    let cold_seconds = cold.elapsed().as_secs_f64();
    let warm = Instant::now();
    device()?;
    let device_seconds = warm.elapsed().as_secs_f64();
    record(&shape, cpu_seconds, device_seconds);
    log::debug!(
        "[GPU row-kernel race] kernel={} rows={} widths={:?} threads={} cpu={cpu_seconds:.6}s \
         device_cold={cold_seconds:.6}s device_warm={device_seconds:.6}s selects={}",
        shape.kernel.as_str(),
        shape.rows,
        shape.widths,
        shape.threads,
        if device_seconds < cpu_seconds {
            "device"
        } else {
            "cpu"
        },
    );
    Ok(out)
}

/// One call of a row kernel that `auto` admitted to its race and that runs as
/// a sequence of tiles: the executor recorded for `shape` when a race of its
/// law has since settled it (an earlier tile of the same pass), otherwise a
/// race of this tile. `device` returns the device executor's value, which is
/// used when the device is the recorded faster.
pub fn run_measured_row_kernel<T, E>(
    shape: RowKernelShape,
    cpu: impl FnOnce() -> Result<T, E>,
    mut device: impl FnMut() -> Result<T, E>,
) -> Result<T, E> {
    match measured_executor(&shape) {
        Some(MeasuredExecutor::Cpu) => cpu(),
        Some(MeasuredExecutor::Device) => device(),
        None => race_row_kernel(shape, cpu, || device().map(|_| ())),
    }
}

/// The race of a row kernel whose CPU executor builds state once and then
/// applies against it many times, while the device executor re-reads its
/// inputs on every apply and keeps nothing: the SAE residual-curvature HVP,
/// whose CPU executor keeps per-state contractions that each CG apply reads.
/// Timing one apply would leave out the build the CPU choice pays and the
/// reuse it buys, so the race times the whole state: the CPU build, then both
/// executors on every apply against it. The device executor's first apply
/// also runs once untimed, which pays compilation, allocation and first touch
/// as [`race_row_kernel`]'s cold call does. Every apply returns the CPU
/// executor's value, and the totals are recorded when the state is dropped,
/// so the next state of this shape reads them. A state with no apply, or one
/// an executor faulted in, records nothing.
pub struct ReusedStateRace {
    shape: RowKernelShape,
    ledger: Mutex<ReusedStateLedger>,
}

struct ReusedStateLedger {
    cpu_seconds: f64,
    device_seconds: f64,
    applies: usize,
    faulted: bool,
}

impl ReusedStateRace {
    /// Build the CPU executor's state for `shape`, timed.
    pub fn build<C, E>(
        shape: RowKernelShape,
        cpu_state: impl FnOnce() -> Result<C, E>,
    ) -> Result<(Self, C), E> {
        let started = Instant::now();
        let state = cpu_state()?;
        let race = Self {
            shape,
            ledger: Mutex::new(ReusedStateLedger {
                cpu_seconds: started.elapsed().as_secs_f64(),
                device_seconds: 0.0,
                applies: 0,
                faulted: false,
            }),
        };
        Ok((race, state))
    }

    /// One apply against the state: both executors, timed; the CPU value.
    pub fn apply<T, E>(
        &self,
        cpu: impl FnOnce() -> Result<T, E>,
        mut device: impl FnMut() -> Result<(), E>,
    ) -> Result<T, E> {
        let cold = self.lock().applies == 0;
        let timed = (|| {
            let started = Instant::now();
            let out = cpu()?;
            let cpu_seconds = started.elapsed().as_secs_f64();
            if cold {
                device()?;
            }
            let started = Instant::now();
            device()?;
            Ok((out, cpu_seconds, started.elapsed().as_secs_f64()))
        })();
        let mut ledger = self.lock();
        match timed {
            Ok((out, cpu_seconds, device_seconds)) => {
                ledger.cpu_seconds += cpu_seconds;
                ledger.device_seconds += device_seconds;
                ledger.applies += 1;
                Ok(out)
            }
            Err(error) => {
                ledger.faulted = true;
                Err(error)
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ReusedStateLedger> {
        self.ledger
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for ReusedStateRace {
    fn drop(&mut self) {
        let ledger = self
            .ledger
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if ledger.faulted || ledger.applies == 0 {
            return;
        }
        record(&self.shape, ledger.cpu_seconds, ledger.device_seconds);
        log::debug!(
            "[GPU row-kernel race] kernel={} rows={} widths={:?} threads={} applies={} \
             cpu_build_and_applies={:.6}s device_applies={:.6}s selects={}",
            self.shape.kernel.as_str(),
            self.shape.rows,
            self.shape.widths,
            self.shape.threads,
            ledger.applies,
            ledger.cpu_seconds,
            ledger.device_seconds,
            if ledger.device_seconds < ledger.cpu_seconds {
                "device"
            } else {
                "cpu"
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timing(rows: usize, cpu_seconds: f64, device_seconds: f64) -> RaceTiming {
        RaceTiming {
            rows,
            cpu_seconds,
            device_seconds,
        }
    }

    /// An exact point decides its own row count; two row counts fix each
    /// executor's `a + b·n`, and a row count between or beyond them is decided
    /// from those laws, including a device that never wins because its
    /// per-row cost exceeds the CPU's.
    #[test]
    fn a_measured_law_decides_every_row_count_3024() {
        // Device: 0.1 s overhead, 1 µs per row; CPU: no overhead, 3 µs per row.
        // They cross at n = 0.1 / 2e-6 = 50,000 rows.
        let crossing = [timing(10_000, 0.03, 0.11), timing(100_000, 0.3, 0.2)];
        assert_eq!(
            executor_from_timings(&crossing, 10_000),
            Some(MeasuredExecutor::Cpu)
        );
        assert_eq!(
            executor_from_timings(&crossing, 100_000),
            Some(MeasuredExecutor::Device)
        );
        assert_eq!(
            executor_from_timings(&crossing, 40_000),
            Some(MeasuredExecutor::Cpu)
        );
        assert_eq!(
            executor_from_timings(&crossing, 60_000),
            Some(MeasuredExecutor::Device)
        );
        assert_eq!(
            executor_from_timings(&crossing, 10_000_000),
            Some(MeasuredExecutor::Device)
        );
        // The A40 shape of gam#3024: the device's per-row cost is 1.9x the
        // CPU's, so no row count admits it.
        let never = [timing(50_000, 0.63, 1.17), timing(100_000, 1.26, 2.30)];
        for rows in [1, 50_000, 75_000, 10_000_000] {
            assert_eq!(
                executor_from_timings(&never, rows),
                Some(MeasuredExecutor::Cpu),
                "rows={rows}"
            );
        }
    }

    /// One row count cannot fix a law, so another row count is raced; a
    /// negative fitted per-row cost is timing noise and is raced too.
    #[test]
    fn an_unfixed_law_is_raced_3024() {
        let one = [timing(50_000, 0.63, 1.17)];
        assert_eq!(
            executor_from_timings(&one, 50_000),
            Some(MeasuredExecutor::Cpu)
        );
        assert_eq!(executor_from_timings(&one, 60_000), None);
        let noisy = [timing(50_000, 0.63, 0.5), timing(50_100, 0.62, 0.5)];
        assert_eq!(executor_from_timings(&noisy, 80_000), None);
        assert_eq!(executor_from_timings(&[], 1), None);
    }

    /// The race returns the CPU executor's value, runs the device executor
    /// twice, records the shape, and propagates a device fault.
    #[test]
    fn a_race_returns_the_cpu_value_and_records_the_shape_3024() {
        let shape = RowKernelShape {
            kernel: GpuKernel::FinalInference,
            rows: 3_024,
            widths: [30, 24, 0, 0],
            threads: 1,
        };
        assert_eq!(measured_executor(&shape), None);
        let mut device_runs = 0;
        let value = race_row_kernel(
            shape,
            || Ok::<_, String>(7.5),
            || {
                device_runs += 1;
                Ok(())
            },
        )
        .expect("both executors succeed");
        assert_eq!(value, 7.5);
        assert_eq!(device_runs, 2, "the device runs cold and then warm");
        assert!(
            measured_executor(&shape).is_some(),
            "the race recorded its shape"
        );
        let other = RowKernelShape {
            widths: [30, 25, 0, 0],
            ..shape
        };
        assert_eq!(
            measured_executor(&other),
            None,
            "other widths are another law"
        );
        let fault = race_row_kernel(
            RowKernelShape {
                rows: 3_025,
                ..shape
            },
            || Ok::<_, String>(1.0),
            || Err("device fault".to_string()),
        );
        assert_eq!(fault, Err("device fault".to_string()));
    }

    /// A tiled pass races its first tile and reads the record on every later
    /// tile of the same shape.
    #[test]
    fn a_tiled_pass_races_once_per_shape_3024() {
        let shape = RowKernelShape {
            kernel: GpuKernel::FinalInference,
            rows: 3_048,
            widths: [30, 48, 0, 0],
            threads: 1,
        };
        let mut device_runs = 0;
        for _ in 0..3 {
            let value = run_measured_row_kernel(
                shape,
                || Ok::<_, String>(2.0),
                || {
                    device_runs += 1;
                    Ok(2.0)
                },
            )
            .expect("both executors succeed");
            assert_eq!(value, 2.0);
        }
        let first_tile = 2;
        assert!(
            device_runs == first_tile || device_runs == first_tile + 2,
            "the first tile races (two device calls), each later one runs the \
             recorded executor once; got {device_runs} device calls"
        );
        assert!(measured_executor(&shape).is_some());
    }

    /// A reused-state race times the CPU build plus every apply against the
    /// device's applies, records only when its state drops, returns the CPU
    /// value, and records nothing for a state an executor faulted in or that
    /// was never applied.
    #[test]
    fn a_reused_state_race_records_its_whole_state_on_drop_3024() {
        let shape = RowKernelShape {
            kernel: GpuKernel::FinalInference,
            rows: 3_348,
            widths: [33, 48, 0, 0],
            threads: 1,
        };
        let (race, kept) =
            ReusedStateRace::build(shape, || Ok::<_, String>(4.0)).expect("the build succeeds");
        let mut device_runs = 0;
        for _ in 0..3 {
            let value = race
                .apply(
                    || Ok::<_, String>(kept * 2.0),
                    || {
                        device_runs += 1;
                        Ok(())
                    },
                )
                .expect("both executors succeed");
            assert_eq!(value, 8.0);
        }
        assert_eq!(device_runs, 4, "one untimed cold call, then one per apply");
        assert_eq!(
            measured_executor(&shape),
            None,
            "nothing is recorded while the state lives"
        );
        drop(race);
        assert!(measured_executor(&shape).is_some());

        let unapplied = RowKernelShape {
            rows: 3_349,
            ..shape
        };
        drop(ReusedStateRace::build(unapplied, || Ok::<_, String>(())).expect("built"));
        assert_eq!(measured_executor(&unapplied), None);
        let faulted = RowKernelShape {
            widths: [33, 49, 0, 0],
            ..shape
        };
        let (race, ()) = ReusedStateRace::build(faulted, || Ok::<_, String>(())).expect("built");
        assert_eq!(
            race.apply(|| Ok(1.0), || Err("device fault".to_string())),
            Err("device fault".to_string())
        );
        drop(race);
        assert_eq!(measured_executor(&faulted), None);
    }
}
